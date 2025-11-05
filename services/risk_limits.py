"""Extended risk and limit management for PDT-safe trading"""

from __future__ import annotations

from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Set

from services.agents.messages import TradingMode



@dataclass
class TradingLimits:
    """Per-mode trading limits configuration"""
    max_trades_per_day: int = 6
    min_cooldown_minutes: int = 30  # Min time between trades for same ticker
    max_concurrent_positions: int = 3
    max_daily_loss_pct: float = 0.04  # 4%
    max_consecutive_losses: int = 2
    ticker_cooldown_after_exit_minutes: int = 60  # Avoid re-entry in chop


@dataclass
class RiskState:
    """Current risk state tracking"""
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    last_reset_date: date = field(default_factory=lambda: datetime.now().date())
    
    # Symbol -> last trade timestamp
    symbol_last_trade: Dict[str, datetime] = field(default_factory=dict)
    
    # Symbol -> exit timestamp (for cooldown)
    symbol_exit_times: Dict[str, datetime] = field(default_factory=dict)
    
    # Currently open positions
    open_positions: Set[str] = field(default_factory=set)


class RiskManager:
    """
    Extended risk manager with per-mode caps and cooldowns.
    Enforces daily loss limits and consecutive loss halts.
    """
    
    def __init__(self, mode_limits: Optional[Dict[TradingMode, TradingLimits]] = None):
        """
        Args:
            mode_limits: Per-mode limit configuration
        """
        # Default limits per mode
        default_limits = {
            TradingMode.SLOW_SCALPER: TradingLimits(
                max_trades_per_day=6,
                min_cooldown_minutes=30,
                max_concurrent_positions=3,
                max_daily_loss_pct=0.04,
                max_consecutive_losses=2,
                ticker_cooldown_after_exit_minutes=60
            ),
            TradingMode.MICRO_SWING: TradingLimits(
                max_trades_per_day=4,
                min_cooldown_minutes=60,
                max_concurrent_positions=5,
                max_daily_loss_pct=0.05,
                max_consecutive_losses=3,
                ticker_cooldown_after_exit_minutes=120
            ),
            TradingMode.STOCKS: TradingLimits(
                max_trades_per_day=10,
                min_cooldown_minutes=15,
                max_concurrent_positions=8,
                max_daily_loss_pct=0.06,
                max_consecutive_losses=3,
                ticker_cooldown_after_exit_minutes=30
            )
        }
        
        self.limits = mode_limits or default_limits
        self.state = RiskState()
    
    def reset_daily_state(self):
        """Reset daily counters"""
        today = datetime.now().date()
        if today != self.state.last_reset_date:
            self.state.daily_pnl = 0.0
            self.state.consecutive_losses = 0
            self.state.trades_today = 0
            self.state.last_reset_date = today
            self.state.symbol_last_trade.clear()
            self.state.symbol_exit_times.clear()
            logger.info("Daily risk state reset")
    
    def can_enter_trade(self, symbol: str, mode: TradingMode) -> tuple[bool, str]:
        """
        Check if we can enter a new trade.
        Returns (allowed, reason)
        """
        self.reset_daily_state()
        
        limits = self.limits.get(mode)
        if not limits:
            return False, f"No limits configured for mode {mode}"
        
        # Check daily loss limit
        if self.state.daily_pnl <= -abs(limits.max_daily_loss_pct * 100):
            return False, f"Daily loss limit reached ({limits.max_daily_loss_pct*100:.1f}%)"
        
        # Check consecutive losses
        if self.state.consecutive_losses >= limits.max_consecutive_losses:
            return False, f"Max consecutive losses reached ({limits.max_consecutive_losses})"
        
        # Check daily trade count
        if self.state.trades_today >= limits.max_trades_per_day:
            return False, f"Max daily trades reached ({limits.max_trades_per_day})"
        
        # Check concurrent positions
        if len(self.state.open_positions) >= limits.max_concurrent_positions:
            return False, f"Max concurrent positions reached ({limits.max_concurrent_positions})"
        
        # Check symbol cooldown (min time between trades)
        if symbol in self.state.symbol_last_trade:
            last_trade = self.state.symbol_last_trade[symbol]
            minutes_since = (datetime.now() - last_trade).total_seconds() / 60
            if minutes_since < limits.min_cooldown_minutes:
                return False, f"Symbol cooldown active ({limits.min_cooldown_minutes - minutes_since:.0f} min remaining)"
        
        # Check post-exit cooldown
        if symbol in self.state.symbol_exit_times:
            exit_time = self.state.symbol_exit_times[symbol]
            minutes_since_exit = (datetime.now() - exit_time).total_seconds() / 60
            if minutes_since_exit < limits.ticker_cooldown_after_exit_minutes:
                return False, f"Post-exit cooldown active ({limits.ticker_cooldown_after_exit_minutes - minutes_since_exit:.0f} min remaining)"
        
        return True, "OK"
    
    def record_entry(self, symbol: str, mode: TradingMode):
        """Record a trade entry"""
        self.state.trades_today += 1
        self.state.symbol_last_trade[symbol] = datetime.now()
        self.state.open_positions.add(symbol)
        logger.info(f"Recorded entry: {symbol} ({mode}), trades_today={self.state.trades_today}")
    
    def record_exit(self, symbol: str, pnl: float):
        """Record a trade exit"""
        self.state.daily_pnl += pnl
        self.state.symbol_exit_times[symbol] = datetime.now()
        
        if symbol in self.state.open_positions:
            self.state.open_positions.discard(symbol)
        
        # Update consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        logger.info(f"Recorded exit: {symbol}, PnL=${pnl:.2f}, daily_pnl=${self.state.daily_pnl:.2f}, consecutive_losses={self.state.consecutive_losses}")
    
    def get_risk_adjusted_sizing_multiplier(self) -> float:
        """
        Get position sizing multiplier based on current risk state.
        Returns value between 0.5 and 1.0
        """
        # If down >2% on day, halve position size
        if self.state.daily_pnl <= -2.0:
            return 0.5
        
        # If one consecutive loss, reduce size by 25%
        if self.state.consecutive_losses == 1:
            return 0.75
        
        return 1.0
    
    def get_state_summary(self) -> Dict:
        """Get current state summary"""
        return {
            'daily_pnl': self.state.daily_pnl,
            'consecutive_losses': self.state.consecutive_losses,
            'trades_today': self.state.trades_today,
            'open_positions': len(self.state.open_positions),
            'sizing_multiplier': self.get_risk_adjusted_sizing_multiplier()
        }


# Singleton instance
_risk_manager: Optional[RiskManager] = None


def get_risk_manager() -> RiskManager:
    """Get or create singleton risk manager"""
    global _risk_manager
    if _risk_manager is None:
        _risk_manager = RiskManager()
    return _risk_manager


"""
DEX Fast Position Monitor - Real-time monitoring for held DEX positions

This module provides seconds-level monitoring for held meme coin / pump token positions.
Unlike the DexLaunchHunter (which discovers new tokens every 60s), this monitor:
- Tracks HELD positions only (1-5 tokens typically)
- Runs every 2-3 seconds for real-time price updates
- Uses pure math rules (no AI) for instant decision making
- Sends immediate Discord alerts for exit signals

CRITICAL: This is separate from DexLaunchHunter to avoid slowing down discovery.

Features:
- 2-second price monitoring loop
- Trailing stop detection (default 12%)
- Hard stop loss detection (default 30%)
- Pump spike detection (5%+ in 2 seconds)
- Profit target alerts
- Discord webhook integration for instant notifications
- Profitability calculator with realistic slippage/fees

Author: Sentient Trader
Created: December 2025
"""

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httpx
from loguru import logger


class AlertType(Enum):
    """Types of position alerts"""
    SELL_NOW = "SELL_NOW"           # Trailing stop hit
    EMERGENCY_SELL = "EMERGENCY_SELL"  # Hard stop hit
    PUMP_DETECTED = "PUMP_DETECTED"    # 5%+ spike detected
    PROFIT_TARGET = "PROFIT_TARGET"    # Hit profit target
    TAKE_PARTIAL = "TAKE_PARTIAL"      # Suggest partial exit
    PRICE_UPDATE = "PRICE_UPDATE"      # Regular update
    POSITION_ADDED = "POSITION_ADDED"  # New position tracking
    POSITION_CLOSED = "POSITION_CLOSED"  # Position closed
    SLIPPAGE_WARNING = "SLIPPAGE_WARNING"  # High slippage warning


class PositionStatus(Enum):
    """Position lifecycle status"""
    ACTIVE = "ACTIVE"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    ERROR = "ERROR"


@dataclass
class HeldPosition:
    """
    Represents a held DEX position (meme coin / pump token)
    
    This tracks a position you've already bought on Phantom/DEX.
    The monitor watches prices and alerts you when to sell.
    """
    # Position identification
    token_address: str
    symbol: str
    chain: str = "solana"
    
    # Entry details
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    tokens_held: float = 0.0
    investment_usd: float = 0.0
    
    # Current state
    current_price: float = 0.0
    peak_price: float = 0.0
    lowest_price: float = 0.0
    last_price: float = 0.0  # Price from previous check
    last_update: datetime = field(default_factory=datetime.now)
    
    # Risk management
    trailing_stop_pct: float = 12.0  # Default 12% trailing stop
    hard_stop_pct: float = 30.0      # Default 30% hard stop from entry
    profit_target_pct: float = 50.0  # Default 50% profit target
    partial_exit_pct: float = 25.0   # Suggest partial exit at 25%
    
    # Profitability tracking (REAL costs, not displayed)
    liquidity_usd: float = 0.0
    breakeven_gain_needed_pct: float = 0.0
    real_profit_pct: float = 0.0
    estimated_slippage_pct: float = 0.0
    
    # Status
    status: str = PositionStatus.ACTIVE.value
    partial_exit_taken: bool = False
    alerts_sent: int = 0
    last_alert_type: str = ""
    last_alert_time: Optional[datetime] = None
    
    # Performance
    unrealized_pnl_pct: float = 0.0
    unrealized_pnl_usd: float = 0.0
    max_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    @property
    def current_value_usd(self) -> float:
        """Current position value in USD"""
        return self.tokens_held * self.current_price
    
    @property
    def drawdown_from_peak_pct(self) -> float:
        """Current drawdown from peak price"""
        if self.peak_price <= 0:
            return 0.0
        return ((self.peak_price - self.current_price) / self.peak_price) * 100


@dataclass
class ProfitabilityAnalysis:
    """
    Real profitability analysis accounting for DEX trading costs.
    
    This shows what you ACTUALLY get after slippage, price impact, and fees.
    Not what DexScreener displays!
    """
    displayed_gain_pct: float = 0.0
    real_profit_pct: float = 0.0
    real_profit_usd: float = 0.0
    buy_costs_pct: float = 0.0
    sell_costs_pct: float = 0.0
    total_round_trip_cost_pct: float = 0.0
    breakeven_gain_needed_pct: float = 0.0
    min_display_gain_for_profit: float = 0.0
    is_profitable: bool = False
    liquidity_tier: str = ""
    warning: Optional[str] = None
    
    # Detailed breakdown
    buy_slippage_pct: float = 0.0
    sell_slippage_pct: float = 0.0
    buy_price_impact_pct: float = 0.0
    sell_price_impact_pct: float = 0.0
    dex_fee_pct: float = 0.3
    priority_fee_usd: float = 0.50


class FastPositionMonitor:
    """
    Seconds-level price monitoring for HELD DEX positions.
    
    This is a separate loop from DexLaunchHunter focused on:
    - Real-time price tracking (every 2 seconds)
    - Trailing stop detection
    - Pump/dump detection
    - Discord alert integration
    - Profitability calculations
    
    Usage:
        monitor = FastPositionMonitor()
        await monitor.add_position(token_address, entry_price, tokens_held, ...)
        await monitor.run_fast_loop()  # Runs forever
    """
    
    # File to persist held positions
    POSITIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "dex_held_positions.json"
    
    def __init__(
        self,
        check_interval: float = 2.0,  # 2 second default
        discord_webhook_url: Optional[str] = None,
        default_trailing_stop_pct: float = 12.0,
        default_hard_stop_pct: float = 30.0,
        default_profit_target_pct: float = 50.0
    ):
        """
        Initialize Fast Position Monitor
        
        Args:
            check_interval: Seconds between price checks (default: 2.0)
            discord_webhook_url: Discord webhook for alerts (optional, uses env var)
            default_trailing_stop_pct: Default trailing stop percentage
            default_hard_stop_pct: Default hard stop loss percentage
            default_profit_target_pct: Default profit target percentage
        """
        self.check_interval = check_interval
        self.discord_webhook_url = discord_webhook_url or os.getenv("DISCORD_DEX_ALERTS_WEBHOOK") or os.getenv("DISCORD_WEBHOOK_URL")
        
        # Default risk parameters
        self.default_trailing_stop_pct = default_trailing_stop_pct
        self.default_hard_stop_pct = default_hard_stop_pct
        self.default_profit_target_pct = default_profit_target_pct
        
        # Held positions (token_address -> HeldPosition)
        self.held_positions: Dict[str, HeldPosition] = {}
        
        # Rate limiting for DexScreener
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests
        
        # Alert cooldown (prevent spam)
        self.alert_cooldown_seconds = 30  # Minimum 30s between same alert types
        
        # Statistics
        self.is_running = False
        self.total_price_checks = 0
        self.total_alerts_sent = 0
        self.start_time: Optional[datetime] = None
        
        # Load persisted positions
        self._load_positions()
        
        logger.info("=" * 60)
        logger.info("🚀 DEX FAST POSITION MONITOR INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   Check Interval: {check_interval}s")
        logger.info(f"   Trailing Stop: {default_trailing_stop_pct}%")
        logger.info(f"   Hard Stop: {default_hard_stop_pct}%")
        logger.info(f"   Profit Target: {default_profit_target_pct}%")
        logger.info(f"   Discord Alerts: {'✅ Enabled' if self.discord_webhook_url else '❌ Disabled'}")
        logger.info(f"   Loaded Positions: {len(self.held_positions)}")
        logger.info("=" * 60)
    
    # ========================================================================
    # PROFITABILITY CALCULATOR
    # ========================================================================
    
    def calculate_real_profitability(
        self,
        liquidity_usd: float,
        trade_size_usd: float = 100.0,
        displayed_gain_pct: float = 50.0
    ) -> ProfitabilityAnalysis:
        """
        Calculate REAL profit after slippage, price impact, and fees.
        
        This is what you ACTUALLY get, not what DexScreener shows!
        
        Args:
            liquidity_usd: Token liquidity in USD
            trade_size_usd: Your trade size in USD
            displayed_gain_pct: The gain % shown on DexScreener
            
        Returns:
            ProfitabilityAnalysis with real profit breakdown
        """
        # Realistic slippage for meme coins (based on liquidity tier)
        if liquidity_usd < 5000:
            buy_slippage_pct = 12.0  # Very low liq = brutal slippage
            sell_slippage_pct = 15.0
            liquidity_tier = "MICRO (High Risk)"
        elif liquidity_usd < 20000:
            buy_slippage_pct = 8.0
            sell_slippage_pct = 10.0
            liquidity_tier = "LOW"
        elif liquidity_usd < 100000:
            buy_slippage_pct = 5.0
            sell_slippage_pct = 7.0
            liquidity_tier = "MEDIUM"
        elif liquidity_usd < 500000:
            buy_slippage_pct = 3.0
            sell_slippage_pct = 4.0
            liquidity_tier = "GOOD"
        else:
            buy_slippage_pct = 2.0
            sell_slippage_pct = 2.5
            liquidity_tier = "HIGH (Safer)"
        
        # DEX fees (Raydium/Jupiter ~0.25-0.3%)
        dex_fee_pct = 0.3
        
        # Priority fees (Solana)
        priority_fee_usd = 0.50
        
        # 1. Price Impact (AMM constant product approximation)
        # Impact ≈ trade_size / (2 * liquidity) for small trades
        buy_price_impact_pct = (trade_size_usd / (2 * max(liquidity_usd, 1))) * 100
        
        # 2. Total buy-side costs
        buy_costs_pct = buy_slippage_pct + buy_price_impact_pct + dex_fee_pct
        
        # 3. Effective entry (what you actually paid per token)
        effective_entry_value = trade_size_usd * (1 - buy_costs_pct / 100)
        
        # 4. Value after displayed gain
        value_before_sell = effective_entry_value * (1 + displayed_gain_pct / 100)
        
        # 5. Sell-side costs (often WORSE - everyone trying to exit)
        sell_price_impact_pct = (value_before_sell / (2 * max(liquidity_usd, 1))) * 100
        sell_costs_pct = sell_slippage_pct + sell_price_impact_pct + dex_fee_pct
        
        # 6. Final value after selling
        final_value = value_before_sell * (1 - sell_costs_pct / 100) - (priority_fee_usd * 2)
        
        # 7. REAL profit
        real_profit_usd = final_value - trade_size_usd
        real_profit_pct = (real_profit_usd / trade_size_usd) * 100
        
        # 8. Minimum gain needed to break even
        breakeven_multiplier = 1 / ((1 - buy_costs_pct/100) * (1 - sell_costs_pct/100))
        breakeven_gain_pct = (breakeven_multiplier - 1) * 100
        
        # 9. Is this trade even worth it?
        is_profitable = real_profit_pct > 5  # Need at least 5% REAL profit
        min_display_gain_for_profit = breakeven_gain_pct + 10  # Need 10% buffer
        
        # Generate warning if not profitable
        warning = None
        if not is_profitable:
            warning = f"⚠️ Need {min_display_gain_for_profit:.0f}%+ displayed gain to profit"
        
        return ProfitabilityAnalysis(
            displayed_gain_pct=displayed_gain_pct,
            real_profit_pct=round(real_profit_pct, 2),
            real_profit_usd=round(real_profit_usd, 2),
            buy_costs_pct=round(buy_costs_pct, 2),
            sell_costs_pct=round(sell_costs_pct, 2),
            total_round_trip_cost_pct=round(buy_costs_pct + sell_costs_pct, 2),
            breakeven_gain_needed_pct=round(breakeven_gain_pct, 2),
            min_display_gain_for_profit=round(min_display_gain_for_profit, 2),
            is_profitable=is_profitable,
            liquidity_tier=liquidity_tier,
            warning=warning,
            buy_slippage_pct=buy_slippage_pct,
            sell_slippage_pct=sell_slippage_pct,
            buy_price_impact_pct=round(buy_price_impact_pct, 2),
            sell_price_impact_pct=round(sell_price_impact_pct, 2),
            dex_fee_pct=dex_fee_pct,
            priority_fee_usd=priority_fee_usd
        )
    
    def get_liquidity_tier(self, liquidity_usd: float) -> str:
        """Categorize liquidity for risk assessment"""
        if liquidity_usd < 5000:
            return "MICRO (High Risk)"
        elif liquidity_usd < 20000:
            return "LOW"
        elif liquidity_usd < 100000:
            return "MEDIUM"
        elif liquidity_usd < 500000:
            return "GOOD"
        else:
            return "HIGH (Safer)"
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    async def add_position(
        self,
        token_address: str,
        symbol: str,
        entry_price: float,
        tokens_held: float,
        investment_usd: float,
        liquidity_usd: float = 0.0,
        trailing_stop_pct: Optional[float] = None,
        hard_stop_pct: Optional[float] = None,
        profit_target_pct: Optional[float] = None,
        chain: str = "solana"
    ) -> HeldPosition:
        """
        Add a new position to monitor
        
        Args:
            token_address: Token contract address
            symbol: Token symbol
            entry_price: Entry price in USD
            tokens_held: Number of tokens held
            investment_usd: Total USD invested (including fees paid)
            liquidity_usd: Token liquidity in USD (for profitability calc)
            trailing_stop_pct: Custom trailing stop (default: 12%)
            hard_stop_pct: Custom hard stop (default: 30%)
            profit_target_pct: Custom profit target (default: 50%)
            chain: Blockchain (default: solana)
            
        Returns:
            HeldPosition object
        """
        # Calculate profitability thresholds
        profitability = self.calculate_real_profitability(
            liquidity_usd=liquidity_usd,
            trade_size_usd=investment_usd,
            displayed_gain_pct=50.0  # Test with 50% displayed gain
        )
        
        position = HeldPosition(
            token_address=token_address.lower(),
            symbol=symbol,
            chain=chain,
            entry_price=entry_price,
            entry_time=datetime.now(),
            tokens_held=tokens_held,
            investment_usd=investment_usd,
            current_price=entry_price,
            peak_price=entry_price,
            lowest_price=entry_price,
            last_price=entry_price,
            trailing_stop_pct=trailing_stop_pct or self.default_trailing_stop_pct,
            hard_stop_pct=hard_stop_pct or self.default_hard_stop_pct,
            profit_target_pct=profit_target_pct or self.default_profit_target_pct,
            liquidity_usd=liquidity_usd,
            breakeven_gain_needed_pct=profitability.breakeven_gain_needed_pct,
            estimated_slippage_pct=profitability.buy_slippage_pct
        )
        
        self.held_positions[token_address.lower()] = position
        self._save_positions()
        
        logger.info(f"➕ Added position: {symbol} @ ${entry_price:.8f} ({tokens_held:.2f} tokens)")
        logger.info(f"   └─ Breakeven needs: {profitability.breakeven_gain_needed_pct:.1f}% displayed gain")
        
        # Send Discord alert
        await self._send_alert(position, AlertType.POSITION_ADDED, 
            f"Started monitoring {symbol}")
        
        return position
    
    async def close_position(self, token_address: str, exit_price: float = 0.0) -> Optional[HeldPosition]:
        """
        Mark position as closed and remove from monitoring
        
        Args:
            token_address: Token contract address
            exit_price: Exit price (optional, for final P&L calculation)
            
        Returns:
            Closed position or None if not found
        """
        token_address = token_address.lower()
        if token_address not in self.held_positions:
            logger.warning(f"Position not found: {token_address}")
            return None
        
        position = self.held_positions[token_address]
        position.status = PositionStatus.CLOSED.value
        
        if exit_price > 0:
            position.current_price = exit_price
            position.unrealized_pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * 100
            position.unrealized_pnl_usd = (exit_price - position.entry_price) * position.tokens_held
        
        # Remove from active monitoring
        del self.held_positions[token_address]
        self._save_positions()
        
        logger.info(f"🔒 Closed position: {position.symbol} | P&L: {position.unrealized_pnl_pct:.2f}%")
        
        await self._send_alert(position, AlertType.POSITION_CLOSED,
            f"Closed {position.symbol} | P&L: {position.unrealized_pnl_pct:.2f}%")
        
        return position
    
    def get_position(self, token_address: str) -> Optional[HeldPosition]:
        """Get a specific position by token address"""
        return self.held_positions.get(token_address.lower())
    
    def get_all_positions(self) -> List[HeldPosition]:
        """Get all active positions"""
        return list(self.held_positions.values())
    
    # ========================================================================
    # PRICE FETCHING
    # ========================================================================
    
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """
        Fetch current token price from DexScreener
        
        Args:
            token_address: Token contract address
            
        Returns:
            Current price in USD or None if failed
        """
        try:
            # Rate limiting
            elapsed = datetime.now().timestamp() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                self.last_request_time = datetime.now().timestamp()
                
                if response.status_code == 200:
                    data = response.json()
                    pairs = data.get("pairs", [])
                    
                    if pairs:
                        # Get highest liquidity pair
                        best_pair = max(pairs, key=lambda p: p.get("liquidity", {}).get("usd", 0))
                        price = float(best_pair.get("priceUsd", 0))
                        
                        # Also update liquidity
                        liquidity = best_pair.get("liquidity", {}).get("usd", 0)
                        if token_address.lower() in self.held_positions:
                            self.held_positions[token_address.lower()].liquidity_usd = liquidity
                        
                        return price
                elif response.status_code == 429:
                    logger.warning("DexScreener rate limit hit, waiting...")
                    await asyncio.sleep(5)
                    return None
                else:
                    logger.debug(f"DexScreener returned {response.status_code} for {token_address[:8]}...")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching price for {token_address[:8]}...: {e}")
            return None
        
        return None
    
    # ========================================================================
    # FAST MONITORING LOOP
    # ========================================================================
    
    async def run_fast_loop(self):
        """
        Main fast monitoring loop - runs every 2 seconds.
        
        This is the core loop that:
        1. Fetches prices for all held positions
        2. Updates position state (peak, drawdown, P&L)
        3. Checks exit rules (trailing stop, hard stop, profit target)
        4. Sends Discord alerts when rules trigger
        """
        logger.info("🚀 Starting Fast Position Monitor loop...")
        self.is_running = True
        self.start_time = datetime.now()
        
        while self.is_running:
            try:
                if not self.held_positions:
                    # No positions to monitor, sleep longer
                    await asyncio.sleep(10)
                    continue
                
                # Check each position
                for token_address, position in list(self.held_positions.items()):
                    if position.status != PositionStatus.ACTIVE.value:
                        continue
                    
                    # 1. Fetch current price
                    current_price = await self.get_token_price(token_address)
                    
                    if current_price is None:
                        continue
                    
                    self.total_price_checks += 1
                    
                    # 2. Update position state
                    position.last_price = position.current_price
                    position.current_price = current_price
                    position.last_update = datetime.now()
                    
                    # Update peak/lowest
                    if current_price > position.peak_price:
                        position.peak_price = current_price
                    if current_price < position.lowest_price or position.lowest_price == 0:
                        position.lowest_price = current_price
                    
                    # Calculate P&L
                    position.unrealized_pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    position.unrealized_pnl_usd = (current_price - position.entry_price) * position.tokens_held
                    
                    # Track max profit/drawdown
                    if position.unrealized_pnl_pct > position.max_profit_pct:
                        position.max_profit_pct = position.unrealized_pnl_pct
                    if position.unrealized_pnl_pct < 0 and abs(position.unrealized_pnl_pct) > position.max_drawdown_pct:
                        position.max_drawdown_pct = abs(position.unrealized_pnl_pct)
                    
                    # 3. Check exit rules (PURE MATH, NO AI)
                    await self._check_exit_rules(position)
                    
                    # Small delay between tokens to avoid rate limits
                    await asyncio.sleep(0.5)
                
                # Save updated positions
                self._save_positions()
                
                # Wait before next check cycle
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in fast monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Wait before retry
    
    def stop_monitoring(self):
        """Stop the fast monitoring loop"""
        logger.info("Stopping Fast Position Monitor...")
        self.is_running = False
    
    # ========================================================================
    # EXIT RULES (PURE MATH - NO AI)
    # ========================================================================
    
    async def _check_exit_rules(self, position: HeldPosition):
        """
        Check all exit rules for a position.
        Uses pure math - no AI calls for speed.
        
        Rules:
        1. Trailing Stop: Price dropped 12% from peak
        2. Hard Stop: Price dropped 30% from entry
        3. Pump Spike: Price jumped 5%+ in 2 seconds
        4. Profit Target: Hit target profit %
        5. Partial Exit: Suggest partial at 25%+ gain
        """
        
        # Calculate key metrics
        drawdown_from_peak = position.drawdown_from_peak_pct
        loss_from_entry = ((position.entry_price - position.current_price) / position.entry_price) * 100 if position.current_price < position.entry_price else 0
        price_change_pct = ((position.current_price - position.last_price) / position.last_price) * 100 if position.last_price > 0 else 0
        
        # 1. TRAILING STOP (12% default)
        if drawdown_from_peak >= position.trailing_stop_pct:
            await self._send_alert(
                position, 
                AlertType.SELL_NOW,
                f"⚠️ TRAILING STOP HIT\n"
                f"Dropped {drawdown_from_peak:.1f}% from peak ${position.peak_price:.8f}\n"
                f"Current: ${position.current_price:.8f}"
            )
            return
        
        # 2. HARD STOP (30% default)
        if loss_from_entry >= position.hard_stop_pct:
            await self._send_alert(
                position,
                AlertType.EMERGENCY_SELL,
                f"🚨 HARD STOP HIT\n"
                f"Down {loss_from_entry:.1f}% from entry ${position.entry_price:.8f}\n"
                f"Current: ${position.current_price:.8f}\n"
                f"SELL IMMEDIATELY"
            )
            return
        
        # 3. PUMP SPIKE (5%+ in 2 seconds)
        if price_change_pct >= 5.0:
            await self._send_alert(
                position,
                AlertType.PUMP_DETECTED,
                f"🚀 PUMP DETECTED!\n"
                f"+{price_change_pct:.1f}% spike in {self.check_interval}s\n"
                f"Price: ${position.last_price:.8f} → ${position.current_price:.8f}\n"
                f"Consider taking profits"
            )
            return
        
        # 4. PROFIT TARGET HIT
        if position.unrealized_pnl_pct >= position.profit_target_pct:
            # Calculate real profit
            profitability = self.calculate_real_profitability(
                liquidity_usd=position.liquidity_usd,
                trade_size_usd=position.investment_usd,
                displayed_gain_pct=position.unrealized_pnl_pct
            )
            
            await self._send_alert(
                position,
                AlertType.PROFIT_TARGET,
                f"🎯 PROFIT TARGET HIT!\n"
                f"Displayed: +{position.unrealized_pnl_pct:.1f}%\n"
                f"Real profit: +{profitability.real_profit_pct:.1f}% (${profitability.real_profit_usd:.2f})\n"
                f"Consider selling NOW"
            )
            return
        
        # 5. PARTIAL EXIT SUGGESTION (25%+ gain, not taken yet)
        if not position.partial_exit_taken and position.unrealized_pnl_pct >= position.partial_exit_pct:
            # Calculate real profit
            profitability = self.calculate_real_profitability(
                liquidity_usd=position.liquidity_usd,
                trade_size_usd=position.investment_usd * 0.5,  # Half position
                displayed_gain_pct=position.unrealized_pnl_pct
            )
            
            await self._send_alert(
                position,
                AlertType.TAKE_PARTIAL,
                f"💰 PARTIAL EXIT SUGGESTION\n"
                f"Up {position.unrealized_pnl_pct:.1f}% - consider taking 50%\n"
                f"Real profit (if sell half): ${profitability.real_profit_usd:.2f}\n"
                f"Let the rest ride with tighter stop"
            )
            position.partial_exit_taken = True
            return
    
    # ========================================================================
    # DISCORD ALERTS
    # ========================================================================
    
    async def _send_alert(
        self, 
        position: HeldPosition, 
        alert_type: AlertType, 
        message: str
    ):
        """
        Send Discord webhook alert for position events
        
        Args:
            position: HeldPosition object
            alert_type: Type of alert
            message: Alert message
        """
        # Check cooldown (prevent spam)
        if position.last_alert_time:
            elapsed = (datetime.now() - position.last_alert_time).total_seconds()
            if elapsed < self.alert_cooldown_seconds and position.last_alert_type == alert_type.value:
                logger.debug(f"Alert cooldown active for {position.symbol} ({alert_type.value})")
                return
        
        # Update alert tracking
        position.last_alert_time = datetime.now()
        position.last_alert_type = alert_type.value
        position.alerts_sent += 1
        self.total_alerts_sent += 1
        
        # Build embed color based on alert type
        color_map = {
            AlertType.SELL_NOW: 0xFF6600,       # Orange
            AlertType.EMERGENCY_SELL: 0xFF0000,  # Red
            AlertType.PUMP_DETECTED: 0x00FF00,   # Green
            AlertType.PROFIT_TARGET: 0x00FF88,   # Teal
            AlertType.TAKE_PARTIAL: 0xFFFF00,    # Yellow
            AlertType.PRICE_UPDATE: 0x808080,    # Gray
            AlertType.POSITION_ADDED: 0x0088FF,  # Blue
            AlertType.POSITION_CLOSED: 0x8800FF, # Purple
            AlertType.SLIPPAGE_WARNING: 0xFF8800 # Orange
        }
        color = color_map.get(alert_type, 0x808080)
        
        # Calculate profitability for context
        profitability = self.calculate_real_profitability(
            liquidity_usd=position.liquidity_usd,
            trade_size_usd=position.investment_usd,
            displayed_gain_pct=max(position.unrealized_pnl_pct, 0)
        )
        
        # Build Discord embed
        embed = {
            "title": f"🎰 {alert_type.value}: {position.symbol}",
            "description": message,
            "color": color,
            "fields": [
                {
                    "name": "📊 Position",
                    "value": f"Entry: ${position.entry_price:.8f}\nCurrent: ${position.current_price:.8f}\nPeak: ${position.peak_price:.8f}",
                    "inline": True
                },
                {
                    "name": "💰 P&L",
                    "value": f"Displayed: {position.unrealized_pnl_pct:+.2f}%\nReal: {profitability.real_profit_pct:+.2f}%\nUSD: ${profitability.real_profit_usd:+.2f}",
                    "inline": True
                },
                {
                    "name": "⚠️ Costs",
                    "value": f"Round-trip: ~{profitability.total_round_trip_cost_pct:.1f}%\nBreakeven: {profitability.breakeven_gain_needed_pct:.1f}%+",
                    "inline": True
                },
                {
                    "name": "💧 Liquidity",
                    "value": f"${position.liquidity_usd:,.0f} ({profitability.liquidity_tier})",
                    "inline": True
                },
                {
                    "name": "📈 Drawdown",
                    "value": f"From Peak: {position.drawdown_from_peak_pct:.1f}%\nMax DD: {position.max_drawdown_pct:.1f}%",
                    "inline": True
                },
                {
                    "name": "🔗 Links",
                    "value": f"[DexScreener](https://dexscreener.com/{position.chain}/{position.token_address})",
                    "inline": True
                }
            ],
            "footer": {
                "text": f"Fast Monitor | {datetime.now().strftime('%H:%M:%S')} | Check interval: {self.check_interval}s"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add warning if profitability is bad
        if profitability.warning:
            embed["fields"].append({
                "name": "⚠️ Profitability Warning",
                "value": profitability.warning,
                "inline": False
            })
        
        # Log the alert
        logger.info(f"📢 {alert_type.value}: {position.symbol} | P&L: {position.unrealized_pnl_pct:.2f}%")
        print(f"[FAST_MONITOR] 📢 {alert_type.value}: {position.symbol} | P&L: {position.unrealized_pnl_pct:.2f}%", flush=True)
        
        # Send to Discord
        if self.discord_webhook_url:
            try:
                payload = {"embeds": [embed]}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        self.discord_webhook_url,
                        json=payload
                    )
                    if response.status_code not in [200, 204]:
                        logger.warning(f"Discord webhook returned {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to send Discord alert: {e}")
        else:
            logger.debug("Discord webhook not configured, alert logged only")
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def _save_positions(self):
        """Save positions to JSON file for persistence"""
        try:
            self.POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for addr, pos in self.held_positions.items():
                pos_dict = asdict(pos)
                # Convert datetime to ISO format
                for key in ['entry_time', 'last_update', 'last_alert_time']:
                    if pos_dict.get(key):
                        pos_dict[key] = pos_dict[key].isoformat() if isinstance(pos_dict[key], datetime) else pos_dict[key]
                data[addr] = pos_dict
            
            with open(self.POSITIONS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save positions: {e}")
    
    def _load_positions(self):
        """Load positions from JSON file"""
        try:
            if self.POSITIONS_FILE.exists():
                with open(self.POSITIONS_FILE, 'r') as f:
                    data = json.load(f)
                
                for addr, pos_dict in data.items():
                    # Convert ISO strings back to datetime
                    for key in ['entry_time', 'last_update', 'last_alert_time']:
                        if pos_dict.get(key):
                            try:
                                pos_dict[key] = datetime.fromisoformat(pos_dict[key])
                            except (ValueError, TypeError):
                                pos_dict[key] = datetime.now() if key == 'entry_time' else None
                    
                    # Only load ACTIVE positions
                    if pos_dict.get('status') == PositionStatus.ACTIVE.value:
                        self.held_positions[addr] = HeldPosition(**pos_dict)
                
                logger.info(f"Loaded {len(self.held_positions)} active positions from file")
                
        except Exception as e:
            logger.warning(f"Failed to load positions: {e}")
            self.held_positions = {}
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "positions_monitored": len(self.held_positions),
            "total_price_checks": self.total_price_checks,
            "total_alerts_sent": self.total_alerts_sent,
            "check_interval_seconds": self.check_interval,
            "positions": [
                {
                    "symbol": p.symbol,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "pnl_pct": p.unrealized_pnl_pct,
                    "drawdown_pct": p.drawdown_from_peak_pct
                }
                for p in self.held_positions.values()
            ]
        }


# ========================================================================
# SINGLETON INSTANCE
# ========================================================================

_monitor_instance: Optional[FastPositionMonitor] = None


def get_fast_position_monitor(
    check_interval: float = 2.0,
    discord_webhook_url: Optional[str] = None
) -> FastPositionMonitor:
    """Get or create FastPositionMonitor singleton"""
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = FastPositionMonitor(
            check_interval=check_interval,
            discord_webhook_url=discord_webhook_url
        )
    
    return _monitor_instance


# ========================================================================
# CLI ENTRY POINT
# ========================================================================

async def main():
    """CLI entry point for running the fast monitor standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DEX Fast Position Monitor")
    parser.add_argument("--interval", type=float, default=2.0, help="Check interval in seconds")
    parser.add_argument("--add", type=str, help="Add position: token_address,symbol,entry_price,tokens,investment,liquidity")
    parser.add_argument("--close", type=str, help="Close position by token address")
    parser.add_argument("--list", action="store_true", help="List all positions")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    args = parser.parse_args()
    
    monitor = get_fast_position_monitor(check_interval=args.interval)
    
    if args.add:
        parts = args.add.split(",")
        if len(parts) >= 5:
            await monitor.add_position(
                token_address=parts[0],
                symbol=parts[1],
                entry_price=float(parts[2]),
                tokens_held=float(parts[3]),
                investment_usd=float(parts[4]),
                liquidity_usd=float(parts[5]) if len(parts) > 5 else 10000
            )
        else:
            print("Usage: --add token_address,symbol,entry_price,tokens,investment,liquidity")
    elif args.close:
        await monitor.close_position(args.close)
    elif args.list:
        positions = monitor.get_all_positions()
        print(f"\n{'='*60}")
        print(f"Active Positions: {len(positions)}")
        print(f"{'='*60}")
        for pos in positions:
            print(f"\n{pos.symbol} ({pos.token_address[:8]}...)")
            print(f"  Entry: ${pos.entry_price:.8f} | Current: ${pos.current_price:.8f}")
            print(f"  P&L: {pos.unrealized_pnl_pct:+.2f}% | Drawdown: {pos.drawdown_from_peak_pct:.2f}%")
            print(f"  Breakeven needs: {pos.breakeven_gain_needed_pct:.1f}%")
    elif args.stats:
        stats = monitor.get_stats()
        print(json.dumps(stats, indent=2))
    else:
        # Run the monitoring loop
        print("Starting Fast Position Monitor...")
        print("Press Ctrl+C to stop")
        await monitor.run_fast_loop()


if __name__ == "__main__":
    asyncio.run(main())

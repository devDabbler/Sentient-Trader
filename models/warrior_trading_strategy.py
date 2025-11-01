"""
Warrior Trading Scalping Strategy Model
Based on Ross Cameron's Gap & Go approach

Supports:
- Gap & Go
- 1-minute Micro Pullback
- Red-to-Green
- Bull Flag Breakout
- Momentum Scalping
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime


class WarriorSetupType(Enum):
    """Warrior Trading setup types"""
    GAP_AND_GO = "GAP_AND_GO"
    MICRO_PULLBACK = "MICRO_PULLBACK"
    RED_TO_GREEN = "RED_TO_GREEN"
    BULL_FLAG = "BULL_FLAG"
    MOMENTUM_SCALP = "MOMENTUM_SCALP"


@dataclass
class GapAndGoSetup:
    """Gap & Go strategy setup"""
    ticker: str
    premarket_gap_pct: float  # 4-10% gap from previous close
    relative_volume: float  # 2-3x average volume
    price_range: tuple  # ($2-$20)
    entry_trigger: str  # "Breakout above premarket high", "Bounce off premarket low"
    stop_loss_pct: float  # 1% below low of breakout candle
    profit_target_pct: float  # 2% (scale out)
    confidence: float  # 0-100
    timestamp: datetime


@dataclass
class MicroPullbackSetup:
    """1-minute Micro Pullback setup"""
    ticker: str
    pullback_pct: float  # 0.2-0.5% pullback from high
    volume_spike: bool  # Volume > 1.5x on entry
    ema_support: Optional[float]  # Price above 9 EMA
    entry_price: float
    stop_loss: float  # Low of pullback candle
    profit_target: float  # 2R (2x risk)
    confidence: float
    timestamp: datetime


@dataclass
class RedToGreenSetup:
    """Red-to-Green reversal setup"""
    ticker: str
    premarket_low: float
    current_price: float
    reversal_candle: bool  # Green candle after red candles
    volume_confirmation: bool  # Volume spike on reversal
    entry_price: float
    stop_loss: float  # Below premarket low
    profit_target: float  # 2% target
    confidence: float
    timestamp: datetime


@dataclass
class BullFlagSetup:
    """Bull Flag breakout setup"""
    ticker: str
    flag_pole_high: float
    flag_low: float
    breakout_price: float
    volume_breakout: bool  # Volume > 2x on breakout
    entry_price: float
    stop_loss: float  # Below flag low
    profit_target: float  # Flag pole height target
    confidence: float
    timestamp: datetime


@dataclass
class MomentumScalpSetup:
    """Momentum scalping setup"""
    ticker: str
    momentum_pct: float  # Strong intraday move
    vwap_position: str  # "Above VWAP" or "Below VWAP"
    volume_ratio: float  # Relative volume
    entry_price: float
    stop_loss: float  # Tight stop (0.5-1%)
    profit_target: float  # Quick 1-2% target
    confidence: float
    timestamp: datetime


@dataclass
class WarriorTradingSignal:
    """Complete Warrior Trading signal with all setup details"""
    ticker: str
    setup_type: WarriorSetupType
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    stop_loss: float
    profit_target: float
    risk_reward_ratio: float  # R:R ratio
    confidence: float  # 0-100
    reasoning: str
    metadata: Dict  # Additional setup-specific data
    timestamp: datetime
    
    # Position sizing
    position_size_pct: float = 3.0  # Default 3% per trade
    max_loss_per_trade: float = 1.0  # Max 1% loss per trade
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            'ticker': self.ticker,
            'setup_type': self.setup_type.value,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'profit_target': self.profit_target,
            'risk_reward': self.risk_reward_ratio,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


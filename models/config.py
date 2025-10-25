"""Trading configuration and parameters."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TradingConfig:
    """Configuration for trading parameters and guardrails"""
    max_daily_orders: int = 10
    max_position_per_ticker: int = 5
    max_daily_risk: float = 1000.0
    min_dte: int = 7
    max_dte: int = 45
    min_iv_rank: float = 20.0
    max_iv_rank: float = 80.0
    min_volume: int = 100
    max_bid_ask_spread: float = 0.50
    allowed_strategies: Optional[List[str]] = None
    trading_start_hour: int = 9
    trading_end_hour: int = 15
    trading_start_minute: int = 45
    trading_end_minute: int = 45
    
    def __post_init__(self):
        if self.allowed_strategies is None:
            self.allowed_strategies = [
                "SELL_CALL", "SELL_PUT", "BUY_CALL", "BUY_PUT",
                "IRON_CONDOR", "CREDIT_SPREAD", "DEBIT_SPREAD",
                "LONG_STRADDLE", "WHEEL_STRATEGY"
            ]

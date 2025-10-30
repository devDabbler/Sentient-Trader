"""Message dataclasses for event-driven agent communication"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class TradingMode(str, Enum):
    """Trading mode types"""
    SLOW_SCALPER = "SLOW_SCALPER"
    MICRO_SWING = "MICRO_SWING"
    STOCKS = "STOCKS"
    OPTIONS = "OPTIONS"
    ALL = "ALL"


class SetupType(str, Enum):
    """Setup/Strategy types"""
    ORB = "ORB"  # Opening Range Breakout
    VWAP_BOUNCE = "VWAP_BOUNCE"
    KEY_LEVEL_REJECTION = "KEY_LEVEL_REJECTION"
    CUSTOM = "CUSTOM"


class Timeframe(str, Enum):
    """Data timeframes"""
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_60 = "60min"
    DAILY = "daily"


@dataclass
class BarEvent:
    """New bar/candle data event"""
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    event_id: str = field(default_factory=lambda: f"bar_{datetime.now().timestamp()}")


@dataclass
class IndicatorEvent:
    """Technical indicator update event"""
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    indicators: Dict[str, Any]  # e.g., {"ema_9": 150.5, "ema_20": 149.2, "rsi": 65.5, "vwap": 150.0}
    event_id: str = field(default_factory=lambda: f"ind_{datetime.now().timestamp()}")


@dataclass
class TradeCandidate:
    """Potential trade identified by setup detectors"""
    symbol: str
    setup_type: SetupType
    mode: TradingMode
    timestamp: datetime
    entry_price: float
    stop_price: float
    target_price: float
    confidence: float  # 0-100
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"candidate_{datetime.now().timestamp()}")


@dataclass
class ApprovedOrder:
    """Order approved by risk manager"""
    symbol: str
    setup_type: SetupType
    mode: TradingMode
    timestamp: datetime
    side: str  # BUY or SELL
    quantity: int
    entry_price: float
    stop_price: float
    target_price: float
    bucket_index: int
    estimated_settlement: datetime
    tag: str
    duration: str  # day or gtc
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: f"approved_{datetime.now().timestamp()}")


@dataclass
class OrderUpdate:
    """Order status update from broker"""
    symbol: str
    order_id: Optional[str]
    status: str  # pending, filled, cancelled, rejected
    timestamp: datetime
    filled_qty: int = 0
    filled_price: float = 0.0
    message: str = ""
    event_id: str = field(default_factory=lambda: f"order_{datetime.now().timestamp()}")


@dataclass
class JournalEntry:
    """Trade journal record"""
    symbol: str
    setup_type: SetupType
    mode: TradingMode
    side: str
    entry_time: datetime
    entry_price: float
    stop_price: float
    target_price: float
    shares: int
    bucket_index: int
    settlement_date: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    r_multiple: Optional[float] = None
    hold_time_minutes: Optional[int] = None
    settled_cash_after: Optional[float] = None
    exit_reason: Optional[str] = None
    event_id: str = field(default_factory=lambda: f"journal_{datetime.now().timestamp()}")


@dataclass
class ControlEvent:
    """Control signal for orchestrator/agents"""
    command: str  # start, stop, pause, resume, update_settings
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: f"control_{datetime.now().timestamp()}")


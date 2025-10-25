"""
Data models for trading alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict


class AlertType(Enum):
    """Types of trading alerts"""
    # Technical setup alerts
    EMA_RECLAIM = "EMA_RECLAIM"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    TIMEFRAME_ALIGNED = "TIMEFRAME_ALIGNED"
    SECTOR_LEADER = "SECTOR_LEADER"
    FIBONACCI_SETUP = "FIBONACCI_SETUP"
    DEMARKER_ENTRY = "DEMARKER_ENTRY"
    
    # Position/trade monitoring alerts
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    POSITION_UPDATE = "POSITION_UPDATE"
    TRAILING_STOP = "TRAILING_STOP"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class TradingAlert:
    """Represents a trading setup alert"""
    ticker: str
    alert_type: AlertType
    priority: AlertPriority
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            "ticker": self.ticker,
            "alert_type": self.alert_type.value,
            "priority": self.priority.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "confidence_score": self.confidence_score,
            "details": self.details
        }

    def __str__(self) -> str:
        """String representation"""
        return f"[{self.priority.value}] {self.ticker}: {self.message}"

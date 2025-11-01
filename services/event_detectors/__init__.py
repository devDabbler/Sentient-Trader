"""Event detection modules for trading alerts"""

from .base_detector import BaseEventDetector
from .earnings_detector import EarningsDetector
from .news_detector import NewsDetector
from .sec_detector import SECDetector
from .economic_detector import EconomicDetector

__all__ = [
    'BaseEventDetector',
    'EarningsDetector',
    'NewsDetector',
    'SECDetector',
    'EconomicDetector'
]

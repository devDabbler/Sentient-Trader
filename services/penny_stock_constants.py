"""
Centralized Penny Stock Constants and Definitions

This module provides centralized constants for penny stock classification
to ensure consistency across the entire application.
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class PennyStockThresholds:
    """
    Penny stock classification thresholds used throughout the application.
    
    These thresholds define what constitutes a penny stock and various
    price tiers for scoring and filtering purposes.
    """
    
    # Primary penny stock definition
    MAX_PENNY_STOCK_PRICE: float = 5.0  # Stocks below $5 are considered penny stocks
    
    # Price tier classifications
    ULTRA_LOW_PRICE: float = 1.0  # Stocks below $1 (highest risk/reward)
    LOW_PENNY_PRICE: float = 2.0  # $1-$2 range
    MID_PENNY_PRICE: float = 3.0  # $2-$3 range
    HIGH_PENNY_PRICE: float = 5.0  # $3-$5 range
    
    # Market cap thresholds
    MAX_MARKET_CAP: int = 300_000_000  # $300M - typical penny stock max market cap
    MICRO_CAP_THRESHOLD: int = 50_000_000  # $50M - nano-cap vs micro-cap divider
    
    # Float thresholds
    MIN_FLOAT: int = 1_000_000  # 1M shares minimum float
    LOW_FLOAT_THRESHOLD: int = 10_000_000  # < 10M shares considered low float
    
    # Trading volume thresholds
    MIN_VOLUME: int = 100_000  # Minimum daily volume for tradability
    GOOD_VOLUME: int = 500_000  # Good volume for penny stocks
    HIGH_VOLUME: int = 1_000_000  # High volume threshold
    
    # Scoring bonuses for different price tiers
    ULTRA_LOW_BONUS: int = 15  # Bonus points for < $1
    LOW_PENNY_BONUS: int = 10  # Bonus points for $1-$2
    MID_PENNY_BONUS: int = 5   # Bonus points for $2-$3


# Global instance for easy import
PENNY_THRESHOLDS = PennyStockThresholds()


# Filter preset definitions
PENNY_STOCK_FILTER_PRESETS: Dict[str, Dict[str, float]] = {
    "Ultra-Low Price (<$1)": {
        "min_price": None,
        "max_price": PENNY_THRESHOLDS.ULTRA_LOW_PRICE,
    },
    "Penny Stocks ($1-$5)": {
        "min_price": PENNY_THRESHOLDS.ULTRA_LOW_PRICE,
        "max_price": PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE,
    },
    "All Penny Stocks (<$5)": {
        "min_price": None,
        "max_price": PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE,
    },
}


def is_penny_stock(price: float) -> bool:
    """
    Check if a stock price qualifies as a penny stock.
    
    Args:
        price: Current stock price
        
    Returns:
        True if price < $5.00, False otherwise
    """
    return price < PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE


def is_ultra_low_price(price: float) -> bool:
    """
    Check if a stock is in the ultra-low price category.
    
    Args:
        price: Current stock price
        
    Returns:
        True if price < $1.00, False otherwise
    """
    return price < PENNY_THRESHOLDS.ULTRA_LOW_PRICE


def get_penny_stock_tier(price: float) -> str:
    """
    Get the penny stock price tier classification.
    
    Args:
        price: Current stock price
        
    Returns:
        String classification: "ULTRA_LOW", "LOW", "MID", "HIGH", or "NOT_PENNY"
    """
    if price < PENNY_THRESHOLDS.ULTRA_LOW_PRICE:
        return "ULTRA_LOW"  # < $1
    elif price < PENNY_THRESHOLDS.LOW_PENNY_PRICE:
        return "LOW"  # $1-$2
    elif price < PENNY_THRESHOLDS.MID_PENNY_PRICE:
        return "MID"  # $2-$3
    elif price < PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE:
        return "HIGH"  # $3-$5
    else:
        return "NOT_PENNY"  # >= $5


def get_price_tier_bonus(price: float) -> int:
    """
    Get the scoring bonus for a given price tier.
    
    Lower-priced stocks get higher bonuses due to higher breakout potential.
    
    Args:
        price: Current stock price
        
    Returns:
        Bonus points (0-15)
    """
    if price < PENNY_THRESHOLDS.ULTRA_LOW_PRICE:
        return PENNY_THRESHOLDS.ULTRA_LOW_BONUS
    elif price < PENNY_THRESHOLDS.LOW_PENNY_PRICE:
        return PENNY_THRESHOLDS.LOW_PENNY_BONUS
    elif price < PENNY_THRESHOLDS.MID_PENNY_PRICE:
        return PENNY_THRESHOLDS.MID_PENNY_BONUS
    else:
        return 0


def get_penny_stock_description(price: float) -> str:
    """
    Get a human-readable description of the penny stock tier.
    
    Args:
        price: Current stock price
        
    Returns:
        Description string
    """
    tier = get_penny_stock_tier(price)
    
    descriptions = {
        "ULTRA_LOW": f"Ultra-low penny stock (${price:.4f}) - Highest risk/reward potential",
        "LOW": f"Low-priced penny stock (${price:.2f}) - High risk/reward",
        "MID": f"Mid-range penny stock (${price:.2f}) - Moderate penny stock risk",
        "HIGH": f"Higher-priced penny stock (${price:.2f}) - Lower penny stock risk",
        "NOT_PENNY": f"Not a penny stock (${price:.2f})"
    }
    
    return descriptions.get(tier, f"Price: ${price:.2f}")


# Export all for easy access
__all__ = [
    'PennyStockThresholds',
    'PENNY_THRESHOLDS',
    'PENNY_STOCK_FILTER_PRESETS',
    'is_penny_stock',
    'is_ultra_low_price',
    'get_penny_stock_tier',
    'get_price_tier_bonus',
    'get_penny_stock_description',
]


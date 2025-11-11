"""
Fractional Share Manager
Handles fractional share position sizing, configuration, and ROI calculations
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from loguru import logger
import json
import os


@dataclass
class FractionalShareConfig:
    """Configuration for fractional shares"""
    enabled: bool = True  # Enable fractional shares
    min_price_threshold: float = 100.0  # Auto-use fractional for stocks > this price
    max_price_threshold: Optional[float] = None  # Maximum stock price to trade
    min_dollar_amount: float = 50.0  # Minimum dollar amount per trade
    max_dollar_amount: Optional[float] = 1000.0  # Maximum dollar amount for fractional trades
    preferred_dollar_amounts: List[float] = None  # Preferred amounts like [100, 250, 500, 1000]
    allow_manual_fractional: bool = True  # Allow manually specifying fractional amounts
    
    def __post_init__(self):
        if self.preferred_dollar_amounts is None:
            self.preferred_dollar_amounts = [50.0, 100.0, 250.0, 500.0, 1000.0]


@dataclass
class FractionalPosition:
    """Represents a fractional share position"""
    symbol: str
    quantity: float  # Fractional quantity
    cost_basis: float  # Total cost basis
    current_price: float
    entry_price: float  # Price per share at entry
    current_value: float  # Current market value
    unrealized_pnl: float  # Unrealized P&L
    unrealized_pnl_pct: float  # Unrealized P&L percentage
    is_fractional: bool  # Whether this is a fractional position


class FractionalShareManager:
    """
    Manager for fractional share trading
    Handles position sizing, configuration, and ROI calculations
    """
    
    def __init__(self, config: Optional[FractionalShareConfig] = None):
        """
        Initialize fractional share manager
        
        Args:
            config: FractionalShareConfig or None for defaults
        """
        self.config = config or FractionalShareConfig()
        self.custom_amounts: Dict[str, float] = {}  # Symbol -> custom dollar amount
        self.state_file = "fractional_share_config.json"
        
        logger.info("📊 Fractional Share Manager initialized")
        logger.info(f"   Enabled: {self.config.enabled}")
        logger.info(f"   Price threshold: ${self.config.min_price_threshold:.2f}")
        logger.info(f"   Min dollar amount: ${self.config.min_dollar_amount:.2f}")
        
        # Load saved custom amounts
        self._load_state()
    
    def should_use_fractional(self, symbol: str, price: float) -> bool:
        """
        Determine if fractional shares should be used for this stock
        
        Args:
            symbol: Stock symbol
            price: Current stock price
            
        Returns:
            True if fractional shares should be used
        """
        if not self.config.enabled:
            return False
        
        # Check if there's a custom amount set (indicates user wants fractional)
        if symbol in self.custom_amounts:
            return True
        
        # Auto-detect based on price threshold
        if price >= self.config.min_price_threshold:
            return True
        
        # Check max price threshold if set
        if self.config.max_price_threshold and price > self.config.max_price_threshold:
            logger.warning(f"{symbol} price ${price:.2f} exceeds max threshold ${self.config.max_price_threshold:.2f}")
            return False
        
        return False
    
    def calculate_fractional_quantity(
        self,
        symbol: str,
        price: float,
        available_capital: float,
        target_dollar_amount: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Calculate fractional quantity based on dollar amount
        
        Args:
            symbol: Stock symbol
            price: Current stock price
            available_capital: Available capital
            target_dollar_amount: Target dollar amount (or None to use custom/default)
            
        Returns:
            Tuple of (quantity, actual_dollar_amount)
        """
        try:
            # Determine target dollar amount
            if target_dollar_amount is None:
                # Check for custom amount
                if symbol in self.custom_amounts:
                    target_dollar_amount = self.custom_amounts[symbol]
                else:
                    # Use default max or available capital
                    target_dollar_amount = min(
                        self.config.max_dollar_amount or float('inf'),
                        available_capital
                    )
            
            # Ensure within limits
            target_dollar_amount = max(self.config.min_dollar_amount, target_dollar_amount)
            target_dollar_amount = min(available_capital, target_dollar_amount)
            
            if self.config.max_dollar_amount:
                target_dollar_amount = min(target_dollar_amount, self.config.max_dollar_amount)
            
            # Calculate quantity
            quantity = target_dollar_amount / price
            
            # Round to 2 decimal places (0.01 share minimum)
            quantity = round(quantity, 2)
            
            # Calculate actual dollar amount
            actual_dollar_amount = quantity * price
            
            logger.info(f"📊 Fractional sizing for {symbol}:")
            logger.info(f"   Price: ${price:.2f}")
            logger.info(f"   Target amount: ${target_dollar_amount:.2f}")
            logger.info(f"   Calculated quantity: {quantity}")
            logger.info(f"   Actual cost: ${actual_dollar_amount:.2f}")
            
            return quantity, actual_dollar_amount
            
        except Exception as e:
            logger.error(f"Error calculating fractional quantity: {e}")
            return 0.0, 0.0
    
    def set_custom_amount(self, symbol: str, dollar_amount: float):
        """
        Set custom dollar amount for a specific symbol
        
        Args:
            symbol: Stock symbol
            dollar_amount: Dollar amount to invest
        """
        if dollar_amount < self.config.min_dollar_amount:
            logger.warning(f"Dollar amount ${dollar_amount:.2f} below minimum ${self.config.min_dollar_amount:.2f}")
            dollar_amount = self.config.min_dollar_amount
        
        if self.config.max_dollar_amount and dollar_amount > self.config.max_dollar_amount:
            logger.warning(f"Dollar amount ${dollar_amount:.2f} above maximum ${self.config.max_dollar_amount:.2f}")
            dollar_amount = self.config.max_dollar_amount
        
        self.custom_amounts[symbol] = dollar_amount
        logger.info(f"📝 Set custom amount for {symbol}: ${dollar_amount:.2f}")
        
        self._save_state()
    
    def get_custom_amount(self, symbol: str) -> Optional[float]:
        """Get custom dollar amount for symbol"""
        return self.custom_amounts.get(symbol)
    
    def remove_custom_amount(self, symbol: str):
        """Remove custom dollar amount for symbol"""
        if symbol in self.custom_amounts:
            del self.custom_amounts[symbol]
            logger.info(f"🗑️ Removed custom amount for {symbol}")
            self._save_state()
    
    def calculate_position_roi(self, position: FractionalPosition) -> Dict:
        """
        Calculate detailed ROI metrics for a fractional position
        
        Args:
            position: FractionalPosition object
            
        Returns:
            Dict with ROI metrics
        """
        try:
            roi = {
                'symbol': position.symbol,
                'quantity': position.quantity,
                'is_fractional': position.is_fractional,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'cost_basis': position.cost_basis,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl,
                'unrealized_pnl_pct': position.unrealized_pnl_pct,
                'price_change': position.current_price - position.entry_price,
                'price_change_pct': ((position.current_price - position.entry_price) / position.entry_price) * 100 if position.entry_price > 0 else 0,
            }
            
            # Additional metrics
            roi['dollar_gain_per_share'] = position.current_price - position.entry_price
            roi['total_dollar_gain'] = position.unrealized_pnl
            
            # Annualized return (would need entry date for accurate calculation)
            # For now, just showing current return
            roi['current_return_pct'] = position.unrealized_pnl_pct
            
            return roi
            
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return {}
    
    def create_fractional_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float
    ) -> FractionalPosition:
        """
        Create a FractionalPosition object from raw data
        
        Args:
            symbol: Stock symbol
            quantity: Share quantity
            entry_price: Entry price per share
            current_price: Current price per share
            
        Returns:
            FractionalPosition object
        """
        is_fractional = (quantity % 1 != 0)
        cost_basis = quantity * entry_price
        current_value = quantity * current_price
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        return FractionalPosition(
            symbol=symbol,
            quantity=quantity,
            cost_basis=cost_basis,
            current_price=current_price,
            entry_price=entry_price,
            current_value=current_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            is_fractional=is_fractional
        )
    
    def get_preferred_amounts(self) -> List[float]:
        """Get list of preferred dollar amounts"""
        return self.config.preferred_dollar_amounts.copy()
    
    def _save_state(self):
        """Save custom amounts to file"""
        try:
            state = {
                'custom_amounts': self.custom_amounts,
                'config': {
                    'enabled': self.config.enabled,
                    'min_price_threshold': self.config.min_price_threshold,
                    'max_price_threshold': self.config.max_price_threshold,
                    'min_dollar_amount': self.config.min_dollar_amount,
                    'max_dollar_amount': self.config.max_dollar_amount,
                    'preferred_dollar_amounts': self.config.preferred_dollar_amounts
                }
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"💾 Saved fractional share config to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error saving fractional share state: {e}")
    
    def _load_state(self):
        """Load custom amounts from file"""
        try:
            if not os.path.exists(self.state_file):
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            self.custom_amounts = state.get('custom_amounts', {})
            
            # Optionally update config from saved state
            saved_config = state.get('config', {})
            if saved_config:
                self.config.enabled = saved_config.get('enabled', self.config.enabled)
                self.config.min_price_threshold = saved_config.get('min_price_threshold', self.config.min_price_threshold)
                self.config.max_price_threshold = saved_config.get('max_price_threshold', self.config.max_price_threshold)
                self.config.min_dollar_amount = saved_config.get('min_dollar_amount', self.config.min_dollar_amount)
                self.config.max_dollar_amount = saved_config.get('max_dollar_amount', self.config.max_dollar_amount)
                self.config.preferred_dollar_amounts = saved_config.get('preferred_dollar_amounts', self.config.preferred_dollar_amounts)
            
            if self.custom_amounts:
                logger.info(f"📂 Loaded {len(self.custom_amounts)} custom fractional amounts")
                for symbol, amount in self.custom_amounts.items():
                    logger.debug(f"   {symbol}: ${amount:.2f}")
            
        except Exception as e:
            logger.error(f"Error loading fractional share state: {e}")


# Singleton instance
_fractional_share_manager = None

def get_fractional_share_manager(config: Optional[FractionalShareConfig] = None) -> FractionalShareManager:
    """Get or create singleton FractionalShareManager"""
    global _fractional_share_manager
    
    if _fractional_share_manager is None:
        _fractional_share_manager = FractionalShareManager(config)
    
    return _fractional_share_manager


"""Signal validation and guardrails."""

from loguru import logger
from datetime import datetime
from typing import Dict, Tuple
from models.config import TradingConfig



class SignalValidator:
    """Validates trading signals against guardrails"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_orders = 0
        self.daily_risk = 0.0
        self.ticker_positions = {}
        self.last_reset = datetime.now().date()
    
    def reset_daily_counters(self):
        """Reset counters at start of new trading day"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_orders = 0
            self.daily_risk = 0.0
            self.ticker_positions = {}
            self.last_reset = today
            logger.info("Daily counters reset")
    
    def is_trading_hours(self) -> Tuple[bool, str]:
        """Check if current time is within trading hours"""
        now = datetime.now()
        start_time = now.replace(hour=self.config.trading_start_hour, 
                                 minute=self.config.trading_start_minute, second=0)
        end_time = now.replace(hour=self.config.trading_end_hour, 
                               minute=self.config.trading_end_minute, second=0)
        
        if now < start_time:
            return False, f"Before trading hours (starts at {start_time.strftime('%H:%M')})"
        if now > end_time:
            return False, f"After trading hours (ends at {end_time.strftime('%H:%M')})"
        
        return True, "Within trading hours"
    
    def validate_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Comprehensive signal validation"""
        self.reset_daily_counters()
        
        in_hours, hours_msg = self.is_trading_hours()
        if not in_hours:
            return False, hours_msg
        
        if self.daily_orders >= self.config.max_daily_orders:
            return False, f"Daily order limit reached ({self.config.max_daily_orders})"
        
        estimated_risk = signal.get('estimated_risk', 0)
        if self.daily_risk + estimated_risk > self.config.max_daily_risk:
            return False, f"Daily risk limit would be exceeded (${self.config.max_daily_risk})"
        
        ticker = signal.get('ticker', '').upper()
        if not ticker:
            return False, "Ticker is required"
        
        current_positions = self.ticker_positions.get(ticker, 0)
        if current_positions >= self.config.max_position_per_ticker:
            return False, f"Max positions for {ticker} reached ({self.config.max_position_per_ticker})"
        
        action = signal.get('action', '').upper()
        if action not in self.config.allowed_strategies:
            return False, f"Strategy {action} not in allowed list"
        
        dte = signal.get('dte')
        if dte is not None:
            if dte < self.config.min_dte or dte > self.config.max_dte:
                return False, f"DTE {dte} outside allowed range ({self.config.min_dte}-{self.config.max_dte})"
        
        qty = signal.get('qty', 0)
        if qty <= 0 or qty > 10:
            return False, f"Quantity {qty} must be between 1 and 10"
        
        return True, "Signal validated successfully"
    
    def record_order(self, signal: Dict):
        """Record an order for tracking"""
        self.daily_orders += 1
        self.daily_risk += signal.get('estimated_risk', 0)
        ticker = signal.get('ticker', '').upper()
        self.ticker_positions[ticker] = self.ticker_positions.get(ticker, 0) + 1

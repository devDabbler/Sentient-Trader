"""
Earnings Event Detector

Monitors upcoming earnings announcements and generates alerts
based on proximity to earnings date and position status.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf
from models.alerts import TradingAlert, AlertType, AlertPriority
from .base_detector import BaseEventDetector

logger = logging.getLogger(__name__)


class EarningsDetector(BaseEventDetector):
    """Detects upcoming earnings events for watchlist tickers"""
    
    def __init__(self, alert_system, ticker_manager=None, my_tickers_only: bool = True,
                 position_tracker=None, tradier_client=None, ibkr_client=None):
        """
        Initialize earnings detector
        
        Args:
            alert_system: AlertSystem instance
            ticker_manager: TickerManager instance
            my_tickers_only: Filter to watchlist only
            position_tracker: Optional position tracker to check active positions
            tradier_client: Optional Tradier client for position checking
            ibkr_client: Optional IBKR client for position checking
        """
        super().__init__(alert_system, ticker_manager, my_tickers_only)
        
        # Setup position tracker
        if position_tracker:
            self.position_tracker = position_tracker
        elif tradier_client or ibkr_client:
            from services.position_tracker import get_position_tracker
            self.position_tracker = get_position_tracker(tradier_client, ibkr_client)
        else:
            self.position_tracker = None
    
    def get_earnings_date(self, ticker: str) -> Optional[Dict]:
        """
        Get earnings date and info for a ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dict with earnings info or None
        """
        try:
            stock = yf.Ticker(ticker)
            calendar = stock.calendar
            
            if calendar is None or calendar.empty:
                return None
            
            # Extract earnings date
            if 'Earnings Date' in calendar.index:
                earnings_dates = calendar.loc['Earnings Date']
                
                # Handle multiple dates (range)
                if isinstance(earnings_dates, list) or hasattr(earnings_dates, '__iter__'):
                    earnings_date = earnings_dates[0] if len(earnings_dates) > 0 else None
                else:
                    earnings_date = earnings_dates
                
                if earnings_date is None:
                    return None
                
                # Convert to datetime if needed
                if isinstance(earnings_date, str):
                    earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d')
                
                # Get additional info
                info = stock.info
                
                return {
                    'ticker': ticker,
                    'earnings_date': earnings_date,
                    'days_until': (earnings_date - datetime.now()).days,
                    'eps_estimate': calendar.loc['EPS Estimate'].iloc[0] if 'EPS Estimate' in calendar.index else None,
                    'revenue_estimate': calendar.loc['Revenue Estimate'].iloc[0] if 'Revenue Estimate' in calendar.index else None,
                    'iv_rank': info.get('impliedVolatility', 0) * 100 if info.get('impliedVolatility') else None,
                    'last_eps': info.get('trailingEps'),
                    'forward_eps': info.get('forwardEps')
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not get earnings date for {ticker}: {e}")
            return None
    
    def has_active_position(self, ticker: str) -> bool:
        """
        Check if there's an active position for this ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            True if active position exists
        """
        if not self.position_tracker:
            return False
        
        try:
            return self.position_tracker.has_position(ticker)
        except Exception as e:
            logger.debug(f"Error checking position for {ticker}: {e}")
            return False
    
    def determine_priority(self, earnings_info: Dict, has_position: bool) -> AlertPriority:
        """
        Determine alert priority based on earnings proximity and position status
        
        Args:
            earnings_info: Earnings information dict
            has_position: Whether there's an active position
            
        Returns:
            AlertPriority level
        """
        days_until = earnings_info['days_until']
        
        # CRITICAL: Earnings in 1-3 days
        if 1 <= days_until <= 3:
            return AlertPriority.CRITICAL
        
        # HIGH: Earnings in 4-7 days OR has active position
        if 4 <= days_until <= 7 or (has_position and days_until <= 14):
            return AlertPriority.HIGH
        
        # MEDIUM: Earnings in 8-14 days
        if 8 <= days_until <= 14:
            return AlertPriority.MEDIUM
        
        # LOW: Earnings in 15-30 days
        return AlertPriority.LOW
    
    def create_earnings_alert(self, earnings_info: Dict) -> Optional[TradingAlert]:
        """
        Create an earnings alert
        
        Args:
            earnings_info: Earnings information dict
            
        Returns:
            TradingAlert or None
        """
        ticker = earnings_info['ticker']
        days_until = earnings_info['days_until']
        
        # Skip if earnings are too far out (>30 days) or in the past
        if days_until > 30 or days_until < 0:
            return None
        
        has_position = self.has_active_position(ticker)
        priority = self.determine_priority(earnings_info, has_position)
        
        # Build message
        date_str = earnings_info['earnings_date'].strftime('%Y-%m-%d')
        
        if days_until == 0:
            timing = "TODAY"
            emoji = "🔥"
        elif days_until == 1:
            timing = "TOMORROW"
            emoji = "⚠️"
        else:
            timing = f"in {days_until} days"
            emoji = "📅"
        
        message_parts = [f"{emoji} Earnings {timing} ({date_str})"]
        
        if earnings_info.get('eps_estimate'):
            message_parts.append(f"EPS Est: ${earnings_info['eps_estimate']:.2f}")
        
        if earnings_info.get('iv_rank'):
            message_parts.append(f"IV: {earnings_info['iv_rank']:.1f}%")
        
        if has_position:
            message_parts.append("⚠️ ACTIVE POSITION")
        
        message = " | ".join(message_parts)
        
        # Create alert
        alert = TradingAlert(
            ticker=ticker,
            alert_type=AlertType.EARNINGS_UPCOMING,
            priority=priority,
            message=message,
            confidence_score=0.0,
            details={
                'earnings_date': date_str,
                'days_until': days_until,
                'eps_estimate': earnings_info.get('eps_estimate'),
                'revenue_estimate': earnings_info.get('revenue_estimate'),
                'iv_rank': earnings_info.get('iv_rank'),
                'last_eps': earnings_info.get('last_eps'),
                'forward_eps': earnings_info.get('forward_eps'),
                'has_position': has_position
            }
        )
        
        return alert
    
    def detect(self) -> List[TradingAlert]:
        """
        Detect upcoming earnings events
        
        Returns:
            List of earnings alerts
        """
        alerts = []
        
        # Get watchlist tickers
        my_tickers = self._get_my_tickers()
        
        if not my_tickers:
            logger.warning("No tickers in watchlist for earnings detection")
            return alerts
        
        logger.info(f"Checking earnings for {len(my_tickers)} tickers...")
        
        for ticker in my_tickers:
            try:
                earnings_info = self.get_earnings_date(ticker)
                
                if earnings_info:
                    alert = self.create_earnings_alert(earnings_info)
                    
                    if alert:
                        alerts.append(alert)
                        self.trigger_alert(alert)
                        logger.info(f"Earnings alert: {ticker} in {earnings_info['days_until']} days")
            
            except Exception as e:
                logger.error(f"Error processing earnings for {ticker}: {e}")
                continue
        
        return alerts

"""
Economic Calendar Detector

Monitors major economic events that could impact market sectors
and watchlist tickers.
"""

from loguru import logger
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import requests
from models.alerts import TradingAlert, AlertType, AlertPriority
from .base_detector import BaseEventDetector



class EconomicDetector(BaseEventDetector):
    """Detects economic calendar events affecting watchlist sectors"""
    
    # Major economic indicators and their impact
    MAJOR_EVENTS = {
        'FOMC': {'priority': AlertPriority.CRITICAL, 'sectors': 'all'},
        'Fed Rate Decision': {'priority': AlertPriority.CRITICAL, 'sectors': 'all'},
        'CPI': {'priority': AlertPriority.CRITICAL, 'sectors': 'all'},
        'Non-Farm Payrolls': {'priority': AlertPriority.CRITICAL, 'sectors': 'all'},
        'Unemployment Rate': {'priority': AlertPriority.HIGH, 'sectors': 'all'},
        'GDP': {'priority': AlertPriority.HIGH, 'sectors': 'all'},
        'Retail Sales': {'priority': AlertPriority.HIGH, 'sectors': ['Consumer Cyclical', 'Consumer Defensive']},
        'Housing Starts': {'priority': AlertPriority.MEDIUM, 'sectors': ['Real Estate', 'Industrials']},
        'ISM Manufacturing': {'priority': AlertPriority.MEDIUM, 'sectors': ['Industrials', 'Materials']},
        'Consumer Confidence': {'priority': AlertPriority.MEDIUM, 'sectors': ['Consumer Cyclical']},
        'Oil Inventory': {'priority': AlertPriority.MEDIUM, 'sectors': ['Energy']},
    }
    
    def __init__(self, alert_system, ticker_manager=None, my_tickers_only: bool = True):
        """
        Initialize economic detector
        
        Args:
            alert_system: AlertSystem instance
            ticker_manager: TickerManager instance
            my_tickers_only: Filter to watchlist only
        """
        super().__init__(alert_system, ticker_manager, my_tickers_only)
        self.trading_economics_key = os.getenv('TRADING_ECONOMICS_API_KEY')
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    
    def get_watchlist_sectors(self) -> Set[str]:
        """
        Get unique sectors from watchlist tickers
        
        Returns:
            Set of sector names
        """
        sectors = set()
        
        if not self.ticker_manager:
            return sectors
        
        try:
            tickers = self.ticker_manager.get_all_tickers(limit=1000)
            for ticker_info in tickers:
                sector = ticker_info.get('sector')
                if sector:
                    sectors.add(sector)
        except Exception as e:
            logger.error(f"Error getting watchlist sectors: {e}")
        
        return sectors
    
    def get_finnhub_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """
        Get economic calendar from Finnhub
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of economic events
        """
        if not self.finnhub_api_key:
            return []
        
        try:
            from_date = datetime.now().strftime('%Y-%m-%d')
            to_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
            
            url = "https://finnhub.io/api/v1/calendar/economic"
            params = {
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            events = data.get('economicCalendar', [])
            
            return [{
                'event': item.get('event', ''),
                'date': datetime.strptime(item.get('time', ''), '%Y-%m-%d %H:%M:%S') if item.get('time') else None,
                'country': item.get('country', 'US'),
                'actual': item.get('actual'),
                'estimate': item.get('estimate'),
                'previous': item.get('previous'),
                'impact': item.get('impact', 'medium')
            } for item in events if item.get('country') == 'US']
            
        except Exception as e:
            logger.debug(f"Error getting Finnhub economic calendar: {e}")
            return []
    
    def get_mock_calendar(self, days_ahead: int = 7) -> List[Dict]:
        """
        Get mock economic calendar (fallback when no API available)
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of economic events
        """
        # Common recurring events (approximate schedule)
        today = datetime.now()
        events = []
        
        # CPI - usually mid-month
        if 10 <= today.day <= 15:
            cpi_date = today.replace(day=13)
            if cpi_date >= today and (cpi_date - today).days <= days_ahead:
                events.append({
                    'event': 'CPI',
                    'date': cpi_date.replace(hour=8, minute=30),
                    'country': 'US',
                    'impact': 'high'
                })
        
        # Non-Farm Payrolls - first Friday of month
        first_day = today.replace(day=1)
        days_until_friday = (4 - first_day.weekday()) % 7
        nfp_date = first_day + timedelta(days=days_until_friday)
        
        if nfp_date >= today and (nfp_date - today).days <= days_ahead:
            events.append({
                'event': 'Non-Farm Payrolls',
                'date': nfp_date.replace(hour=8, minute=30),
                'country': 'US',
                'impact': 'high'
            })
        
        return events
    
    def match_event_to_sectors(self, event_name: str) -> Optional[Set[str]]:
        """
        Match economic event to affected sectors
        
        Args:
            event_name: Name of economic event
            
        Returns:
            Set of affected sectors or None for all sectors
        """
        for key, info in self.MAJOR_EVENTS.items():
            if key.lower() in event_name.lower():
                sectors = info['sectors']
                if sectors == 'all':
                    return None  # Affects all sectors
                return set(sectors)
        
        return set()
    
    def is_relevant_to_watchlist(self, event_name: str) -> bool:
        """
        Check if event is relevant to watchlist sectors
        
        Args:
            event_name: Name of economic event
            
        Returns:
            True if relevant
        """
        affected_sectors = self.match_event_to_sectors(event_name)
        
        # If affects all sectors, always relevant
        if affected_sectors is None:
            return True
        
        # If no specific sectors, skip
        if not affected_sectors:
            return False
        
        # Check if any watchlist sectors are affected
        watchlist_sectors = self.get_watchlist_sectors()
        return bool(affected_sectors & watchlist_sectors)
    
    def determine_priority(self, event_name: str, days_until: int) -> AlertPriority:
        """
        Determine alert priority based on event importance and timing
        
        Args:
            event_name: Name of economic event
            days_until: Days until event
            
        Returns:
            AlertPriority level
        """
        # Get base priority from event type
        base_priority = AlertPriority.MEDIUM
        
        for key, info in self.MAJOR_EVENTS.items():
            if key.lower() in event_name.lower():
                base_priority = info['priority']
                break
        
        # Adjust based on timing
        if days_until == 0:
            # Event today - bump up priority
            if base_priority == AlertPriority.HIGH:
                return AlertPriority.CRITICAL
            return base_priority
        elif days_until == 1:
            # Event tomorrow - keep priority
            return base_priority
        elif days_until <= 3:
            # Event in 2-3 days - lower by one level
            if base_priority == AlertPriority.CRITICAL:
                return AlertPriority.HIGH
            elif base_priority == AlertPriority.HIGH:
                return AlertPriority.MEDIUM
            return AlertPriority.LOW
        else:
            # Event 4+ days away - informational only
            return AlertPriority.LOW
    
    def create_economic_alert(self, event: Dict) -> Optional[TradingAlert]:
        """
        Create an economic event alert
        
        Args:
            event: Economic event dict
            
        Returns:
            TradingAlert or None
        """
        event_name = event['event']
        event_date = event['date']
        
        if not event_date:
            return None
        
        # Check if relevant to watchlist
        if not self.is_relevant_to_watchlist(event_name):
            return None
        
        # Calculate days until event
        days_until = (event_date - datetime.now()).days
        
        # Skip events too far in future
        if days_until > 7:
            return None
        
        priority = self.determine_priority(event_name, days_until)
        
        # Build message
        if days_until == 0:
            timing = "TODAY"
            emoji = "🔥"
        elif days_until == 1:
            timing = "TOMORROW"
            emoji = "⚠️"
        else:
            timing = f"in {days_until} days"
            emoji = "📅"
        
        time_str = event_date.strftime('%I:%M %p ET')
        message = f"{emoji} Economic Event {timing} at {time_str}: {event_name}"
        
        # Add estimate/previous if available
        if event.get('estimate'):
            message += f" (Est: {event['estimate']})"
        
        # Determine affected sectors
        affected_sectors = self.match_event_to_sectors(event_name)
        sector_text = "All Sectors" if affected_sectors is None else ", ".join(list(affected_sectors)[:3])
        
        # Create alert (use ticker "SPY" for market-wide events)
        alert = TradingAlert(
            ticker="SPY",  # Market-wide indicator
            alert_type=AlertType.ECONOMIC_EVENT,
            priority=priority,
            message=message,
            confidence_score=0.0,
            details={
                'event': event_name,
                'date': event_date.isoformat(),
                'days_until': days_until,
                'country': event.get('country', 'US'),
                'estimate': event.get('estimate'),
                'previous': event.get('previous'),
                'impact': event.get('impact', 'medium'),
                'affected_sectors': sector_text
            }
        )
        
        return alert
    
    def detect(self) -> List[TradingAlert]:
        """
        Detect upcoming economic events
        
        Returns:
            List of economic event alerts
        """
        alerts = []
        
        logger.info("Checking economic calendar...")
        
        # Try Finnhub first
        events = self.get_finnhub_calendar(days_ahead=7)
        
        # Fallback to mock calendar
        if not events:
            logger.info("Using mock economic calendar (no API key)")
            events = self.get_mock_calendar(days_ahead=7)
        
        for event in events:
            try:
                alert = self.create_economic_alert(event)
                
                if alert:
                    alerts.append(alert)
                    self.trigger_alert(alert)
                    logger.info(f"Economic alert: {event['event']} in {event.get('days_until', '?')} days")
            
            except Exception as e:
                logger.error(f"Error processing economic event {event.get('event', 'Unknown')}: {e}")
                continue
        
        return alerts

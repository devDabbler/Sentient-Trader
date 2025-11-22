"""
Base Event Detector Class

Provides common functionality for all event detectors including
watchlist filtering, error handling, and logging.
"""

from loguru import logger
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Set, Optional
from models.alerts import TradingAlert, AlertType, AlertPriority



class BaseEventDetector(ABC):
    """Abstract base class for event detectors"""
    
    def __init__(self, alert_system, ticker_manager=None, my_tickers_only: bool = True):
        """
        Initialize base event detector
        
        Args:
            alert_system: AlertSystem instance to send notifications
            ticker_manager: TickerManager instance for filtering
            my_tickers_only: If True, only detect events for watchlist tickers
        """
        self.alert_system = alert_system
        self.ticker_manager = ticker_manager
        self.my_tickers_only = my_tickers_only
        self._my_tickers_cache: Optional[Set[str]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes
    
    def _get_my_tickers(self) -> Set[str]:
        """Get 'My Tickers' with caching (5 minute TTL)"""
        now = datetime.now()
        
        # Refresh cache if older than TTL
        if self._my_tickers_cache is None or self._cache_time is None or \
           (now - self._cache_time).total_seconds() > self._cache_ttl_seconds:
            if self.ticker_manager:
                try:
                    tickers_list = self.ticker_manager.get_all_tickers(limit=1000)
                    self._my_tickers_cache = set(t['ticker'].upper() for t in tickers_list)
                    self._cache_time = now
                    logger.info(f"My Tickers cache refreshed: {len(self._my_tickers_cache)} tickers")
                except Exception as e:
                    logger.error(f"Error loading My Tickers: {e}")
                    self._my_tickers_cache = set()
            else:
                self._my_tickers_cache = set()
        
        return self._my_tickers_cache or set()
    
    def filter_by_watchlist(self, tickers: List[str]) -> List[str]:
        """
        Filter tickers to only include those in watchlist
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Filtered list of tickers
        """
        if not self.my_tickers_only:
            return tickers
        
        my_tickers = self._get_my_tickers()
        filtered = [t.upper() for t in tickers if t.upper() in my_tickers]
        
        if len(filtered) < len(tickers):
            logger.debug(f"Filtered {len(tickers)} tickers to {len(filtered)} watchlist tickers")
        
        return filtered
    
    def should_alert(self, ticker: str) -> bool:
        """
        Check if alert should be generated for this ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            True if alert should be generated
        """
        if not self.my_tickers_only:
            return True
        
        my_tickers = self._get_my_tickers()
        return ticker.upper() in my_tickers
    
    def normalize_data(self, raw_data: Dict) -> Dict:
        """
        Normalize raw API data to standard format
        
        Args:
            raw_data: Raw data from API
            
        Returns:
            Normalized data dictionary
        """
        # Override in subclasses for specific data formats
        return raw_data
    
    def trigger_alert(self, alert: TradingAlert):
        """
        Trigger an alert through the alert system
        
        Args:
            alert: TradingAlert instance
        """
        try:
            self.alert_system.trigger_alert(alert)
            logger.info(f"Alert triggered: {alert.ticker} - {alert.alert_type.value}")
        except Exception as e:
            logger.error(f"Failed to trigger alert for {alert.ticker}: {e}")
    
    @abstractmethod
    def detect(self) -> List[TradingAlert]:
        """
        Detect events and generate alerts
        
        Returns:
            List of generated alerts
        """
        pass
    
    def run_detection(self) -> List[TradingAlert]:
        """
        Run detection with error handling
        
        Returns:
            List of generated alerts
        """
        try:
            logger.info(f"Running {self.__class__.__name__} detection...")
            alerts = self.detect()
            pass  # logger.info(f"{self.__class__.__name__} generated {len(alerts))} alerts")
            return alerts
        except Exception as e:
            logger.error("Error in {self.__class__.__name__}.detect(): {}", str(e), exc_info=True)
            return []

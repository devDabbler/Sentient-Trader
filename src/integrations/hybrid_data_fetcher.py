"""
Hybrid Data Fetcher - Combines IBKR and yfinance for optimal real-time data
Falls back to yfinance when IBKR provides delayed data (paper trading)
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)


class HybridDataFetcher:
    """
    Fetches market data from IBKR first, falls back to yfinance
    Useful for paper trading where IBKR provides delayed data
    """
    
    def __init__(self, ibkr_client=None, prefer_yfinance: bool = False):
        """
        Initialize hybrid data fetcher
        
        Args:
            ibkr_client: IBKR client instance (optional)
            prefer_yfinance: If True, use yfinance as primary source
        """
        self.ibkr_client = ibkr_client
        self.prefer_yfinance = prefer_yfinance
        self._cache = {}
        self._cache_duration = timedelta(seconds=5)  # 5-second cache
        
    def get_quote(self, symbol: str, use_ibkr: bool = True) -> Optional[Dict]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock ticker symbol
            use_ibkr: Whether to try IBKR first
            
        Returns:
            Dictionary with quote data or None
        """
        # Check cache first
        cache_key = f"{symbol}_{datetime.now().timestamp() // 5}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {symbol}")
            return self._cache[cache_key]
        
        quote = None
        
        # Try IBKR first if available and preferred
        if not self.prefer_yfinance and use_ibkr and self.ibkr_client:
            try:
                quote = self._get_ibkr_quote(symbol)
                
                # Check if data is delayed (more than 15 minutes old)
                if quote and self._is_delayed_data(quote):
                    logger.info(f"📊 IBKR data delayed for {symbol}, falling back to yfinance")
                    quote = None  # Force fallback
                    
            except Exception as e:
                logger.debug(f"IBKR quote fetch failed for {symbol}: {e}")
        
        # Fallback to yfinance
        if quote is None:
            try:
                quote = self._get_yfinance_quote(symbol)
            except Exception as e:
                logger.error(f"yfinance quote fetch failed for {symbol}: {e}")
        
        # Cache the result
        if quote:
            self._cache[cache_key] = quote
            # Clean old cache entries
            self._clean_cache()
        
        return quote
    
    def _get_ibkr_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote from IBKR"""
        if not self.ibkr_client:
            return None
        
        market_data = self.ibkr_client.get_market_data(symbol)
        if not market_data:
            return None
        
        return {
            'symbol': symbol,
            'last': market_data.get('last', 0),
            'bid': market_data.get('bid', 0),
            'ask': market_data.get('ask', 0),
            'volume': market_data.get('volume', 0),
            'timestamp': datetime.now(),
            'source': 'IBKR'
        }
    
    def _get_yfinance_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get fast_info for real-time data
            info = ticker.fast_info
            
            # Get current price
            last_price = info.get('lastPrice', 0)
            if not last_price:
                # Fallback to regular info
                info = ticker.info
                last_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            if not last_price:
                logger.warning(f"No price data available for {symbol} from yfinance")
                return None
            
            return {
                'symbol': symbol,
                'last': last_price,
                'bid': info.get('bid', last_price * 0.999),  # Estimate if not available
                'ask': info.get('ask', last_price * 1.001),  # Estimate if not available
                'volume': info.get('volume', 0),
                'timestamp': datetime.now(),
                'source': 'yfinance'
            }
            
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")
            return None
    
    def _is_delayed_data(self, quote: Dict) -> bool:
        """
        Check if quote data is delayed (>15 minutes old)
        
        Args:
            quote: Quote dictionary
            
        Returns:
            True if data appears delayed
        """
        # Check timestamp if available
        timestamp = quote.get('timestamp')
        if timestamp and isinstance(timestamp, datetime):
            age = datetime.now() - timestamp
            if age > timedelta(minutes=15):
                return True
        
        # Check for common delayed data indicators
        # IBKR delayed data often has 0 bid/ask or very old prices
        if quote.get('bid', 0) == 0 and quote.get('ask', 0) == 0:
            return True
        
        return False
    
    def _clean_cache(self):
        """Remove old cache entries"""
        current_time = datetime.now().timestamp() // 5
        keys_to_remove = [
            key for key in self._cache.keys()
            if not key.endswith(str(int(current_time)))
            and not key.endswith(str(int(current_time - 1)))
        ]
        for key in keys_to_remove:
            del self._cache[key]
    
    def get_historical_data(self, symbol: str, period: str = "1d", interval: str = "1m") -> Optional[Dict]:
        """
        Get historical/intraday data
        
        Args:
            symbol: Stock ticker
            period: Data period (1d, 5d, 1mo, etc.)
            interval: Data interval (1m, 5m, 1h, etc.)
            
        Returns:
            Dictionary with historical data or None
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return None
            
            return {
                'symbol': symbol,
                'data': hist,
                'source': 'yfinance'
            }
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

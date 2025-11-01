"""
News Event Detector

Monitors news feeds for major events affecting watchlist tickers.
Supports multiple news sources: Finnhub, NewsAPI, and yfinance.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
import yfinance as yf
from models.alerts import TradingAlert, AlertType, AlertPriority
from .base_detector import BaseEventDetector

logger = logging.getLogger(__name__)


class NewsDetector(BaseEventDetector):
    """Detects major news events for watchlist tickers"""
    
    # Keywords that indicate critical news
    CRITICAL_KEYWORDS = [
        'bankruptcy', 'sec investigation', 'fraud', 'lawsuit', 'recall',
        'fda rejection', 'delisting', 'investigation', 'scandal', 'halt'
    ]
    
    # Keywords that indicate positive news
    POSITIVE_KEYWORDS = [
        'breakthrough', 'approval', 'partnership', 'acquisition', 'merger',
        'beat estimates', 'record revenue', 'expansion', 'innovation', 'award'
    ]
    
    def __init__(self, alert_system, ticker_manager=None, my_tickers_only: bool = True):
        """
        Initialize news detector
        
        Args:
            alert_system: AlertSystem instance
            ticker_manager: TickerManager instance
            my_tickers_only: Filter to watchlist only
        """
        super().__init__(alert_system, ticker_manager, my_tickers_only)
        self.finnhub_api_key = os.getenv('FINNHUB_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
    
    def get_yfinance_news(self, ticker: str, hours_back: int = 24) -> List[Dict]:
        """
        Get news from yfinance (free, no API key needed)
        
        Args:
            ticker: Ticker symbol
            hours_back: How many hours back to check
            
        Returns:
            List of news items
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return []
            
            # Filter by time
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_news = []
            
            for item in news:
                pub_time = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                if pub_time >= cutoff_time:
                    recent_news.append({
                        'title': item.get('title', ''),
                        'publisher': item.get('publisher', 'Unknown'),
                        'link': item.get('link', ''),
                        'published': pub_time,
                        'summary': item.get('summary', '')
                    })
            
            return recent_news
            
        except Exception as e:
            logger.debug(f"Error getting yfinance news for {ticker}: {e}")
            return []
    
    def get_finnhub_news(self, ticker: str, hours_back: int = 24) -> List[Dict]:
        """
        Get news from Finnhub API
        
        Args:
            ticker: Ticker symbol
            hours_back: How many hours back to check
            
        Returns:
            List of news items
        """
        if not self.finnhub_api_key:
            return []
        
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': from_date,
                'to': to_date,
                'token': self.finnhub_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            news_items = response.json()
            
            return [{
                'title': item.get('headline', ''),
                'publisher': item.get('source', 'Unknown'),
                'link': item.get('url', ''),
                'published': datetime.fromtimestamp(item.get('datetime', 0)),
                'summary': item.get('summary', '')
            } for item in news_items]
            
        except Exception as e:
            logger.debug(f"Error getting Finnhub news for {ticker}: {e}")
            return []
    
    def analyze_sentiment(self, news_item: Dict) -> Dict:
        """
        Analyze news sentiment based on keywords
        
        Args:
            news_item: News item dict
            
        Returns:
            Dict with sentiment analysis
        """
        title = news_item.get('title', '').lower()
        summary = news_item.get('summary', '').lower()
        text = f"{title} {summary}"
        
        # Check for critical keywords
        critical_matches = [kw for kw in self.CRITICAL_KEYWORDS if kw in text]
        positive_matches = [kw for kw in self.POSITIVE_KEYWORDS if kw in text]
        
        if critical_matches:
            return {
                'sentiment': 'negative',
                'severity': 'critical',
                'keywords': critical_matches,
                'score': -0.8
            }
        elif positive_matches:
            return {
                'sentiment': 'positive',
                'severity': 'high',
                'keywords': positive_matches,
                'score': 0.7
            }
        else:
            return {
                'sentiment': 'neutral',
                'severity': 'medium',
                'keywords': [],
                'score': 0.0
            }
    
    def determine_priority(self, sentiment: Dict, news_count: int) -> AlertPriority:
        """
        Determine alert priority based on sentiment and news volume
        
        Args:
            sentiment: Sentiment analysis dict
            news_count: Number of news items
            
        Returns:
            AlertPriority level
        """
        if sentiment['severity'] == 'critical':
            return AlertPriority.CRITICAL
        
        if sentiment['sentiment'] == 'positive' and news_count >= 3:
            return AlertPriority.HIGH
        
        if sentiment['sentiment'] == 'positive':
            return AlertPriority.MEDIUM
        
        return AlertPriority.LOW
    
    def create_news_alert(self, ticker: str, news_items: List[Dict]) -> Optional[TradingAlert]:
        """
        Create a news alert
        
        Args:
            ticker: Ticker symbol
            news_items: List of news items
            
        Returns:
            TradingAlert or None
        """
        if not news_items:
            return None
        
        # Analyze most recent news item
        latest_news = news_items[0]
        sentiment = self.analyze_sentiment(latest_news)
        priority = self.determine_priority(sentiment, len(news_items))
        
        # Build message
        emoji_map = {
            'critical': '🚨',
            'high': '📰',
            'medium': '📢',
            'low': 'ℹ️'
        }
        emoji = emoji_map.get(sentiment['severity'], '📰')
        
        sentiment_text = sentiment['sentiment'].upper()
        if sentiment['keywords']:
            keyword_text = f" ({', '.join(sentiment['keywords'][:2])})"
        else:
            keyword_text = ""
        
        message = f"{emoji} {sentiment_text} News{keyword_text}: {latest_news['title'][:100]}"
        
        # Create alert
        alert = TradingAlert(
            ticker=ticker,
            alert_type=AlertType.MAJOR_NEWS,
            priority=priority,
            message=message,
            confidence_score=abs(sentiment['score']) * 100,
            details={
                'news_count': len(news_items),
                'sentiment': sentiment['sentiment'],
                'sentiment_score': sentiment['score'],
                'keywords': sentiment['keywords'],
                'title': latest_news['title'],
                'publisher': latest_news['publisher'],
                'link': latest_news['link'],
                'published': latest_news['published'].isoformat()
            }
        )
        
        return alert
    
    def detect(self) -> List[TradingAlert]:
        """
        Detect major news events
        
        Returns:
            List of news alerts
        """
        alerts = []
        
        # Get watchlist tickers
        my_tickers = self._get_my_tickers()
        
        if not my_tickers:
            logger.warning("No tickers in watchlist for news detection")
            return alerts
        
        logger.info(f"Checking news for {len(my_tickers)} tickers...")
        
        for ticker in my_tickers:
            try:
                # Try yfinance first (free)
                news_items = self.get_yfinance_news(ticker, hours_back=24)
                
                # Fallback to Finnhub if available
                if not news_items and self.finnhub_api_key:
                    news_items = self.get_finnhub_news(ticker, hours_back=24)
                
                if news_items:
                    alert = self.create_news_alert(ticker, news_items)
                    
                    if alert and alert.priority in [AlertPriority.CRITICAL, AlertPriority.HIGH]:
                        # Only alert on significant news
                        alerts.append(alert)
                        self.trigger_alert(alert)
                        logger.info(f"News alert: {ticker} - {len(news_items)} items")
            
            except Exception as e:
                logger.error(f"Error processing news for {ticker}: {e}")
                continue
        
        return alerts

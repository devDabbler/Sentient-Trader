"""Caching utilities for stock data and news."""

import pandas as pd
import yfinance as yf
from loguru import logger
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import threading
import os
import requests
from utils.helpers import fetch_ticker_history, fetch_ticker_info, fetch_ticker_news

# Manual cache for background processes (non-Streamlit context)
_cache_lock = threading.Lock()
_manual_cache: Dict[str, Tuple[any, datetime]] = {}
_cache_ttl = {
    'stock_data': 60,      # 1 minute
    'news': 180,           # 3 minutes
}

def _get_from_manual_cache(key: str, ttl_seconds: int) -> any:
    """Get value from manual cache if not expired"""
    with _cache_lock:
        if key in _manual_cache:
            value, timestamp = _manual_cache[key]
            if datetime.now() - timestamp < timedelta(seconds=ttl_seconds):
                return value
            else:
                del _manual_cache[key]  # Expired, remove it
    return None

def _set_manual_cache(key: str, value: any) -> None:
    """Set value in manual cache with timestamp"""
    with _cache_lock:
        _manual_cache[key] = (value, datetime.now())

def _try_streamlit_cache(func, ttl: int):
    """Try to use Streamlit cache if available, otherwise use manual cache"""
    try:
        import streamlit as st
        return st.cache_data(ttl=ttl)(func)
    except (ImportError, RuntimeError):
        # Streamlit not available or not in Streamlit context
        return func

# Stock data caching
def get_cached_stock_data(ticker: str):
    """Cache stock data to improve performance - refreshes every minute for real-time accuracy"""
    # Try manual cache first (for background processes)
    cache_key = f"stock_data_{ticker}"
    cached = _get_from_manual_cache(cache_key, _cache_ttl['stock_data'])
    if cached is not None:
        return cached
    
    try:
        stock = yf.Ticker(ticker)
        hist = fetch_ticker_history(stock, period="3mo", timeout=10)  # 10s timeout
        info = fetch_ticker_info(stock)
        result = (hist, info)
        _set_manual_cache(cache_key, result)
        return result
    except Exception as e:
        logger.error(f"Error fetching cached data for {ticker}: {e}")
        # Return empty DataFrame and empty info to avoid NoneType errors
        return pd.DataFrame(), {}


def get_cached_news(ticker: str) -> List[Dict]:
    """Cache news data to improve performance - works in both Streamlit and background contexts
    
    Uses yfinance as primary source with Finnhub fallback for OTC/penny stocks
    that often have no news coverage on Yahoo Finance.
    """
    # Try manual cache first (for background processes)
    cache_key = f"news_{ticker}"
    cached = _get_from_manual_cache(cache_key, _cache_ttl['news'])
    if cached is not None:
        logger.debug(f"Using cached news for {ticker}")
        return cached
    
    try:
        logger.info(f"Fetching news for {ticker}")
        stock = yf.Ticker(ticker)
        
        # Add timeout to prevent hanging
        news = None
        try:
            news = fetch_ticker_news(stock)
        except Exception as fetch_error:
            logger.warning(f"Failed to fetch news for {ticker}: {fetch_error}")
        
        # If yfinance returns no news, try Finnhub as fallback (especially for OTC stocks)
        if not news:
            logger.info(f"No yfinance news for {ticker}, trying Finnhub fallback...")
            news = _get_finnhub_news_fallback(ticker)
        
        if not news:
            logger.warning(f"No news found for {ticker} from any source")
            _set_manual_cache(cache_key, [])
            return []
        
        # Limit to 10 articles for better coverage
        result = news[:10] if isinstance(news, list) else []
        logger.info(f"Retrieved {len(result)} news articles for {ticker}")
        
        _set_manual_cache(cache_key, result)
        return result
        
    except Exception as e:
        logger.error("Error fetching cached news for {ticker}: {}", str(e), exc_info=True)
        return []


def _get_finnhub_news_fallback(ticker: str) -> List[Dict]:
    """Fallback to Finnhub API for news when yfinance has no coverage (OTC/penny stocks)"""
    finnhub_api_key = os.getenv('FINNHUB_API_KEY')
    if not finnhub_api_key:
        logger.debug(f"Finnhub API key not configured, skipping fallback for {ticker}")
        return []
    
    try:
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': ticker,
            'from': from_date,
            'to': to_date,
            'token': finnhub_api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        news_items = response.json()
        
        if not news_items:
            logger.debug(f"Finnhub returned no news for {ticker}")
            return []
        
        # Convert Finnhub format to yfinance-compatible format
        result = []
        for item in news_items:
            result.append({
                'title': item.get('headline', ''),
                'publisher': item.get('source', 'Unknown'),
                'link': item.get('url', ''),
                'providerPublishTime': item.get('datetime', 0),
                'summary': item.get('summary', ''),
                'source': 'finnhub'  # Mark source for debugging
            })
        
        logger.info(f"Finnhub returned {len(result)} news articles for {ticker}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.debug(f"Finnhub API request failed for {ticker}: {e}")
        return []
    except Exception as e:
        logger.debug(f"Error getting Finnhub news for {ticker}: {e}")
        return []

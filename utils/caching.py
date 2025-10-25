"""Caching utilities for stock data and news."""

import streamlit as st
import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


@st.cache_data(ttl=60)  # Cache for 1 minute for more real-time data
def get_cached_stock_data(ticker: str):
    """Cache stock data to improve performance - refreshes every minute for real-time accuracy"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        info = stock.info
        return hist, info
    except Exception as e:
        logger.error(f"Error fetching cached data for {ticker}: {e}")
        # Return empty DataFrame and empty info to avoid NoneType errors
        return pd.DataFrame(), {}


@st.cache_data(ttl=180)  # Cache for 3 minutes
def get_cached_news(ticker: str):
    """Cache news data to improve performance"""
    try:
        logger.info(f"Fetching news for {ticker}")
        stock = yf.Ticker(ticker)
        news = stock.news
        logger.info(f"Retrieved {len(news) if news else 0} news articles for {ticker}")
        
        if not news:
            logger.warning(f"No news found for {ticker}")
            return []
            
        return news[:10]  # Increased to 10 articles for better coverage
    except Exception as e:
        logger.error(f"Error fetching cached news for {ticker}: {e}")
        return []

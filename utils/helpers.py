"""Helper utility functions."""

import functools
import time
import yfinance as yf
from datetime import datetime
from loguru import logger



def calculate_dte(expiry_date: str) -> int:
    """Calculate days to expiration"""
    try:
        exp_date = datetime.strptime(expiry_date, '%Y-%m-%d')
        return (exp_date - datetime.now()).days
    except Exception as e:
        logger.error(f"Error calculating DTE: {e}")
        return 0

def retry_on_rate_limit(max_retries=3, initial_delay=2.0, backoff_factor=2.0):
    """Decorator to retry functions on API rate limit errors"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for i in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check for common rate limit indicators
                    is_rate_limit = (
                        "too many requests" in error_msg or 
                        "rate limited" in error_msg or 
                        "429" in error_msg
                    )
                    
                    if is_rate_limit and i < max_retries:
                        logger.warning(f"Rate limited in {func.__name__}. Retrying in {delay:.1f}s... (Attempt {i+1}/{max_retries})")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise e
            
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator

@retry_on_rate_limit()
def fetch_ticker_info(ticker_obj):
    """Safely fetch ticker info with retries"""
    return ticker_obj.info

@retry_on_rate_limit()
def fetch_ticker_history(ticker_obj, **kwargs):
    """Safely fetch ticker history with retries"""
    return ticker_obj.history(**kwargs)

@retry_on_rate_limit()
def fetch_ticker_news(ticker_obj):
    """Safely fetch ticker news with retries"""
    return ticker_obj.news

@retry_on_rate_limit()
def fetch_ticker_options(ticker_obj):
    """Safely fetch ticker options dates with retries"""
    return ticker_obj.options

@retry_on_rate_limit()
def fetch_ticker_option_chain(ticker_obj, date):
    """Safely fetch ticker option chain with retries"""
    return ticker_obj.option_chain(date)

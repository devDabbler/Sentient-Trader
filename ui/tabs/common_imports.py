"""
Common Imports for Tab Modules
Centralized imports to reduce duplication across tabs
"""

# Standard library imports
import sys
import os
import asyncio
import io
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from loguru import logger

# Windows-specific asyncio policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# Broker/Trading integrations
try:
    from src.integrations.tradier_client import TradierClient, validate_tradier_connection, validate_all_trading_modes
except ImportError:
    logger.debug("Tradier client imports not available")

try:
    from src.integrations.trading_config import get_trading_mode_manager, TradingMode, switch_to_paper_mode, switch_to_production_mode
except ImportError:
    logger.warning("Trading config imports not available")

# Service imports
try:
    from services.llm_strategy_analyzer import LLMStrategyAnalyzer, StrategyAnalysis
    from services.watchlist_manager import WatchlistManager
    from services.ai_trading_signals import AITradingSignalGenerator, TradingSignal
    from services.ticker_manager import TickerManager
    from services.top_trades_scanner import TopTradesScanner, TopTrade
    from services.ai_confidence_scanner import AIConfidenceScanner, AIConfidenceTrade
    from services.alpha_factors import AlphaFactorCalculator
    from services.ml_enhanced_scanner import MLEnhancedScanner, MLEnhancedTrade
    from services.penny_stock_analyzer import PennyStockScorer, PennyStockAnalyzer, StockScores
    from services.unified_penny_stock_analysis import UnifiedPennyStockAnalysis
    from services.penny_stock_constants import PENNY_THRESHOLDS, is_penny_stock, PENNY_STOCK_FILTER_PRESETS
    from services.advanced_opportunity_scanner import AdvancedOpportunityScanner, ScanType, ScanFilters, OpportunityResult
except ImportError as e:
    logger.warning(f"Some service imports not available: {e}")
    # Provide fallback for ScanType if import fails
    class ScanType:
        ALL = "ALL"
        OPTIONS = "OPTIONS"
        PENNY_STOCKS = "PENNY_STOCKS"
        BREAKOUTS = "BREAKOUTS"
        MOMENTUM = "MOMENTUM"
        BUZZING = "BUZZING"
        HOTTEST_STOCKS = "HOTTEST_STOCKS"

# Analyzer imports
try:
    from analyzers.comprehensive import ComprehensiveAnalyzer
    from models.analysis import StockAnalysis
    from analyzers.trading_styles import TradingStyleAnalyzer
    from analyzers.news import NewsAnalyzer
    from services.social_sentiment_analyzer import SocialSentimentAnalyzer
    from services.event_detectors.sec_detector import SECDetector
    from services.enhanced_catalyst_detector import EnhancedCatalystDetector
except ImportError as e:
    logger.warning(f"Some analyzer imports not available: {e}")

# Crypto imports
try:
    from clients.kraken_client import KrakenClient
except ImportError:
    logger.debug("Kraken client not available")

try:
    # Note: display_crypto_quick_trade doesn't exist, use render_quick_trade_tab instead
    from ui.crypto_ai_monitor_ui import display_crypto_ai_monitors
    from ui.crypto_entry_monitors_ui import display_crypto_entry_monitors
    from ui.crypto_signal_ui import display_crypto_signals
    from ui.crypto_watchlist_ui import display_crypto_watchlist
except ImportError as e:
    logger.debug(f"Crypto UI imports not available: {e}")

try:
    from services.crypto_scanner import CryptoOpportunityScanner
    from services.ai_crypto_scanner import AICryptoScanner
    from services.penny_crypto_scanner import PennyCryptoScanner
    from services.sub_penny_discovery import SubPennyDiscovery
    from services.crypto_sentiment_analyzer import CryptoSentimentAnalyzer
except ImportError as e:
    logger.debug(f"Crypto service imports not available: {e}")

# Caching functions
@st.cache_data(ttl=60)
def get_cached_stock_data(ticker: str):
    """Cache stock data to improve performance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        info = stock.info
        return hist, info
    except Exception as e:
        logger.error(f"Error fetching cached data for {ticker}: {e}")
        return pd.DataFrame(), {}

@st.cache_resource
def get_ticker_manager():
    """Cached TickerManager instance"""
    try:
        from services.ticker_manager import TickerManager
        return TickerManager()
    except Exception as e:
        logger.error(f"Error creating TickerManager: {e}")
        return None

@st.cache_resource
def get_base_scanner():
    """Cached TopTradesScanner"""
    try:
        from services.top_trades_scanner import TopTradesScanner
        return TopTradesScanner()
    except Exception as e:
        logger.error(f"Error creating scanner: {e}")
        return None

@st.cache_resource
def get_llm_analyzer():
    """Cached LLMStrategyAnalyzer"""
    try:
        from services.llm_strategy_analyzer import LLMStrategyAnalyzer
        from utils.config_loader import get_api_key
        api_key = get_api_key('OPENROUTER_API_KEY', 'openrouter')
        model = os.getenv('AI_ANALYZER_MODEL') or get_api_key('AI_ANALYZER_MODEL', 'models') or 'google/gemini-2.5-flash'
        return LLMStrategyAnalyzer(api_key=api_key, model=model)
    except Exception as e:
        logger.error(f"Error creating LLM analyzer: {e}")
        return None

@st.cache_resource
def get_social_analyzer():
    """Cached SocialSentimentAnalyzer"""
    try:
        from services.social_sentiment_analyzer import SocialSentimentAnalyzer
        return SocialSentimentAnalyzer()
    except Exception as e:
        logger.error(f"Error creating social analyzer: {e}")
        return None

@st.cache_resource
def get_ai_scanner():
    """Cached AIConfidenceScanner - reuses shared base scanner and LLM analyzer"""
    try:
        from services.ai_confidence_scanner import AIConfidenceScanner
        base = get_base_scanner()
        llm = get_llm_analyzer()
        return AIConfidenceScanner(base_scanner=base, llm_analyzer=llm)
    except Exception as e:
        logger.error(f"Error creating AI scanner: {e}")
        return None

@st.cache_resource
def get_ml_scanner():
    """Cached MLEnhancedScanner - reuses AI scanner"""
    try:
        from services.ml_enhanced_scanner import MLEnhancedScanner
        ai = get_ai_scanner()
        return MLEnhancedScanner(ai_scanner=ai)
    except Exception as e:
        logger.error(f"Error creating ML scanner: {e}")
        return None

@st.cache_resource
def get_advanced_scanner():
    """Cached AdvancedOpportunityScanner - reuses shared scanners"""
    try:
        from services.advanced_opportunity_scanner import AdvancedOpportunityScanner
        base = get_base_scanner()
        ai = get_ai_scanner()
        social = get_social_analyzer()
        return AdvancedOpportunityScanner(use_ai=True, base_scanner=base, ai_scanner=ai, social_analyzer=social)
    except Exception as e:
        logger.error(f"Error creating advanced scanner: {e}")
        return None

# Export all
__all__ = [
    # Standard modules
    'sys', 'os', 'asyncio', 'io', 'json', 'time',
    'datetime', 'timedelta', 'timezone',
    'Dict', 'List', 'Optional', 'Tuple', 'Any',
    'dataclass', 'Enum',
    # Third party
    'st', 'pd', 'np', 'yf', 'requests', 'logger',
    # Cache functions
    'get_cached_stock_data', 'get_ticker_manager', 'get_base_scanner', 'get_llm_analyzer',
    'get_social_analyzer', 'get_ai_scanner', 'get_ml_scanner', 'get_advanced_scanner',
    # Scanner types and classes
    'ScanType', 'ComprehensiveAnalyzer', 'PENNY_THRESHOLDS'
]

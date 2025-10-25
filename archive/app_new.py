"""
Sentient Trader Platform
Now with Modular Architecture for Better Maintainability
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging and compatibility
from utils.logging_config import setup_logging, logger
setup_logging()

from utils.streamlit_compat import setup_streamlit_compatibility
setup_streamlit_compatibility()

# Import modular components
from models import MarketCondition, StockAnalysis, StrategyRecommendation, TradingConfig
from analyzers import TechnicalAnalyzer, NewsAnalyzer, ComprehensiveAnalyzer, StrategyAdvisor
from clients import OptionAlphaClient, SignalValidator
from utils.caching import get_cached_stock_data, get_cached_news
from utils.styling import apply_custom_styling
from utils.helpers import calculate_dte

# Import external modules
from tradier_client import TradierClient, validate_tradier_connection
from llm_strategy_analyzer import LLMStrategyAnalyzer, StrategyAnalysis, extract_bot_config_from_screenshot, create_strategy_comparison
from penny_stock_analyzer import PennyStockScorer, PennyStockAnalyzer, StockScores
from watchlist_manager import WatchlistManager
from ticker_manager import TickerManager
from top_trades_scanner import TopTradesScanner, TopTrade
from ai_confidence_scanner import AIConfidenceScanner, AIConfidenceTrade

# All class definitions and analyzer logic are now in modular files:
# - models/ : Data models (MarketCondition, StockAnalysis, StrategyRecommendation, TradingConfig)
# - analyzers/ : Analysis logic (TechnicalAnalyzer, NewsAnalyzer, ComprehensiveAnalyzer, StrategyAdvisor)
# - clients/ : External clients (OptionAlphaClient, SignalValidator)
# - utils/ : Utilities (caching, logging, styling, helpers)


"""
Sentient Trader - Main Entry Point (Navigation Only)
Refactored from 13,393-line monolithic file to modular architecture
"""

from dotenv import load_dotenv
load_dotenv()

from utils.logging_config import setup_logging, get_broker_specific_log_file
setup_logging(log_file=get_broker_specific_log_file("logs/sentient_trader.log"))

import streamlit as st
from loguru import logger
import os
import sys

# Add src directory to path for integrations
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Windows-specific asyncio policy
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

st.set_page_config(page_title="Sentient Trader", page_icon="📈", layout="wide")

# Import all tab modules
try:
    from ui.tabs import (
        dashboard_tab, scanner_tab, watchlist_tab, strategy_advisor_tab,
        generate_signal_tab, signal_history_tab, strategy_guide_tab,
        tradier_tab, ibkr_tab, scalping_tab, strategy_analyzer_tab,
        autotrader_tab, crypto_tab, dca_tab
    )
except ImportError as e:
    st.error(f"Failed to import tab modules: {e}")
    st.info("Make sure all tab modules have been extracted from app.py")
    st.stop()

def get_default_tab():
    """Get default tab based on DEFAULT_BROKER env var"""
    broker = os.getenv('DEFAULT_BROKER', '').upper()
    if broker == 'TRADIER':
        return "🏦 Tradier Account"
    elif broker == 'IBKR':
        return "📈 IBKR Trading"
    elif broker == 'KRAKEN':
        return "₿ Crypto Trading"
    return "🔍 Stock Intelligence"

def initialize_session_state():
    """Initialize all session state variables with graceful error handling"""
    try:
        # Basic state variables (always initialize these)
        if 'signal_history' not in st.session_state:
            st.session_state.signal_history = []
        
        if 'paper_mode' not in st.session_state:
            st.session_state.paper_mode = True
        
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        
        if 'show_quick_trade' not in st.session_state:
            st.session_state.show_quick_trade = False
        
        if 'selected_recommendation' not in st.session_state:
            st.session_state.selected_recommendation = None
        
        # Auto-refresh settings
        if 'auto_refresh_enabled' not in st.session_state:
            st.session_state.auto_refresh_enabled = False
        
        if 'last_auto_refresh_time' not in st.session_state:
            st.session_state.last_auto_refresh_time = 0
        
        # Ticker cache
        if 'ticker_cache' not in st.session_state:
            st.session_state.ticker_cache = {}
            st.session_state.ticker_cache_timestamp = None
            st.session_state.ticker_cache_ttl = 300  # 5 minutes
        
        # Analysis update cache
        if 'analysis_update_cache' not in st.session_state:
            st.session_state.analysis_update_cache = {}
            st.session_state.analysis_update_cache_timestamp = {}
        
        # Trading mode (with error handling for import)
        if 'trading_mode' not in st.session_state:
            try:
                from src.integrations.trading_config import TradingMode
                st.session_state.trading_mode = TradingMode.PAPER
                logger.info("🔒 Initialized trading_mode to PAPER (safe default)")
            except ImportError as e:
                logger.warning(f"Could not import TradingMode: {e}. Using string fallback.")
                st.session_state.trading_mode = "PAPER"
        
        # Background update queue
        if 'update_queue' not in st.session_state:
            from queue import Queue
            st.session_state.update_queue = Queue()
            st.session_state.background_processor_started = False
            st.session_state.background_update_results = {}
        
        # Cached services (with error handling)
        if 'ticker_manager' not in st.session_state:
            try:
                from ui.tabs.common_imports import get_ticker_manager
                st.session_state.ticker_manager = get_ticker_manager()
            except Exception as e:
                logger.debug(f"Could not initialize ticker_manager: {e}")
                st.session_state.ticker_manager = None
        
        if 'llm_analyzer' not in st.session_state:
            try:
                from ui.tabs.common_imports import get_llm_analyzer
                st.session_state.llm_analyzer = get_llm_analyzer()
            except Exception as e:
                logger.debug(f"Could not initialize llm_analyzer: {e}")
                st.session_state.llm_analyzer = None
        
        if 'news_analyzer' not in st.session_state:
            try:
                from analyzers.news import NewsAnalyzer
                st.session_state.news_analyzer = NewsAnalyzer(llm_analyzer=st.session_state.llm_analyzer)
            except Exception as e:
                logger.debug(f"Could not initialize news_analyzer: {e}")
                st.session_state.news_analyzer = None
        
        if 'social_sentiment_analyzer' not in st.session_state:
            try:
                from services.social_sentiment_analyzer import SocialSentimentAnalyzer
                st.session_state.social_sentiment_analyzer = SocialSentimentAnalyzer()
            except Exception as e:
                logger.debug(f"Could not initialize social_sentiment_analyzer: {e}")
                st.session_state.social_sentiment_analyzer = None
        
        # Scanner services (use cached getter functions)
        if 'ai_scanner' not in st.session_state:
            try:
                from ui.tabs.common_imports import get_ai_scanner
                st.session_state.ai_scanner = get_ai_scanner()
            except Exception as e:
                logger.debug(f"Could not initialize ai_scanner: {e}")
                st.session_state.ai_scanner = None
        
        if 'ml_scanner' not in st.session_state:
            try:
                from ui.tabs.common_imports import get_ml_scanner
                st.session_state.ml_scanner = get_ml_scanner()
            except Exception as e:
                logger.debug(f"Could not initialize ml_scanner: {e}")
                st.session_state.ml_scanner = None
        
        if 'advanced_scanner' not in st.session_state:
            try:
                from ui.tabs.common_imports import get_advanced_scanner
                st.session_state.advanced_scanner = get_advanced_scanner()
            except Exception as e:
                logger.debug(f"Could not initialize advanced_scanner: {e}")
                st.session_state.advanced_scanner = None
        
        logger.info("✅ Session state initialized successfully")
        
    except Exception as e:
        logger.error("Error during session state initialization: {}", str(e), exc_info=True)
        # Ensure critical variables are still set
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'paper_mode' not in st.session_state:
            st.session_state.paper_mode = True

def main():
    # Initialize session state (must be first!)
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📈 Sentient Trader")
        st.caption("AI-Driven Trading Platform")
        st.divider()
    
    # Tab names
    tab_names = [
        "🔍 Stock Intelligence",
        "🚀 Advanced Scanner",
        "⭐ My Tickers",
        "💰 Fractional DCA",
        "🎯 Strategy Advisor",
        "📊 Generate Signal",
        "📜 Signal History",
        "📚 Strategy Guide",
        "🏦 Tradier Account",
        "📈 IBKR Trading",
        "⚡ Scalping/Day Trade",
        "🤖 Strategy Analyzer",
        "🤖 Auto-Trader",
        "₿ Crypto Trading"
    ]
    
    # Initialize active tab
    if 'active_main_tab' not in st.session_state:
        st.session_state.active_main_tab = get_default_tab()
    
    # Navigation radio
    selected_tab = st.radio(
        "Select Section:",
        tab_names,
        index=tab_names.index(st.session_state.active_main_tab),
        horizontal=True
    )
    st.session_state.active_main_tab = selected_tab
    st.divider()
    
    # Render selected tab
    try:
        if selected_tab == "🔍 Stock Intelligence":
            dashboard_tab.render_tab()
        elif selected_tab == "🚀 Advanced Scanner":
            scanner_tab.render_tab()
        elif selected_tab == "⭐ My Tickers":
            watchlist_tab.render_tab()
        elif selected_tab == "💰 Fractional DCA":
            # Get Supabase client if available
            supabase_client = st.session_state.get('supabase_client', None)
            dca_tab.render_tab(supabase_client)
        elif selected_tab == "🎯 Strategy Advisor":
            strategy_advisor_tab.render_tab()
        elif selected_tab == "📊 Generate Signal":
            generate_signal_tab.render_tab()
        elif selected_tab == "📜 Signal History":
            signal_history_tab.render_tab()
        elif selected_tab == "📚 Strategy Guide":
            strategy_guide_tab.render_tab()
        elif selected_tab == "🏦 Tradier Account":
            tradier_tab.render_tab()
        elif selected_tab == "📈 IBKR Trading":
            ibkr_tab.render_tab()
        elif selected_tab == "⚡ Scalping/Day Trade":
            scalping_tab.render_tab()
        elif selected_tab == "🤖 Strategy Analyzer":
            strategy_analyzer_tab.render_tab()
        elif selected_tab == "🤖 Auto-Trader":
            autotrader_tab.render_tab()
        elif selected_tab == "₿ Crypto Trading":
            crypto_tab.render_tab()
        else:
            st.error(f"Unknown tab: {selected_tab}")
            
    except Exception as e:
        logger.error("Error rendering tab {}: {}", selected_tab, str(e), exc_info=True)
        st.error(f"Failed to load {selected_tab}")
        st.exception(e)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Application error: {}", str(e), exc_info=True)
        st.error("Application encountered an error. Check logs for details.")
        st.exception(e)

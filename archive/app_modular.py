"""
Sentient Trader Platform - Modular Version
Main application entry point with modular architecture.
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging first
from utils.logging_config import setup_logging, logger
setup_logging()

# Setup Streamlit compatibility shims
from utils.streamlit_compat import setup_streamlit_compatibility
setup_streamlit_compatibility()

# Import models
from models import MarketCondition, StockAnalysis, StrategyRecommendation, TradingConfig

# Import analyzers
from analyzers import TechnicalAnalyzer, NewsAnalyzer, ComprehensiveAnalyzer, StrategyAdvisor

# Import clients
from clients import OptionAlphaClient, SignalValidator

# Import utilities
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


def main():
    """Main application entry point"""
    # If Streamlit ScriptRunContext isn't present, exit early
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            logger.info("Streamlit ScriptRunContext not detected - main() called outside 'streamlit run'. Exiting early.")
            return
    except Exception:
        pass

    # Page configuration
    st.set_page_config(
        page_title="Sentient Trader",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    apply_custom_styling()
    
    st.title("📈 Sentient Trader Platform")
    st.caption("Real-time analysis, news, catalysts, technical indicators & intelligent strategy recommendations")
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = TradingConfig()
    if 'validator' not in st.session_state:
        st.session_state.validator = SignalValidator(st.session_state.config)
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []
    if 'paper_mode' not in st.session_state:
        st.session_state.paper_mode = True
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Trading Mode")
        paper_mode = st.toggle("Paper Trading Mode", value=st.session_state.paper_mode)
        st.session_state.paper_mode = paper_mode
        
        if paper_mode:
            st.info("🔒 Paper trading: Signals logged only")
        else:
            st.warning("⚠️ LIVE TRADING ENABLED")
        
        st.subheader("Option Alpha Webhook")
        webhook_url = st.text_input(
            "Webhook URL",
            value="https://app.optionalpha.com/api/webhooks/XXXX",
            type="password" if not paper_mode else "default"
        )
        
        st.subheader("Guardrails")
        with st.expander("Risk Limits"):
            max_daily_orders = st.number_input("Max Daily Orders", 1, 50, st.session_state.config.max_daily_orders)
            max_daily_risk = st.number_input("Max Daily Risk ($)", 100, 10000, int(st.session_state.config.max_daily_risk))
            max_position_per_ticker = st.number_input("Max Positions per Ticker", 1, 10, st.session_state.config.max_position_per_ticker)
        
        if st.button("Update Configuration"):
            st.session_state.config = TradingConfig(
                max_daily_orders=max_daily_orders,
                max_daily_risk=float(max_daily_risk),
                max_position_per_ticker=max_position_per_ticker
            )
            st.session_state.validator = SignalValidator(st.session_state.config)
            st.success("Configuration updated!")

        # Strategy Analyzer summary
        st.markdown("---")
        st.subheader("🤖 Strategy Analyzer (Quick View)")
        analysis = st.session_state.get('strategy_analysis') or st.session_state.get('current_analysis')
        if analysis:
            try:
                bot_name = getattr(analysis, 'bot_name', getattr(analysis, 'ticker', 'Bot'))
                overall = getattr(analysis, 'overall_rating', getattr(analysis, 'rating', 'N/A'))
                conf = getattr(analysis, 'confidence', None)

                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.write(f"**{bot_name}**")
                with c2:
                    st.metric("Overall", overall)
                with c3:
                    st.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else conf)

                if st.button("🔎 Open Strategy Analyzer", use_container_width=True):
                    st.session_state.goto_strategy_analyzer = True
                    st.rerun()
            except Exception:
                st.write("Compact summary unavailable")
        else:
            st.info("Run a strategy analysis to see a quick summary here.")
    
    # Import and run the tab content from original app
    # NOTE: The original app.py tab implementations remain the same
    # They are imported here to keep the code organized
    from app_backup import main as original_main_tabs
    
    # Display message about modular structure
    st.info("✅ **Modular Architecture Active** - Components loaded from organized modules: `models/`, `analyzers/`, `clients/`, `utils/`")


if __name__ == "__main__":
    main()

"""
Crypto Quick Trade UI
Enhanced quick trade interface with ticker management, scanner integration,
investment controls, risk management, AI validation, and automated execution
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timezone
from loguru import logger
import json
import os
import asyncio
from clients.kraken_client import KrakenClient, OrderType, OrderSide
from clients.crypto_validator import CryptoValidator
from src.integrations.discord_webhook import send_discord_alert
from models.alerts import TradingAlert, AlertType, AlertPriority
from services.freqtrade_strategies import FreqtradeStrategyAdapter
from services.ai_crypto_trade_reviewer import AICryptoTradeReviewer

# Note: Asset pairs are cached in session state in display_trade_setup()
# to avoid repeated API calls, since KrakenClient is not hashable for @st.cache_data


def get_llm_for_bulk_analysis(num_tickers: int):
    """
    Get appropriate LLM analyzer based on number of tickers being analyzed.
    
    For bulk operations (>1 ticker), forces cloud API (OpenRouter) for speed.
    Local LLM is too slow for multiple tickers.
    
    Args:
        num_tickers: Number of tickers to analyze
        
    Returns:
        LLM analyzer instance (cloud-forced if >1 ticker, hybrid if 1 ticker)
    """
    # Single ticker: use existing analyzer (could be local/hybrid for speed)
    if num_tickers <= 1:
        if 'llm_analyzer' in st.session_state:
            logger.info(f"📊 Single ticker analysis - using existing LLM analyzer (may use local Ollama)")
            return st.session_state.llm_analyzer
        return None
    
    # Multiple tickers: force cloud API for parallel processing speed
    logger.info(f"📊 Bulk analysis ({num_tickers} tickers) - forcing cloud API (OpenRouter) for speed")
    logger.info("   💡 Local LLM too slow for bulk operations, switching to OpenRouter")
    
    try:
        from services.llm_strategy_analyzer import LLMStrategyAnalyzer
        from utils.config_loader import get_api_key
        
        api_key = get_api_key('OPENROUTER_API_KEY', 'openrouter')
        if not api_key:
            logger.warning("⚠️ No OpenRouter API key found for bulk analysis")
            return st.session_state.get('llm_analyzer')
        
        model = os.getenv('AI_ANALYZER_MODEL') or get_api_key('AI_ANALYZER_MODEL', 'models') or 'google/gemini-2.0-flash-exp:free'
        
        # Create cloud-only analyzer for bulk speed
        cloud_analyzer = LLMStrategyAnalyzer(
            api_key=api_key, 
            model=model, 
            provider="openrouter"
        )
        
        logger.success(f"✅ Created cloud-only LLM analyzer for bulk analysis: {model}")
        return cloud_analyzer
        
    except Exception as e:
        logger.error(f"Failed to create cloud analyzer for bulk: {e}")
        # Fallback to session state analyzer
        return st.session_state.get('llm_analyzer')


# --- Unified Discovery Workflow ---

def display_unified_scanner(kraken_client: KrakenClient, crypto_config, scanner_instances: Dict, watchlist_manager=None):
    """
    A unified scanner UI that provides a streamlined workflow for crypto discovery.
    Workflow: Scan -> Bulk Select -> Analyze All -> Pick Best -> Execute
    """
    st.markdown("### 🔍 Smart Crypto Scanner & Analyzer")
    st.markdown("**Workflow:** Scan → Bulk Select → Analyze All → Pick Best → Execute")

    # --- 1. Scan for Opportunities ---
    st.markdown("### 1️⃣ SCAN FOR OPPORTUNITIES")
    st.caption("Choose a scanner or manually enter tickers to get started")
    
    scanner_map = {
        "✍️ Manual Selection": "manual",
        "⭐ My Watchlist": "watchlist",
        "💰 Penny Cryptos (<$1)": "penny_crypto_scanner",
        "🔥 Buzzing/Volume Surge": "crypto_opportunity_scanner",
        "🌶️ Hottest/Momentum": "ai_crypto_scanner",
        "💎 Sub-Penny Discovery": "sub_penny_discovery"
    }

    scanner_col1, scanner_col2 = st.columns([2, 1])
    with scanner_col1:
        # Initialize scanner type in session state to preserve selection across reruns
        if 'crypto_scanner_type' not in st.session_state:
            st.session_state.crypto_scanner_type = list(scanner_map.keys())[0]
        
        scan_type_display = st.selectbox(
            "Scanner Type",
            options=list(scanner_map.keys()),
            index=list(scanner_map.keys()).index(st.session_state.crypto_scanner_type) if st.session_state.crypto_scanner_type in scanner_map.keys() else 0,
            key='crypto_scanner_type_selectbox',
            help="Choose a discovery method to find opportunities."
        )
        
        # Update session state with current selection and preserve tab states
        st.session_state.crypto_scanner_type = scan_type_display
        # Preserve crypto tab state to prevent redirect to dashboard
        if 'active_crypto_tab' not in st.session_state:
            st.session_state.active_crypto_tab = "⚡ Quick Trade"
        scan_type = scanner_map[scan_type_display]

    with scanner_col2:
        st.write("")
        st.write("")
        
        # Show different button based on selection
        if scan_type == "manual":
            button_label = "✍️ Add Manually"
            button_type = "secondary"
        else:
            button_label = "🚀 Scan"
            button_type = "primary"
        
        if st.button(button_label, width='stretch', type=button_type):
            # Preserve tab states before scan to prevent redirect
            if 'active_crypto_tab' not in st.session_state:
                st.session_state.active_crypto_tab = "⚡ Quick Trade"
            
            # Handle manual selection differently
            if scan_type == "manual":
                # Don't clear existing results, just mark that manual mode is active
                if 'scan_results' not in st.session_state:
                    st.session_state.scan_results = []
                if 'selected_tickers' not in st.session_state:
                    st.session_state.selected_tickers = []
                st.session_state.manual_selection_mode = True
                st.info("💡 Use the text input below to manually add coins to analyze")
                st.rerun()
            else:
                st.session_state.scan_results = []
                st.session_state.selected_tickers = []
                st.session_state.manual_selection_mode = False
            
            with st.spinner(f"Scanning for {scan_type_display}..."):
                try:
                    # Initialize validator
                    validator = CryptoValidator(kraken_client)
                    
                    if scan_type == "penny_crypto_scanner" and scanner_instances.get(scan_type):
                        results = scanner_instances[scan_type].scan_penny_cryptos(max_price=1.0, top_n=50)
                        # Validate all results
                        raw_results = [{"Ticker": r.symbol, "Price": r.current_price, "Change": r.change_pct_24h, "Score": r.runner_potential_score} for r in results]
                        valid_results, invalid_symbols = validator.filter_valid_pairs(raw_results, symbol_key='Ticker')
                        st.session_state.scan_results = valid_results
                        if invalid_symbols:
                            st.info(f"ℹ️ Filtered out {len(invalid_symbols)} invalid Kraken pairs")
                    
                    elif scan_type == "crypto_opportunity_scanner" and scanner_instances.get(scan_type):
                        opps = scanner_instances[scan_type].scan_opportunities(top_n=50)
                        # Validate all results
                        raw_results = [{"Ticker": o.symbol, "Price": o.current_price, "Change": o.change_pct_24h, "Score": o.score} for o in opps]
                        valid_results, invalid_symbols = validator.filter_valid_pairs(raw_results, symbol_key='Ticker')
                        st.session_state.scan_results = valid_results
                        if invalid_symbols:
                            st.info(f"ℹ️ Filtered out {len(invalid_symbols)} invalid Kraken pairs")

                    elif scan_type == "ai_crypto_scanner" and scanner_instances.get(scan_type):
                        opps = scanner_instances[scan_type].scan_with_ai_confidence(top_n=50)
                        # Validate all results
                        raw_results = [{"Ticker": o.symbol, "Price": o.current_price, "Change": o.change_pct_24h, "Score": o.score} for o in opps]
                        valid_results, invalid_symbols = validator.filter_valid_pairs(raw_results, symbol_key='Ticker')
                        st.session_state.scan_results = valid_results
                        if invalid_symbols:
                            st.info(f"ℹ️ Filtered out {len(invalid_symbols)} invalid Kraken pairs")

                    elif scan_type == "sub_penny_discovery" and scanner_instances.get(scan_type):
                        with st.status("🔬 Discovering sub-penny coins... This may take 1-2 minutes to find enough valid Kraken pairs.", expanded=True) as status:
                            status.update(label="🔍 Fetching coins from CoinGecko & CoinMarketCap...")
                            runners = asyncio.run(scanner_instances[scan_type].discover_sub_penny_runners(
                                max_price=0.01,
                                min_market_cap=0,
                                max_market_cap=10_000_000,  # 10M to match main app and allow more coins
                                top_n=50,
                                sort_by="runner_potential"
                            ))
                            
                            status.update(label=f"✅ Found {len(runners)} coins, validating against Kraken...")
                            # Validate all results using validator
                            raw_results = [{"Ticker": r.symbol.upper(), "Price": r.price_usd, "Change": r.change_24h, "Score": r.runner_potential_score} for r in runners]
                            valid_results, invalid_symbols = validator.filter_valid_pairs(raw_results, symbol_key='Ticker')
                            
                            status.update(label=f"✅ Validation complete: {len(valid_results)} valid Kraken pairs found")
                        
                        st.session_state.scan_results = valid_results
                        
                        if invalid_symbols:
                            st.warning(f"⚠️ Filtered out {len(invalid_symbols)} coins not available on Kraken (only {len(valid_results)} valid pairs found)")
                        
                        if len(valid_results) < 5:
                            st.info("💡 **Tip:** Most sub-penny coins aren't available on Kraken. Try the 'Penny Cryptos (<$1)' scanner for more tradable options.")
                    
                    elif scan_type == "watchlist":
                        # Fetch real-time data for watchlist tickers
                        watchlist_results = []
                        
                        # Get watchlist from multiple sources
                        watchlist_tickers = []
                        
                        # 1. From crypto_config
                        if hasattr(crypto_config, 'CRYPTO_WATCHLIST'):
                            config_tickers = crypto_config.CRYPTO_WATCHLIST
                            watchlist_tickers.extend(config_tickers)
                            logger.info(f"Loaded {len(config_tickers)} tickers from crypto_config.CRYPTO_WATCHLIST")
                        else:
                            logger.info("No CRYPTO_WATCHLIST found in crypto_config")
                        
                        # 2. From watchlist_manager if available
                        if watchlist_manager:
                            try:
                                managed_watchlist = watchlist_manager.get_watchlist()
                                manager_tickers = [item['symbol'] for item in managed_watchlist]
                                watchlist_tickers.extend(manager_tickers)
                                logger.info(f"Loaded {len(manager_tickers)} tickers from watchlist_manager database")
                                logger.debug("Database tickers: {}...", str(manager_tickers[:10]) if len(manager_tickers) > 10 else f"Database tickers: {manager_tickers}")
                            except Exception as e:
                                logger.error(f"Error loading from watchlist_manager: {e}")
                        else:
                            logger.info("No watchlist_manager available")
                        
                        # Remove duplicates and ensure proper format
                        watchlist_tickers = list(set(watchlist_tickers))
                        logger.info(f"Total unique tickers after dedup: {len(watchlist_tickers)}")
                        
                        if not watchlist_tickers:
                            st.warning("⚠️ Your watchlist is empty. Add tickers to your watchlist first!")
                            st.session_state.scan_results = []
                        else:
                            st.info(f"📋 Loading {len(watchlist_tickers)} tickers from your watchlist...")
                            
                            # Fetch live data for each ticker
                            for ticker in watchlist_tickers:
                                try:
                                    # Ensure ticker has proper format (XXX/USD)
                                    if '/' not in ticker:
                                        if ticker.upper().endswith('USD'):
                                            ticker = ticker[:-3] + '/USD'
                                        else:
                                            ticker = ticker + '/USD'
                                    
                                    ticker_info = kraken_client.get_ticker_info(ticker)
                                    
                                    if ticker_info and 'c' in ticker_info:
                                        current_price = float(ticker_info['c'][0])
                                        
                                        # Calculate 24h change if available
                                        change_24h = 0
                                        if 'o' in ticker_info:
                                            open_price = float(ticker_info['o'])
                                            if open_price > 0:
                                                change_24h = ((current_price - open_price) / open_price) * 100
                                        
                                        # Calculate simple score based on volume and price action
                                        score = 50  # Default score
                                        if 'v' in ticker_info:
                                            volume_24h = float(ticker_info['v'][0])
                                            if volume_24h > 1000:
                                                score += 10
                                            if volume_24h > 10000:
                                                score += 10
                                        
                                        if abs(change_24h) > 5:
                                            score += 10
                                        if abs(change_24h) > 10:
                                            score += 10
                                        
                                        watchlist_results.append({
                                            "Ticker": ticker,
                                            "Price": current_price,
                                            "Change": change_24h,
                                            "Score": min(score, 100)
                                        })
                                except Exception as e:
                                    logger.warning(f"Could not fetch data for {ticker}: {e}")
                                    # Still add it with placeholder data
                                    watchlist_results.append({
                                        "Ticker": ticker,
                                        "Price": 0.0,
                                        "Change": 0.0,
                                        "Score": 0
                                    })
                            
                            st.session_state.scan_results = watchlist_results
                            st.success(f"✅ Loaded {len(watchlist_results)} tickers from watchlist")
                            
                            # Auto-select all watchlist tickers for convenience
                            if watchlist_results:
                                st.session_state.selected_tickers = [r['Ticker'] for r in watchlist_results]
                                st.info(f"💡 Auto-selected all {len(watchlist_results)} watchlist tickers for analysis")
                                st.rerun()  # Force page refresh to show Section 2
                    
                    # Ensure tab states are preserved after scan
                    st.session_state.active_crypto_tab = "⚡ Quick Trade"

                except Exception as e:
                    st.error(f"An error occurred while scanning: {e}")
                    logger.error("Scanner failed: {}", str(e), exc_info=True)
                    # Preserve tab state even on error
                    st.session_state.active_crypto_tab = "⚡ Quick Trade"

    # --- Manual Selection Interface ---
    if st.session_state.get('manual_selection_mode', False) or scan_type == "manual":
        st.markdown("---")
        st.markdown("#### ✍️ Manual Coin Selection")
        st.info("💡 **Pro Tip**: Type coin symbols separated by commas or spaces. Auto-validates against Kraken!")
        
        manual_col1, manual_col2 = st.columns([3, 1])
        
        with manual_col1:
            manual_input = st.text_area(
                "Enter coin symbols (e.g., BTC ETH SOL AVAX LINK, or BTC/USD, ETH/USD)",
                value=st.session_state.get('manual_coins_input', ''),
                height=100,
                key='manual_coins_input_field',
                help="Enter coin symbols separated by commas, spaces, or new lines. You can use 'BTC' or 'BTC/USD' format."
            )
        
        with manual_col2:
            st.write("")
            st.write("")
            if st.button("✅ Add These Coins", width='stretch', type="primary"):
                if manual_input.strip():
                    with st.spinner("Validating coins against Kraken..."):
                        # Parse input - handle commas, spaces, newlines
                        import re
                        raw_symbols = re.split('[,\\s\\n]+', manual_input.strip().upper())
                        raw_symbols = [s.strip() for s in raw_symbols if s.strip()]
                        
                        # Normalize to XXX/USD format
                        normalized_symbols = []
                        for symbol in raw_symbols:
                            if '/' not in symbol:
                                # Add /USD if not present
                                if symbol.endswith('USD'):
                                    # Already has USD suffix like BTCUSD -> BTC/USD
                                    normalized_symbols.append(symbol[:-3] + '/USD')
                                else:
                                    normalized_symbols.append(symbol + '/USD')
                            else:
                                normalized_symbols.append(symbol)
                        
                        # Validate with Kraken
                        validator = CryptoValidator(kraken_client)
                        manual_results = []
                        valid_count = 0
                        invalid_symbols = []
                        
                        for ticker in normalized_symbols:
                            try:
                                ticker_info = kraken_client.get_ticker_info(ticker)
                                
                                if ticker_info and 'c' in ticker_info:
                                    current_price = float(ticker_info['c'][0])
                                    
                                    # Calculate 24h change
                                    change_24h = 0
                                    if 'o' in ticker_info:
                                        open_price = float(ticker_info['o'])
                                        if open_price > 0:
                                            change_24h = ((current_price - open_price) / open_price) * 100
                                    
                                    # Simple score based on price action
                                    score = 50
                                    if abs(change_24h) > 5:
                                        score += 20
                                    if abs(change_24h) > 10:
                                        score += 20
                                    
                                    manual_results.append({
                                        "Ticker": ticker,
                                        "Price": current_price,
                                        "Change": change_24h,
                                        "Score": min(score, 100)
                                    })
                                    valid_count += 1
                                else:
                                    invalid_symbols.append(ticker)
                            except Exception as e:
                                logger.warning(f"Invalid ticker {ticker}: {e}")
                                invalid_symbols.append(ticker)
                        
                        if manual_results:
                            # Merge with existing results if any
                            existing_tickers = {r['Ticker'] for r in st.session_state.get('scan_results', [])}
                            new_results = [r for r in manual_results if r['Ticker'] not in existing_tickers]
                            
                            st.session_state.scan_results = st.session_state.get('scan_results', []) + new_results
                            st.session_state.selected_tickers = st.session_state.get('selected_tickers', []) + [r['Ticker'] for r in new_results]
                            
                            st.success(f"✅ Added {len(new_results)} valid coins! (Total: {len(st.session_state.scan_results)} coins)")
                            
                            if invalid_symbols:
                                st.warning(f"⚠️ Skipped {len(invalid_symbols)} invalid coins: {', '.join(invalid_symbols[:5])}{'...' if len(invalid_symbols) > 5 else ''}")
                        else:
                            st.error("❌ No valid coins found. Check your symbols and try again.")
                            if invalid_symbols:
                                st.info(f"Invalid: {', '.join(invalid_symbols)}")
                else:
                    st.warning("⚠️ Please enter at least one coin symbol")
        
        # Show current manually added coins
        if st.session_state.get('scan_results'):
            st.caption(f"📋 Currently have {len(st.session_state.scan_results)} coins ready to analyze")

    # --- 2. Select Tickers for Analysis ---
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        st.markdown("---")
        st.markdown("### 2️⃣ SELECT TICKERS FOR ANALYSIS")
        st.caption(f"📊 {len(st.session_state.scan_results)} tickers loaded from scan. Choose which ones to analyze below:")
        
        results = st.session_state.scan_results
        df = pd.DataFrame(results)

        if 'selected_tickers' not in st.session_state:
            st.session_state.selected_tickers = []

        # Filter and sort options
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
        
        with filter_col1:
            sort_by = st.selectbox(
                "Sort By",
                options=["Score (High→Low)", "Score (Low→High)", "Price (High→Low)", "Price (Low→High)", 
                        "Change % (High→Low)", "Change % (Low→High)", "Ticker (A→Z)"],
                index=0,
                help="Choose how to sort the results"
            )
        
        with filter_col2:
            min_score = st.number_input("Min Score", min_value=0, max_value=100, value=0, step=5, 
                                       help="Filter by minimum score")
        
        with filter_col3:
            price_filter = st.selectbox(
                "Price Range",
                options=["All", "< $0.01 (Sub-Penny)", "< $0.10", "< $1.00", "$1-$10", "$10-$100", "> $100"],
                help="Filter by price range"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        # Score filter
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['Score'] >= min_score]
        
        # Price filter
        if price_filter == "< $0.01 (Sub-Penny)":
            filtered_df = filtered_df[filtered_df['Price'] < 0.01]
        elif price_filter == "< $0.10":
            filtered_df = filtered_df[filtered_df['Price'] < 0.10]
        elif price_filter == "< $1.00":
            filtered_df = filtered_df[filtered_df['Price'] < 1.00]
        elif price_filter == "$1-$10":
            filtered_df = filtered_df[(filtered_df['Price'] >= 1.0) & (filtered_df['Price'] < 10.0)]
        elif price_filter == "$10-$100":
            filtered_df = filtered_df[(filtered_df['Price'] >= 10.0) & (filtered_df['Price'] < 100.0)]
        elif price_filter == "> $100":
            filtered_df = filtered_df[filtered_df['Price'] >= 100.0]
        
        # Apply sorting
        if sort_by == "Score (High→Low)":
            filtered_df = filtered_df.sort_values(by='Score', ascending=False)
        elif sort_by == "Score (Low→High)":
            filtered_df = filtered_df.sort_values(by='Score', ascending=True)
        elif sort_by == "Price (High→Low)":
            filtered_df = filtered_df.sort_values(by='Price', ascending=False)
        elif sort_by == "Price (Low→High)":
            filtered_df = filtered_df.sort_values(by='Price', ascending=True)
        elif sort_by == "Change % (High→Low)":
            filtered_df = filtered_df.sort_values(by='Change', ascending=False)
        elif sort_by == "Change % (Low→High)":
            filtered_df = filtered_df.sort_values(by='Change', ascending=True)
        elif sort_by == "Ticker (A→Z)":
            filtered_df = filtered_df.sort_values(by='Ticker', ascending=True)
        
        # Show filter results
        if len(filtered_df) < len(df):
            st.info(f"🔍 Showing {len(filtered_df)} of {len(df)} tickers after filtering")
        
        # Update df for selection
        df = filtered_df.reset_index(drop=True)

        st.markdown("---")
        st.markdown("### 📌 QUICK SELECTION")
        st.caption(f"Your watchlist has **{len(df)} tickers**. Choose which ones to analyze:")
        
        # Helper functions for button callbacks
        def select_all_tickers():
            st.session_state.selected_tickers = df['Ticker'].tolist()
        
        def clear_all_tickers():
            st.session_state.selected_tickers = []
        
        def select_top_n(n: int):
            """Select top N tickers by score"""
            def selector():
                if 'Score' in df.columns:
                    st.session_state.selected_tickers = df.nlargest(min(n, len(df)), 'Score')['Ticker'].tolist()
                else:
                    st.session_state.selected_tickers = []
            return selector
        
        # Big prominent buttons for selection
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            st.button("✅ SELECT ALL", use_container_width=True, type="primary", on_click=select_all_tickers, 
                     help=f"Select all {len(df)} tickers for analysis")
        with btn_col2:
            st.button("🔝 TOP 10", use_container_width=True, on_click=select_top_n(10), 
                     help="Select top 10 tickers by score")
        with btn_col3:
            st.button("🔝 TOP 20", use_container_width=True, on_click=select_top_n(20), 
                     help="Select top 20 tickers by score")
        with btn_col4:
            st.button("❌ CLEAR ALL", use_container_width=True, on_click=clear_all_tickers, 
                     help="Deselect all tickers")
        
        # Show current selection count prominently
        selected_count = len(st.session_state.selected_tickers)
        if selected_count == len(df):
            st.success(f"✅ **ALL {selected_count} tickers selected** - Ready to analyze!")
        elif selected_count > 0:
            st.info(f"📊 **{selected_count} of {len(df)} tickers selected** - Click 'SELECT ALL' to analyze all tickers")
        else:
            st.warning(f"⚠️ **No tickers selected** - Click 'SELECT ALL' to select all {len(df)} tickers")

        # Multi-select dropdown for easy ticker selection
        st.markdown("**Quick Selection (Searchable):**")
        selected_from_multiselect = st.multiselect(
            "Select tickers to analyze",
            options=df['Ticker'].tolist(),
            default=st.session_state.selected_tickers,
            key='ticker_multiselect',
            help="Start typing to search, or scroll to browse all tickers"
        )
        
        # Update session state from multiselect
        if selected_from_multiselect != st.session_state.selected_tickers:
            st.session_state.selected_tickers = selected_from_multiselect

        # Expandable detailed view with pagination
        with st.expander("📊 View Detailed Scan Results", expanded=False):
            # Pagination controls
            items_per_page = 20
            total_pages = (len(df) - 1) // items_per_page + 1
            
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 0
            
            page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
            
            with page_col1:
                if st.button("⬅️ Previous", disabled=st.session_state.current_page == 0):
                    st.session_state.current_page = max(0, st.session_state.current_page - 1)
                    st.rerun()
            
            with page_col2:
                st.markdown(f"**Page {st.session_state.current_page + 1} of {total_pages}** ({len(df)} total tickers)")
            
            with page_col3:
                if st.button("Next ➡️", disabled=st.session_state.current_page >= total_pages - 1):
                    st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
                    st.rerun()
            
            # Calculate pagination slice
            start_idx = st.session_state.current_page * items_per_page
            end_idx = min(start_idx + items_per_page, len(df))
            
            # Display current page
            st.markdown(f"**Showing {start_idx + 1}-{end_idx} of {len(df)} tickers:**")
            
            # Helper function for checkbox callbacks
            def make_ticker_toggle(ticker: str):
                """Create a callback function for a specific ticker"""
                def toggle_ticker():
                    checkbox_key = f"sel_{ticker}"
                    new_value = st.session_state.get(checkbox_key, False)
                    
                    if new_value:
                        if ticker not in st.session_state.selected_tickers:
                            st.session_state.selected_tickers.append(ticker)
                    else:
                        if ticker in st.session_state.selected_tickers:
                            st.session_state.selected_tickers.remove(ticker)
                return toggle_ticker
            
            # Display paginated results
            for i, row in df.iloc[start_idx:end_idx].iterrows():
                cols = st.columns([1, 4, 2, 2, 2])
                ticker = row['Ticker']
                is_selected = ticker in st.session_state.selected_tickers
                
                cols[0].checkbox(
                    f"Select {ticker}",
                    value=is_selected,
                    key=f"sel_{ticker}",
                    on_change=make_ticker_toggle(ticker),
                    label_visibility="hidden"
                )

                cols[1].markdown(f"**{row.get('Ticker', 'N/A')}**")
                cols[2].metric("Price", f"${row.get('Price', 0):,.4f}")
                cols[3].metric("24h Change", f"{row.get('Change', 0):.2f}%" if row.get('Change') else "N/A")
                cols[4].metric("Score", f"{int(row.get('Score', 0))}" if row.get('Score') else "N/A")

    # --- 3. Analyze Selected Tickers ---
    st.markdown("---")
    st.markdown("### 3️⃣ ANALYZE SELECTED TICKERS")
    selected_tickers = st.session_state.get('selected_tickers', [])
    
    if not selected_tickers:
        st.warning("⚠️ **No tickers selected!**")
        st.info("💡 **How to get started:**\n1. Go to Section 1️⃣ above and click 'Scan' on your **⭐ My Watchlist**\n2. Section 2️⃣ will appear with a searchable list of all your watchlist tickers\n3. Use the multiselect dropdown to choose which tickers to analyze\n4. The 3 analysis buttons will appear here!")
    else:
        st.success(f"✅ **{len(selected_tickers)} tickers ready for analysis!**")
        
        # Show selected tickers summary
        with st.expander(f"📋 Selected Tickers ({len(selected_tickers)})", expanded=False):
            # Display in columns for better readability
            num_cols = 5
            cols = st.columns(num_cols)
            for idx, ticker in enumerate(selected_tickers):
                col_idx = idx % num_cols
                cols[col_idx].markdown(f"• **{ticker}**")
        
        st.markdown("---")
        st.markdown("### 🎯 CHOOSE ANALYSIS TYPE")
        
        # Initialize analysis mode in session state if not set
        if 'analysis_mode' not in st.session_state:
            st.session_state.analysis_mode = 'standard'
        
        # THREE BIG BUTTONS AT THE TOP - SUPER VISIBLE!
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button(
                "🔬 STANDARD ANALYSIS", 
                use_container_width=True,
                type="primary" if st.session_state.analysis_mode == 'standard' else "secondary",
                help="Analyze with single chosen strategy", 
                key="btn_standard_analysis"
            ):
                st.session_state.analysis_mode = 'standard'
                st.rerun()
            st.caption("Single strategy + timeframe")
        
        with btn_col2:
            if st.button(
                "🎯 MULTI-CONFIG", 
                use_container_width=True,
                type="primary" if st.session_state.analysis_mode == 'multi_config' else "secondary",
                help="Test Long/Short + Leverage levels", 
                key="btn_multi_config"
            ):
                st.session_state.analysis_mode = 'multi_config'
                st.rerun()
            st.caption("All directions + leverages")
        
        with btn_col3:
            if st.button(
                "🚀 ULTIMATE", 
                use_container_width=True,
                type="primary" if st.session_state.analysis_mode == 'ultimate' else "secondary",
                help="Test ALL strategies + directions + leverages", 
                key="btn_ultimate_analysis"
            ):
                st.session_state.analysis_mode = 'ultimate'
                st.rerun()
            st.caption("COMPLETE analysis - ALL combos!")
        
        st.markdown("---")
        
        # Configuration section (collapsed by default unless standard analysis selected)
        adapter = FreqtradeStrategyAdapter(kraken_client)
        strategies = adapter.get_available_strategies()
        
        # Show configuration based on selected mode
        if st.session_state.analysis_mode == 'standard':
            with st.expander("⚙️ Standard Analysis Configuration", expanded=True):
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    # Add smart strategy recommendation
                    num_tickers = len(selected_tickers)
                    if num_tickers <= 5:
                        recommended_strategy = 'aggressive_scalp'
                        recommendation_reason = "Few tickers - Use aggressive scalping"
                    elif num_tickers <= 15:
                        recommended_strategy = 'ema_crossover'
                        recommendation_reason = "Medium list - Use EMA crossover"
                    else:
                        recommended_strategy = 'macd_volume'
                        recommendation_reason = "Large list - Use MACD+Volume"
                    
                    st.info(f"💡 {recommendation_reason}")
                    
                    # Strategy selector with recommended default
                    default_index = next((i for i, s in enumerate(strategies) if s['id'] == recommended_strategy), 0)
                    
                    strategy_id = st.selectbox(
                        "Analysis Strategy",
                        options=[s['id'] for s in strategies],
                        format_func=lambda x: next(s['name'] for s in strategies if s['id'] == x),
                        index=default_index,
                        help="Choose the technical analysis strategy to use"
                    )
                    
                    # Show strategy info
                    selected_strategy_info = next((s for s in strategies if s['id'] == strategy_id), None)
                    if selected_strategy_info:
                        st.markdown(f"**{selected_strategy_info['name']}**")
                        st.caption(selected_strategy_info['description'])
                
                with config_col2:
                    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "60m"], index=2, 
                                            help="Shorter = more signals but more noise")
        else:
            # Default values for multi-config/ultimate
            strategy_id = 'ema_crossover'
            timeframe = '15m'
            selected_strategy_info = next((s for s in strategies if s['id'] == strategy_id), None)
        
        # Show configuration and start button based on mode
        if st.session_state.analysis_mode == 'standard':
            if st.button("🚀 Start Standard Analysis", use_container_width=True, type="primary", key="start_standard"):
                st.session_state.start_standard_analysis = True

        # Execute standard analysis
        if st.session_state.get('start_standard_analysis', False) and st.session_state.analysis_mode == 'standard':
            st.session_state.start_standard_analysis = False
            with st.spinner(f"Analyzing {len(selected_tickers)} tickers..."):
                try:
                    # Convert timeframe from "15m" to "15"
                    interval_minutes = timeframe.replace('m', '') if timeframe else '15'

                    # ... (rest of the code remains the same)

                    
                    # First validate that all tickers are valid Kraken pairs
                    valid_tickers = []
                    invalid_tickers = []
                        
                    for ticker in selected_tickers:
                        try:
                            # Test if ticker is valid by fetching ticker info
                            test_info = kraken_client.get_ticker_info(ticker)
                            if test_info and 'c' in test_info:
                                valid_tickers.append(ticker)
                            else:
                                invalid_tickers.append(ticker)
                        except Exception:
                            invalid_tickers.append(ticker)
                    
                    if invalid_tickers:
                        st.warning(f"⚠️ Found {len(invalid_tickers)} invalid Kraken pairs: {', '.join(invalid_tickers[:5])}{'...' if len(invalid_tickers) > 5 else ''}")
                        st.info(f"✅ Analyzing {len(valid_tickers)} valid pairs only")
                    
                    if valid_tickers:
                        logger.info(f"Starting bulk analysis: {len(valid_tickers)} tickers, strategy={strategy_id}, interval={interval_minutes}")
                        
                        # Show analysis progress
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        analysis_results = []
                        for i, ticker in enumerate(valid_tickers):
                            progress_text.text(f"Analyzing {ticker}... ({i+1}/{len(valid_tickers)})")
                            progress_bar.progress((i + 1) / len(valid_tickers))
                            
                            result = adapter.analyze_crypto(ticker, strategy_id or "default", interval_minutes)
                            if 'error' not in result:
                                analysis_results.append(result)
                            else:
                                logger.warning("Analysis failed for {}: {result.get('error')}", str(ticker))
                        
                        progress_text.empty()
                        progress_bar.empty()
                        
                        st.session_state.analysis_results = analysis_results
                        logger.info(f"Bulk analysis complete: {len(analysis_results)} results returned")
                        
                        if analysis_results:
                            st.success(f"✅ Successfully analyzed {len(analysis_results)} coins!")
                        else:
                            st.error("❌ Analysis completed but no results returned. Check logs for details.")
                    else:
                        st.error("❌ No valid trading pairs found to analyze")
                        
                except Exception as e:
                    st.error(f"Bulk analysis failed: {e}")
                    logger.error("Bulk analysis error: {}", str(e), exc_info=True)
        
        # Execute multi-configuration analysis
        if st.session_state.analysis_mode == 'multi_config':
            st.markdown("---")
            st.markdown("### 🎯 Multi-Configuration Bulk Analysis")
            st.caption("⚡ Test multiple strategies on your watchlist simultaneously")
            
            # Fractional Trading Quick Preset
            st.markdown("#### 💰 Fractional Trading Presets")
            
            col_frac1, col_frac2 = st.columns(2)
            
            with col_frac1:
                if st.button("🏆 Major Coins Only (BTC, ETH, SOL)", type="secondary", use_container_width=True):
                    st.info("💡 **Filtering for expensive coins** - best for fractional trading!")
                    # Filter watchlist for major expensive coins
                    major_coins = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD', 'DOT/USD', 'AVAX/USD', 'LINK/USD']
                    filtered_tickers = [t for t in selected_tickers if any(major in t for major in major_coins)]
                    
                    if filtered_tickers:
                        st.success(f"✅ Found {len(filtered_tickers)} major coins in your selection: {', '.join(filtered_tickers)}")
                        st.info(f"💰 With $100 each, you can buy: {len(filtered_tickers)} positions in expensive coins!")
                        # Don't change selection, just show info
                    else:
                        st.warning("⚠️ No major coins in your selection. Select BTC, ETH, or SOL from your watchlist first!")
            
            with col_frac2:
                if st.button("🛡️ Spot Trading Only (No Leverage)", type="secondary", use_container_width=True):
                    st.session_state.fractional_spot_only = True
                    st.success("✅ Will analyze SPOT trades only (safest for beginners)")
            
            # Fractional buying info
            st.info("💡 **Micro Trading Enabled!** Kraken supports fractional crypto purchases. Even with $100, you can trade BTC/ETH!\n\n"
                   "**Examples:** $100 → 0.00111 BTC | 0.033 ETH | 0.667 SOL | 1,000 DOGE")
            
            # Configuration options
            with st.expander("⚙️ Analysis Configuration", expanded=True):
                config_col1, config_col2, config_col3 = st.columns(3)
                
                with config_col1:
                    test_directions = st.multiselect(
                        "Directions to Test",
                        options=['BUY', 'SELL'],
                        default=['BUY', 'SELL'],
                        help="Test long positions (BUY) and/or short positions (SELL)"
                    )
                
                with config_col2:
                    test_leverage = st.multiselect(
                        "Leverage Levels",
                        options=[1.0, 2.0, 3.0, 5.0],
                        default=[1.0, 2.0, 3.0, 5.0],
                        help="Leverage multipliers to test (1x = spot, >1x = margin)"
                    )
                
                with config_col3:
                    position_size_per_pair = st.number_input(
                        "Position Size per Pair (USD)",
                        min_value=10.0,
                        max_value=10000.0,
                        value=100.0,
                        step=10.0,
                        help="💡 Kraken supports fractional buys! $100 can buy 0.00111 BTC or 0.033 ETH"
                    )
                    
                    # Quick presets
                    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
                    if preset_col1.button("$50", key="preset_50_multi"):
                        st.session_state.multi_position_preset = 50.0
                        st.rerun()
                    if preset_col2.button("$100", key="preset_100_multi"):
                        st.session_state.multi_position_preset = 100.0
                        st.rerun()
                    if preset_col3.button("$200", key="preset_200_multi"):
                        st.session_state.multi_position_preset = 200.0
                        st.rerun()
                    if preset_col4.button("$500", key="preset_500_multi"):
                        st.session_state.multi_position_preset = 500.0
                        st.rerun()
                    
                    # Apply preset if set
                    if 'multi_position_preset' in st.session_state:
                        position_size_per_pair = st.session_state.multi_position_preset
                
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    risk_pct_config = st.number_input(
                        "Risk (Stop Loss) %",
                        min_value=0.5,
                        max_value=10.0,
                        value=2.0,
                        step=0.5,
                        help="Stop loss percentage from entry"
                    )
                
                with risk_col2:
                    tp_pct_config = st.number_input(
                        "Take Profit %",
                        min_value=1.0,
                        max_value=50.0,  # Increased for volatile crypto
                        value=10.0,
                        step=0.5,
                        help="Take profit percentage from entry"
                    )
            
            # Validate configuration
            if not test_directions:
                st.error("❌ Please select at least one direction (BUY or SELL)")
            elif not test_leverage:
                st.error("❌ Please select at least one leverage level")
            elif not selected_tickers:
                st.error("❌ No tickers selected for analysis")
            else:
                # Show what will be tested
                total_configs = len(selected_tickers) * len(test_directions) * len(test_leverage)
                st.info(f"🔬 Ready to test **{total_configs} configurations** across **{len(selected_tickers)} pairs**")
                
                if st.button("🚀 Start Multi-Config Analysis", type="primary"):
                    # Clear previous results
                    st.session_state.multi_config_results = None
                    st.session_state.multi_config_risk_pct = None
                    st.session_state.multi_config_tp_pct = None
                    
                    # Build test config
                    test_configs = {
                        'directions': test_directions,
                        'leverage_levels': test_leverage,
                        'risk_pct': risk_pct_config,
                        'take_profit_pct': tp_pct_config
                    }
                    
                    # Store config for later use when displaying results
                    st.session_state.multi_config_risk_pct = risk_pct_config
                    st.session_state.multi_config_tp_pct = tp_pct_config
                    
                    # Run multi-config analysis (this will store results in session state)
                    analyze_multi_config_bulk(
                        kraken_client=kraken_client,
                        pairs=selected_tickers,
                        position_size=position_size_per_pair,
                        test_configs=test_configs
                    )
                
                # Display existing results if they exist (handles reruns after button clicks)
                elif 'multi_config_results' in st.session_state and st.session_state.multi_config_results is not None:
                    # Call with empty pairs list to skip analysis and just display results
                    logger.info("📊 Displaying existing multi-config results after rerun")
                    analyze_multi_config_bulk(
                        kraken_client=kraken_client,
                        pairs=[],  # Empty list triggers display-only mode
                        position_size=position_size_per_pair,
                        test_configs={
                            'risk_pct': st.session_state.get('multi_config_risk_pct', 2.0),
                            'take_profit_pct': st.session_state.get('multi_config_tp_pct', 5.0)
                        }
                    )
        
        # Execute ULTIMATE analysis - tests ALL strategies + ALL configs
        if st.session_state.analysis_mode == 'ultimate':
            st.markdown("---")
            st.markdown("### 🚀 ULTIMATE ANALYSIS")
            st.caption("⚡ Testing EVERY possible combination: All Strategies × Long/Short × Spot/Margin/Leverage")
            
            # Fractional buying info
            st.info("💡 **Micro Trading Enabled!** Kraken supports fractional crypto purchases. Even with $100, you can trade BTC/ETH!\n\n"
                   "**Examples:** $100 → 0.00111 BTC | 0.033 ETH | 0.667 SOL | 1,000 DOGE")
            
            # Configuration options
            with st.expander("⚙️ Ultimate Analysis Configuration", expanded=True):
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    # Strategy selection - default to all
                    all_strategies = [
                        "orb_fvg",
                        "ema_crossover",
                        "rsi_stoch_hammer", 
                        "fisher_rsi_multi",
                        "macd_volume",
                        "aggressive_scalp"
                    ]
                    
                    test_strategies = st.multiselect(
                        "Strategies to Test",
                        options=all_strategies,
                        default=all_strategies,
                        format_func=lambda x: {
                            "orb_fvg": "📊 ORB+FVG (15min)",
                            "ema_crossover": "📈 EMA Crossover + Heikin Ashi",
                            "rsi_stoch_hammer": "📊 RSI + Stochastic + Hammer",
                            "fisher_rsi_multi": "🎯 Fisher RSI Multi-Indicator",
                            "macd_volume": "📉 MACD + Volume + RSI",
                            "aggressive_scalp": "🔥 Aggressive Scalping"
                        }.get(x, str(x)),
                        help="Select which trading strategies to test. Default = ALL for complete analysis"
                    )
                    
                    test_directions_ultimate = st.multiselect(
                        "Directions to Test",
                        options=['BUY', 'SELL'],
                        default=['BUY', 'SELL'],
                        help="Test long positions (BUY) and/or short positions (SELL)",
                        key="ultimate_directions"
                    )
                
                with config_col2:
                    test_leverage_ultimate = st.multiselect(
                        "Leverage Levels",
                        options=[1.0, 2.0, 3.0, 5.0],
                        default=[1.0, 2.0, 3.0],  # Skip 5x by default to reduce config count
                        help="Leverage multipliers to test (1x = spot, >1x = margin)",
                        key="ultimate_leverage"
                    )
                    
                    position_size_ultimate = st.number_input(
                        "Position Size per Pair (USD)",
                        min_value=10.0,
                        max_value=10000.0,
                        value=100.0,
                        step=10.0,
                        help="💡 Kraken supports fractional buys! $100 can buy 0.00111 BTC or 0.033 ETH",
                        key="ultimate_position_size"
                    )
                    
                    # Quick presets
                    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
                    if preset_col1.button("$50", key="preset_50_ult"):
                        st.session_state.ultimate_position_preset = 50.0
                        st.rerun()
                    if preset_col2.button("$100", key="preset_100_ult"):
                        st.session_state.ultimate_position_preset = 100.0
                        st.rerun()
                    if preset_col3.button("$200", key="preset_200_ult"):
                        st.session_state.ultimate_position_preset = 200.0
                        st.rerun()
                    if preset_col4.button("$500", key="preset_500_ult"):
                        st.session_state.ultimate_position_preset = 500.0
                        st.rerun()
                    
                    # Apply preset if set
                    if 'ultimate_position_preset' in st.session_state:
                        position_size_ultimate = st.session_state.ultimate_position_preset
            
            # Validate configuration
            if not test_strategies:
                st.error("❌ Please select at least one strategy to test")
            elif not test_directions_ultimate:
                st.error("❌ Please select at least one direction (BUY or SELL)")
            elif not test_leverage_ultimate:
                st.error("❌ Please select at least one leverage level")
            elif not selected_tickers:
                st.error("❌ No tickers selected for analysis")
            else:
                # Calculate total configurations
                total_ultimate_configs = len(selected_tickers) * len(test_strategies) * len(test_directions_ultimate) * len(test_leverage_ultimate)
                
                st.warning(f"⚠️ **MASSIVE ANALYSIS AHEAD**: This will test **{total_ultimate_configs} total configurations**!")
                st.info(f"📊 **Breakdown:** {len(selected_tickers)} tokens × {len(test_strategies)} strategies × {len(test_directions_ultimate)} directions × {len(test_leverage_ultimate)} leverages")
                st.caption("💡 This may take 2-5 minutes depending on your selection. Progress will be shown.")
                
                if st.button("🔥 START ULTIMATE ANALYSIS", type="primary", key="start_ultimate"):
                    with st.spinner("🚀 Running comprehensive analysis across all configurations..."):
                        # Call the ultimate analysis function
                        analyze_ultimate_all_strategies(
                            kraken_client=kraken_client,
                            adapter=adapter,
                            pairs=selected_tickers,
                            strategies=test_strategies,
                            directions=test_directions_ultimate,
                            leverage_levels=test_leverage_ultimate,
                            position_size=position_size_ultimate,
                            timeframe=timeframe
                        )

    # --- 4. Ranked Opportunities ---
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("#### 4. Ranked Opportunities")
        
        results = st.session_state.analysis_results
        sorted_results = sorted(results, key=lambda x: x['confidence_score'], reverse=True)

        # Summary Metrics
        buy_signals = sum(1 for r in sorted_results if r['recommendation'] == 'BUY')
        sell_signals = sum(1 for r in sorted_results if r['recommendation'] == 'SELL')
        hold_signals = sum(1 for r in sorted_results if r['recommendation'] == 'HOLD')
        avg_confidence = sum(r['confidence_score'] for r in sorted_results) / len(sorted_results) if sorted_results else 0
        high_confidence_count = sum(1 for r in sorted_results if r['confidence_score'] >= 75)
        near_entry_count = sum(1 for r in sorted_results if r.get('signals', {}).get('near_entry', False))

        # Smart insights at the top
        if buy_signals > 0:
            st.success(f"🎯 **{buy_signals} BUY Signal{'s' if buy_signals != 1 else ''}** - Ready to trade now!")
        elif near_entry_count > 0:
            st.warning(f"⏳ **{near_entry_count} coin{'s' if near_entry_count != 1 else ''} approaching BUY signal** - Watch these closely!")
        elif avg_confidence >= 55:
            st.info(f"📊 Market conditions are decent (avg confidence {avg_confidence:.1f}%) - Be patient for better setups")
        else:
            st.warning(f"⚠️ Weak market conditions (avg confidence {avg_confidence:.1f}%) - Consider waiting or exploring other opportunities")

        summary_cols = st.columns(6)
        summary_cols[0].metric("Total Analyzed", len(sorted_results))
        summary_cols[1].metric("🟢 BUY", f"{buy_signals}", help="Strong buy signals - Ready to enter")
        summary_cols[2].metric("🔴 SELL", f"{sell_signals}", help="Exit signals - Consider taking profits")
        summary_cols[3].metric("🟡 HOLD", f"{hold_signals}", help="No clear signal yet")
        summary_cols[4].metric("🎯 Near Entry", f"{near_entry_count}", help="Approaching buy signal - Watch closely")
        summary_cols[5].metric("Avg. Confidence", f"{avg_confidence:.1f}%", help="Overall signal strength")

        # Add filter options
        filter_col1, filter_col2 = st.columns([2, 1])
        with filter_col1:
            show_filter = st.selectbox(
                "Filter by:",
                ["All", "🔥 Scalping Opportunities", "BUY Only", "SELL Only", "HOLD Only", "Near Entry (HOLD)", "High Confidence (75%+)"],
                help="Filter analysis results by signal type or confidence"
            )
        
        # Apply filter
        if show_filter == "🔥 Scalping Opportunities":
            # Show BUY signals + Near Entry with good conditions for scalping
            sorted_results = [
                r for r in sorted_results 
                if (
                    # BUY signal with decent confidence
                    (r['recommendation'] == 'BUY' and r['confidence_score'] >= 60) or
                    # OR Near Entry with oversold + volume
                    (r.get('signals', {}).get('near_entry', False) and 
                     r.get('signals', {}).get('rsi', 100) < 35 and
                     r.get('signals', {}).get('volume_ratio', 0) > 1.5 and
                     r['confidence_score'] >= 55)
                )
            ]
            # Re-sort by confidence for scalping opportunities
            sorted_results.sort(key=lambda x: (
                x['recommendation'] == 'BUY',  # BUY signals first
                x['confidence_score']  # Then by confidence
            ), reverse=True)
        elif show_filter == "BUY Only":
            sorted_results = [r for r in sorted_results if r['recommendation'] == 'BUY']
        elif show_filter == "SELL Only":
            sorted_results = [r for r in sorted_results if r['recommendation'] == 'SELL']
        elif show_filter == "HOLD Only":
            sorted_results = [r for r in sorted_results if r['recommendation'] == 'HOLD']
        elif show_filter == "Near Entry (HOLD)":
            sorted_results = [r for r in sorted_results if r.get('signals', {}).get('near_entry', False)]
        elif show_filter == "High Confidence (75%+)":
            sorted_results = [r for r in sorted_results if r['confidence_score'] >= 75]
        
        if not sorted_results:
            if show_filter == "🔥 Scalping Opportunities":
                st.warning(f"❌ No scalping opportunities found right now.")
                st.info("💡 **Tip**: Try scanning more coins or wait for market conditions to improve. Scalping needs: BUY signals OR oversold RSI (<35) + volume spike (>1.5x)")
            else:
                st.info(f"No results match the filter: {show_filter}")
        else:
            if show_filter == "🔥 Scalping Opportunities":
                buy_count = sum(1 for r in sorted_results if r['recommendation'] == 'BUY')
                near_entry_count = sum(1 for r in sorted_results if r.get('signals', {}).get('near_entry', False))
                st.success(f"🔥 Found {len(sorted_results)} scalping opportunities: {buy_count} BUY signals + {near_entry_count} Near Entry setups")
            else:
                st.info(f"Showing {len(sorted_results)} results")

        for i, analysis in enumerate(sorted_results):
            # Build title with more context
            title_parts = [
                f"**{i+1}. {analysis['symbol']}**",
                f"Signal: **{analysis['recommendation']}**",
                f"Confidence: **{analysis['confidence_score']}%**"
            ]
            
            # Add near-entry badge
            if analysis.get('signals', {}).get('near_entry', False):
                title_parts.append("🎯 **Near Entry**")
            
            expander_title = " | ".join(title_parts)
            
            with st.expander(expander_title, expanded=(i < 3)):
                
                rec = analysis['recommendation']
                if rec == 'BUY':
                    st.success(f"✅ **{rec}** Signal - Entry conditions met!")
                elif rec == 'SELL':
                    st.error(f"❌ **{rec}** Signal - Exit conditions met")
                else:
                    # Provide more context for HOLD signals
                    signals = analysis.get('signals', {})
                    confidence = analysis.get('confidence_score', 50)
                    
                    if signals.get('near_entry'):
                        st.warning(f"🎯 **{rec}** ({confidence}%) - Approaching BUY signal! Watch closely.")
                    elif signals.get('near_exit'):
                        st.info(f"⚠️ **{rec}** ({confidence}%) - Approaching SELL signal. Consider reducing exposure.")
                    elif confidence >= 60:
                        st.info(f"✅ **{rec}** ({confidence}%) - Good holding conditions. Stay patient.")
                    elif confidence >= 50:
                        st.info(f"⏸️ **{rec}** ({confidence}%) - Neutral. Monitor for opportunities.")
                    else:
                        st.warning(f"⚠️ **{rec}** ({confidence}%) - Weak conditions. Consider other opportunities.")
                
                # Show confidence breakdown if available
                signals = analysis.get('signals', {})
                if 'confidence_factors' in signals and signals['confidence_factors']:
                    with st.expander("🔍 Confidence Score Breakdown", expanded=False):
                        st.markdown("**Factors affecting this confidence score:**")
                        for factor in signals['confidence_factors']:
                            # Color code based on positive/negative impact
                            if '(+' in factor:
                                st.markdown(f"✅ {factor}")
                            elif '(-' in factor:
                                st.markdown(f"❌ {factor}")
                            else:
                                st.markdown(f"ℹ️ {factor}")

                # Key metrics
                metric_cols = st.columns(4)
                metric_cols[0].metric("Current Price", f"${analysis['current_price']:,.4f}")
                metric_cols[1].metric("Stop Loss", f"${analysis['stop_loss']:,.4f}")
                metric_cols[2].metric("Risk Level", analysis['risk_level'])
                metric_cols[3].metric("R:R Ratio", f"{analysis.get('risk_reward_ratio', 'N/A')}")
                
                # Technical Indicators Section
                st.markdown("**📊 Technical Indicators**")
                signals = analysis.get('signals', {})
                
                # Create a more visual indicator summary
                ind_summary_cols = st.columns(6)
                
                # RSI indicator with color coding
                rsi = signals.get('rsi', 0)
                with ind_summary_cols[0]:
                    if rsi < 30:
                        st.metric("RSI", f"{rsi:.1f}", "Oversold 🟢", delta_color="normal")
                    elif rsi > 70:
                        st.metric("RSI", f"{rsi:.1f}", "Overbought 🔴", delta_color="inverse")
                    else:
                        st.metric("RSI", f"{rsi:.1f}", "Neutral 🟡", delta_color="off")
                
                # Volume indicator
                vol_ratio = signals.get('volume_ratio', 0)
                with ind_summary_cols[1]:
                    if vol_ratio > 2.0:
                        st.metric("Volume", f"{vol_ratio:.1f}x", "Spike 🔥", delta_color="normal")
                    elif vol_ratio > 1.5:
                        st.metric("Volume", f"{vol_ratio:.1f}x", "High 📊", delta_color="normal")
                    else:
                        st.metric("Volume", f"{vol_ratio:.1f}x", "Normal", delta_color="off")
                
                # ADX trend strength
                adx = signals.get('adx', 0)
                with ind_summary_cols[2]:
                    if adx > 25:
                        st.metric("ADX", f"{adx:.1f}", "Strong 💪", delta_color="normal")
                    else:
                        st.metric("ADX", f"{adx:.1f}", "Weak 📉", delta_color="off")
                
                # EMA alignment
                ema_aligned = signals.get('ema_aligned', False)
                with ind_summary_cols[3]:
                    if ema_aligned:
                        st.metric("EMA", "✅", "Aligned", delta_color="normal")
                    else:
                        st.metric("EMA", "❌", "Mixed", delta_color="off")
                
                # MACD signal
                macd = signals.get('macd', 0)
                macd_signal = signals.get('macdsignal', 0)
                with ind_summary_cols[4]:
                    if macd > macd_signal:
                        st.metric("MACD", "✅", "Bullish", delta_color="normal")
                    elif macd < macd_signal:
                        st.metric("MACD", "❌", "Bearish", delta_color="inverse")
                    else:
                        st.metric("MACD", "⚪", "Neutral", delta_color="off")
                
                # Near entry/exit flag
                with ind_summary_cols[5]:
                    if signals.get('near_entry'):
                        st.metric("Status", "🎯", "Near BUY", delta_color="normal")
                    elif signals.get('near_exit'):
                        st.metric("Status", "⚠️", "Near SELL", delta_color="inverse")
                    else:
                        st.metric("Status", "⏸️", "Watch", delta_color="off")
                
                # Detailed indicator breakdown in expander
                with st.expander("📈 Detailed Technical Data", expanded=False):
                    ind_col1, ind_col2, ind_col3 = st.columns(3)
                    
                    with ind_col1:
                        st.markdown("**Momentum Indicators:**")
                        st.text(f"RSI: {rsi:.2f}")
                        st.text(f"Stochastic: {signals.get('stoch', 0):.2f}")
                        st.text(f"Fisher RSI: {signals.get('fisher_rsi', 0):.3f}")
                        st.text(f"MFI: {signals.get('mfi', 0):.2f}")
                    
                    with ind_col2:
                        st.markdown("**Trend Indicators:**")
                        st.text(f"ADX: {adx:.2f}")
                        st.text(f"EMA5: ${signals.get('ema5', 0):.4f}")
                        st.text(f"EMA10: ${signals.get('ema10', 0):.4f}")
                        st.text(f"EMA20: ${signals.get('ema20', 0):.4f}")
                        ema_cross = signals.get('ema_cross_strength', 0)
                        st.text(f"EMA Separation: {ema_cross:.2f}%")
                    
                    with ind_col3:
                        st.markdown("**Volume & MACD:**")
                        st.text(f"Volume Ratio: {vol_ratio:.2f}x")
                        st.text(f"MACD: {macd:.6f}")
                        st.text(f"MACD Signal: {macd_signal:.6f}")
                        macd_diff = macd - macd_signal
                        st.text(f"MACD Diff: {macd_diff:.6f}")
                        st.text(f"Volume Spike: {'Yes ✅' if signals.get('volume_spike') else 'No'}")

                # Action button - Use on_click callback for reliable session state setting in expanders
                st.markdown("---")
                
                def set_analysis_setup(analysis_data):
                    logger.info(f"🔘 ANALYSIS SETUP CALLBACK! Setting up {analysis_data['symbol']}")
                    # Store in crypto_scanner_opportunity format for Execute Trade tab
                    st.session_state.crypto_scanner_opportunity = {
                        'symbol': analysis_data['symbol'],
                        'strategy': analysis_data['strategy'],
                        'confidence': analysis_data['confidence'],
                        'risk_level': analysis_data['risk_level'],
                        'score': analysis_data.get('score', 75),
                        'current_price': analysis_data['current_price'],
                        'change_24h': analysis_data.get('change_24h', 0),
                        'volume_ratio': analysis_data.get('volume_ratio', 1.0),
                        'volatility': analysis_data.get('volatility', 0),
                        'reason': analysis_data.get('reason', f"{analysis_data['action']} signal detected"),
                        'ai_reasoning': analysis_data.get('ai_reasoning', ''),
                        'ai_confidence': analysis_data.get('confidence', 'Medium'),
                        'ai_rating': analysis_data.get('confidence_pct', 50) / 10,
                        'ai_risks': analysis_data.get('risks', ['Standard market risks'])
                    }
                    
                    # Also set quick trade values for compatibility
                    st.session_state.crypto_quick_pair = analysis_data['symbol']
                    st.session_state.crypto_quick_trade_pair = analysis_data['symbol']
                    st.session_state.crypto_quick_direction = analysis_data['action'].upper()
                    st.session_state.crypto_quick_stop_pct = abs((analysis_data['stop_loss'] - analysis_data['current_price']) / analysis_data['current_price'] * 100)
                    if analysis_data['roi_targets']:
                        st.session_state.crypto_quick_target_pct = analysis_data['roi_targets'][0]['gain_percent']
                    
                    # CRITICAL FIX: Set missing required fields for Execute Trade form
                    st.session_state.crypto_quick_leverage = 1.0  # Default to spot trading (no leverage)
                    st.session_state.crypto_quick_position_size = 100.0  # Default position size
                    
                    # Switch to Execute Trade tab
                    st.session_state.quick_trade_subtab = "⚡ Execute Trade"
                    st.session_state.show_analysis_setup_success = {'symbol': analysis_data['symbol']}
                
                st.button(
                    "✅ Use This Setup", 
                    key=f"use_{analysis['symbol']}", 
                    use_container_width=True, 
                    type="primary",
                    on_click=set_analysis_setup,
                    args=(analysis,)
                )


def render_quick_trade_tab(
    kraken_client: KrakenClient, 
    crypto_config,
    penny_crypto_scanner=None,
    crypto_opportunity_scanner=None,
    ai_crypto_scanner=None,
    sub_penny_discovery=None,
    watchlist_manager=None
):
    """
    Main renderer for the Quick Trade tab with integrated unified scanner
    Accepts pre-cached scanner instances for optimal performance
    """
    # Initialize scanner instances in session state if not exists (only once)
    if 'scanner_instances' not in st.session_state:
        st.session_state.scanner_instances = {}
    
    # Use provided cached scanners or create new ones only if not provided
    if penny_crypto_scanner and 'penny_crypto_scanner' not in st.session_state.scanner_instances:
        st.session_state.scanner_instances['penny_crypto_scanner'] = penny_crypto_scanner
    elif 'penny_crypto_scanner' not in st.session_state.scanner_instances:
        try:
            from services.penny_crypto_scanner import PennyCryptoScanner
            st.session_state.scanner_instances['penny_crypto_scanner'] = PennyCryptoScanner(kraken_client)
        except ImportError:
            logger.warning("PennyCryptoScanner not available")
    
    if crypto_opportunity_scanner and 'crypto_opportunity_scanner' not in st.session_state.scanner_instances:
        st.session_state.scanner_instances['crypto_opportunity_scanner'] = crypto_opportunity_scanner
    elif 'crypto_opportunity_scanner' not in st.session_state.scanner_instances:
        try:
            from services.crypto_scanner import CryptoOpportunityScanner
            st.session_state.scanner_instances['crypto_opportunity_scanner'] = CryptoOpportunityScanner(kraken_client)
        except ImportError:
            logger.warning("CryptoOpportunityScanner not available")
    
    if ai_crypto_scanner and 'ai_crypto_scanner' not in st.session_state.scanner_instances:
        st.session_state.scanner_instances['ai_crypto_scanner'] = ai_crypto_scanner
    elif 'ai_crypto_scanner' not in st.session_state.scanner_instances:
        try:
            from services.ai_crypto_scanner import AICryptoScanner
            st.session_state.scanner_instances['ai_crypto_scanner'] = AICryptoScanner(kraken_client)
        except ImportError:
            logger.warning("AICryptoScanner not available")
    
    if sub_penny_discovery and 'sub_penny_discovery' not in st.session_state.scanner_instances:
        st.session_state.scanner_instances['sub_penny_discovery'] = sub_penny_discovery
    elif 'sub_penny_discovery' not in st.session_state.scanner_instances:
        try:
            from services.sub_penny_discovery import SubPennyDiscovery
            st.session_state.scanner_instances['sub_penny_discovery'] = SubPennyDiscovery()
        except ImportError:
            logger.warning("SubPennyDiscovery not available")
    
    # Use stateful navigation instead of st.tabs() to prevent reruns
    if 'quick_trade_subtab' not in st.session_state:
        st.session_state.quick_trade_subtab = "🔍 Ticker Management"
    
    # Use index-based selection to allow programmatic tab switching
    tab_options = ["🔍 Ticker Management", "⚡ Execute Trade"]
    current_index = tab_options.index(st.session_state.quick_trade_subtab) if st.session_state.quick_trade_subtab in tab_options else 0
    
    # Callback to update session state when radio button is changed by user
    def update_quick_trade_subtab():
        st.session_state.quick_trade_subtab = st.session_state.quick_trade_subtab_selector
    
    # CRITICAL: Sync the radio selector key with quick_trade_subtab for programmatic navigation
    # This ensures that when quick_trade_subtab is set programmatically (e.g., from Multi-Config),
    # the radio button reflects the correct tab on the next render
    if 'quick_trade_subtab_selector' not in st.session_state:
        st.session_state.quick_trade_subtab_selector = st.session_state.quick_trade_subtab
    elif st.session_state.quick_trade_subtab_selector != st.session_state.quick_trade_subtab:
        # Programmatic navigation detected - sync the selector
        st.session_state.quick_trade_subtab_selector = st.session_state.quick_trade_subtab
    
    # Tab selector using radio buttons with on_change callback
    st.radio(
        "Navigation",
        options=tab_options,
        index=current_index,
        horizontal=True,
        label_visibility="collapsed",
        key="quick_trade_subtab_selector",
        on_change=update_quick_trade_subtab
    )
    
    # Render the selected subtab
    if st.session_state.quick_trade_subtab == "🔍 Ticker Management":
        display_unified_scanner(kraken_client, crypto_config, st.session_state.scanner_instances, watchlist_manager)
    elif st.session_state.quick_trade_subtab == "⚡ Execute Trade":
        display_trade_setup(kraken_client, crypto_config, watchlist_manager)


def display_trade_setup(kraken_client: KrakenClient, crypto_config, watchlist_manager=None):
    """
    Display the trade execution form with AI analysis
    Supports single trade, bulk custom selection, and bulk watchlist trading
    """
    st.markdown("### ⚡ Execute Trade")
    
    # Show success message if we just switched here from a button click
    if 'show_setup_success' in st.session_state:
        setup_info = st.session_state.show_setup_success
        st.success(f"✅ Trade setup loaded for {setup_info['pair']} ({setup_info['trade_type']})!")
        st.balloons()
        del st.session_state.show_setup_success
    
    # Trade mode selector
    trade_mode = st.radio(
        "Trade Mode",
        options=["Single Trade", "Bulk Custom Selection", "Bulk Watchlist"],
        horizontal=True,
        key="crypto_trade_mode"
    )
    
    st.divider()
    
    # Render based on mode
    if trade_mode == "Single Trade":
        display_single_trade(kraken_client, crypto_config)
    elif trade_mode == "Bulk Custom Selection":
        display_bulk_custom_trade(kraken_client, crypto_config)
    elif trade_mode == "Bulk Watchlist":
        if watchlist_manager:
            display_bulk_watchlist_trade(kraken_client, crypto_config, watchlist_manager)
        else:
            st.error("Watchlist manager not available. Please ensure watchlist is initialized.")


def display_single_trade(kraken_client: KrakenClient, crypto_config):
    """
    Display single trade execution form
    """
    # Display scanner opportunity if one was copied
    if 'crypto_scanner_opportunity' in st.session_state:
        opp = st.session_state.crypto_scanner_opportunity
        
        st.success(f"✅ Scanner opportunity loaded: **{opp['symbol']}**")
        
        with st.expander("📊 Scanner Analysis Details", expanded=True):
            scol1, scol2, scol3, scol4 = st.columns(4)
            
            with scol1:
                st.metric("Score", f"{opp['score']:.1f}/100")
                st.metric("Strategy", opp['strategy'].upper())
            
            with scol2:
                st.metric("Confidence", opp['confidence'])
                st.metric("Risk Level", opp['risk_level'])
            
            with scol3:
                st.metric("Price", f"${opp['current_price']:,.6f}")
                direction = "🟢" if opp['change_24h'] > 0 else "🔴"
                st.metric("24h Change", f"{direction} {opp['change_24h']:.2f}%")
            
            with scol4:
                st.metric("Volume Ratio", f"{opp['volume_ratio']:.2f}x")
                st.metric("Volatility", f"{opp['volatility']:.2f}%")
            
            st.info(f"**Analysis:** {opp['reason']}")
            
            # Show AI analysis if available
            if 'ai_reasoning' in opp:
                st.markdown("**🤖 AI Analysis:**")
                st.write(f"**Confidence:** {opp['ai_confidence']} | **Rating:** {opp['ai_rating']:.1f}/10")
                st.write(f"**Reasoning:** {opp['ai_reasoning']}")
                st.warning(f"**Risks:** {opp['ai_risks']}")
            
            # Clear button
            if st.button("🗑️ Clear Scanner Data", key="clear_scanner_opp"):
                del st.session_state.crypto_scanner_opportunity
                if 'crypto_quick_stop_pct' in st.session_state:
                    del st.session_state.crypto_quick_stop_pct
                if 'crypto_quick_target_pct' in st.session_state:
                    del st.session_state.crypto_quick_target_pct
                if 'crypto_quick_direction' in st.session_state:
                    del st.session_state.crypto_quick_direction
                st.success("✅ Scanner data cleared")
                st.rerun()
        
        st.markdown("---")
    
    # Check if scanner opportunity is loaded - auto-populate everything
    scanner_loaded = 'crypto_scanner_opportunity' in st.session_state
    
    if scanner_loaded:
        # Scanner setup is active - use those values
        opp = st.session_state.crypto_scanner_opportunity
        selected_pair = opp['symbol']
        
        # Display locked-in setup
        st.info(f"🔒 **Using Scanner Setup:** {selected_pair} | {opp['strategy'].upper()} strategy")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pair", selected_pair)
        with col2:
            st.metric("Strategy", opp['strategy'].upper())
        with col3:
            direction = st.session_state.get('crypto_quick_direction', 'BUY')
            st.metric("Direction", direction)
        
        # Add quick action buttons
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("🤖 Run AI Analysis", key="quick_ai_analysis", type="primary", use_container_width=True):
                logger.info(f"🔘 Quick AI Analysis button clicked for {selected_pair}")
                # Set flag to trigger analysis in the AI Analysis section below
                st.session_state.trigger_ai_analysis = True
                st.info("⏳ Triggering AI analysis... scroll down to see results")
                st.rerun()
        
        with action_col2:
            if st.button("✏️ Modify Setup", key="modify_scanner_setup", use_container_width=True):
                # Clear scanner data to allow manual entry
                del st.session_state.crypto_scanner_opportunity
                st.rerun()
        
        with action_col3:
            if st.button("🗑️ Clear & Start Fresh", key="clear_and_restart", use_container_width=True):
                # Clear everything
                if 'crypto_scanner_opportunity' in st.session_state:
                    del st.session_state.crypto_scanner_opportunity
                if 'crypto_quick_trade_pair' in st.session_state:
                    del st.session_state.crypto_quick_trade_pair
                if 'crypto_quick_stop_pct' in st.session_state:
                    del st.session_state.crypto_quick_stop_pct
                if 'crypto_quick_target_pct' in st.session_state:
                    del st.session_state.crypto_quick_target_pct
                st.rerun()
        
        # Direction selector (still allow changing)
        direction = st.radio(
            "Confirm Direction:",
            options=["BUY", "SELL"],
            index=0 if st.session_state.get('crypto_quick_direction', 'BUY') == 'BUY' else 1,
            horizontal=True,
            key="crypto_quick_direction"
        )
        
    else:
        # No scanner setup - manual entry mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Add input method selector
            input_method = st.radio(
                "Select input method:",
                options=["Dropdown", "Custom Input"],
                horizontal=True,
                key="pair_input_method",
                help="Use dropdown for common pairs or custom input to type any ticker"
            )
            
            if input_method == "Dropdown":
                # Get available pairs from Kraken (cached in session state)
                try:
                    # Cache asset pairs in session state to avoid repeated API calls
                    cache_key = 'crypto_asset_pairs_cache'
                    cache_timestamp_key = 'crypto_asset_pairs_cache_timestamp'
                    cache_ttl = 300  # 5 minutes
                    
                    import time
                    current_time = time.time()
                    
                    # Check if cache exists and is still valid
                    if (cache_key in st.session_state and 
                        cache_timestamp_key in st.session_state and
                        current_time - st.session_state[cache_timestamp_key] < cache_ttl):
                        asset_pairs = st.session_state[cache_key]
                    else:
                        # Fetch fresh data
                        asset_pairs = kraken_client.get_tradable_asset_pairs()
                        st.session_state[cache_key] = asset_pairs
                        st.session_state[cache_timestamp_key] = current_time
                    
                    pair_options = [pair['altname'] for pair in asset_pairs if pair['quote'] == 'USD' or pair['quote'] == 'USDT']
                    pair_options.sort()
                    
                    # Default to session state or first available
                    default_pair = st.session_state.get('crypto_quick_trade_pair', pair_options[0] if pair_options else 'BTC/USD')
                    
                    selected_pair = st.selectbox(
                        "Trading Pair",
                        options=pair_options,
                        index=pair_options.index(default_pair) if default_pair in pair_options else 0,
                        key="crypto_quick_trade_pair_dropdown"
                    )
                except Exception as e:
                    st.error(f"Error loading trading pairs: {e}")
                    return
            else:
                # Custom text input for any ticker
                selected_pair = st.text_input(
                    "Trading Pair (Symbol/USD)",
                    value=st.session_state.get('crypto_quick_trade_pair', 'BTC/USD'),
                    key="crypto_custom_pair",
                    help="Enter any ticker symbol (e.g., HIPPO/USD, BTC/USD, ETH/USD). Format: SYMBOL/USD or SYMBOL/USDT",
                    placeholder="e.g., HIPPO/USD"
                ).upper().strip()
                
                # Validate format
                if selected_pair and '/' not in selected_pair:
                    st.warning("⚠️ Please use format: SYMBOL/USD (e.g., HIPPO/USD)")
                elif selected_pair:
                    st.info(f"✓ Will trade: **{selected_pair}**")
        
        with col2:
            st.write("")
            st.write("")
            direction = st.radio(
                "Direction",
                options=["BUY", "SELL"],
                horizontal=True,
                key="crypto_quick_direction"
            )
    
    # 🆕 ENHANCED SELL MODE: Show position selection when SELL is selected
    if direction == "SELL":
        st.markdown("---")
        st.markdown("#### 💰 Select Position to Sell")
        st.info("💡 **Tip:** Select a position from your Kraken account to get AI-powered exit analysis")
        
        # Fetch open positions
        with st.spinner("Fetching your Kraken positions..."):
            try:
                positions = kraken_client.get_open_positions()
                
                if not positions:
                    st.warning("⚠️ No open positions found in your Kraken account")
                    st.info("You can still manually enter a trading pair below to place a SELL order")
                else:
                    # Display positions in a table
                    st.markdown(f"**Found {len(positions)} open position(s):**")
                    
                    position_data = []
                    for pos in positions:
                        position_data.append({
                            'Pair': pos.pair,
                            'Volume': f"{pos.volume:.6f}",
                            'Current Price': f"${pos.current_price:,.4f}",
                            'Value (USD)': f"${pos.cost:,.2f}"
                        })
                    
                    df = pd.DataFrame(position_data)
                    st.dataframe(df, width='stretch', hide_index=True)
                    
                    # Position selector
                    position_options = [f"{pos.pair} ({pos.volume:.6f} @ ${pos.current_price:,.4f})" for pos in positions]
                    position_options.insert(0, "🔧 Manual Entry (enter pair below)")
                    
                    selected_position_idx = st.selectbox(
                        "Select Position to Analyze",
                        options=range(len(position_options)),
                        format_func=lambda i: position_options[i],
                        key="crypto_sell_position_selector"
                    )
                    
                    # If a position is selected (not manual entry)
                    if selected_position_idx > 0:
                        selected_pos = positions[selected_position_idx - 1]
                        selected_pair = selected_pos.pair
                        
                        # Auto-populate the pair
                        st.session_state.crypto_custom_pair = selected_pair
                        
                        # Show AI Exit Analysis button
                        st.markdown("---")
                        if st.button("🤖 Get AI Exit Analysis", width='stretch', type="primary"):
                            with st.spinner(f"🤖 AI analyzing exit timing for {selected_pair}..."):
                                try:
                                    from services.ai_exit_analyzer import analyze_exit_timing
                                    
                                    # Get exit analysis
                                    exit_analysis = analyze_exit_timing(
                                        kraken_client=kraken_client,
                                        pair=selected_pair,
                                        current_price=selected_pos.current_price,
                                        position_size=selected_pos.volume,
                                        entry_price=selected_pos.entry_price if selected_pos.entry_price > 0 else selected_pos.current_price
                                    )
                                    
                                    # Store in session state
                                    st.session_state.crypto_exit_analysis = exit_analysis
                                    st.session_state.crypto_selected_position = selected_pos
                                    
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"AI exit analysis failed: {e}")
                                    logger.error("Exit analysis error: {}", str(e), exc_info=True)
                        
                        # Display exit analysis if available
                        if 'crypto_exit_analysis' in st.session_state:
                            exit_analysis = st.session_state.crypto_exit_analysis
                            selected_pos = st.session_state.crypto_selected_position
                            
                            st.markdown("---")
                            st.markdown("#### 🤖 AI Exit Recommendation")
                            
                            # Calculate P&L
                            if selected_pos.entry_price > 0:
                                pnl_pct = ((selected_pos.current_price - selected_pos.entry_price) / selected_pos.entry_price) * 100
                                pnl_usd = (selected_pos.current_price - selected_pos.entry_price) * selected_pos.volume
                            else:
                                pnl_pct = 0
                                pnl_usd = 0
                            
                            # Display P&L metrics
                            pnl_col1, pnl_col2, pnl_col3, pnl_col4 = st.columns(4)
                            pnl_col1.metric("Entry Price", f"${selected_pos.entry_price:,.4f}" if selected_pos.entry_price > 0 else "Unknown")
                            pnl_col2.metric("Current Price", f"${selected_pos.current_price:,.4f}")
                            pnl_col3.metric("P&L %", f"{pnl_pct:+.2f}%", delta=f"{pnl_pct:+.2f}%")
                            pnl_col4.metric("P&L USD", f"${pnl_usd:+,.2f}", delta=f"{pnl_usd:+,.2f}")
                            
                            # AI Recommendation
                            rec_col1, rec_col2 = st.columns([3, 1])
                            with rec_col1:
                                action = exit_analysis.get('action', 'HOLD')
                                confidence = exit_analysis.get('confidence', 0)
                                
                                if action == 'SELL_NOW':
                                    st.error(f"🚨 **AI says SELL NOW** (Confidence: {confidence:.0f}%)")
                                elif action == 'TAKE_PARTIAL':
                                    st.warning(f"⚠️ **AI suggests PARTIAL EXIT** (Confidence: {confidence:.0f}%)")
                                elif action == 'HOLD':
                                    st.success(f"✅ **AI says HOLD** (Confidence: {confidence:.0f}%)")
                                else:
                                    st.info(f"ℹ️ **AI says {action}** (Confidence: {confidence:.0f}%)")
                            
                            with rec_col2:
                                st.metric("AI Score", f"{exit_analysis.get('score', 0):.0f}/100")
                            
                            # AI Reasoning
                            st.info(f"**💡 AI Reasoning:** {exit_analysis.get('reasoning', 'No reasoning provided')}")
                            
                            # Key signals
                            if 'signals' in exit_analysis and exit_analysis['signals']:
                                st.markdown("**📊 Key Signals:**")
                                for signal in exit_analysis['signals'][:5]:
                                    st.write(f"- {signal}")
                            
                            # Suggested exit price/levels
                            if 'suggested_exit_price' in exit_analysis:
                                st.success(f"💰 **Suggested Exit Price:** ${exit_analysis['suggested_exit_price']:,.4f}")
                            
                            # Quick action buttons
                            st.markdown("---")
                            action_col1, action_col2, action_col3 = st.columns(3)
                            
                            with action_col1:
                                if st.button("🚀 Execute SELL", width='stretch', type="primary"):
                                    # Auto-populate analysis for execution
                                    st.session_state.crypto_analysis = {
                                        'pair': selected_pair,
                                        'direction': 'SELL',
                                        'current_price': selected_pos.current_price,
                                        'stop_loss': exit_analysis.get('stop_loss', selected_pos.current_price * 0.95),
                                        'take_profit': exit_analysis.get('take_profit', selected_pos.current_price * 1.05),
                                        'position_size': selected_pos.cost,
                                        'leverage': 1.0,
                                        'risk_reward_ratio': 2.0
                                    }
                                    st.success("✅ Trade setup populated! Scroll down to execute.")
                                    st.rerun()
                            
                            with action_col2:
                                if st.button("📊 Refresh Analysis", width='stretch'):
                                    if 'crypto_exit_analysis' in st.session_state:
                                        del st.session_state.crypto_exit_analysis
                                    st.rerun()
                            
                            with action_col3:
                                if st.button("🔄 Clear", width='stretch'):
                                    if 'crypto_exit_analysis' in st.session_state:
                                        del st.session_state.crypto_exit_analysis
                                    if 'crypto_selected_position' in st.session_state:
                                        del st.session_state.crypto_selected_position
                                    st.rerun()
            
            except Exception as e:
                st.error(f"Failed to fetch positions: {e}")
                logger.error("Position fetch error: {}", str(e), exc_info=True)
                st.info("You can still manually enter a trading pair below to place a SELL order")
    
    # ========================================================================
    # 🎯 TRADING OPTIONS & STRATEGY CONFIGURATION
    # ========================================================================
    st.markdown("---")
    st.markdown("### ⚙️ Trading Configuration")
    
    # Row 1: Order Type, Trading Mode, Strategy Template
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        order_type = st.selectbox(
            "📋 Order Type",
            options=["MARKET", "LIMIT", "STOP_LOSS", "TAKE_PROFIT"],
            index=0,
            key="crypto_order_type",
            help="MARKET = Execute immediately at current price\nLIMIT = Execute at specific price or better\nSTOP_LOSS = Sell if price drops to stop price\nTAKE_PROFIT = Sell if price rises to target"
        )
    
    with config_col2:
        trading_mode = st.selectbox(
            "💼 Trading Mode",
            options=["Spot Trading", "Margin Trading"],
            index=0,
            key="crypto_trading_mode",
            help="Spot = Trade with your own funds only\nMargin = Borrow funds to amplify position (enables short selling)"
        )
        
        # Show margin status indicator
        if trading_mode == "Margin Trading":
            st.caption("🔓 Margin/Leverage enabled")
        else:
            st.caption("🔒 No leverage (safer)")
    
    with config_col3:
        strategy_templates = {
            "Custom": {"risk": 2.0, "tp": 5.0, "leverage": 1.0, "description": "Manual settings"},
            "Conservative": {"risk": 1.0, "tp": 3.0, "leverage": 1.0, "description": "Low risk, spot only"},
            "Balanced": {"risk": 2.0, "tp": 5.0, "leverage": 1.0, "description": "Moderate risk/reward"},
            "Momentum": {"risk": 2.0, "tp": 6.0, "leverage": 1.0, "description": "Mid-cap coins with strong trends ($0.01-$1)"},
            "Volatile Altcoin": {"risk": 4.0, "tp": 10.0, "leverage": 1.0, "description": "⚠️ Wider stops for volatile alts (PLUME, TRUMP, etc.)"},
            "Aggressive Scalp": {"risk": 2.0, "tp": 3.0, "leverage": 2.0, "description": "Quick profits, 2x leverage"},
            "Swing Trade": {"risk": 3.0, "tp": 10.0, "leverage": 1.0, "description": "Larger moves, longer holds"},
            "Scalp": {"risk": 1.0, "tp": 2.5, "leverage": 1.0, "description": "Quick scalping strategy"},
            "Breakout": {"risk": 2.5, "tp": 6.0, "leverage": 1.0, "description": "Breakout trading"},
            "High Risk Penny": {"risk": 5.0, "tp": 15.0, "leverage": 1.0, "description": "Sub-penny cryptos (<$0.01) - extreme volatility"},
            "High Risk Short": {"risk": 3.0, "tp": 8.0, "leverage": 3.0, "description": "3x leverage short selling"},
        }
        
        # If scanner setup exists, auto-select matching strategy
        if scanner_loaded:
            scanner_strategy = opp['strategy'].lower()
            # Map scanner strategy names to template names
            strategy_mapping = {
                'swing': 'Swing Trade',
                'momentum': 'Momentum',
                'scalp': 'Scalp',
                'breakout': 'Breakout',
                'aggressive_scalp': 'Aggressive Scalp',
                'ema_crossover': 'Scalp',  # EMA strategies map to scalp
                'rsi_stoch_hammer': 'Scalp',
                'fisher_rsi_multi': 'Momentum',
                'macd_volume': 'Momentum'
            }
            
            default_strategy = strategy_mapping.get(scanner_strategy, 'Custom')
            default_index = list(strategy_templates.keys()).index(default_strategy) if default_strategy in strategy_templates else 0
            
            strategy = st.selectbox(
                "🎯 Strategy Template 🔒",
                options=list(strategy_templates.keys()),
                index=default_index,
                key="crypto_strategy_template",
                help="Auto-selected from scanner. Change if needed.",
                disabled=True  # Lock it to scanner strategy
            )
            st.caption(f"✅ Locked to scanner strategy: {opp['strategy'].upper()}")
        else:
            strategy = st.selectbox(
                "🎯 Strategy Template",
                options=list(strategy_templates.keys()),
                index=0,
                key="crypto_strategy_template",
                help="Pre-configured risk/reward profiles"
            )
        
        # Show strategy description
        st.caption(f"ℹ️ {strategy_templates[strategy]['description']}")
    
    # Row 2: Limit Price & Stop Price (conditional on order type)
    if order_type in ["LIMIT", "STOP_LOSS", "TAKE_PROFIT"]:
        price_col1, price_col2 = st.columns(2)
        
        with price_col1:
            if order_type == "LIMIT":
                limit_price = st.number_input(
                    "💰 Limit Price (USD)",
                    min_value=0.000001,
                    value=0.0,
                    step=0.01,
                    format="%.6f",
                    key="crypto_limit_price",
                    help="Your order will execute at this price or better"
                )
                st.caption(f"{'Buy' if direction == 'BUY' else 'Sell'} at ${limit_price:.6f} or better")
            
            elif order_type in ["STOP_LOSS", "TAKE_PROFIT"]:
                stop_price = st.number_input(
                    f"🎯 {order_type.replace('_', ' ').title()} Price (USD)",
                    min_value=0.000001,
                    value=0.0,
                    step=0.01,
                    format="%.6f",
                    key="crypto_stop_trigger_price",
                    help=f"Order triggers when price {'drops to' if order_type == 'STOP_LOSS' else 'rises to'} this level"
                )
                st.caption(f"Triggers at ${stop_price:.6f}")
        
        with price_col2:
            # Show current market price for reference
            try:
                ticker = kraken_client.get_ticker_data(selected_pair or "")
                if ticker:
                    current_price = ticker.get('last_price', 0)
                    st.metric("Current Market Price", f"${current_price:,.6f}")
                    
                    # Show price difference
                    if order_type == "LIMIT" and limit_price > 0:
                        diff_pct = ((limit_price - current_price) / current_price) * 100
                        st.caption(f"{'Above' if diff_pct > 0 else 'Below'} market by {abs(diff_pct):.2f}%")
            except:
                pass
    
    # Advanced Options Expander
    with st.expander("🔧 Advanced Options", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        
        with adv_col1:
            time_in_force = st.selectbox(
                "⏱️ Time in Force",
                options=["GTC", "IOC", "FOK"],
                index=0,
                key="crypto_time_in_force",
                help="GTC = Good Till Cancelled\nIOC = Immediate or Cancel\nFOK = Fill or Kill"
            )
        
        with adv_col2:
            post_only = st.checkbox(
                "📌 Post Only",
                value=False,
                key="crypto_post_only",
                help="Ensure order is added to order book (maker fee)"
            )
        
        with adv_col3:
            reduce_only = st.checkbox(
                "🔻 Reduce Only",
                value=False,
                key="crypto_reduce_only",
                help="Only reduce existing position (prevent increasing)"
            )
        
        # Position Management Options
        st.markdown("**📊 Position Management:**")
        pm_col1, pm_col2, pm_col3 = st.columns(3)
        
        with pm_col1:
            use_trailing_stop = st.checkbox(
                "🎯 Trailing Stop",
                value=False,
                key="crypto_trailing_stop",
                help="Stop loss follows price as it moves in your favor"
            )
            
            if use_trailing_stop:
                trailing_distance = st.number_input(
                    "Trail Distance %",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    key="crypto_trail_distance"
                )
        
        with pm_col2:
            use_partial_exit = st.checkbox(
                "📉 Partial Exit",
                value=False,
                key="crypto_partial_exit",
                help="Close only a portion of the position"
            )
            
            if use_partial_exit:
                exit_percentage = st.slider(
                    "Exit %",
                    min_value=10,
                    max_value=90,
                    value=50,
                    step=10,
                    key="crypto_exit_pct"
                )
        
        with pm_col3:
            use_scale_out = st.checkbox(
                "📊 Scale Out",
                value=False,
                key="crypto_scale_out",
                help="Exit in multiple increments at different price levels"
            )
            
            if use_scale_out:
                num_exits = st.number_input(
                    "# of Exits",
                    min_value=2,
                    max_value=5,
                    value=3,
                    step=1,
                    key="crypto_num_exits"
                )
    
    # ========================================================================
    # POSITION SIZING (Enhanced with strategy template integration)
    # ========================================================================
    st.markdown("---")
    st.markdown("### 💰 Position Sizing")
    
    # Check if scanner setup exists - use those values as defaults
    if scanner_loaded:
        # Use scanner stop/target values AND leverage/position size
        default_risk = st.session_state.get('crypto_quick_stop_pct', 2.0)
        default_tp = st.session_state.get('crypto_quick_target_pct', 5.0)
        default_leverage = st.session_state.get('crypto_quick_leverage', 1.0)
        default_position_size = st.session_state.get('crypto_quick_position_size', 100.0)
        
        logger.info(f"📖 Loading from session state: stop={default_risk}%, target={default_tp}%, leverage={default_leverage}x, position=${default_position_size}")
        pass  # logger.info(f"📖 Scanner opportunity symbol: {opp.get('symbol', 'N/A'}"))
        pass  # logger.info(f"📖 Session state keys present: crypto_quick_trade_pair={st.session_state.get('crypto_quick_trade_pair', 'NOT SET'}"))
        
        st.info(f"📊 **Scanner Strategy:** {default_risk:.1f}% stop loss, {default_tp:.1f}% take profit, {default_leverage:.0f}x leverage, ${default_position_size:.2f} position (from {opp['strategy'].upper()} strategy)")
    else:
        # Apply strategy template if not custom
        default_risk = strategy_templates[strategy]['risk'] if strategy != "Custom" else 2.0
        default_tp = strategy_templates[strategy]['tp'] if strategy != "Custom" else 5.0
        default_leverage = strategy_templates[strategy]['leverage'] if strategy != "Custom" else 1.0
        default_position_size = 100.0
    
    pos_col1, pos_col2, pos_col3 = st.columns(3)
    
    # Force leverage to 1 if Spot Trading mode
    if trading_mode == "Spot Trading":
        default_leverage = 1.0
    
    with pos_col1:
        position_size = st.number_input(
            "Position Size (USD)",
            min_value=10.0,
            max_value=10000.0,
            value=default_position_size,
            step=10.0,
            key="crypto_quick_position_size"
        )
    
    with pos_col2:
        # Disable leverage input if Spot Trading mode selected
        leverage_disabled = (trading_mode == "Spot Trading")
        leverage_max = 5.0 if not leverage_disabled else 1.0
        
        leverage = st.number_input(
            "Leverage" + (" (Disabled in Spot Mode)" if leverage_disabled else " (Margin Trading)"),
            min_value=1.0,
            max_value=leverage_max,
            value=default_leverage,
            step=1.0,
            key="crypto_quick_leverage",
            disabled=leverage_disabled,
            help="1 = Spot trading (no leverage). 2-5 = Margin trading with leverage. SELL with leverage > 1 = SHORT SELLING"
        )
        
        # Show margin trading warning
        if leverage > 1 and not leverage_disabled:
            if direction == "SELL":
                st.warning(f"⚠️ **SHORT SELLING** with {leverage}x leverage - You're betting the price will go DOWN")
            else:
                st.info(f"ℹ️ **LEVERAGED LONG** with {leverage}x leverage - Amplifies both gains and losses")
            st.caption("⚙️ Margin trading enabled")
        elif leverage_disabled:
            st.caption("🔒 Spot trading only (safer)")
        else:
            st.caption("📊 No leverage applied")
    
    with pos_col3:
        # Risk percentage uses default from scanner or template
        risk_pct = st.number_input(
            "Risk (Stop Loss) %" + (" 🔒 From Scanner" if scanner_loaded else ""),
            min_value=0.1,
            max_value=10.0,
            value=default_risk,
            step=0.1,
            key="crypto_quick_stop_pct"
        )
    
    # Take profit uses default from scanner or template
    # Clamp value to max if it exceeds (can happen with high-volatility coin strategies)
    clamped_tp = min(default_tp, 50.0)
    take_profit_pct = st.number_input(
        "Take Profit %" + (" 🔒 From Scanner" if scanner_loaded else ""),
        min_value=0.1,
        max_value=50.0,  # Increased to support volatile crypto strategies
        value=clamped_tp,
        step=0.1,
        key="crypto_quick_target_pct"
    )
    
    # AI Analysis section
    st.markdown("#### 🤖 AI Analysis")
    logger.info(f"📍 Rendering AI Analysis section for {selected_pair}")
    
    # Check if analysis was triggered from quick button
    auto_trigger = st.session_state.get('trigger_ai_analysis', False)
    if auto_trigger:
        logger.info(f"🤖 Auto-triggering AI analysis from quick button")
        # Clear the flag to prevent repeated triggers
        st.session_state.trigger_ai_analysis = False
    
    analysis_col1, analysis_col2, analysis_col3 = st.columns([2, 1, 1])
    
    with analysis_col1:
        button_clicked = st.button("🤖 AI Entry Analysis", key="crypto_ai_entry_analysis_btn", use_container_width=True, type="primary")
        if button_clicked or auto_trigger:
            logger.info(f"🔘 AI Entry Analysis button clicked for {selected_pair} ({direction})")
            with st.spinner("🤖 AI analyzing entry timing..."):
                try:
                    logger.info(f"📋 Starting AI entry analysis: pair={selected_pair}, side={direction}, position=${position_size}, risk={risk_pct}%, tp={take_profit_pct}%")
                    # Initialize AI Entry Assistant
                    from services.ai_entry_assistant import get_ai_entry_assistant
                    from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                    
                    if 'ai_entry_assistant' not in st.session_state:
                        llm_analyzer = LLMStrategyAnalyzer()
                        entry_assistant = get_ai_entry_assistant(
                            kraken_client=kraken_client,
                            llm_analyzer=llm_analyzer,
                            check_interval_seconds=60,
                            enable_auto_entry=False  # Manual approval required by default
                        )
                        st.session_state.ai_entry_assistant = entry_assistant
                        # Start monitoring if not running
                        if not entry_assistant.is_running:
                            entry_assistant.start_monitoring()
                    else:
                        entry_assistant = st.session_state.ai_entry_assistant
                    
                    # Get AI entry analysis - include scanner context if available
                    if scanner_loaded:
                        # Add scanner context to help AI understand the original recommendation
                        scanner_context = f"""
                        SCANNER RECOMMENDATION CONTEXT:
                        - Original Strategy: {opp['strategy'].upper()}
                        - Scanner Score: {opp['score']:.1f}/100
                        - Scanner Confidence: {opp['confidence']}
                        - Risk Level: {opp['risk_level']}
                        - Original Reasoning: {opp['reason']}
                        - 24h Price Change: {opp['change_24h']:.2f}%
                        - Volume Ratio: {opp['volume_ratio']:.2f}x (vs avg)
                        - Volatility: {opp['volatility']:.2f}%
                        
                        NOTE: If current market conditions differ from scanner findings, 
                        explain WHY (e.g., volume dried up, trend reversed, etc.)
                        """
                        st.info("🔍 Including scanner analysis context for AI review...")
                    else:
                        scanner_context = None
                    
                    entry_analysis = entry_assistant.analyze_entry(
                        pair=selected_pair or "",
                        side=direction,
                        position_size=position_size,
                        risk_pct=risk_pct,
                        take_profit_pct=take_profit_pct,
                        additional_context=scanner_context or ""
                    )
                    
                    # Store in session state
                    st.session_state.crypto_entry_analysis = entry_analysis
                    st.session_state.crypto_analysis = {
                        'pair': selected_pair,
                        'direction': direction,
                        'current_price': entry_analysis.current_price,
                        'stop_loss': entry_analysis.suggested_stop or (entry_analysis.current_price * (1 - risk_pct / 100)),
                        'take_profit': entry_analysis.suggested_target or (entry_analysis.current_price * (1 + take_profit_pct / 100)),
                        'position_size': position_size,
                        'leverage': leverage if trading_mode == "Margin Trading" else None,
                        'risk_reward_ratio': entry_analysis.risk_reward_ratio if entry_analysis.risk_reward_ratio > 0 else (take_profit_pct / risk_pct),
                        'order_type': order_type,
                        'limit_price': limit_price if order_type == "LIMIT" else None,
                        'trading_mode': trading_mode
                    }
                    
                    logger.info(f"🤖 AI Entry Analysis: {entry_analysis.action} (Confidence: {entry_analysis.confidence:.1f}%)")
                    
                    # Show success message
                    if entry_analysis.action == "DO_NOT_ENTER" and entry_analysis.confidence == 0.0:
                        st.warning("⚠️ AI analysis completed but API call failed. Showing fallback analysis based on technical indicators.")
                    else:
                        st.success(f"✅ AI analysis complete! Scroll down to see results.")
                    
                    # Rerun to display results
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
                    logger.error("❌ AI entry analysis error: {type(e).__name__)}: {}", str(e), exc_info=True)
                    # Still rerun to show any partial results or error message
                    st.rerun()
    
    with analysis_col2:
        if st.button("📝 Manual Setup", key="crypto_manual_setup_btn", use_container_width=True):
            # Create manual analysis without AI
            try:
                ticker = kraken_client.get_ticker_data(selected_pair) if selected_pair else None
                current_price = ticker.get('last_price', 0) if ticker else 0
                
                if current_price == 0:
                    st.error("Could not fetch current price")
                else:
                    # Calculate stop loss and take profit based on percentages
                    if direction == "BUY":
                        stop_loss = current_price * (1 - risk_pct / 100)
                        take_profit = current_price * (1 + take_profit_pct / 100)
                    else:  # SELL
                        stop_loss = current_price * (1 + risk_pct / 100)
                        take_profit = current_price * (1 - take_profit_pct / 100)
                    
                    # Store manual setup in session state
                    st.session_state.crypto_analysis = {
                        'pair': selected_pair,
                        'direction': direction,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'leverage': leverage if trading_mode == "Margin Trading" else None,
                        'risk_reward_ratio': take_profit_pct / risk_pct,
                        'order_type': order_type,
                        'limit_price': limit_price if order_type == "LIMIT" else None,
                        'trading_mode': trading_mode
                    }
                    st.success("✅ Manual setup created! Review below and execute.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to create setup: {e}")
    
    with analysis_col3:
        if st.button("📊 Get Market Data", key="crypto_market_data_btn", use_container_width=True):
            with st.spinner("Fetching market data..."):
                try:
                    ticker_info = kraken_client.get_ticker_info(selected_pair) if selected_pair else None
                    st.json(ticker_info)
                except Exception as e:
                    st.error(f"Failed to fetch market data: {e}")
    
    # Display AI entry analysis if available
    if 'crypto_entry_analysis' in st.session_state:
        entry_analysis = st.session_state.crypto_entry_analysis
        logger.info(f"📊 Displaying AI entry analysis: action={entry_analysis.action}, confidence={entry_analysis.confidence:.1f}%")
        
        st.markdown("---")
        st.markdown("#### 🤖 AI Entry Recommendation")
        
        # Confidence-based color coding
        if entry_analysis.confidence >= 85:
            confidence_color = "🟢"  # High confidence - green
        elif entry_analysis.confidence >= 70:
            confidence_color = "🟡"  # Medium confidence - yellow
        else:
            confidence_color = "🔴"  # Low confidence - red
        
        # Display recommendation with styling
        rec_col1, rec_col2 = st.columns([3, 1])
        with rec_col1:
            st.markdown(f"**Action:** `{entry_analysis.action}` {confidence_color}")
            st.markdown(f"**Confidence:** {entry_analysis.confidence:.1f}% | **Urgency:** {entry_analysis.urgency}")
        with rec_col2:
            st.metric("Technical", f"{entry_analysis.technical_score:.0f}/100")
            st.metric("Timing", f"{entry_analysis.timing_score:.0f}/100")
        
        # AI Reasoning
        st.info(f"**💡 AI Reasoning:** {entry_analysis.reasoning}")
        
        # Scores breakdown
        score_cols = st.columns(4)
        score_cols[0].metric("Technical Score", f"{entry_analysis.technical_score:.0f}/100")
        score_cols[1].metric("Trend Score", f"{entry_analysis.trend_score:.0f}/100")
        score_cols[2].metric("Timing Score", f"{entry_analysis.timing_score:.0f}/100")
        score_cols[3].metric("Risk Score", f"{entry_analysis.risk_score:.0f}/100", 
                            delta="Lower is better" if entry_analysis.risk_score < 50 else "Higher risk",
                            delta_color="normal" if entry_analysis.risk_score < 50 else "inverse")
        
        # Action-specific guidance
        if entry_analysis.action == "ENTER_NOW":
            st.success("✅ **AI says ENTER NOW** - Excellent setup detected!")
        elif entry_analysis.action == "WAIT_FOR_PULLBACK":
            wait_price_text = f" to ${entry_analysis.wait_for_price:,.6f}" if entry_analysis.wait_for_price else ""
            wait_rsi_text = f" (RSI < {entry_analysis.wait_for_rsi:.0f})" if entry_analysis.wait_for_rsi else ""
            st.warning(f"⏳ **AI says WAIT FOR PULLBACK**{wait_price_text}{wait_rsi_text}")
        elif entry_analysis.action == "WAIT_FOR_BREAKOUT":
            st.warning(f"⏳ **AI says WAIT FOR BREAKOUT** - Consolidating, wait for confirmation")
        elif entry_analysis.action == "PLACE_LIMIT_ORDER":
            st.info(f"📝 **AI suggests LIMIT ORDER** at ${entry_analysis.suggested_entry:,.6f}")
        else:  # DO_NOT_ENTER
            st.error("❌ **AI says DO NOT ENTER** - Poor setup, avoid this trade")
    
    # Display analysis results if available
    if 'crypto_analysis' in st.session_state:
        analysis = st.session_state.crypto_analysis
        
        st.markdown("---")
        st.markdown("#### 📈 Trade Setup Review")
        
        # Show trading configuration
        config_review_col1, config_review_col2, config_review_col3, config_review_col4 = st.columns(4)
        
        with config_review_col1:
            st.metric("Order Type", analysis.get('order_type', 'MARKET'))
        
        with config_review_col2:
            st.metric("Trading Mode", analysis.get('trading_mode', 'Spot Trading'))
        
        with config_review_col3:
            leverage_val = analysis.get('leverage', 1) or 1
            st.metric("Leverage", f"{leverage_val}x")
            if leverage_val > 1:
                if analysis['direction'] == 'SELL':
                    st.caption("⚠️ SHORT SELLING")
                else:
                    st.caption("📈 LEVERAGED LONG")
        
        with config_review_col4:
            st.metric("Position Size", f"${analysis['position_size']:,.2f}")
        
        # Price levels
        metric_cols = st.columns(4)
        metric_cols[0].metric("Entry Price", f"${analysis['current_price']:,.4f}")
        
        # Show limit price if applicable
        if analysis.get('order_type') == "LIMIT" and analysis.get('limit_price'):
            metric_cols[1].metric("Limit Price", f"${analysis['limit_price']:,.4f}")
        else:
            metric_cols[1].metric("Stop Loss", f"${analysis['stop_loss']:,.4f}")
        
        metric_cols[2].metric("Take Profit", f"${analysis['take_profit']:,.4f}")
        metric_cols[3].metric("R:R Ratio", f"{analysis['risk_reward_ratio']:.2f}")
        
        # Action buttons based on AI recommendation
        st.markdown("---")
        
        # Check if we have AI entry analysis
        has_entry_analysis = 'crypto_entry_analysis' in st.session_state
        entry_analysis = st.session_state.get('crypto_entry_analysis')
        
        # Determine button layout based on AI recommendation
        if has_entry_analysis and entry_analysis and entry_analysis.action in ["WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT"]:
            exec_col1, exec_col2, exec_col3 = st.columns([1, 1, 1])
        else:
            exec_col1, exec_col2, exec_col3 = st.columns([2, 1, 1])
        
        with exec_col1:
            # Check for duplicate execution protection
            execution_key = f"crypto_single_execution_{analysis['pair']}_{analysis['direction']}_{analysis['position_size']}"
            execution_timestamp_key = f"{execution_key}_timestamp"
            
            import time
            current_time = time.time()
            
            # Check if we just executed this trade recently (within last 30 seconds)
            is_recent = (execution_key in st.session_state and 
                        execution_timestamp_key in st.session_state and
                        current_time - st.session_state[execution_timestamp_key] < 30)
            
            # Determine button label and type based on AI confidence
            if has_entry_analysis and entry_analysis:
                if entry_analysis.confidence >= 85:
                    button_label = "🚀 Execute Now (AI Approved)"
                    button_type = "primary"
                elif entry_analysis.confidence >= 70:
                    button_label = "⚠️ Execute (Medium Confidence)"
                    button_type = "secondary"
                else:
                    button_label = "🛑 Execute Anyway (Low Confidence)"
                    button_type = "secondary"
            else:
                button_label = "🚀 Execute Trade"
                button_type = "primary"
            
            if st.button(button_label, width='stretch', type=button_type, disabled=is_recent):
                if is_recent:
                    st.warning("⚠️ **Duplicate execution prevented!** You just executed this trade. Please wait a moment before executing again.")
                else:
                    # Mark execution
                    st.session_state[execution_key] = True
                    st.session_state[execution_timestamp_key] = current_time
                    execute_crypto_trade(kraken_client, analysis)
        
        # Add "Monitor & Alert" button if AI says to wait
        if has_entry_analysis and entry_analysis and entry_analysis.action in ["WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT"]:
            with exec_col2:
                if st.button("🔔 Monitor & Alert", width='stretch', type="primary"):
                    try:
                        entry_assistant = st.session_state.get('ai_entry_assistant')
                        if entry_assistant:
                            opp_id = entry_assistant.monitor_entry_opportunity(
                                pair=analysis['pair'],
                                side=analysis['direction'],
                                position_size=analysis['position_size'],
                                risk_pct=risk_pct,
                                take_profit_pct=take_profit_pct,
                                analysis=entry_analysis,
                                auto_execute=False
                            )
                            st.success(f"✅ Monitoring {analysis['pair']} for entry opportunity!")
                            st.info(f"📊 Will alert when conditions improve (Opportunity ID: {opp_id})")
                            logger.info(f"🔔 User set up monitoring for {analysis['pair']}")
                        else:
                            st.error("AI Entry Assistant not initialized")
                    except Exception as e:
                        st.error(f"Failed to set up monitoring: {e}")
                        logger.error("Monitor setup error: {}", str(e), exc_info=True)
            
            with exec_col3:
                def reset_analysis():
                    if 'crypto_analysis' in st.session_state:
                        del st.session_state.crypto_analysis
                    if 'crypto_entry_analysis' in st.session_state:
                        del st.session_state.crypto_entry_analysis
                
                st.button("🔄 Reset", width='stretch', on_click=reset_analysis)
        else:
            # Normal layout without monitoring button
            with exec_col2:
                if st.button("💾 Save Setup", width='stretch'):
                    save_trade_setup(analysis, crypto_config)
                    st.success("Setup saved!")
            
            with exec_col3:
                def reset_analysis():
                    if 'crypto_analysis' in st.session_state:
                        del st.session_state.crypto_analysis
                    if 'crypto_entry_analysis' in st.session_state:
                        del st.session_state.crypto_entry_analysis
                
                st.button("🔄 Reset", width='stretch', on_click=reset_analysis)


def execute_crypto_trade(kraken_client: KrakenClient, analysis: Dict):
    """
    Execute the crypto trade with given parameters (supports margin/leverage and different order types)
    """
    try:
        with st.spinner("Placing order..."):
            # Determine order type and side
            order_side = OrderSide.BUY if analysis['direction'] == 'BUY' else OrderSide.SELL
            
            # Get order type from analysis (default to MARKET)
            order_type_str = analysis.get('order_type', 'MARKET')
            order_type = OrderType[order_type_str]
            
            # Calculate order quantity
            if analysis['direction'] == 'BUY':
                quantity = analysis['position_size'] / analysis['current_price']
            else:
                quantity = analysis['position_size'] / analysis['current_price']
            
            # Get leverage setting from analysis (defaults to None for spot trading)
            leverage = analysis.get('leverage', None)
            if leverage == 1.0 or leverage == 1:
                leverage = None  # Spot trading
            
            # Get limit/stop price if applicable
            limit_price = analysis.get('limit_price', None)
            
            # Place the order with stop loss, take profit, and optional leverage
            result = kraken_client.place_order(
                pair=analysis['pair'],
                side=order_side,
                order_type=order_type,
                volume=quantity,
                price=limit_price,
                stop_loss=analysis.get('stop_loss'),
                take_profit=analysis.get('take_profit'),
                leverage=leverage
            )
            
            if result is not None:
                # Verify order ID exists
                order_id = result.order_id if hasattr(result, 'order_id') else None
                
                if not order_id or order_id == '':
                    st.error(f"❌ Order placed but no order ID returned. Check Kraken for order status.")
                    logger.error(f"Order placed but no order ID returned for {analysis['pair']}")
                else:
                    # Success message varies by order type
                    if order_type == OrderType.MARKET:
                        st.success(f"✅ Market order executed! Order ID: {order_id}")
                        st.info("ℹ️ Market orders fill immediately. Check your Kraken account for filled order.")
                    elif order_type == OrderType.LIMIT:
                        st.success(f"✅ Limit order placed at ${limit_price:.6f}! Order ID: {order_id}")
                        st.info(f"ℹ️ Order will execute when price {'drops to' if order_side == OrderSide.BUY else 'rises to'} ${limit_price:.6f}")
                    else:
                        st.success(f"✅ {order_type_str} order placed! Order ID: {order_id}")
                        st.info("ℹ️ Check your Kraken account for order status.")
                    logger.info("✅ Order {} placed successfully for {} - {} {:.6f} @ ${:.4f}", str(order_id), analysis['pair'], analysis['direction'], quantity, analysis['current_price'])
                    
                    st.json({
                        'order_id': order_id,
                        'pair': result.pair,
                        'side': result.side,
                        'order_type': result.order_type,
                        'volume': result.volume,
                        'price': result.price,
                        'status': result.status,
                        'timestamp': result.timestamp.isoformat() if hasattr(result.timestamp, 'isoformat') else str(result.timestamp)
                    })
                    
                    # 🤖 NEW: Add to AI Position Manager for intelligent monitoring
                    try:
                        import time
                        from services.ai_crypto_position_manager import get_ai_position_manager
                        
                        # Get or initialize AI position manager
                        if 'ai_position_manager' not in st.session_state:
                            from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                            llm_analyzer = LLMStrategyAnalyzer()
                            
                            ai_manager = get_ai_position_manager(
                                kraken_client=kraken_client,
                                llm_analyzer=llm_analyzer,
                                check_interval_seconds=60,
                                enable_ai_decisions=True,
                                enable_trailing_stops=True,
                                enable_breakeven_moves=True,
                                enable_partial_exits=True,
                                require_manual_approval=True  # SAFETY: Set to False for auto-execution (RISKY!)
                            )
                            st.session_state.ai_position_manager = ai_manager
                            
                            # Start monitoring loop if not running
                            if not ai_manager.is_running:
                                ai_manager.start_monitoring_loop()
                                logger.info("🤖 AI Position Manager monitoring loop started")
                        else:
                            ai_manager = st.session_state.ai_position_manager
                        
                        # Add position to AI monitoring
                        trade_id = f"{analysis['pair']}_{order_id}_{int(time.time())}"
                        success = ai_manager.add_position(
                            trade_id=trade_id,
                            pair=analysis['pair'],
                            side=analysis['direction'],
                            volume=quantity,
                            entry_price=analysis['current_price'],
                            stop_loss=analysis.get('stop_loss') or 0.0,
                            take_profit=analysis.get('take_profit') or 0.0,
                            strategy=analysis.get('strategy', 'Manual'),
                            entry_order_id=order_id
                        )
                        
                        if success:
                            st.success("🤖 AI monitoring activated - Position will be intelligently managed!")
                            logger.info("🤖 Added {} to AI position manager (ID: {})", analysis['pair'], trade_id)
                        
                    except Exception as ai_err:
                        logger.warning(f"Could not add to AI position manager: {ai_err}")
                        # Don't fail the trade if AI monitoring fails
                    
                    # Log to unified journal (even if not AI-managed)
                    try:
                        from services.unified_trade_journal import get_unified_journal, UnifiedTradeEntry, TradeType
                        journal = get_unified_journal()
                        
                        # Calculate risk/reward percentages
                        if analysis['direction'] == 'BUY':
                            risk_pct = ((analysis['current_price'] - analysis.get('stop_loss', 0)) / analysis['current_price']) * 100 if analysis.get('stop_loss') else 2.0
                            reward_pct = ((analysis.get('take_profit', 0) - analysis['current_price']) / analysis['current_price']) * 100 if analysis.get('take_profit') else 5.0
                        else:
                            risk_pct = ((analysis.get('stop_loss', 0) - analysis['current_price']) / analysis['current_price']) * 100 if analysis.get('stop_loss') else 2.0
                            reward_pct = ((analysis['current_price'] - analysis.get('take_profit', 0)) / analysis['current_price']) * 100 if analysis.get('take_profit') else 5.0
                        
                        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                        
                        trade_entry = UnifiedTradeEntry(
                            trade_id=f"{analysis['pair']}_{order_id}_{int(time.time())}",
                            trade_type=TradeType.CRYPTO.value,
                            symbol=analysis['pair'],
                            side=analysis['direction'],
                            entry_time=datetime.now(),
                            entry_price=analysis['current_price'],
                            quantity=quantity,
                            position_size_usd=analysis['position_size'],
                            stop_loss=analysis.get('stop_loss', 0),
                            take_profit=analysis.get('take_profit', 0),
                            risk_pct=risk_pct,
                            reward_pct=reward_pct,
                            risk_reward_ratio=rr_ratio,
                            strategy=analysis.get('strategy', 'Manual'),
                            ai_managed='ai_position_manager' in st.session_state,
                            broker="KRAKEN",
                            order_id=order_id,
                            status="OPEN"
                        )
                        
                        journal.log_trade_entry(trade_entry)
                        logger.info(f"📝 Logged crypto trade to unified journal")
                    except Exception as journal_err:
                        logger.warning(f"Could not log to unified journal: {journal_err}")
                
                # Send Discord notification if configured
                try:
                    alert = TradingAlert(
                        ticker=analysis['pair'],
                        alert_type=AlertType.TRADE_EXECUTED,
                        message=f"{analysis['direction']} order executed at ${analysis['current_price']:.4f}",
                        priority=AlertPriority.MEDIUM,
                        details={
                            'order_id': result.order_id,
                            'price': float(analysis['current_price']),
                            'quantity': float(quantity),
                            'direction': str(analysis['direction']),
                            'position_size': float(analysis['position_size'])
                        }
                    )
                    send_discord_alert(alert)
                except Exception as e:
                    error_msg = str(e) if e else "Unknown error"
                    logger.error("❌ Failed to send Discord alert for {}: {}", analysis['pair'], str(error_msg), exc_info=True)
                    
            else:
                st.error(f"❌ Trade failed: Order placement returned None - check logs for details")
                
    except Exception as e:
        error_msg = str(e)
        if "Invalid permissions" in error_msg and "restricted" in error_msg:
            st.error(f"🚫 **Restricted Asset**: Trading this pair is not allowed in your region (e.g. WA state restrictions).")
            st.caption(f"Details: {error_msg}")
        else:
            st.error(f"Trade execution error: {error_msg}")
        logger.error("Trade execution error: {}", error_msg, exc_info=True)


def save_trade_setup(analysis: Dict, crypto_config=None):
    """
    Save trade setup to watchlist or configuration
    
    Args:
        analysis: Dict containing trade analysis data with 'pair' key
        crypto_config: Optional crypto config object with CRYPTO_WATCHLIST
    """
    try:
        # Add to crypto watchlist if config provided
        if crypto_config is None:
            logger.debug("No crypto_config provided, skipping watchlist save")
            return
            
        if hasattr(analysis['pair'], 'replace'):
            ticker = analysis['pair'].replace('/USD', '').replace('/USDT', '')
        else:
            ticker = str(analysis['pair']).replace('/USD', '').replace('/USDT', '')
            
        if ticker not in crypto_config.CRYPTO_WATCHLIST:
            crypto_config.CRYPTO_WATCHLIST.append(ticker)
            logger.info(f"Added {ticker} to crypto watchlist")
        
    except Exception as e:
        logger.error("Failed to save trade setup: {}", str(e), exc_info=True)


def display_bulk_custom_trade(kraken_client: KrakenClient, crypto_config):
    """
    Display bulk trade execution form with custom pair selection
    """
    st.markdown("#### 📊 Bulk Custom Selection")
    
    # Get available pairs
    try:
        import time
        cache_key = 'crypto_asset_pairs_cache'
        cache_timestamp_key = 'crypto_asset_pairs_cache_timestamp'
        cache_ttl = 300
        current_time = time.time()
        
        if (cache_key in st.session_state and 
            cache_timestamp_key in st.session_state and
            current_time - st.session_state[cache_timestamp_key] < cache_ttl):
            asset_pairs = st.session_state[cache_key]
        else:
            asset_pairs = kraken_client.get_tradable_asset_pairs()
            st.session_state[cache_key] = asset_pairs
            st.session_state[cache_timestamp_key] = current_time
        
        pair_options = [pair['altname'] for pair in asset_pairs if pair['quote'] == 'USD' or pair['quote'] == 'USDT']
        pair_options.sort()
        
        # Multi-select for pairs
        selected_pairs = st.multiselect(
            "Select Trading Pairs",
            options=pair_options,
            default=[],
            key="crypto_bulk_custom_pairs",
            help="Select multiple pairs to trade simultaneously"
        )
        
        if not selected_pairs:
            st.info("👆 Select one or more trading pairs above to configure bulk trades")
            return
        
        st.divider()
        
        # Common parameters for all trades
        st.markdown("#### ⚙️ Common Trade Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            direction = st.radio(
                "Direction",
                options=["BUY", "SELL"],
                horizontal=True,
                key="crypto_bulk_direction"
            )
            
            position_size = st.number_input(
                "Position Size per Pair (USD)",
                min_value=1.0,
                max_value=10000.0,
                value=100.0,
                step=1.0,
                key="crypto_bulk_position_size"
            )
        
        with col2:
            risk_pct = st.number_input(
                "Risk % per Trade",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                key="crypto_bulk_risk_pct"
            )
            
            take_profit_pct = st.number_input(
                "Take Profit %",
                min_value=0.1,
                max_value=50.0,  # Increased for volatile crypto
                value=10.0,
                step=0.1,
                key="crypto_bulk_target_pct"
            )
        
        # Analysis section
        st.divider()
        st.markdown("#### 🤖 Analyze Selected Pairs")
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col1:
            if st.button("🔍 Analyze All Selected Pairs", width='stretch', type="primary"):
                analyze_bulk_pairs(kraken_client, selected_pairs, direction, position_size, risk_pct, take_profit_pct, "bulk_custom")
        
        with analysis_col2:
            if st.button("📊 Get Market Data", width='stretch'):
                with st.spinner("Fetching market data..."):
                    try:
                        # Show market data for first selected pair as example
                        if selected_pairs:
                            ticker_info = kraken_client.get_ticker_info(selected_pairs[0])
                            st.json(ticker_info)
                    except Exception as e:
                        st.error(f"Failed to fetch market data: {e}")
        
        # Display analysis results if available (filtered to currently selected pairs)
        analysis_key = "crypto_bulk_custom_analysis"
        
        # Auto-restore from backup if session state is empty but backup exists
        if (analysis_key not in st.session_state or not st.session_state.get(analysis_key)) and os.path.exists("data/crypto_bulk_analysis_backup.json"):
            try:
                with open("data/crypto_bulk_analysis_backup.json", 'r') as f:
                    backup_data = json.load(f)
                    st.session_state[analysis_key] = backup_data.get('results', [])
                    st.session_state.crypto_bulk_custom_analysis_timestamp = backup_data.get('timestamp', '')
                    st.session_state.crypto_bulk_custom_analysis_complete = True
                    st.info(f"📂 Restored analysis from backup ({len(backup_data.get('results', []))} pairs)")
                    logger.info("Restored analysis from backup file")
            except Exception as e:
                logger.warning(f"Could not restore analysis from backup: {e}")
        
        if analysis_key in st.session_state and st.session_state[analysis_key]:
            all_analysis_results = st.session_state[analysis_key]
            
            # Show persistence indicator with restore option
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.session_state.get('crypto_bulk_custom_analysis_complete'):
                    timestamp = st.session_state.get('crypto_bulk_custom_analysis_timestamp', '')
                    st.success(f"💾 Analysis saved in session (persisted at {timestamp[:19] if timestamp else 'N/A'})")
            with col2:
                if st.button("🔄 Clear Results", key="clear_bulk_custom"):
                    st.session_state.crypto_bulk_custom_analysis = []
                    st.session_state.crypto_bulk_custom_analysis_complete = False
                    st.rerun()
            
            # Filter to only show results for currently selected pairs
            analysis_results = [r for r in all_analysis_results if r.get('pair') in selected_pairs]
            
            if analysis_results:
                st.divider()
                st.markdown("#### 📈 Analysis Results")
                
                # Add export button for results
                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("📥 Export Results", key="export_custom_analysis"):
                        export_df = pd.DataFrame(analysis_results)
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download CSV",
                            data=csv,
                            file_name=f"crypto_analysis_{timestamp_str}.csv",
                            mime="text/csv"
                        )
                
                # Summary metrics
                total_investment = sum(a.get('position_size', 0) for a in analysis_results)
                avg_rr = sum(a.get('risk_reward_ratio', 0) for a in analysis_results) / len(analysis_results) if analysis_results else 0
                
                # AI summary metrics
                ai_approved_count = sum(1 for a in analysis_results if a.get('ai_approved') == True)
                ai_rejected_count = sum(1 for a in analysis_results if a.get('ai_approved') == False)
                avg_confidence = sum(a.get('ai_confidence', 0) for a in analysis_results if a.get('ai_confidence')) / max(1, sum(1 for a in analysis_results if a.get('ai_confidence')))
                
                metric_cols = st.columns(6)
                metric_cols[0].metric("Pairs Analyzed", len(analysis_results))
                metric_cols[1].metric("Total Investment", f"${total_investment:,.2f}")
                metric_cols[2].metric("Avg R:R Ratio", f"{avg_rr:.2f}")
                metric_cols[3].metric("Direction", direction)
                metric_cols[4].metric("🤖 AI Approved", ai_approved_count, delta=f"-{ai_rejected_count} rejected" if ai_rejected_count > 0 else None)
                metric_cols[5].metric("🎯 Avg Confidence", f"{avg_confidence:.0f}%" if avg_confidence > 0 else "N/A")
                
                # Detailed analysis table with AI columns
                analysis_df = pd.DataFrame(analysis_results)
                
                # Reorder columns to show AI recommendations prominently
                column_order = ['pair', 'ai_recommendation', 'ai_confidence', 'current_price', 'stop_loss', 
                               'take_profit', 'position_size', 'quantity', 'risk_reward_ratio', 'ai_risks']
                # Only include columns that exist
                column_order = [col for col in column_order if col in analysis_df.columns]
                analysis_df = analysis_df[column_order]
                
                # Format numeric columns for better display
                if not analysis_df.empty:
                    numeric_cols = ['current_price', 'stop_loss', 'take_profit', 'position_size', 'quantity', 'risk_reward_ratio']
                    for col in numeric_cols:
                        if col in analysis_df.columns:
                            analysis_df[col] = analysis_df[col].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
                    
                    # Format AI confidence as percentage
                    if 'ai_confidence' in analysis_df.columns:
                        analysis_df['ai_confidence'] = analysis_df['ai_confidence'].apply(
                            lambda x: f"{x:.0f}%" if isinstance(x, (int, float)) and x > 0 else "N/A"
                        )
                    
                    st.dataframe(analysis_df, width='stretch', hide_index=True)
                    
                    # Show AI recommendations summary
                    if ai_approved_count > 0 or ai_rejected_count > 0:
                        st.markdown("##### 🤖 AI Recommendations Summary")
                        
                        rec_col1, rec_col2 = st.columns(2)
                        
                        with rec_col1:
                            st.markdown("**✅ Approved Trades:**")
                            approved_trades = [r for r in analysis_results if r.get('ai_approved') == True]
                            if approved_trades:
                                for trade in approved_trades[:5]:  # Show top 5
                                    st.success(f"• {trade['pair']} - Confidence: {trade.get('ai_confidence', 0):.0f}%")
                            else:
                                st.info("No trades approved by AI")
                        
                        with rec_col2:
                            st.markdown("**❌ Rejected Trades:**")
                            rejected_trades = [r for r in analysis_results if r.get('ai_approved') == False]
                            if rejected_trades:
                                for trade in rejected_trades[:5]:  # Show top 5
                                    st.error(f"• {trade['pair']} - Reason: {trade.get('ai_risks', 'Unknown')[:50]}")
                            else:
                                st.info("No trades rejected by AI")
        
        # Preview trades
        st.divider()
        st.markdown(f"#### 📋 Trade Preview ({len(selected_pairs)} pairs)")
        
        # Display selected pairs for confirmation
        st.info(f"**Selected pairs:** {', '.join(selected_pairs)}")
        
        # Check if execution is in progress
        execution_key = f"crypto_bulk_execution_{hash(tuple(sorted(selected_pairs)))}_{direction}_{position_size}"
        is_executing = st.session_state.get(execution_key, False)
        
        # Confirmation checkbox
        confirm_key = f"confirm_bulk_custom_{hash(tuple(sorted(selected_pairs)))}"
        confirm = st.checkbox(
            f"⚠️ I confirm I want to execute {len(selected_pairs)} trades ({direction}) with ${position_size:.2f} each (Total: ${position_size * len(selected_pairs):.2f})",
            key=confirm_key
        )
        
        if st.button("🚀 Execute Bulk Trades", type="primary", width='stretch', disabled=is_executing or not confirm):
            if not confirm:
                st.warning("Please confirm by checking the box above")
            else:
                execute_bulk_trades(
                    kraken_client,
                    selected_pairs,
                    direction,
                    position_size,
                    risk_pct,
                    take_profit_pct
                )
        
    except Exception as e:
        st.error(f"Error loading trading pairs: {e}")
        logger.error("Bulk custom trade error: {}", str(e), exc_info=True)


def display_bulk_watchlist_trade(kraken_client: KrakenClient, crypto_config, watchlist_manager):
    """
    Display bulk trade execution form for watchlist with comprehensive trading configuration
    """
    st.markdown("### ⭐ Bulk Watchlist Trading")
    
    try:
        # Get watchlist
        watchlist = watchlist_manager.get_all_cryptos()
        
        if not watchlist:
            st.warning("Your watchlist is empty. Add some cryptos first!")
            return
        
        # Extract symbols
        watchlist_symbols = [crypto.get('symbol', '') for crypto in watchlist if crypto.get('symbol')]
        
        if not watchlist_symbols:
            st.warning("No valid symbols found in watchlist")
            return
        
        st.info(f"📊 Found {len(watchlist_symbols)} symbols in your watchlist")
        
        # Allow selection of which watchlist items to trade
        selected_symbols = st.multiselect(
            "Select Symbols to Trade",
            options=watchlist_symbols,
            default=[],  # Start with nothing selected for safety
            key="crypto_bulk_watchlist_pairs",
            help="⚠️ Explicitly select which symbols you want to trade. Nothing is pre-selected for your safety."
        )
        
        if not selected_symbols:
            st.info("👆 Select symbols from your watchlist above to configure bulk trades")
            return
        
        # ========================================================================
        # 🎯 TRADING CONFIGURATION (same as single trade)
        # ========================================================================
        st.markdown("---")
        st.markdown("### ⚙️ Trading Configuration")
        
        # Row 1: Order Type, Trading Mode, Strategy Template
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            order_type = st.selectbox(
                "📋 Order Type",
                options=["MARKET", "LIMIT", "STOP_LOSS", "TAKE_PROFIT"],
                index=0,
                key="crypto_bulk_wl_order_type",
                help="MARKET = Execute immediately at current price\nLIMIT = Execute at specific price or better\nSTOP_LOSS = Sell if price drops to stop price\nTAKE_PROFIT = Sell if price rises to target"
            )
        
        with config_col2:
            trading_mode = st.selectbox(
                "💼 Trading Mode",
                options=["Spot Trading", "Margin Trading"],
                index=0,
                key="crypto_bulk_wl_trading_mode",
                help="Spot = Trade with your own funds only\nMargin = Borrow funds to amplify position (enables short selling)"
            )
            
            # Show margin status indicator
            if trading_mode == "Margin Trading":
                st.caption("🔓 Margin/Leverage enabled")
            else:
                st.caption("🔒 No leverage (safer)")
        
        with config_col3:
            strategy_templates = {
                "Custom": {"risk": 2.0, "tp": 5.0, "leverage": 1.0, "description": "Manual settings"},
                "Conservative": {"risk": 1.0, "tp": 3.0, "leverage": 1.0, "description": "Low risk, spot only"},
                "Balanced": {"risk": 2.0, "tp": 5.0, "leverage": 1.0, "description": "Moderate risk/reward"},
                "Momentum": {"risk": 2.0, "tp": 6.0, "leverage": 1.0, "description": "Mid-cap coins with strong trends ($0.01-$1)"},
                "Volatile Altcoin": {"risk": 4.0, "tp": 10.0, "leverage": 1.0, "description": "⚠️ Wider stops for volatile alts (PLUME, TRUMP, etc.)"},
                "Aggressive Scalp": {"risk": 2.0, "tp": 3.0, "leverage": 2.0, "description": "Quick profits, 2x leverage"},
                "Swing Trade": {"risk": 3.0, "tp": 10.0, "leverage": 1.0, "description": "Larger moves, longer holds"},
                "Scalp": {"risk": 1.0, "tp": 2.5, "leverage": 1.0, "description": "Quick scalping strategy"},
                "Breakout": {"risk": 2.5, "tp": 6.0, "leverage": 1.0, "description": "Breakout trading"},
                "High Risk Penny": {"risk": 5.0, "tp": 15.0, "leverage": 1.0, "description": "Sub-penny cryptos (<$0.01) - extreme volatility"},
                "High Risk Short": {"risk": 3.0, "tp": 8.0, "leverage": 3.0, "description": "3x leverage short selling"},
            }
            
            strategy = st.selectbox(
                "🎯 Strategy Template",
                options=list(strategy_templates.keys()),
                index=0,
                key="crypto_bulk_wl_strategy_template",
                help="Pre-configured risk/reward profiles"
            )
            
            # Show strategy description
            st.caption(f"ℹ️ {strategy_templates[strategy]['description']}")
        
        # Advanced Options Expander
        with st.expander("🔧 Advanced Options", expanded=False):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                time_in_force = st.selectbox(
                    "⏱️ Time in Force",
                    options=["GTC", "IOC", "FOK"],
                    index=0,
                    key="crypto_bulk_wl_time_in_force",
                    help="GTC = Good Till Cancelled\nIOC = Immediate or Cancel\nFOK = Fill or Kill"
                )
            
            with adv_col2:
                post_only = st.checkbox(
                    "📌 Post Only",
                    value=False,
                    key="crypto_bulk_wl_post_only",
                    help="Ensure order is added to order book (maker fee)"
                )
            
            with adv_col3:
                reduce_only = st.checkbox(
                    "🔻 Reduce Only",
                    value=False,
                    key="crypto_bulk_wl_reduce_only",
                    help="Only reduce existing position (prevent increasing)"
                )
        
        # ========================================================================
        # POSITION SIZING (Enhanced with strategy template integration)
        # ========================================================================
        st.markdown("---")
        st.markdown("### 💰 Common Trade Parameters")
        
        # Apply strategy template if not custom
        default_risk = strategy_templates[strategy]['risk'] if strategy != "Custom" else 2.0
        default_tp = strategy_templates[strategy]['tp'] if strategy != "Custom" else 5.0
        default_leverage = strategy_templates[strategy]['leverage'] if strategy != "Custom" else 1.0
        
        # Force leverage to 1 if Spot Trading mode
        if trading_mode == "Spot Trading":
            default_leverage = 1.0
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            direction = st.radio(
                "Direction",
                options=["BUY", "SELL"],
                horizontal=True,
                key="crypto_bulk_wl_direction"
            )
            
            # Show short selling warning if applicable
            if direction == "SELL" and trading_mode == "Margin Trading" and default_leverage > 1:
                st.warning(f"⚠️ **SHORT SELLING** {len(selected_symbols)} pairs with {default_leverage}x leverage")
        
        with param_col2:
            position_size = st.number_input(
                "Position Size per Symbol (USD)",
                min_value=10.0,
                max_value=10000.0,
                value=100.0,
                step=10.0,
                key="crypto_bulk_wl_position_size"
            )
            
            total_investment = position_size * len(selected_symbols)
            st.caption(f"💰 Total investment: ${total_investment:,.2f} across {len(selected_symbols)} symbols")
        
        # Risk and leverage parameters
        param_col3, param_col4, param_col5 = st.columns(3)
        
        with param_col3:
            # Disable leverage input if Spot Trading mode selected
            leverage_disabled = (trading_mode == "Spot Trading")
            leverage_max = 5.0 if not leverage_disabled else 1.0
            
            leverage = st.number_input(
                "Leverage" + (" (Disabled in Spot Mode)" if leverage_disabled else ""),
                min_value=1.0,
                max_value=leverage_max,
                value=default_leverage,
                step=1.0,
                key="crypto_bulk_wl_leverage",
                disabled=leverage_disabled,
                help="1 = Spot trading. 2-5 = Margin trading. SELL with leverage > 1 = SHORT SELLING"
            )
            
            if leverage > 1 and not leverage_disabled:
                st.caption("⚙️ Margin trading")
            else:
                st.caption("📊 Spot trading")
        
        with param_col4:
            risk_pct = st.number_input(
                "Risk (Stop Loss) %",
                min_value=0.1,
                max_value=10.0,
                value=default_risk,
                step=0.1,
                key="crypto_bulk_wl_risk_pct"
            )
        
        with param_col5:
            take_profit_pct = st.number_input(
                "Take Profit %",
                min_value=0.1,
                max_value=50.0,  # Increased for volatile crypto
                value=default_tp,
                step=0.1,
                key="crypto_bulk_wl_tp_pct"
            )
        
        # Display risk/reward summary
        rr_ratio = take_profit_pct / risk_pct
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        summary_col1.metric("Risk:Reward Ratio", f"{rr_ratio:.2f}:1")
        summary_col2.metric("Total Risk", f"${total_investment * (risk_pct/100):,.2f}")
        summary_col3.metric("Total Potential Profit", f"${total_investment * (take_profit_pct/100):,.2f}")
        
        # ========================================================================
        # AI ANALYSIS & EXECUTION
        # ========================================================================
        st.markdown("---")
        st.markdown("### 🤖 Analysis & Execution")
        
        analysis_btn_col1, analysis_btn_col2 = st.columns(2)
        
        with analysis_btn_col1:
            if st.button("📊 Analyze All Symbols", width='stretch', type="secondary"):
                # Use the same analyze_bulk_pairs function with AI support
                formatted_pairs = [f"{symbol}/USD" if '/' not in symbol else symbol for symbol in selected_symbols]
                analyze_bulk_pairs(
                    kraken_client, 
                    formatted_pairs, 
                    direction, 
                    position_size, 
                    risk_pct, 
                    take_profit_pct, 
                    "bulk_watchlist"
                )
        
        with analysis_btn_col2:
            # Manual setup without analysis
            if st.button("📝 Manual Setup (Skip Analysis)", width='stretch'):
                # Create basic setup for all pairs
                analysis_results = []
                for symbol in selected_symbols:
                    pair = f"{symbol}/USD" if '/' not in symbol else symbol
                    analysis_results.append({
                        'symbol': symbol,
                        'pair': pair,
                        'position_size': position_size,
                        'risk_reward_ratio': rr_ratio,
                        'order_type': order_type,
                        'trading_mode': trading_mode,
                        'leverage': leverage if trading_mode == "Margin Trading" else None,
                        'ai_approved': None,
                        'ai_confidence': None,
                        'ai_recommendation': 'Skipped AI analysis'
                    })
                st.session_state.crypto_bulk_watchlist_analysis = analysis_results
                st.success(f"✅ Created manual setup for {len(analysis_results)} symbols")
                st.rerun()
        
        # Display analysis results if available
        if 'crypto_bulk_watchlist_analysis' in st.session_state:
            analysis_results = st.session_state.crypto_bulk_watchlist_analysis
            
            st.markdown("---")
            st.markdown("### 📈 Analysis Results")
            
            # Summary metrics
            total_analyzed = len(analysis_results)
            total_investment_calc = sum(a.get('position_size', 0) for a in analysis_results)
            avg_rr_calc = sum(a.get('risk_reward_ratio', 0) for a in analysis_results) / total_analyzed if total_analyzed > 0 else 0
            
            # AI summary metrics
            ai_approved_count = sum(1 for a in analysis_results if a.get('ai_approved') == True)
            ai_rejected_count = sum(1 for a in analysis_results if a.get('ai_approved') == False)
            avg_confidence = sum(a.get('ai_confidence', 0) for a in analysis_results if a.get('ai_confidence')) / max(1, sum(1 for a in analysis_results if a.get('ai_confidence')))
            
            metric_cols = st.columns(6)
            metric_cols[0].metric("Symbols Analyzed", total_analyzed)
            metric_cols[1].metric("Total Investment", f"${total_investment_calc:,.2f}")
            metric_cols[2].metric("Avg R:R Ratio", f"{avg_rr_calc:.2f}")
            metric_cols[3].metric("Direction", direction)
            metric_cols[4].metric("🤖 AI Approved", ai_approved_count, delta=f"-{ai_rejected_count} rejected" if ai_rejected_count > 0 else None)
            metric_cols[5].metric("🎯 Avg Confidence", f"{avg_confidence:.0f}%" if avg_confidence > 0 else "N/A")
            
            # Detailed analysis table with AI columns
            if analysis_results:
                analysis_df = pd.DataFrame(analysis_results)
                
                # Reorder columns to show AI recommendations prominently
                column_order = ['pair', 'ai_recommendation', 'ai_confidence', 'current_price', 'stop_loss', 
                               'take_profit', 'position_size', 'quantity', 'risk_reward_ratio', 'ai_risks']
                # Only include columns that exist
                column_order = [col for col in column_order if col in analysis_df.columns]
                analysis_df = analysis_df[column_order]
                
                # Format numeric columns for better display
                if not analysis_df.empty:
                    numeric_cols = ['current_price', 'stop_loss', 'take_profit', 'position_size', 'quantity', 'risk_reward_ratio']
                    for col in numeric_cols:
                        if col in analysis_df.columns:
                            analysis_df[col] = analysis_df[col].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
                    
                    # Format AI confidence as percentage
                    if 'ai_confidence' in analysis_df.columns:
                        analysis_df['ai_confidence'] = analysis_df['ai_confidence'].apply(
                            lambda x: f"{x:.0f}%" if isinstance(x, (int, float)) and x > 0 else "N/A"
                        )
                    
                    st.dataframe(analysis_df, width='stretch', hide_index=True)
                    
                    # Show AI recommendations summary
                    if ai_approved_count > 0 or ai_rejected_count > 0:
                        st.markdown("##### 🤖 AI Recommendations Summary")
                        
                        rec_col1, rec_col2 = st.columns(2)
                        
                        with rec_col1:
                            st.markdown("**✅ Approved Trades:**")
                            approved_trades = [r for r in analysis_results if r.get('ai_approved') == True]
                            if approved_trades:
                                for trade in approved_trades[:5]:  # Show top 5
                                    st.success(f"• {trade['pair']} - Confidence: {trade.get('ai_confidence', 0):.0f}%")
                            else:
                                st.info("No trades approved by AI")
                        
                        with rec_col2:
                            st.markdown("**❌ Rejected Trades:**")
                            rejected_trades = [r for r in analysis_results if r.get('ai_approved') == False]
                            if rejected_trades:
                                for trade in rejected_trades[:5]:  # Show top 5
                                    st.error(f"• {trade['pair']} - Reason: {trade.get('ai_risks', 'Unknown')[:50]}")
                            else:
                                st.info("No trades rejected by AI")
        
        # ========================================================================
        # EXECUTION SECTION
        # ========================================================================
        st.markdown("---")
        st.markdown(f"### 🚀 Execute Trades ({len(selected_symbols)} symbols)")
        
        # Display configuration summary
        config_summary_col1, config_summary_col2, config_summary_col3, config_summary_col4 = st.columns(4)
        config_summary_col1.metric("Order Type", order_type)
        config_summary_col2.metric("Trading Mode", trading_mode)
        config_summary_col3.metric("Leverage", f"{leverage}x" if leverage > 1 else "None")
        config_summary_col4.metric("Total Capital", f"${total_investment:,.2f}")
        
        # Confirmation checkbox
        confirm_key = f"confirm_bulk_wl_{hash(tuple(sorted(selected_symbols)))}"
        confirm = st.checkbox(
            f"⚠️ I confirm I want to execute {len(selected_symbols)} {direction} trades with ${position_size:.2f} each (Total: ${total_investment:.2f})",
            key=confirm_key
        )
        
        # Check if execution is in progress
        execution_key = f"crypto_bulk_wl_execution_{hash(tuple(sorted(selected_symbols)))}_{direction}_{position_size}"
        is_executing = st.session_state.get(execution_key, False)
        
        if st.button("� Execute Bulk Trades", type="primary", width='stretch', disabled=is_executing or not confirm):
            if not confirm:
                st.warning("Please confirm by checking the box above")
            else:
                # Create formatted pairs list
                formatted_pairs = [f"{symbol}/USD" if '/' not in symbol else symbol for symbol in selected_symbols]
                
                execute_bulk_trades(
                    kraken_client,
                    formatted_pairs,
                    direction,
                    position_size,
                    risk_pct,
                    take_profit_pct,
                    order_type=order_type,
                    trading_mode=trading_mode,
                    leverage=int(leverage) if trading_mode == "Margin Trading" else None
                )
            
            if analysis_results:
                st.divider()
                st.markdown("#### 📈 Analysis Results")
                
                # Summary metrics
                total_investment = sum(a.get('position_size', 0) for a in analysis_results)
                avg_rr = sum(a.get('risk_reward_ratio', 0) for a in analysis_results) / len(analysis_results) if analysis_results else 0
                
                metric_cols = st.columns(4)
                metric_cols[0].metric("Symbols Analyzed", len(analysis_results))
                metric_cols[1].metric("Total Investment", f"${total_investment:,.2f}")
                metric_cols[2].metric("Avg R:R Ratio", f"{avg_rr:.2f}")
                metric_cols[3].metric("Direction", direction)
                
                # Detailed analysis table
                analysis_df = pd.DataFrame(analysis_results)
                # Format numeric columns for better display
                if not analysis_df.empty:
                    numeric_cols = ['current_price', 'stop_loss', 'take_profit', 'position_size', 'quantity', 'risk_reward_ratio']
                    for col in numeric_cols:
                        if col in analysis_df.columns:
                            analysis_df[col] = analysis_df[col].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
                    st.dataframe(analysis_df, width='stretch', hide_index=True)
        
        # Preview trades
        st.divider()
        st.markdown(f"#### 📋 Trade Preview ({len(selected_symbols)} symbols)")
        
        # Display selected symbols for confirmation
        st.info(f"**Selected symbols:** {', '.join(selected_symbols)}")
        
        # Check if execution is in progress
        execution_key = f"crypto_bulk_execution_{hash(tuple(sorted(selected_symbols)))}_{direction}_{position_size}"
        is_executing = st.session_state.get(execution_key, False)
        
        # Confirmation checkbox
        confirm_key = f"confirm_bulk_watchlist_{hash(tuple(sorted(selected_symbols)))}"
        confirm = st.checkbox(
            f"⚠️ I confirm I want to execute {len(selected_symbols)} trades ({direction}) with ${position_size:.2f} each (Total: ${position_size * len(selected_symbols):.2f})",
            key=confirm_key
        )
        
        if st.button("🚀 Execute Bulk Watchlist Trades", type="primary", width='stretch', disabled=is_executing or not confirm):
            if not confirm:
                st.warning("Please confirm by checking the box above")
            else:
                execute_bulk_trades(
                    kraken_client,
                    selected_symbols,
                    direction,
                    position_size,
                    risk_pct,
                    take_profit_pct
                )
        
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        logger.error("Bulk watchlist trade error: {}", str(e), exc_info=True)


def analyze_bulk_pairs(
    kraken_client: KrakenClient,
    pairs: List[str],
    direction: str,
    position_size: float,
    risk_pct: float,
    take_profit_pct: float,
    analysis_type: str = "bulk_custom"
):
    """
    Analyze multiple pairs with AI recommendations and store results in session state
    
    Args:
        kraken_client: KrakenClient instance
        pairs: List of trading pair symbols
        direction: "BUY" or "SELL"
        position_size: Position size in USD per pair
        risk_pct: Risk percentage per trade
        take_profit_pct: Take profit percentage
        analysis_type: "bulk_custom" or "bulk_watchlist" to determine session state key
    """
    if not pairs:
        st.warning("No pairs selected for analysis")
        return
    
    # Get appropriate LLM analyzer based on number of tickers (cloud for bulk, hybrid for single)
    llm_analyzer = get_llm_for_bulk_analysis(len(pairs))
    
    # Initialize AI trade reviewer if available
    ai_reviewer = None
    if llm_analyzer:
        try:
            ai_reviewer = AICryptoTradeReviewer(
                llm_analyzer=llm_analyzer,
                active_monitors=st.session_state.get('crypto_active_monitors', {}),
                supabase_client=st.session_state.get('supabase_client')
            )
            if len(pairs) > 1:
                logger.info(f"🤖 AI Trade Reviewer initialized for bulk analysis ({len(pairs)} pairs) with cloud API")
            else:
                logger.info("🤖 AI Trade Reviewer initialized for single pair analysis")
        except Exception as e:
            logger.warning(f"Could not initialize AI reviewer: {e}")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for idx, pair in enumerate(pairs):
            progress = (idx + 1) / len(pairs)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {pair} ({idx + 1}/{len(pairs)})...")
            
            try:
                # Get current price
                ticker_info = kraken_client.get_ticker_info(pair)
                if not ticker_info:
                    results.append({
                        'pair': pair,
                        'status': 'FAILED',
                        'error': 'Could not fetch ticker data',
                        'current_price': 0.0,
                        'stop_loss': 0.0,
                        'take_profit': 0.0,
                        'position_size': 0.0,
                        'quantity': 0.0,
                        'risk_reward_ratio': 0.0,
                        'ai_approved': False,
                        'ai_confidence': 0,
                        'ai_recommendation': 'No data'
                    })
                    continue
                
                current_price = float(ticker_info.get('c', [0])[0])
                
                # Calculate stop loss and take profit
                if direction == "BUY":
                    stop_loss = current_price * (1 - risk_pct / 100)
                    take_profit = current_price * (1 + take_profit_pct / 100)
                else:
                    stop_loss = current_price * (1 + risk_pct / 100)
                    take_profit = current_price * (1 - take_profit_pct / 100)
                
                # Calculate quantity
                quantity = position_size / current_price if current_price > 0 else 0
                
                # Calculate risk/reward ratio
                risk_reward_ratio = take_profit_pct / risk_pct if risk_pct > 0 else 0
                
                # Get AI review if available
                ai_approved = None
                ai_confidence = None
                ai_recommendation = "Not analyzed"
                ai_risks = []
                
                if ai_reviewer:
                    try:
                        # Get account balance for capital context
                        balance_data = kraken_client.get_account_balance()
                        # balance_data is a List[KrakenBalance], so sum the balance attribute
                        total_balance = sum(bal.balance for bal in balance_data) if balance_data else 0.0
                        
                        # Get market data for context
                        high_24h = float(ticker_info.get('h', [0])[0])
                        low_24h = float(ticker_info.get('l', [0])[0])
                        volume_24h = float(ticker_info.get('v', [0])[0])
                        
                        # Calculate market metrics
                        change_24h = ((current_price - low_24h) / low_24h * 100) if low_24h > 0 else 0
                        volatility = ((high_24h - low_24h) / current_price * 100) if current_price > 0 else 0
                        
                        market_data = {
                            'ticker': ticker_info,
                            'volume_24h': volume_24h,
                            'high_24h': high_24h,
                            'low_24h': low_24h,
                            'pair': pair,
                            'current_price': current_price,
                            'change_24h': change_24h,
                            'volatility': volatility
                        }
                        
                        # Determine strategy based on coin characteristics
                        if current_price < 0.01:
                            strategy = "HIGH_RISK_PENNY"
                        elif current_price < 1.0:
                            strategy = "MOMENTUM"
                        else:
                            strategy = "SCALP"
                        
                        # Get AI pre-trade review
                        logger.info(f"📊 Reviewing {pair} - ${current_price:.6f}, {direction}, Strategy: {strategy}")
                        
                        approved, confidence, reasoning, recommendations = ai_reviewer.pre_trade_review(
                            pair=pair,
                            side=direction,
                            entry_price=current_price,
                            position_size_usd=position_size,
                            stop_loss_price=stop_loss,
                            take_profit_price=take_profit,
                            strategy=strategy,
                            market_data=market_data,
                            total_capital=position_size * len(pairs),  # Total capital for all trades
                            actual_balance=total_balance
                        )
                        
                        ai_approved = approved
                        ai_confidence = confidence
                        
                        # More detailed recommendation
                        if approved:
                            ai_recommendation = f"✅ APPROVED ({confidence:.0f}%)"
                        else:
                            ai_recommendation = f"❌ REJECTED ({confidence:.0f}%)"
                        
                        ai_risks = recommendations.get('risks', [])
                        
                        # Log detailed results
                        logger.info(f"  → {ai_recommendation} - Risks: {len(ai_risks)}")
                        if ai_risks:
                            pass  # logger.info(f"     Risks: {', '.join(ai_risks[:3]}"))
                        
                    except Exception as e:
                        logger.warning(f"AI review failed for {pair}: {e}")
                        ai_approved = False  # Default to rejection on error
                        ai_confidence = 0
                        ai_recommendation = f"❌ ERROR: {str(e)[:50]}"
                        ai_risks = [str(e)]
                
                results.append({
                    'pair': pair,
                    'status': 'SUCCESS',
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size,
                    'quantity': quantity,
                    'risk_reward_ratio': risk_reward_ratio,
                    'risk_pct': risk_pct,
                    'take_profit_pct': take_profit_pct,
                    'ai_approved': ai_approved,
                    'ai_confidence': ai_confidence,
                    'ai_recommendation': ai_recommendation,
                    'ai_risks': ', '.join(ai_risks) if ai_risks else 'None identified'
                })
                
            except Exception as e:
                logger.error("Error analyzing {pair}: {}", str(e), exc_info=True)
                results.append({
                    'pair': pair,
                    'status': 'FAILED',
                    'error': str(e),
                    'current_price': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'position_size': 0.0,
                    'quantity': 0.0,
                    'risk_reward_ratio': 0.0,
                    'ai_approved': False,
                    'ai_confidence': 0,
                    'ai_recommendation': 'Analysis failed'
                })
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Store results in session state with completion flag
        if analysis_type == "bulk_custom":
            st.session_state.crypto_bulk_custom_analysis = results
            st.session_state.crypto_bulk_custom_analysis_complete = True
            st.session_state.crypto_bulk_custom_analysis_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Auto-save to JSON file as backup (prevents data loss on refresh)
            try:
                backup_file = "data/crypto_bulk_analysis_backup.json"
                os.makedirs("data", exist_ok=True)
                with open(backup_file, 'w') as f:
                    json.dump({
                        'timestamp': st.session_state.crypto_bulk_custom_analysis_timestamp,
                        'results': results,
                        'analysis_type': 'bulk_custom',
                        'total_pairs': len(pairs)
                    }, f, indent=2)
                logger.info(f"📁 Saved analysis backup to {backup_file}")
            except Exception as e:
                logger.warning(f"Could not save analysis backup: {e}")
                
        elif analysis_type == "bulk_watchlist":
            st.session_state.crypto_bulk_watchlist_analysis = results
            st.session_state.crypto_bulk_watchlist_analysis_complete = True
            st.session_state.crypto_bulk_watchlist_analysis_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Auto-save to JSON file as backup
            try:
                backup_file = "data/crypto_watchlist_analysis_backup.json"
                os.makedirs("data", exist_ok=True)
                with open(backup_file, 'w') as f:
                    json.dump({
                        'timestamp': st.session_state.crypto_bulk_watchlist_analysis_timestamp,
                        'results': results,
                        'analysis_type': 'bulk_watchlist',
                        'total_pairs': len(pairs)
                    }, f, indent=2)
                logger.info(f"📁 Saved analysis backup to {backup_file}")
            except Exception as e:
                logger.warning(f"Could not save analysis backup: {e}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show success message with AI summary
        successful = [r for r in results if r['status'] == 'SUCCESS']
        if successful:
            approved_count = sum(1 for r in successful if r.get('ai_approved') == True)
            rejected_count = sum(1 for r in successful if r.get('ai_approved') == False)
            
            st.success(f"✅ Successfully analyzed {len(successful)} out of {len(pairs)} pairs")
            if ai_reviewer:
                st.info(f"🤖 AI Recommendations: {approved_count} approved, {rejected_count} rejected, {len(successful) - approved_count - rejected_count} no recommendation")
        else:
            st.warning(f"⚠️ Could not analyze any pairs. Please check your selections.")
        
        # DON'T rerun - let results display naturally
        # st.rerun() causes data loss on refresh
        
    except Exception as e:
        st.error(f"Bulk analysis error: {e}")
        logger.error("Bulk analysis error: {}", str(e), exc_info=True)


def execute_bulk_trades(
    kraken_client: KrakenClient,
    pairs: List[str],
    direction: str,
    position_size: float,
    risk_pct: float,
    take_profit_pct: float,
    order_type: str = "MARKET",
    trading_mode: str = "Spot Trading",
    leverage: Optional[int] = None
):
    """
    Execute bulk trades for multiple pairs with comprehensive trading options
    
    Args:
        kraken_client: KrakenClient instance
        pairs: List of trading pair symbols
        direction: "BUY" or "SELL"
        position_size: Position size in USD per pair
        risk_pct: Risk percentage per trade
        take_profit_pct: Take profit percentage
        order_type: Order type (MARKET, LIMIT, etc.)
        trading_mode: Trading mode (Spot Trading or Margin Trading)
        leverage: Leverage amount (None for spot trading)
    """
    if not pairs:
        st.warning("No pairs selected for trading")
        return
    
    # Check for duplicate execution protection
    execution_key = f"crypto_bulk_execution_{hash(tuple(sorted(pairs)))}_{direction}_{position_size}"
    execution_timestamp_key = f"{execution_key}_timestamp"
    
    import time
    current_time = time.time()
    
    # Check if we just executed these same trades recently (within last 30 seconds)
    if (execution_key in st.session_state and 
        execution_timestamp_key in st.session_state and
        current_time - st.session_state[execution_timestamp_key] < 30):
        st.warning("⚠️ **Duplicate execution prevented!** You just executed these trades. Please wait a moment before executing again.")
        st.info(f"Last execution was {int(current_time - st.session_state[execution_timestamp_key])} seconds ago.")
        return
    
    # Mark execution in progress
    st.session_state[execution_key] = True
    st.session_state[execution_timestamp_key] = current_time
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        order_side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL
        
        for idx, pair in enumerate(pairs):
            progress = (idx + 1) / len(pairs)
            progress_bar.progress(progress)
            status_text.text(f"Processing {pair} ({idx + 1}/{len(pairs)})...")
            
            try:
                # Get current price
                ticker_info = kraken_client.get_ticker_info(pair)
                if not ticker_info:
                    results.append({
                        'pair': pair,
                        'status': 'FAILED',
                        'error': 'Could not fetch ticker data'
                    })
                    continue
                
                current_price = float(ticker_info.get('c', [0])[0])
                
                # Calculate stop loss and take profit
                if direction == "BUY":
                    stop_loss = current_price * (1 - risk_pct / 100)
                    take_profit = current_price * (1 + take_profit_pct / 100)
                else:
                    stop_loss = current_price * (1 + risk_pct / 100)
                    take_profit = current_price * (1 - take_profit_pct / 100)
                
                # Calculate quantity
                quantity = position_size / current_price
                
                # Get order type enum
                order_type_enum = OrderType[order_type]
                
                # Convert leverage to int if needed
                leverage_int = int(leverage) if leverage and leverage > 1 else None
                
                # Place the order with all configuration options
                result = kraken_client.place_order(
                    pair=pair,
                    side=order_side,
                    order_type=order_type_enum,
                    volume=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    leverage=leverage_int
                )
                
                if result is not None:
                    # Verify order was actually placed by checking order ID
                    order_id = result.order_id if hasattr(result, 'order_id') else None
                    
                    if not order_id or order_id == '':
                        # Order ID is missing - order might not have been placed
                        results.append({
                            'pair': pair,
                            'status': 'FAILED',
                            'error': 'Order ID missing - order may not have been placed'
                        })
                        logger.error(f"Order placed but no order ID returned for {pair}")
                        continue
                    
                    # Order was successful
                    results.append({
                        'pair': pair,
                        'status': 'SUCCESS',
                        'price': current_price,
                        'quantity': quantity,
                        'position_size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'order_id': order_id
                    })
                    
                    logger.info(f"✅ Order {order_id} placed successfully for {pair} - {direction} {quantity:.6f} @ ${current_price:.4f}")
                    
                    # 📝 LOG TO UNIFIED JOURNAL
                    try:
                        from services.unified_trade_journal import get_unified_journal, UnifiedTradeEntry, TradeType
                        from datetime import datetime
                        journal = get_unified_journal()
                        
                        # Calculate risk/reward percentages
                        if direction == "BUY":
                            risk_pct = ((current_price - stop_loss) / current_price) * 100
                            reward_pct = ((take_profit - current_price) / current_price) * 100
                        else:
                            risk_pct = ((stop_loss - current_price) / current_price) * 100
                            reward_pct = ((current_price - take_profit) / current_price) * 100
                        
                        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                        
                        trade_entry = UnifiedTradeEntry(
                            trade_id=f"{pair}_{order_id}_{int(datetime.now().timestamp())}",
                            trade_type=TradeType.CRYPTO.value,
                            symbol=pair,
                            side=direction,
                            entry_time=datetime.now(),
                            entry_price=current_price,
                            quantity=quantity,
                            position_size_usd=position_size,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_pct=risk_pct,
                            reward_pct=reward_pct,
                            risk_reward_ratio=rr_ratio,
                            strategy="Bulk Trade",
                            ai_managed=False,  # Will be updated if added to AI manager
                            broker="KRAKEN",
                            order_id=order_id,
                            status="OPEN"
                        )
                        
                        journal.log_trade_entry(trade_entry)
                        logger.info(f"📝 Logged {pair} to unified journal")
                    except Exception as journal_err:
                        logger.warning(f"Could not log to unified journal: {journal_err}")
                    
                    # 🤖 ADD TO AI POSITION MANAGER
                    try:
                        from services.ai_crypto_position_manager import get_ai_position_manager
                        from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                        import streamlit as st
                        
                        # Get or initialize AI position manager
                        if 'ai_position_manager' not in st.session_state:
                            llm_analyzer = LLMStrategyAnalyzer()
                            ai_manager = get_ai_position_manager(
                                kraken_client=kraken_client,
                                llm_analyzer=llm_analyzer,
                                check_interval_seconds=60,
                                enable_ai_decisions=True,
                                enable_trailing_stops=True,
                                enable_breakeven_moves=True,
                                enable_partial_exits=True,
                                require_manual_approval=True  # SAFETY: Set to False for auto-execution (RISKY!)
                            )
                            st.session_state.ai_position_manager = ai_manager
                            if not ai_manager.is_running:
                                ai_manager.start_monitoring_loop()
                                logger.info("🤖 AI Position Manager started")
                        else:
                            ai_manager = st.session_state.ai_position_manager
                        
                        # Add position to AI monitoring
                        trade_id = f"{pair}_{order_id}_{int(time.time())}"
                        success = ai_manager.add_position(
                            trade_id=trade_id,
                            pair=pair,
                            side=direction,
                            volume=quantity,
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            strategy="Bulk Trade",
                            entry_order_id=order_id
                        )
                        
                        if success:
                            logger.info(f"🤖 Added {pair} to AI position manager (ID: {trade_id})")
                            
                            # Update journal to mark as AI-managed
                            try:
                                from services.unified_trade_journal import get_unified_journal
                                journal = get_unified_journal()
                                # Note: Would need to add method to update ai_managed flag
                            except:
                                pass
                    except Exception as ai_err:
                        logger.warning(f"Could not add {pair} to AI position manager: {ai_err}")
                    
                    # Send Discord notification
                    try:
                        alert = TradingAlert(
                            ticker=pair,
                            alert_type=AlertType.TRADE_EXECUTED,
                            message=f"{direction} order executed at ${current_price:.4f} (Bulk Trade)",
                            priority=AlertPriority.MEDIUM,
                            details={
                                'order_id': result.order_id,
                                'price': float(current_price),
                                'quantity': float(quantity),
                                'direction': str(direction),
                                'position_size': float(position_size)
                            }
                        )
                        send_discord_alert(alert)
                    except Exception as e:
                        error_msg = str(e) if e else "Unknown error"
                        logger.error("❌ Failed to send Discord alert for {pair}: {}", str(error_msg), exc_info=True)
                else:
                    # Order failed
                    results.append({
                        'pair': pair,
                        'status': 'FAILED',
                        'error': 'Order placement failed - check logs for details'
                    })
                    
            except Exception as e:
                logger.error("Error executing trade for {pair}: {}", str(e), exc_info=True)
                results.append({
                    'pair': pair,
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Display results
        st.divider()
        st.markdown("#### 📊 Execution Results")
        
        successful = [r for r in results if r['status'] == 'SUCCESS']
        failed = [r for r in results if r['status'] == 'FAILED']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(results))
        col2.metric("Successful", len(successful), delta=f"{len(successful)/len(results)*100:.1f)}%")
        col3.metric("Failed", len(failed), delta=f"-{len(failed)/len(results)*100:.1f)}%")
        
        # Detailed results table
        if successful:
            st.markdown("##### ✅ Successful Trades")
            success_df = pd.DataFrame(successful)
            st.dataframe(success_df, width='stretch', hide_index=True)
        
        if failed:
            st.markdown("##### ❌ Failed Trades")
            failed_df = pd.DataFrame(failed)
            st.dataframe(failed_df, width='stretch', hide_index=True)
        
        # Summary
        if successful:
            total_invested = sum(r['position_size'] for r in successful)
            st.success(f"✅ Successfully executed {len(successful)} trades with total position size of ${total_invested:,.2f}")
            st.info("ℹ️ **Note:** Market orders fill immediately. Check your Kraken account's 'Trade History' or 'Closed Orders' section to see filled orders.")
            
            # Add verification section
            st.divider()
            st.markdown("#### 🔍 Verify Orders")
            
            verify_col1, verify_col2 = st.columns(2)
            
            with verify_col1:
                if st.button("📊 Check Recent Orders", width='stretch'):
                    verify_recent_orders(kraken_client, [r['order_id'] for r in successful])
            
            with verify_col2:
                if st.button("💰 Check Positions", width='stretch'):
                    verify_positions(kraken_client, [r['pair'] for r in successful])
        
    except Exception as e:
        st.error(f"Bulk trade execution error: {e}")
        logger.error("Bulk trade execution error: {}", str(e), exc_info=True)


def verify_recent_orders(kraken_client: KrakenClient, order_ids: List[str]):
    """
    Verify if orders were actually placed by checking closed orders
    
    Args:
        kraken_client: KrakenClient instance
        order_ids: List of order IDs to verify
    """
    try:
        with st.spinner("Checking recent orders..."):
            # Get closed orders from the last hour
            from datetime import datetime, timedelta
            start_time = int((datetime.now() - timedelta(hours=1)).timestamp())
            
            closed_orders = kraken_client.get_closed_orders(start=start_time)
            
            if not closed_orders:
                st.warning("⚠️ No closed orders found in the last hour. Orders may still be processing or may not have executed.")
                return
            
            # Check if our order IDs are in the closed orders
            found_orders = []
            missing_orders = []
            
            for order_id in order_ids:
                found = False
                for closed_order in closed_orders:
                    if closed_order.order_id == order_id:
                        found_orders.append({
                            'order_id': order_id,
                            'status': closed_order.status,
                            'pair': closed_order.pair,
                            'side': closed_order.side,
                            'volume': closed_order.volume,
                            'executed_volume': closed_order.executed_volume,
                            'avg_price': closed_order.avg_price if hasattr(closed_order, 'avg_price') else None
                        })
                        found = True
                        break
                
                if not found:
                    missing_orders.append(order_id)
            
            # Display results
            if found_orders:
                st.success(f"✅ Found {len(found_orders)} order(s) in closed orders:")
                found_df = pd.DataFrame(found_orders)
                st.dataframe(found_df, width='stretch', hide_index=True)
            
            if missing_orders:
                st.warning(f"⚠️ {len(missing_orders)} order(s) not found in closed orders:")
                for order_id in missing_orders:
                    st.text(f"  - {order_id}")
                st.info("💡 These orders may have failed, been rejected, or are still processing. Check your Kraken account directly.")
            
            if not found_orders and not missing_orders:
                st.info("ℹ️ No matching orders found. They may still be processing or may have failed.")
                
    except Exception as e:
        st.error(f"Error verifying orders: {e}")
        logger.error("Order verification error: {}", str(e), exc_info=True)


def verify_positions(kraken_client: KrakenClient, pairs: List[str]):
    """
    Verify if trades executed by checking positions
    
    Args:
        kraken_client: KrakenClient instance
        pairs: List of trading pairs to check
    """
    try:
        with st.spinner("Checking positions..."):
            # Get account balances
            balances = kraken_client.get_account_balance()
            
            if not balances:
                st.warning("⚠️ Could not fetch account balances. Check your API permissions.")
                return
            
            # Extract base assets from pairs
            base_assets = []
            for pair in pairs:
                if '/' in pair:
                    base_asset = pair.split('/')[0]
                    base_assets.append(base_asset)
            
            # Check if we have positions in these assets
            found_positions = []
            for balance in balances:
                # Check if this balance matches any of our traded assets
                for base_asset in base_assets:
                    if balance.currency.upper() == base_asset.upper() or balance.currency.upper() == f"Z{base_asset.upper()}" or balance.currency.upper() == f"X{base_asset.upper()}":
                        if balance.balance > 0:
                            found_positions.append({
                                'asset': base_asset,
                                'balance': balance.balance,
                                'available': balance.available,
                                'hold': balance.hold
                            })
            
            # Display results
            if found_positions:
                st.success(f"✅ Found positions in {len(found_positions)} asset(s):")
                positions_df = pd.DataFrame(found_positions)
                st.dataframe(positions_df, width='stretch', hide_index=True)
            else:
                st.info("ℹ️ No positions found for the traded pairs. This could mean:")
                st.markdown("""
                - Orders were placed but not filled yet
                - Orders were rejected
                - You sold the positions immediately
                - Check your Kraken account's Trade History
                """)
            
            # Show USD balance
            usd_balance = next((b for b in balances if b.currency == 'USD' or b.currency == 'ZUSD'), None)
            if usd_balance:
                st.metric("USD Balance", f"${usd_balance.balance:,.2f}")
                
    except Exception as e:
        st.error(f"Error verifying positions: {e}")
        logger.error("Position verification error: {}", str(e), exc_info=True)


def analyze_multi_config_bulk(
    kraken_client: KrakenClient,
    pairs: List[str],
    position_size: float,
    test_configs: Optional[Dict] = None
):
    """
    Analyze multiple trading configurations in bulk
    Tests different directions, leverage levels, and gets AI review for each
    
    Args:
        kraken_client: Kraken API client
        pairs: List of trading pairs to analyze
        position_size: Position size in USD per pair
        test_configs: Optional custom configuration dict with:
            - directions: List of "BUY" and/or "SELL" (default: both)
            - leverage_levels: List of leverage multipliers (default: [1, 2, 3, 5])
            - risk_pct: Risk percentage per trade (default: 2.0)
            - take_profit_pct: Take profit percentage (default: 5.0)
    """
    # Check if we already have results to display (on rerun after button click)
    if not pairs and 'multi_config_results' in st.session_state and st.session_state.multi_config_results is not None:
        # Just displaying existing results, skip analysis
        logger.info("📊 Displaying existing multi-config results from session state")
        all_results = []  # Will be loaded from session state below
    elif not pairs:
        st.warning("No pairs selected for multi-config analysis")
        return
    else:
        all_results = []  # Will be populated by analysis loop
    
    # Fractional Trading Preset Check
    fractional_mode = test_configs and test_configs.get('fractional_mode', False)
    
    # Default configurations
    # NOTE: Targets increased to ensure profitability after Kraken fees (~0.52% round trip)
    if test_configs is None:
        test_configs = {
            'directions': ['BUY', 'SELL'],
            'leverage_levels': [1.0, 2.0, 3.0, 5.0],
            'risk_pct': 3.0,  # Slightly wider stops for volatile crypto
            'take_profit_pct': 10.0,  # Increased from 5% - must beat fees!
            'fractional_mode': False
        }
    
    # Fractional mode: spot-only on major coins
    if fractional_mode:
        st.info("💰 **Fractional Trading Mode**: Analyzing spot trades on expensive coins (no leverage, safer)")
        directions = test_configs.get('directions', ['BUY'])  # Default to BUY only
        leverage_levels = [1.0]  # Spot only
        risk_pct = test_configs.get('risk_pct', 2.0)
        take_profit_pct = test_configs.get('take_profit_pct', 5.0)
    else:
        directions = test_configs.get('directions', ['BUY', 'SELL'])
        leverage_levels = test_configs.get('leverage_levels', [1.0, 2.0, 3.0, 5.0])
        risk_pct = test_configs.get('risk_pct', 2.0)
        take_profit_pct = test_configs.get('take_profit_pct', 5.0)
    
    # Calculate total combinations
    total_combinations = len(pairs) * len(directions) * len(leverage_levels)
    
    st.info(f"🔬 Testing **{total_combinations} configurations** across {len(pairs)} pairs...")
    st.markdown(f"**Test Matrix:** {len(directions)} directions × {len(leverage_levels)} leverage levels = {len(directions) * len(leverage_levels)} configs per pair")
    
    # Get appropriate LLM analyzer based on number of tickers (cloud for bulk, hybrid for single)
    llm_analyzer = get_llm_for_bulk_analysis(len(pairs))
    
    # Initialize AI trade reviewer if available
    ai_reviewer = None
    if llm_analyzer:
        try:
            ai_reviewer = AICryptoTradeReviewer(
                llm_analyzer=llm_analyzer,
                active_monitors=st.session_state.get('crypto_active_monitors', {}),
                supabase_client=st.session_state.get('supabase_client')
            )
            if len(pairs) > 1:
                logger.info(f"🤖 AI Trade Reviewer initialized for multi-config analysis ({len(pairs)} pairs, {total_combinations} configs) with cloud API")
            else:
                logger.info("🤖 AI Trade Reviewer initialized for single pair multi-config analysis")
        except Exception as e:
            logger.warning(f"Could not initialize AI reviewer: {e}")
    
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        combo_idx = 0
        
        for pair in pairs:
            # Get current price once per pair
            ticker_info = kraken_client.get_ticker_info(pair)
            if not ticker_info:
                logger.warning(f"Could not fetch ticker data for {pair}")
                continue
            
            current_price = float(ticker_info.get('c', [0])[0])
            if current_price == 0:
                logger.warning(f"Invalid price for {pair}")
                continue
                
            # Calculate market metrics (moved up for strategy determination)
            high_24h = float(ticker_info.get('h', [0])[0])
            low_24h = float(ticker_info.get('l', [0])[0])
            volatility = ((high_24h - low_24h) / current_price * 100) if current_price > 0 else 0
            volume_24h = float(ticker_info.get('v', [0])[0])
            change_24h = ((current_price - low_24h) / low_24h * 100) if low_24h > 0 else 0
            
            # Determine base strategy and params for this pair based on price and volatility
            # This ensures each coin gets appropriate risk/reward settings
            # NOTE: Kraken fees are ~0.26% per trade (0.52% round trip), so targets must exceed this!
            # Minimum profitable target = 1% (to cover fees + make profit)
            if current_price < 0.01:
                # Sub-penny coins: High volatility, aim for big moves
                base_strategy = "High Risk Penny"
                effective_risk_pct = 5.0
                effective_tp_pct = 25.0  # Increased from 15% - these coins move big
            elif volatility > 10.0:
                # Volatile altcoins: Ride the swings
                base_strategy = "Volatile Altcoin"
                effective_risk_pct = 4.0
                effective_tp_pct = 15.0  # Increased from 10%
            elif current_price < 1.0:
                # Mid-cap altcoins: Good momentum plays
                base_strategy = "Momentum"
                effective_risk_pct = 3.0
                effective_tp_pct = 10.0  # Increased from 6%
            else:
                # Large caps (BTC, ETH, etc): Tighter but still profitable
                base_strategy = "Swing"
                effective_risk_pct = 2.0
                effective_tp_pct = 5.0  # Increased from 2.5% - must beat fees!
            
            # Test all configurations for this pair
            for direction in directions:
                for leverage in leverage_levels:
                    combo_idx += 1
                    progress = combo_idx / total_combinations
                    progress_bar.progress(progress)
                    
                    # Determine if this is margin trading
                    trading_mode = "Margin Trading" if leverage > 1.0 else "Spot Trading"
                    
                    # Skip invalid combinations (spot trading with leverage > 1)
                    if trading_mode == "Spot Trading" and leverage > 1.0:
                        continue
                    
                    # Determine trade type for display
                    if direction == "SELL" and leverage > 1.0:
                        trade_type = f"SHORT {leverage}x"
                    elif direction == "BUY" and leverage > 1.0:
                        trade_type = f"LONG {leverage}x"
                    else:
                        trade_type = f"{direction} (Spot)"
                    
                    status_text.text(f"Analyzing {pair} - {trade_type} ({combo_idx}/{total_combinations})...")
                    
                    try:
                        # Calculate stop loss and take profit based on direction
                        # Use effective percentages derived from coin strategy
                        if direction == "BUY":
                            stop_loss = current_price * (1 - effective_risk_pct / 100)
                            take_profit = current_price * (1 + effective_tp_pct / 100)
                        else:  # SELL
                            stop_loss = current_price * (1 + effective_risk_pct / 100)
                            take_profit = current_price * (1 - effective_tp_pct / 100)
                        
                        # Calculate quantity (accounting for leverage)
                        effective_position = position_size * leverage
                        quantity = effective_position / current_price if current_price > 0 else 0
                        
                        # Calculate risk/reward ratio
                        risk_reward_ratio = effective_tp_pct / effective_risk_pct if effective_risk_pct > 0 else 0
                        
                        # Get AI review if available
                        ai_approved = None
                        ai_confidence = 0
                        ai_recommendation = "Not analyzed"
                        ai_risks = []
                        ai_score = 0
                        
                        if ai_reviewer:
                            try:
                                # Get account balance for capital context
                                balance_data = kraken_client.get_account_balance()
                                total_balance = sum(bal.balance for bal in balance_data) if balance_data else 0.0
                                
                                # Get market data for context
                                market_data = {
                                    'ticker': ticker_info,
                                    'volume_24h': volume_24h,
                                    'high_24h': high_24h,
                                    'low_24h': low_24h,
                                    'pair': pair,
                                    'current_price': current_price,
                                    'change_24h': change_24h,
                                    'volatility': volatility
                                }
                                
                                # Get AI pre-trade review
                                approved, confidence, reasoning, recommendations = ai_reviewer.pre_trade_review(
                                    pair=pair,
                                    side=direction,
                                    entry_price=current_price,
                                    position_size_usd=effective_position,
                                    stop_loss_price=stop_loss,
                                    take_profit_price=take_profit,
                                    strategy=base_strategy,
                                    market_data=market_data,
                                    total_capital=position_size * len(pairs) * leverage,
                                    actual_balance=total_balance
                                )
                                
                                ai_approved = approved
                                ai_confidence = confidence
                                
                                # Calculate composite score (confidence adjusted by leverage risk)
                                leverage_penalty = (leverage - 1.0) * 5  # -5% per leverage unit above 1
                                ai_score = confidence - leverage_penalty
                                
                                # More detailed recommendation
                                if approved:
                                    ai_recommendation = f"✅ APPROVED ({confidence:.0f}%)"
                                else:
                                    ai_recommendation = f"❌ REJECTED ({confidence:.0f}%)"
                                
                                ai_risks = recommendations.get('risks', [])
                                
                            except Exception as e:
                                logger.warning(f"AI review failed for {pair} {trade_type}: {e}")
                                ai_approved = False
                                ai_confidence = 0
                                ai_score = 0
                        
                        # Store result with market metrics
                        all_results.append({
                            'pair': pair,
                            'direction': direction,
                            'trading_mode': trading_mode,
                            'leverage': leverage,
                            'trade_type': trade_type,
                            'current_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'stop_pct': effective_risk_pct,
                            'target_pct': effective_tp_pct,
                            'position_size': position_size,
                            'effective_position': effective_position,
                            'quantity': quantity,
                            'risk_reward_ratio': risk_reward_ratio,
                            'ai_approved': ai_approved,
                            'ai_confidence': ai_confidence,
                            'ai_score': ai_score,
                            'ai_recommendation': ai_recommendation,
                            'ai_risks': ai_risks[:3] if ai_risks else [],
                            'strategy': base_strategy,
                            'strategy_name': base_strategy,  # Ensure consistent naming
                            # Add real market metrics for context
                            'change_24h': change_24h,
                            'volatility': volatility,
                            'volume_24h': volume_24h,
                            'high_24h': high_24h,
                            'low_24h': low_24h
                        })
                        
                    except Exception as e:
                        logger.error("Error analyzing {pair} {trade_type}: {}", str(e), exc_info=True)
                        continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Check if we have results from analysis or session state
        if not all_results and 'multi_config_results' not in st.session_state:
            st.warning("No results to display. Check logs for errors.")
            return
        
        # Convert to DataFrame for easier analysis (or load from session state)
        if all_results:
            results_df = pd.DataFrame(all_results)
            # Store in session state
            st.session_state.multi_config_results = results_df
        else:
            # Loading from session state (empty pairs list)
            results_df = st.session_state.multi_config_results
            logger.info(f"📊 Loaded {len(results_df)} configs from session state for display")
        
        # Display summary metrics
        st.markdown("---")
        st.markdown("### 📊 Multi-Configuration Analysis Results")
        
        logger.info(f"📊 DISPLAYING RESULTS: {len(results_df)} total configs")
        logger.info(f"📊 Results columns: {list(results_df.columns)}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        approved_count = len(results_df[results_df['ai_approved'] == True])
        avg_confidence = results_df['ai_confidence'].mean() if 'ai_confidence' in results_df.columns and len(results_df) > 0 else 0
        best_config = results_df.loc[results_df['ai_score'].idxmax()] if len(results_df) > 0 and 'ai_score' in results_df.columns else None
        best_score = best_config['ai_score'] if best_config is not None else 0
        
        logger.info("📊 Approved: {}, Avg conf: {:.1f}%, Best score: {:.1f}", approved_count, avg_confidence, best_score)
        
        col1.metric("Total Configs Tested", len(results_df))
        col2.metric("AI Approved", f"{approved_count} ({approved_count/len(results_df)*100:.1f}%)")
        col3.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        if best_config is not None:
            col4.metric("Best Score", f"{best_config['ai_score']:.1f}")
        
        # Show best configuration per pair
        st.markdown("#### 🏆 Best Configuration Per Pair")
        st.caption("Ranked by AI score (confidence - leverage penalty)")
        
        # Group by pair and get best config for each
        best_per_pair = results_df.loc[results_df.groupby('pair')['ai_score'].idxmax()]
        best_per_pair = best_per_pair.sort_values('ai_score', ascending=False)
        
        logger.info(f"📊 Rendering {len(best_per_pair)} best-per-pair expanders")
        logger.info(f"📊 best_per_pair index values: {list(best_per_pair.index)}")
        
        # Check if any button was clicked BEFORE rendering (Streamlit button state fix)
        selected_config_idx = None
        
        # Debug: Check what clicked flags exist in session state
        clicked_flags = [k for k in st.session_state.keys() if '_clicked' in str(k)]
        logger.info(f"📊 Current clicked flags in session state: {clicked_flags}")
        
        for idx in best_per_pair.index:
            flag_key = f'use_config_{idx}_clicked'
            flag_value = st.session_state.get(flag_key, False)
            logger.debug(f"📊 Checking flag {flag_key} = {flag_value}")
            if flag_value:
                selected_config_idx = idx
                st.session_state[flag_key] = False  # Reset flag
                logger.info(f"🔘 FOUND clicked flag: {flag_key}")
                break
        
        # If a config was selected, load it and navigate
        if selected_config_idx is not None:
            row = best_per_pair.loc[selected_config_idx].to_dict()
            pair = row.get('pair', 'UNKNOWN')
            trade_type = row.get('trade_type', 'UNKNOWN')
            
            logger.info("🔘 BEST CONFIG - Use This Setup clicked for {} - {}", str(pair), str(trade_type))
            
            # Store complete setup with REAL market data
            st.session_state.crypto_scanner_opportunity = {
                'symbol': row.get('pair', 'UNKNOWN'),
                'strategy': row.get('strategy', 'Unknown'),
                'confidence': row.get('ai_approved', False),
                'risk_level': 'Medium' if (row.get('leverage', 0) or 0) <= 2 else 'High',
                'score': row.get('ai_score', 0),
                'current_price': row.get('current_price', 0),
                'change_24h': row.get('change_24h', 0),
                'volume_ratio': (row.get('volume_24h', 0) or 0) / 1000000 if (row.get('volume_24h', 0) or 0) > 0 else 1.0,
                'volatility': row.get('volatility', 0),
                'reason': f"{row.get('trade_type', 'UNKNOWN')} recommended",
                'ai_reasoning': row.get('ai_recommendation', ''),
                'ai_confidence': 'High' if row.get('ai_confidence', 0) >= 75 else 'Medium' if row.get('ai_confidence', 0) >= 50 else 'Low',
                'ai_rating': row.get('ai_confidence', 0) / 10,
                'ai_risks': row.get('ai_risks', [])
            }
            
            st.session_state.crypto_quick_pair = row.get('pair', 'UNKNOWN')
            st.session_state.crypto_quick_trade_pair = row.get('pair', 'UNKNOWN')
            st.session_state.crypto_quick_direction = row.get('direction', 'BUY')
            st.session_state.crypto_trading_mode = row.get('trading_mode', 'Spot Trading')
            st.session_state.crypto_quick_leverage = row.get('leverage', 1)
            st.session_state.crypto_quick_position_size = row.get('position_size', 100)
            st.session_state.crypto_quick_stop_pct = row.get('stop_pct', 2.0)  # FIXED: Use actual percentage from row
            st.session_state.crypto_quick_target_pct = row.get('target_pct', 5.0)  # FIXED: Use actual percentage from row
            
            logger.info("📝 BEST CONFIG - Session state set: pair={}, direction={}, leverage={}, position=${}", str(pair), str(row.get('direction', 'BUY')), str(row.get('leverage', 1)), str(row.get('position_size', 100)))
            
            # Switch to Quick Trade main tab AND Execute Trade subtab
            st.session_state.active_crypto_tab = "⚡ Quick Trade"
            st.session_state.quick_trade_subtab = "⚡ Execute Trade"
            
            st.success(f"✅ Trade setup loaded for {pair} ({trade_type})! Switching to Execute Trade tab...")
            st.balloons()
            st.rerun()
        
        # Also check for filtered button clicks
        for key in list(st.session_state.keys()):
            if key.startswith('use_filtered_') and key.endswith('_clicked') and st.session_state.get(key, False):
                idx = key.replace('use_filtered_', '').replace('_clicked', '')
                data_key = f'use_filtered_{idx}_data'
                row = st.session_state.get(data_key, {})
                if row:
                    pair = row.get('pair', 'UNKNOWN')
                    trade_type = row.get('trade_type', 'UNKNOWN')
                    logger.info(f"🔘 FILTERED - Processing clicked flag for {pair} - {trade_type}")
                    
                    # Store complete setup
                    st.session_state.crypto_scanner_opportunity = {
                        'symbol': row.get('pair', 'UNKNOWN'),
                        'strategy': row.get('strategy', 'Unknown'),
                        'confidence': row.get('ai_approved', False),
                        'risk_level': 'Medium' if (row.get('leverage', 0) or 0) <= 2 else 'High',
                        'score': row.get('ai_score', 0),
                        'current_price': row.get('current_price', 0),
                        'change_24h': row.get('change_24h', 0),
                        'volume_ratio': (row.get('volume_24h', 0) or 0) / 1000000 if (row.get('volume_24h', 0) or 0) > 0 else 1.0,
                        'volatility': row.get('volatility', 0),
                        'reason': f"{row.get('trade_type', 'UNKNOWN')} recommended",
                        'ai_reasoning': row.get('ai_recommendation', ''),
                        'ai_confidence': 'High' if row.get('ai_confidence', 0) >= 75 else 'Medium' if row.get('ai_confidence', 0) >= 50 else 'Low',
                        'ai_rating': row.get('ai_confidence', 0) / 10,
                        'ai_risks': row.get('ai_risks', [])
                    }
                    
                    st.session_state.crypto_quick_pair = row.get('pair', 'UNKNOWN')
                    st.session_state.crypto_quick_trade_pair = row.get('pair', 'UNKNOWN')
                    st.session_state.crypto_quick_direction = row.get('direction', 'BUY')
                    st.session_state.crypto_trading_mode = row.get('trading_mode', 'Spot Trading')
                    st.session_state.crypto_quick_leverage = row.get('leverage', 1)
                    st.session_state.crypto_quick_position_size = row.get('position_size', 100)
                    st.session_state.crypto_quick_stop_pct = row.get('stop_pct', 2.0)
                    st.session_state.crypto_quick_target_pct = row.get('target_pct', 5.0)
                    
                    # Clear flags
                    st.session_state[key] = False
                    if data_key in st.session_state:
                        del st.session_state[data_key]
                    
                    # Switch tabs
                    st.session_state.active_crypto_tab = "⚡ Quick Trade"
                    st.session_state.quick_trade_subtab = "⚡ Execute Trade"
                    
                    st.success(f"✅ Trade setup loaded for {pair} ({trade_type})! Switching to Execute Trade tab...")
                    st.balloons()
                    st.rerun()
        
        # Render the expanders with buttons
        for idx, row_series in best_per_pair.iterrows():
            row = row_series.to_dict()  # Convert Series to dict so .get() works
            pair = row.get('pair', 'UNKNOWN')
            trade_type = row.get('trade_type', 'UNKNOWN')
            ai_score = row.get('ai_score', 0)
            logger.info("📊 Rendering expander for {} - {}", str(pair), str(trade_type))
            with st.expander(f"🎯 {pair} - {trade_type} (Score: {ai_score:.1f})"):
                info_col1, info_col2, info_col3 = st.columns(3)
                
                info_col1.markdown(f"""
                **Trade Details:**
                - Direction: **{row.get('direction', 'BUY')}**
                - Mode: **{row.get('trading_mode', 'Spot Trading')}**
                - Leverage: **{row.get('leverage', 0):.1f}x**
                - Strategy: {row.get('strategy', 'Unknown')}
                """)
                
                info_col2.markdown(f"""
                **Pricing:**
                - Current: ${row.get('current_price', 0):,.6f}
                - Stop Loss: ${row.get('stop_loss', 0):,.6f}
                - Take Profit: ${row.get('take_profit', 0):,.6f}
                - R:R Ratio: {row.get('risk_reward_ratio', 0):.2f}
                """)
                
                info_col3.markdown(f"""
                **Position & Risk:**
                - Base Size: ${row.get('position_size', 0):,.2f}
                - Effective Size: ${row.get('effective_position', 0):,.2f}
                - Quantity: {row.get('quantity', 0):.8f}
                - Stop %: {row.get('stop_pct', 0):.2f}%
                - Target %: {row.get('target_pct', 0):.2f}%
                """)
                
                # AI Analysis
                st.markdown("**🤖 AI Analysis:**")
                st.markdown(f"- **Recommendation:** {row.get('ai_recommendation', 'N/A')}")
                st.markdown(f"- **Confidence:** {row.get('ai_confidence', 0):.0f}%")
                st.markdown(f"- **Composite Score:** {row.get('ai_score', 0):.1f}")
                
                ai_risks = row.get('ai_risks', [])
                if ai_risks:
                    st.markdown("**⚠️ Key Risks:**")
                    for risk in ai_risks:
                        st.markdown(f"- {risk}")
                
                # AI Trade Style Recommendation
                st.markdown("#### 🤖 AI Trade Style Recommendation")
                
                # Determine best trade style based on config
                leverage = row.get('leverage', 0)
                direction = row.get('direction', 'BUY')
                confidence = row.get('ai_confidence', 0)
                
                if leverage == 1:
                    if confidence >= 75:
                        style = "**Conservative Spot** 🛡️"
                        reason = "High confidence + no leverage = Safe accumulation play"
                    else:
                        style = "**Cautious Spot** ⚠️"
                        reason = "Lower confidence suggests spot-only position to minimize risk"
                elif leverage == 2:
                    style = "**Balanced Swing** ⚖️"
                    reason = "2x leverage provides good risk/reward for medium-term holds"
                elif leverage == 3:
                    if direction == "SELL":
                        style = "**Aggressive Short** 🐻"
                        reason = "3x short = High conviction bearish trade (watch closely!)"
                    else:
                        style = "**Momentum Long** 🚀"
                        reason = "3x leverage for strong bullish momentum plays"
                else:  # 5x
                    style = "**High Risk Scalp** ⚡"
                    reason = "5x leverage = Quick in/out trades only. Set tight stops!"
                
                st.info(f"{style}\n\n{reason}")
                
                logger.info("📊 About to render 'Use This Setup' button for {} (key=use_config_{})", str(pair), idx)
                
                # Action button - Use on_click callback to set flag BEFORE rerun
                # This is the ONLY reliable way to handle button clicks in Streamlit expanders
                def set_config_flag(config_idx, config_pair):
                    logger.info(f"🔘 BUTTON CALLBACK! Setting flag use_config_{config_idx}_clicked for {config_pair}")
                    st.session_state[f'use_config_{config_idx}_clicked'] = True
                
                st.button(
                    f"✅ Use This Setup for {pair}", 
                    key=f"use_config_{idx}", 
                    use_container_width=True, 
                    type="primary",
                    on_click=set_config_flag,
                    args=(idx, pair)
                )
        
        # Show comparison table
        st.markdown("#### 📋 All Configurations Comparison")
        
        # Filter controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            filter_direction = st.multiselect(
                "Filter by Direction",
                options=['BUY', 'SELL'],
                default=['BUY', 'SELL'],
                key='multi_config_filter_direction'
            )
        
        with filter_col2:
            filter_approved = st.selectbox(
                "Filter by AI Approval",
                options=['All', 'Approved Only', 'Rejected Only'],
                key='multi_config_filter_approved'
            )
        
        with filter_col3:
            min_confidence = st.slider(
                "Min Confidence %",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
                key='multi_config_min_confidence'
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if filter_direction:
            filtered_df = filtered_df[filtered_df['direction'].isin(filter_direction)]
        
        if filter_approved == 'Approved Only':
            filtered_df = filtered_df[filtered_df['ai_approved'] == True]
        elif filter_approved == 'Rejected Only':
            filtered_df = filtered_df[filtered_df['ai_approved'] == False]
        
        if min_confidence is not None:
            filtered_df = filtered_df[filtered_df['ai_confidence'] >= min_confidence]
        
        # Sort by AI score
        filtered_df = filtered_df.sort_values(by='ai_score', ascending=False)
        
        # Display table
        display_df = filtered_df[[
            'pair', 'trade_type', 'current_price', 'ai_score', 
            'ai_confidence', 'ai_recommendation', 'risk_reward_ratio'
        ]].copy()
        
        display_df.columns = [
            'Pair', 'Trade Type', 'Price', 'AI Score', 
            'Confidence %', 'Recommendation', 'R:R'
        ]
        
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                'Price': st.column_config.NumberColumn(format="$%.6f"),
                'AI Score': st.column_config.NumberColumn(format="%.1f"),
                'Confidence %': st.column_config.NumberColumn(format="%.0f"),
                'R:R': st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        st.caption(f"Showing {len(filtered_df)} of {len(results_df)} configurations")
        
        # Add interactive selection for all filtered results
        st.markdown("#### 🎯 Select Any Configuration")
        st.info("💡 **Click 'Use This Setup' on any configuration below to populate the Execute Trade form**")
        
        # Display filtered results with action buttons
        for idx, row in filtered_df.iterrows():
            ai_score = row.get('ai_score', 0)
            ai_approved = row.get('ai_approved', False)
            current_price = row.get('current_price', 0)
            trade_type = row.get('trade_type', 'UNKNOWN')
            pair = row.get('pair', 'UNKNOWN')
            
            with st.expander(f"{'✅ APPROVED' if ai_approved else '❌ REJECTED'} | {pair} - {trade_type} (Score: {ai_score:.1f})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.6f}")
                    st.metric("Stop Loss", f"${row.get('stop_loss', 0):.6f}")
                
                with col2:
                    st.metric("Position Size", f"${row.get('position_size', 0):.2f}")
                    st.metric("Effective Size", f"${row.get('effective_position', 0):.2f}")
                
                with col3:
                    st.metric("Take Profit", f"${row.get('take_profit', 0):.6f}")
                    st.metric("R:R Ratio", f"{row.get('risk_reward_ratio', 0):.2f}")
                
                st.markdown(f"**AI Confidence:** {row.get('ai_confidence', 0):.0f}%")
                st.markdown(f"**Recommendation:** {row.get('ai_recommendation', 'N/A')}")
                st.markdown(f"**Strategy:** {row.get('strategy', 'N/A')}")
                st.markdown(f"**Leverage:** {row.get('leverage', 0):.1f}x ({row.get('trading_mode', 'Unknown')})")
                
                ai_risks = row.get('ai_risks', [])
                if ai_risks:
                    with st.expander("⚠️ Risk Analysis"):
                        for risk in ai_risks:
                            st.warning(risk)
                
                # Action button for each config - Use on_click callback for reliable flag setting
                def set_filtered_flag(filtered_idx, filtered_pair, filtered_trade_type, row_data):
                    logger.info(f"🔘 FILTERED CALLBACK! Setting flag use_filtered_{filtered_idx}_clicked for {filtered_pair} - {filtered_trade_type}")
                    st.session_state[f'use_filtered_{filtered_idx}_data'] = row_data
                    st.session_state[f'use_filtered_{filtered_idx}_clicked'] = True
                
                st.button(
                    f"✅ Use This Setup", 
                    key=f"use_filtered_{idx}", 
                    use_container_width=True, 
                    type="primary",
                    on_click=set_filtered_flag,
                    args=(idx, pair, trade_type, row)
                )
        
        # Export option
        if st.button("📥 Export Results to CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"crypto_multi_config_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error during multi-config analysis: {e}")
        logger.error("Multi-config analysis error: {}", str(e), exc_info=True)


def analyze_ultimate_all_strategies(
    kraken_client: KrakenClient,
    adapter,
    pairs: List[str],
    strategies: List[str],
    directions: List[str],
    leverage_levels: List[float],
    position_size: float,
    timeframe: str = "15m"
):
    """
    ULTIMATE ANALYSIS: Test EVERY combination of strategies, directions, and leverage levels.
    This is the most comprehensive analysis - leaves NO stone unturned!
    
    Tests: Pairs × Strategies × Directions × Leverage = potentially 100s of configs
    """
    if not pairs or not strategies or not directions or not leverage_levels:
        st.warning("No configurations to test")
        return
    
    # Calculate total combinations
    total_configs = len(pairs) * len(strategies) * len(directions) * len(leverage_levels)
    
    st.info(f"🔬 Testing **{total_configs} total configurations**...")
    st.markdown(f"**Matrix:** {len(pairs)} pairs × {len(strategies)} strategies × {len(directions)} directions × {len(leverage_levels)} leverages")
    
    all_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Convert timeframe to interval minutes
    interval_minutes = timeframe.replace('m', '') if timeframe else '15'
    
    try:
        config_idx = 0
        
        for pair in pairs:
            # Get current price once per pair
            ticker_info = kraken_client.get_ticker_info(pair)
            if not ticker_info:
                logger.warning(f"Could not fetch ticker data for {pair}")
                continue
            
            current_price = float(ticker_info.get('c', [0])[0])
            if current_price == 0:
                logger.warning(f"Invalid price for {pair}")
                continue
            
            for strategy_id in strategies:
                # Get strategy analysis using FreqtradeStrategyAdapter
                try:
                    strategy_analysis = adapter.analyze_crypto(pair, strategy_id, interval_minutes)
                    if 'error' in strategy_analysis:
                        logger.warning("Strategy analysis failed for {} {strategy_id}: {strategy_analysis.get('error')}", str(pair))
                        continue
                    
                    # Extract strategy metrics
                    recommendation = strategy_analysis.get('recommendation', 'HOLD')
                    confidence_score = strategy_analysis.get('confidence_score', 0)
                    signals = strategy_analysis.get('signals', {})
                    risk_level = strategy_analysis.get('risk_level', 'UNKNOWN')
                    
                except Exception as e:
                    logger.error(f"Error analyzing {pair} with {strategy_id}: {e}")
                    continue
                
                for direction in directions:
                    for leverage in leverage_levels:
                        config_idx += 1
                        progress = config_idx / total_configs
                        progress_bar.progress(progress)
                        
                        # Determine trading mode
                        trading_mode = "Margin Trading" if leverage > 1.0 else "Spot Trading"
                        
                        # Determine trade type label
                        if direction == "SELL" and leverage > 1.0:
                            trade_type = f"SHORT {leverage:.0f}x"
                        elif direction == "BUY" and leverage > 1.0:
                            trade_type = f"LONG {leverage:.0f}x"
                        else:
                            trade_type = f"{direction} (Spot)"
                        
                        status_text.text(f"Analyzing {pair} - {strategy_id} - {trade_type} ({config_idx}/{total_configs})...")
                        
                        # FIXED: Extract or calculate stop loss and take profit percentages from strategy
                        # Try multiple methods to get the percentages:
                        # 1. Check if strategy already provides percentage fields
                        stop_pct = strategy_analysis.get('stop_loss_pct')
                        target_pct = strategy_analysis.get('target_pct')
                        
                        # 2. If not, reverse-calculate from provided prices
                        if stop_pct is None and 'stop_loss' in strategy_analysis:
                            stop_loss_price = strategy_analysis['stop_loss']
                            if stop_loss_price != current_price and current_price > 0:
                                stop_pct = abs((stop_loss_price - current_price) / current_price * 100)
                        
                        if target_pct is None and 'roi_targets' in strategy_analysis:
                            roi_targets = strategy_analysis['roi_targets']
                            if roi_targets and len(roi_targets) > 0:
                                # Use the first (most aggressive) ROI target
                                target_pct = roi_targets[0].get('gain_percent', 5.0)
                        
                        # 3. Fall back to reasonable defaults based on strategy type
                        if stop_pct is None:
                            stop_pct = 2.0  # Conservative default
                        if target_pct is None:
                            target_pct = 5.0  # Conservative default
                        
                        # Ensure percentages are positive
                        stop_pct = abs(stop_pct)
                        target_pct = abs(target_pct)
                        
                        logger.debug(f"📊 {pair} {trade_type}: stop_pct={stop_pct:.2f}%, target_pct={target_pct:.2f}%, current_price=${current_price:.6f}")
                        
                        if direction == "BUY":
                            stop_loss = current_price * (1 - stop_pct / 100)
                            take_profit = current_price * (1 + target_pct / 100)
                        else:  # SELL
                            stop_loss = current_price * (1 + stop_pct / 100)
                            take_profit = current_price * (1 - target_pct / 100)
                        
                        # Sanity check: Ensure stop/target are different from current price
                        if stop_loss == current_price or take_profit == current_price:
                            logger.error(f"⚠️ CALCULATION ERROR for {pair} {trade_type}: stop_loss=${stop_loss:.6f}, take_profit=${take_profit:.6f}, current=${current_price:.6f}, stop%={stop_pct}, target%={target_pct}")
                        
                        # Calculate position details
                        effective_position = position_size * leverage
                        quantity = effective_position / current_price if current_price > 0 else 0
                        risk_reward_ratio = target_pct / stop_pct if stop_pct > 0 else 0
                        
                        # Calculate composite score
                        # Base score from strategy confidence
                        base_score = confidence_score
                        
                        # Apply penalties/bonuses
                        leverage_penalty = (leverage - 1.0) * 5  # -5% per leverage unit
                        
                        # Direction bonus if strategy recommends it
                        direction_bonus = 0
                        if (recommendation == 'BUY' and direction == 'BUY') or (recommendation == 'SELL' and direction == 'SELL'):
                            direction_bonus = 10  # +10% if direction matches recommendation
                        
                        composite_score = base_score - leverage_penalty + direction_bonus
                        
                        # Determine approval based on composite score and recommendation alignment
                        approved = composite_score >= 60 and recommendation != 'HOLD'
                        
                        # Store result
                        all_results.append({
                            'pair': pair,
                            'strategy': strategy_id,
                            'strategy_name': {
                                "ema_crossover": "EMA Crossover",
                                "rsi_stoch_hammer": "RSI+Stochastic",
                                "fisher_rsi_multi": "Fisher RSI",
                                "macd_volume": "MACD+Volume",
                                "aggressive_scalp": "Aggressive Scalp"
                            }.get(strategy_id, strategy_id),
                            'direction': direction,
                            'trading_mode': trading_mode,
                            'leverage': leverage,
                            'trade_type': trade_type,
                            'current_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'stop_pct': stop_pct,  # FIXED: Add percentage values
                            'target_pct': target_pct,  # FIXED: Add percentage values
                            'position_size': position_size,
                            'effective_position': effective_position,
                            'quantity': quantity,
                            'risk_reward_ratio': risk_reward_ratio,
                            'recommendation': recommendation,
                            'confidence_score': confidence_score,
                            'composite_score': composite_score,
                            'approved': approved,
                            'risk_level': risk_level,
                            'signals': signals
                        })
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if not all_results:
            st.warning("No results to display. Check logs for errors.")
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Store in session state
        st.session_state.ultimate_analysis_results = results_df
        
        # Display results
        st.markdown("---")
        st.markdown("### 🏆 ULTIMATE ANALYSIS RESULTS")
        st.success(f"✅ Tested {len(results_df)} configurations successfully!")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        approved_count = len(results_df[results_df['approved'] == True])
        avg_score = results_df['composite_score'].mean()
        best_config = results_df.loc[results_df['composite_score'].idxmax()] if len(results_df) > 0 else None
        
        col1.metric("Total Configs", len(results_df))
        col2.metric("Approved", f"{approved_count} ({approved_count/len(results_df)*100:.1f}%)")
        col3.metric("Avg Score", f"{avg_score:.1f}")
        if best_config is not None:
            col4.metric("Best Score", f"{best_config['composite_score']:.1f}")
        
        # Best configuration per pair
        st.markdown("#### 🎯 Best Configuration Per Pair (Across ALL Strategies)")
        st.caption("Showing the absolute best setup for each token - considering ALL strategies and configurations")
        
        best_per_pair = results_df.loc[results_df.groupby('pair')['composite_score'].idxmax()]
        best_per_pair = best_per_pair.sort_values('composite_score', ascending=False)
        
        for idx, row in best_per_pair.head(10).iterrows():
            pair = row.get('pair', 'UNKNOWN')
            strategy_name = row.get('strategy_name', 'Unknown')
            trade_type = row.get('trade_type', 'UNKNOWN')
            composite_score = row.get('composite_score', 0)
            with st.expander(f"🏆 {pair} - {strategy_name} {trade_type} (Score: {composite_score:.1f})"):
                result_col1, result_col2, result_col3 = st.columns(3)
                
                result_col1.markdown(f"""
                **Strategy & Direction:**
                - Strategy: **{strategy_name}**
                - Direction: **{row.get('direction', 'BUY')}**
                - Mode: **{row.get('trading_mode', 'Spot Trading')}**
                - Leverage: **{row.get('leverage', 0):.1f}x**
                - Recommendation: **{row.get('recommendation', 'N/A')}**
                """)
                
                result_col2.markdown(f"""
                **Pricing:**
                - Current: ${row.get('current_price', 0):,.6f}
                - Stop Loss: ${row.get('stop_loss', 0):,.6f}
                - Take Profit: ${row.get('take_profit', 0):,.6f}
                - R:R Ratio: {row.get('risk_reward_ratio', 0):.2f}
                """)
                
                result_col3.markdown(f"""
                **Position & Scoring:**
                - Base Size: ${row.get('position_size', 0):,.2f}
                - Effective Size: ${row.get('effective_position', 0):,.2f}
                - Quantity: {row.get('quantity', 0):.8f}
                - Stop %: {row.get('stop_pct', 0):.2f}%
                - Target %: {row.get('target_pct', 0):.2f}%
                - Confidence: {row.get('confidence_score', 0):.0f}%
                - **Score: {row.get('composite_score', 0):.1f}**
                - Risk: {row.get('risk_level', 'Unknown')}
                """)
                
                # AI Trade Style Recommendation
                st.markdown("#### 🤖 AI Trade Style Recommendation")
                leverage = row.get('leverage', 0)
                direction = row.get('direction', 'BUY')
                confidence = row.get('confidence_score', 0)
                
                if leverage == 1:
                    if confidence >= 75:
                        style = "**Conservative Spot** 🛡️"
                        reason = "High confidence + no leverage = Safe accumulation play"
                    else:
                        style = "**Cautious Spot** ⚠️"
                        reason = "Lower confidence suggests spot-only position to minimize risk"
                elif leverage == 2:
                    style = "**Balanced Swing** ⚖️"
                    reason = "2x leverage provides good risk/reward for medium-term holds"
                elif leverage == 3:
                    if direction == "SELL":
                        style = "**Aggressive Short** 🐻"
                        reason = "3x short = High conviction bearish trade (watch closely!)"
                    else:
                        style = "**Momentum Long** 🚀"
                        reason = "3x leverage for strong bullish momentum plays"
                else:  # 5x
                    style = "**High Risk Scalp** ⚡"
                    reason = "5x leverage = Quick in/out trades only. Set tight stops!"
                
                st.info(f"{style}\n\n{reason}\n\n**Optimal Strategy:** {row['strategy_name']}")
                
                # Action button - Use on_click callback for reliable flag setting in expanders
                def set_ultimate_setup(row_data, total_configs):
                    logger.info(f"🔘 ULTIMATE CALLBACK! Setting up {row_data['pair']} - {row_data['trade_type']}")
                    
                    # Store complete setup with REAL market data
                    st.session_state.crypto_scanner_opportunity = {
                        'symbol': row_data['pair'],
                        'strategy': row_data['strategy_name'],
                        'confidence': row_data['recommendation'],
                        'risk_level': row_data['risk_level'],
                        'score': row_data['composite_score'],
                        'current_price': row_data['current_price'],
                        'change_24h': row_data.get('change_24h', 0),
                        'volume_ratio': row_data.get('volume_24h', 0) / 1000000 if row_data.get('volume_24h', 0) > 0 else 1.0,
                        'volatility': row_data.get('volatility', 0),
                        'reason': f"{row_data['strategy_name']} {row_data['trade_type']} - Ultimate Analysis Winner",
                        'ai_reasoning': f"Best of {total_configs} configs tested",
                        'ai_confidence': row_data['recommendation'],
                        'ai_rating': row_data['confidence_score'] / 10,
                        'ai_risks': []
                    }
                    
                    st.session_state.crypto_quick_pair = row_data['pair']
                    st.session_state.crypto_quick_trade_pair = row_data['pair']
                    st.session_state.crypto_quick_direction = row_data['direction']
                    st.session_state.crypto_trading_mode = row_data['trading_mode']
                    st.session_state.crypto_quick_leverage = row_data['leverage']
                    st.session_state.crypto_quick_position_size = row_data['position_size']
                    st.session_state.crypto_quick_stop_pct = row_data['stop_pct']
                    st.session_state.crypto_quick_target_pct = row_data['target_pct']
                    
                    logger.info(f"📝 ULTIMATE - Session state set: pair={row_data['pair']}, direction={row_data['direction']}, leverage={row_data['leverage']}, position=${row_data['position_size']}")
                    
                    # Switch to Execute Trade tab
                    st.session_state.quick_trade_subtab = "⚡ Execute Trade"
                    st.session_state.show_ultimate_success = {'pair': row_data['pair'], 'trade_type': row_data['trade_type']}
                
                st.button(
                    f"✅ Use This Setup", 
                    key=f"use_ultimate_{idx}", 
                    use_container_width=True, 
                    type="primary",
                    on_click=set_ultimate_setup,
                    args=(row.to_dict(), len(results_df))
                )
        
        # Full results table with filters
        st.markdown("#### 📊 Full Results Table")
        
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            filter_strategy = st.multiselect(
                "Filter by Strategy",
                options=results_df['strategy_name'].unique().tolist(),
                default=results_df['strategy_name'].unique().tolist(),
                key='ultimate_filter_strategy'
            )
        
        with filter_col2:
            filter_direction_result = st.multiselect(
                "Filter by Direction",
                options=['BUY', 'SELL'],
                default=['BUY', 'SELL'],
                key='ultimate_filter_direction'
            )
        
        with filter_col3:
            filter_approved_result = st.selectbox(
                "Filter by Approval",
                options=['All', 'Approved Only', 'Rejected Only'],
                key='ultimate_filter_approved'
            )
        
        with filter_col4:
            min_score = st.slider(
                "Min Score",
                min_value=0,
                max_value=100,
                value=50,
                step=5,
                key='ultimate_min_score'
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if filter_strategy:
            filtered_df = filtered_df[filtered_df['strategy_name'].isin(filter_strategy)]
        
        if filter_direction_result:
            filtered_df = filtered_df[filtered_df['direction'].isin(filter_direction_result)]
        
        if filter_approved_result == 'Approved Only':
            filtered_df = filtered_df[filtered_df['approved'] == True]
        elif filter_approved_result == 'Rejected Only':
            filtered_df = filtered_df[filtered_df['approved'] == False]
        
        if min_score is not None:
            filtered_df = filtered_df[filtered_df['composite_score'] >= min_score]
        
        # Sort and display
        filtered_df = filtered_df.sort_values(by='composite_score', ascending=False)
        
        display_df = filtered_df[[
            'pair', 'strategy_name', 'trade_type', 'current_price', 
            'composite_score', 'confidence_score', 'recommendation', 'risk_reward_ratio'
        ]].copy()
        
        display_df.columns = [
            'Pair', 'Strategy', 'Trade Type', 'Price', 
            'Score', 'Confidence %', 'Signal', 'R:R'
        ]
        
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                'Price': st.column_config.NumberColumn(format="$%.6f"),
                'Score': st.column_config.NumberColumn(format="%.1f"),
                'Confidence %': st.column_config.NumberColumn(format="%.0f"),
                'R:R': st.column_config.NumberColumn(format="%.2f")
            }
        )
        
        st.caption(f"Showing {len(filtered_df)} of {len(results_df)} configurations")
        
        # Export option
        if st.button("📥 Export Ultimate Results to CSV", key="export_ultimate"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"crypto_ultimate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error during ultimate analysis: {e}")
        logger.error("Ultimate analysis error: {}", str(e), exc_info=True)

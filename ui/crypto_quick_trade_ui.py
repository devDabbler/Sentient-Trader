"""
Crypto Quick Trade UI
Enhanced quick trade interface with ticker management, scanner integration,
investment controls, risk management, AI validation, and automated execution
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from loguru import logger
import json
import os
import asyncio
from clients.kraken_client import KrakenClient, OrderType, OrderSide
from clients.crypto_validator import CryptoValidator
from src.integrations.discord_webhook import send_discord_alert
from models.alerts import TradingAlert, AlertType, AlertPriority
from services.freqtrade_strategies import FreqtradeStrategyAdapter

# Note: Asset pairs are cached in session state in display_trade_setup()
# to avoid repeated API calls, since KrakenClient is not hashable for @st.cache_data

# --- Unified Discovery Workflow ---

def display_unified_scanner(kraken_client: KrakenClient, crypto_config, scanner_instances: Dict):
    """
    A unified scanner UI that provides a streamlined workflow for crypto discovery.
    Workflow: Scan -> Bulk Select -> Analyze All -> Pick Best -> Execute
    """
    st.markdown("### 🔍 Smart Crypto Scanner & Analyzer")
    st.markdown("**Workflow:** Scan → Bulk Select → Analyze All → Pick Best → Execute")

    # --- 1. Scan for Opportunities ---
    st.markdown("---")
    st.markdown("#### 1. Scan for Opportunities")
    
    scanner_map = {
        "💰 Penny Cryptos (<$1)": "penny_crypto_scanner",
        "🔥 Buzzing/Volume Surge": "crypto_opportunity_scanner",
        "🌶️ Hottest/Momentum": "ai_crypto_scanner",
        "💎 Sub-Penny Discovery": "sub_penny_discovery",
        "⭐ My Watchlist": "watchlist"
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
        if st.button("🚀 Scan", use_container_width=True, type="primary"):
            # Preserve tab states before scan to prevent redirect
            if 'active_crypto_tab' not in st.session_state:
                st.session_state.active_crypto_tab = "⚡ Quick Trade"
            
            st.session_state.scan_results = []
            st.session_state.selected_tickers = []
            
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
                        st.session_state.scan_results = [{"Ticker": t, "Price": 0, "Change": 0, "Score": 0} for t in crypto_config.CRYPTO_WATCHLIST]
                    
                    # Ensure tab states are preserved after scan
                    st.session_state.active_crypto_tab = "⚡ Quick Trade"

                except Exception as e:
                    st.error(f"An error occurred while scanning: {e}")
                    logger.error(f"Scanner failed: {e}", exc_info=True)
                    # Preserve tab state even on error
                    st.session_state.active_crypto_tab = "⚡ Quick Trade"

    # --- 2. Select Tickers for Analysis ---
    if 'scan_results' in st.session_state and st.session_state.scan_results:
        st.markdown("---")
        st.markdown("#### 2. Select Tickers for Analysis")
        
        results = st.session_state.scan_results
        df = pd.DataFrame(results)

        if 'selected_tickers' not in st.session_state:
            st.session_state.selected_tickers = []

        sel_col1, sel_col2, sel_col3 = st.columns([2, 1, 1])
        with sel_col1:
            st.metric("Selected for Analysis", f"{len(st.session_state.selected_tickers)} / {len(df)}")
        
        # Helper functions for button callbacks
        def select_all_tickers():
            st.session_state.selected_tickers = df['Ticker'].tolist()
        
        def clear_all_tickers():
            st.session_state.selected_tickers = []
        
        # Helper function for checkbox callbacks
        def make_ticker_toggle(ticker: str):
            """Create a callback function for a specific ticker"""
            def toggle_ticker():
                # Read the current checkbox value from session state
                checkbox_key = f"sel_{ticker}"
                new_value = st.session_state.get(checkbox_key, False)
                
                if new_value:
                    # Checkbox was checked - add to selected
                    if ticker not in st.session_state.selected_tickers:
                        st.session_state.selected_tickers.append(ticker)
                else:
                    # Checkbox was unchecked - remove from selected
                    if ticker in st.session_state.selected_tickers:
                        st.session_state.selected_tickers.remove(ticker)
            return toggle_ticker
        
        with sel_col2:
            st.button("Select All", use_container_width=True, on_click=select_all_tickers)
        with sel_col3:
            st.button("Clear All", use_container_width=True, on_click=clear_all_tickers)

        # Display table with checkboxes - use on_change to prevent reruns
        for i, row in df.iterrows():
            cols = st.columns([1, 4, 2, 2, 2])
            ticker = row['Ticker']
            is_selected = ticker in st.session_state.selected_tickers
            
            # Use on_change callback to update state without triggering rerun
            cols[0].checkbox(
                f"Select {ticker}",
                value=is_selected,
                key=f"sel_{ticker}",
                on_change=make_ticker_toggle(ticker),
                label_visibility="hidden"
            )

            cols[1].markdown(f"**{row['Ticker']}**")
            cols[2].metric("Price", f"${row['Price']:,.4f}")
            cols[3].metric("24h Change", f"{row['Change']:.2f}%" if row['Change'] else "N/A")
            cols[4].metric("Score", f"{int(row['Score'])}" if row['Score'] else "N/A")

    # --- 3. Analyze Selected Tickers ---
    st.markdown("---")
    st.markdown("#### 3. Analyze Selected Tickers")
    selected_tickers = st.session_state.get('selected_tickers', [])
    
    if not selected_tickers:
        st.info("Select tickers from the scan results above to begin analysis.")
    else:
        st.success(f"**{len(selected_tickers)} tickers ready for analysis.**")
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        with analysis_col1:
            adapter = FreqtradeStrategyAdapter(kraken_client)
            strategies = adapter.get_available_strategies()
            strategy_id = st.selectbox(
                "Analysis Strategy",
                options=[s['id'] for s in strategies],
                format_func=lambda x: next(s['name'] for s in strategies if s['id'] == x)
            )
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "60m"], index=2)
        
        with analysis_col2:
            st.write("")
            st.write("")
            if st.button("🔬 Analyze All", use_container_width=True, type="primary"):
                with st.spinner(f"Analyzing {len(selected_tickers)} tickers..."):
                    try:
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
                            analysis_results = adapter.bulk_analyze(valid_tickers, strategy_id, timeframe)
                            st.session_state.analysis_results = analysis_results
                        else:
                            st.error("❌ No valid trading pairs found to analyze")
                            
                    except Exception as e:
                        st.error(f"Bulk analysis failed: {e}")
                        logger.error(f"Bulk analysis error: {e}", exc_info=True)

    # --- 4. Ranked Opportunities ---
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.markdown("---")
        st.markdown("#### 4. Ranked Opportunities")
        
        results = st.session_state.analysis_results
        sorted_results = sorted(results, key=lambda x: x['confidence_score'], reverse=True)

        # Summary Metrics
        buy_signals = sum(1 for r in sorted_results if r['recommendation'] == 'BUY')
        avg_confidence = sum(r['confidence_score'] for r in sorted_results) / len(sorted_results) if sorted_results else 0
        high_confidence_count = sum(1 for r in sorted_results if r['confidence_score'] >= 75)

        summary_cols = st.columns(3)
        summary_cols[0].metric("Total Opportunities", len(sorted_results))
        summary_cols[1].metric("Buy Signals", f"{buy_signals}")
        summary_cols[2].metric("Avg. Confidence", f"{avg_confidence:.1f}%")

        for i, analysis in enumerate(sorted_results):
            expander_title = f"**{i+1}. {analysis['symbol']}** | Signal: **{analysis['recommendation']}** | Confidence: **{analysis['confidence_score']}%**"
            with st.expander(expander_title, expanded=(i < 3)):
                
                rec = analysis['recommendation']
                if rec == 'BUY':
                    st.success(f"✅ **{rec}** Signal")
                elif rec == 'SELL':
                    st.error(f"❌ **{rec}** Signal")
                else:
                    st.info(f"⏸️ **{rec}** Signal")

                # Key metrics
                metric_cols = st.columns(4)
                metric_cols[0].metric("Current Price", f"${analysis['current_price']:,.4f}")
                metric_cols[1].metric("Stop Loss", f"${analysis['stop_loss']:,.4f}")
                metric_cols[2].metric("Risk Level", analysis['risk_level'])
                metric_cols[3].metric("R:R Ratio", f"{analysis.get('risk_reward_ratio', 'N/A')}")

                if st.button("✅ Use This Setup", key=f"use_{analysis['symbol']}", use_container_width=True):
                    st.session_state.crypto_quick_trade_pair = analysis['symbol']
                    st.session_state.crypto_quick_stop_pct = abs((analysis['stop_loss'] - analysis['current_price']) / analysis['current_price'] * 100)
                    if analysis['roi_targets']:
                        st.session_state.crypto_quick_target_pct = analysis['roi_targets'][0]['gain_percent']
                    
                    st.success(f"Applied {analysis['symbol']} to the 'Execute Trade' form!")
                    st.balloons()


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
    
    # Tab selector using radio buttons (no rerun on selection)
    subtab = st.radio(
        "Navigation",
        options=["🔍 Ticker Management", "⚡ Execute Trade"],
        horizontal=True,
        key="quick_trade_subtab_selector",
        label_visibility="collapsed"
    )
    
    # Update session state if changed
    if subtab != st.session_state.quick_trade_subtab:
        st.session_state.quick_trade_subtab = subtab
    
    # Render the selected subtab
    if st.session_state.quick_trade_subtab == "🔍 Ticker Management":
        display_unified_scanner(kraken_client, crypto_config, st.session_state.scanner_instances)
    elif st.session_state.quick_trade_subtab == "⚡ Execute Trade":
        display_trade_setup(kraken_client, crypto_config, watchlist_manager)


def display_trade_setup(kraken_client: KrakenClient, crypto_config, watchlist_manager=None):
    """
    Display the trade execution form with AI analysis
    Supports single trade, bulk custom selection, and bulk watchlist trading
    """
    st.markdown("### ⚡ Execute Trade")
    
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
    # Trading pair selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
                key="crypto_quick_trade_pair"
            )
        except Exception as e:
            st.error(f"Error loading trading pairs: {e}")
            return
    
    with col2:
        st.write("")
        st.write("")
        direction = st.radio(
            "Direction",
            options=["BUY", "SELL"],
            horizontal=True,
            key="crypto_quick_direction"
        )
    
    # Position sizing
    st.markdown("#### Position Sizing")
    pos_col1, pos_col2, pos_col3 = st.columns(3)
    
    with pos_col1:
        position_size = st.number_input(
            "Position Size (USD)",
            min_value=10.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            key="crypto_quick_position_size"
        )
    
    with pos_col2:
        leverage = st.number_input(
            "Leverage",
            min_value=1.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key="crypto_quick_leverage"
        )
    
    with pos_col3:
        risk_pct = st.number_input(
            "Risk %",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key="crypto_quick_stop_pct"
        )
    
    # Take profit
    take_profit_pct = st.number_input(
        "Take Profit %",
        min_value=0.1,
        max_value=20.0,
        value=5.0,
        step=0.1,
        key="crypto_quick_target_pct"
    )
    
    # AI Analysis section
    st.markdown("#### 🤖 AI Analysis")
    
    analysis_col1, analysis_col2, analysis_col3 = st.columns([2, 1, 1])
    
    with analysis_col1:
        if st.button("🔍 Analyze Opportunity", use_container_width=True):
            with st.spinner("Analyzing opportunity..."):
                try:
                    # Get current price
                    ticker_info = kraken_client.get_ticker_info(selected_pair)
                    current_price = float(ticker_info.get('c', [0])[0])
                    
                    # Calculate stop loss and take profit
                    if direction == "BUY":
                        stop_loss = current_price * (1 - risk_pct / 100)
                        take_profit = current_price * (1 + take_profit_pct / 100)
                    else:
                        stop_loss = current_price * (1 + risk_pct / 100)
                        take_profit = current_price * (1 - take_profit_pct / 100)
                    
                    # Store in session state
                    st.session_state.crypto_analysis = {
                        'pair': selected_pair,
                        'direction': direction,
                        'current_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'leverage': leverage,
                        'risk_reward_ratio': take_profit_pct / risk_pct
                    }
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    logger.error(f"AI analysis error: {e}", exc_info=True)
    
    with analysis_col2:
        if st.button("📊 Get Market Data", use_container_width=True):
            with st.spinner("Fetching market data..."):
                try:
                    ticker_info = kraken_client.get_ticker_info(selected_pair)
                    st.json(ticker_info)
                except Exception as e:
                    st.error(f"Failed to fetch market data: {e}")
    
    # Display analysis results if available
    if 'crypto_analysis' in st.session_state:
        analysis = st.session_state.crypto_analysis
        
        st.markdown("#### 📈 Analysis Results")
        
        metric_cols = st.columns(4)
        metric_cols[0].metric("Current Price", f"${analysis['current_price']:,.4f}")
        metric_cols[1].metric("Stop Loss", f"${analysis['stop_loss']:,.4f}")
        metric_cols[2].metric("Take Profit", f"${analysis['take_profit']:,.4f}")
        metric_cols[3].metric("R:R Ratio", f"{analysis['risk_reward_ratio']:.2f}")
        
        # Execute trade button
        st.markdown("---")
        
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
            
            if st.button("🚀 Execute Trade", use_container_width=True, type="primary", disabled=is_recent):
                if is_recent:
                    st.warning("⚠️ **Duplicate execution prevented!** You just executed this trade. Please wait a moment before executing again.")
                else:
                    # Mark execution
                    st.session_state[execution_key] = True
                    st.session_state[execution_timestamp_key] = current_time
                    execute_crypto_trade(kraken_client, analysis)
        
        with exec_col2:
            if st.button("💾 Save Setup", use_container_width=True):
                save_trade_setup(analysis)
                st.success("Setup saved!")
        
        with exec_col3:
            def reset_analysis():
                if 'crypto_analysis' in st.session_state:
                    del st.session_state.crypto_analysis
            
            st.button("🔄 Reset", use_container_width=True, on_click=reset_analysis)


def execute_crypto_trade(kraken_client: KrakenClient, analysis: Dict):
    """
    Execute the crypto trade with given parameters
    """
    try:
        with st.spinner("Placing order..."):
            # Determine order type and side
            order_side = OrderSide.BUY if analysis['direction'] == 'BUY' else OrderSide.SELL
            
            # Calculate order quantity
            if analysis['direction'] == 'BUY':
                quantity = analysis['position_size'] / analysis['current_price']
            else:
                quantity = analysis['position_size'] / analysis['current_price']
            
            # Place the order with stop loss and take profit
            result = kraken_client.place_order(
                pair=analysis['pair'],
                side=order_side,
                order_type=OrderType.MARKET,
                volume=quantity,
                stop_loss=analysis.get('stop_loss'),
                take_profit=analysis.get('take_profit')
            )
            
            if result is not None:
                # Verify order ID exists
                order_id = result.order_id if hasattr(result, 'order_id') else None
                
                if not order_id or order_id == '':
                    st.error(f"❌ Order placed but no order ID returned. Check Kraken for order status.")
                    logger.error(f"Order placed but no order ID returned for {analysis['pair']}")
                else:
                    st.success(f"✅ Trade executed successfully! Order ID: {order_id}")
                    st.info("ℹ️ **Note:** Market orders fill immediately. Check your Kraken account's 'Trade History' or 'Closed Orders' section to see the filled order.")
                    logger.info(f"✅ Order {order_id} placed successfully for {analysis['pair']} - {analysis['direction']} {quantity:.6f} @ ${analysis['current_price']:.4f}")
                    
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
                    logger.error(f"❌ Failed to send Discord alert for {analysis['pair']}: {error_msg}", exc_info=True)
                    
            else:
                st.error(f"❌ Trade failed: Order placement returned None - check logs for details")
                
    except Exception as e:
        st.error(f"Trade execution error: {e}")
        logger.error(f"Trade execution error: {e}", exc_info=True)


def save_trade_setup(analysis: Dict):
    """
    Save trade setup to watchlist or configuration
    """
    try:
        # Add to crypto watchlist
        if hasattr(analysis['pair'], 'replace'):
            ticker = analysis['pair'].replace('/USD', '').replace('/USDT', '')
        else:
            ticker = str(analysis['pair']).replace('/USD', '').replace('/USDT', '')
            
        if ticker not in crypto_config.CRYPTO_WATCHLIST:
            crypto_config.CRYPTO_WATCHLIST.append(ticker)
            logger.info(f"Added {ticker} to crypto watchlist")
        
    except Exception as e:
        logger.error(f"Failed to save trade setup: {e}", exc_info=True)


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
                max_value=20.0,
                value=5.0,
                step=0.1,
                key="crypto_bulk_target_pct"
            )
        
        # Analysis section
        st.divider()
        st.markdown("#### 🤖 Analyze Selected Pairs")
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col1:
            if st.button("🔍 Analyze All Selected Pairs", use_container_width=True, type="primary"):
                analyze_bulk_pairs(kraken_client, selected_pairs, direction, position_size, risk_pct, take_profit_pct, "bulk_custom")
        
        with analysis_col2:
            if st.button("📊 Get Market Data", use_container_width=True):
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
        if analysis_key in st.session_state and st.session_state[analysis_key]:
            all_analysis_results = st.session_state[analysis_key]
            
            # Filter to only show results for currently selected pairs
            analysis_results = [r for r in all_analysis_results if r.get('pair') in selected_pairs]
            
            if analysis_results:
                st.divider()
                st.markdown("#### 📈 Analysis Results")
                
                # Summary metrics
                total_investment = sum(a.get('position_size', 0) for a in analysis_results)
                avg_rr = sum(a.get('risk_reward_ratio', 0) for a in analysis_results) / len(analysis_results) if analysis_results else 0
                
                metric_cols = st.columns(4)
                metric_cols[0].metric("Pairs Analyzed", len(analysis_results))
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
                    st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        
        # Preview trades
        st.divider()
        st.markdown(f"#### 📋 Trade Preview ({len(selected_pairs)} pairs)")
        
        # Check if execution is in progress
        execution_key = f"crypto_bulk_execution_{hash(tuple(sorted(selected_pairs)))}_{direction}_{position_size}"
        is_executing = st.session_state.get(execution_key, False)
        
        if st.button("🚀 Execute Bulk Trades", type="primary", use_container_width=True, disabled=is_executing):
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
        logger.error(f"Bulk custom trade error: {e}", exc_info=True)


def display_bulk_watchlist_trade(kraken_client: KrakenClient, crypto_config, watchlist_manager):
    """
    Display bulk trade execution form for watchlist
    """
    st.markdown("#### ⭐ Bulk Watchlist Trading")
    
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
            default=watchlist_symbols[:10] if len(watchlist_symbols) > 10 else watchlist_symbols,
            key="crypto_bulk_watchlist_pairs",
            help="Select which watchlist symbols to trade (default: first 10)"
        )
        
        if not selected_symbols:
            st.info("👆 Select symbols from your watchlist above to configure bulk trades")
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
                key="crypto_bulk_watchlist_direction"
            )
            
            position_size = st.number_input(
                "Position Size per Symbol (USD)",
                min_value=1.0,
                max_value=10000.0,
                value=100.0,
                step=1.0,
                key="crypto_bulk_watchlist_position_size"
            )
        
        with col2:
            risk_pct = st.number_input(
                "Risk % per Trade",
                min_value=0.1,
                max_value=10.0,
                value=2.0,
                step=0.1,
                key="crypto_bulk_watchlist_risk_pct"
            )
            
            take_profit_pct = st.number_input(
                "Take Profit %",
                min_value=0.1,
                max_value=20.0,
                value=5.0,
                step=0.1,
                key="crypto_bulk_watchlist_target_pct"
            )
        
        # Analysis section
        st.divider()
        st.markdown("#### 🤖 Analyze Selected Symbols")
        
        analysis_col1, analysis_col2 = st.columns([2, 1])
        
        with analysis_col1:
            if st.button("🔍 Analyze All Selected Symbols", use_container_width=True, type="primary"):
                analyze_bulk_pairs(kraken_client, selected_symbols, direction, position_size, risk_pct, take_profit_pct, "bulk_watchlist")
        
        with analysis_col2:
            if st.button("📊 Get Market Data", use_container_width=True):
                with st.spinner("Fetching market data..."):
                    try:
                        # Show market data for first selected symbol as example
                        if selected_symbols:
                            ticker_info = kraken_client.get_ticker_info(selected_symbols[0])
                            st.json(ticker_info)
                    except Exception as e:
                        st.error(f"Failed to fetch market data: {e}")
        
        # Display analysis results if available (filtered to currently selected symbols)
        analysis_key = "crypto_bulk_watchlist_analysis"
        if analysis_key in st.session_state and st.session_state[analysis_key]:
            all_analysis_results = st.session_state[analysis_key]
            
            # Filter to only show results for currently selected symbols
            analysis_results = [r for r in all_analysis_results if r.get('pair') in selected_symbols]
            
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
                    st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        
        # Preview trades
        st.divider()
        st.markdown(f"#### 📋 Trade Preview ({len(selected_symbols)} symbols)")
        
        # Check if execution is in progress
        execution_key = f"crypto_bulk_execution_{hash(tuple(sorted(selected_symbols)))}_{direction}_{position_size}"
        is_executing = st.session_state.get(execution_key, False)
        
        if st.button("🚀 Execute Bulk Watchlist Trades", type="primary", use_container_width=True, disabled=is_executing):
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
        logger.error(f"Bulk watchlist trade error: {e}", exc_info=True)


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
    Analyze multiple pairs and store results in session state
    
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
                        'risk_reward_ratio': 0.0
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
                    'take_profit_pct': take_profit_pct
                })
                
            except Exception as e:
                logger.error(f"Error analyzing {pair}: {e}", exc_info=True)
                results.append({
                    'pair': pair,
                    'status': 'FAILED',
                    'error': str(e),
                    'current_price': 0.0,
                    'stop_loss': 0.0,
                    'take_profit': 0.0,
                    'position_size': 0.0,
                    'quantity': 0.0,
                    'risk_reward_ratio': 0.0
                })
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Store results in session state
        if analysis_type == "bulk_custom":
            st.session_state.crypto_bulk_custom_analysis = results
        elif analysis_type == "bulk_watchlist":
            st.session_state.crypto_bulk_watchlist_analysis = results
        
        # Show success message
        successful = [r for r in results if r['status'] == 'SUCCESS']
        if successful:
            st.success(f"✅ Successfully analyzed {len(successful)} out of {len(pairs)} pairs")
        else:
            st.warning(f"⚠️ Could not analyze any pairs. Please check your selections.")
        
        # Rerun to display results
        st.rerun()
        
    except Exception as e:
        st.error(f"Bulk analysis error: {e}")
        logger.error(f"Bulk analysis error: {e}", exc_info=True)


def execute_bulk_trades(
    kraken_client: KrakenClient,
    pairs: List[str],
    direction: str,
    position_size: float,
    risk_pct: float,
    take_profit_pct: float
):
    """
    Execute bulk trades for multiple pairs
    
    Args:
        kraken_client: KrakenClient instance
        pairs: List of trading pair symbols
        direction: "BUY" or "SELL"
        position_size: Position size in USD per pair
        risk_pct: Risk percentage per trade
        take_profit_pct: Take profit percentage
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
                
                # Place the order with stop loss and take profit
                result = kraken_client.place_order(
                    pair=pair,
                    side=order_side,
                    order_type=OrderType.MARKET,
                    volume=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit
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
                        logger.error(f"❌ Failed to send Discord alert for {pair}: {error_msg}", exc_info=True)
                else:
                    # Order failed
                    results.append({
                        'pair': pair,
                        'status': 'FAILED',
                        'error': 'Order placement failed - check logs for details'
                    })
                    
            except Exception as e:
                logger.error(f"Error executing trade for {pair}: {e}", exc_info=True)
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
        col2.metric("Successful", len(successful), delta=f"{len(successful)/len(results)*100:.1f}%")
        col3.metric("Failed", len(failed), delta=f"-{len(failed)/len(results)*100:.1f}%")
        
        # Detailed results table
        if successful:
            st.markdown("##### ✅ Successful Trades")
            success_df = pd.DataFrame(successful)
            st.dataframe(success_df, use_container_width=True, hide_index=True)
        
        if failed:
            st.markdown("##### ❌ Failed Trades")
            failed_df = pd.DataFrame(failed)
            st.dataframe(failed_df, use_container_width=True, hide_index=True)
        
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
                if st.button("📊 Check Recent Orders", use_container_width=True):
                    verify_recent_orders(kraken_client, [r['order_id'] for r in successful])
            
            with verify_col2:
                if st.button("💰 Check Positions", use_container_width=True):
                    verify_positions(kraken_client, [r['pair'] for r in successful])
        
    except Exception as e:
        st.error(f"Bulk trade execution error: {e}")
        logger.error(f"Bulk trade execution error: {e}", exc_info=True)


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
                st.dataframe(found_df, use_container_width=True, hide_index=True)
            
            if missing_orders:
                st.warning(f"⚠️ {len(missing_orders)} order(s) not found in closed orders:")
                for order_id in missing_orders:
                    st.text(f"  - {order_id}")
                st.info("💡 These orders may have failed, been rejected, or are still processing. Check your Kraken account directly.")
            
            if not found_orders and not missing_orders:
                st.info("ℹ️ No matching orders found. They may still be processing or may have failed.")
                
    except Exception as e:
        st.error(f"Error verifying orders: {e}")
        logger.error(f"Order verification error: {e}", exc_info=True)


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
                st.dataframe(positions_df, use_container_width=True, hide_index=True)
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
        logger.error(f"Position verification error: {e}", exc_info=True)

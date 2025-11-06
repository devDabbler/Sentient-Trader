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
                    if scan_type == "penny_crypto_scanner" and scanner_instances.get(scan_type):
                        results = scanner_instances[scan_type].scan_penny_cryptos(max_price=1.0, top_n=50)
                        st.session_state.scan_results = [{"Ticker": r.symbol, "Price": r.current_price, "Change": r.change_pct_24h, "Score": r.runner_potential_score} for r in results]
                    
                    elif scan_type == "crypto_opportunity_scanner" and scanner_instances.get(scan_type):
                        opps = scanner_instances[scan_type].scan_opportunities(top_n=50)
                        st.session_state.scan_results = [{"Ticker": o.symbol, "Price": o.current_price, "Change": o.change_pct_24h, "Score": o.score} for o in opps]

                    elif scan_type == "ai_crypto_scanner" and scanner_instances.get(scan_type):
                        opps = scanner_instances[scan_type].scan_with_ai_confidence(top_n=50)
                        st.session_state.scan_results = [{"Ticker": o.symbol, "Price": o.current_price, "Change": o.change_pct_24h, "Score": o.score} for o in opps]

                    elif scan_type == "sub_penny_discovery" and scanner_instances.get(scan_type):
                        runners = asyncio.run(scanner_instances[scan_type].discover_sub_penny_runners(
                            max_price=0.01,
                            min_market_cap=0,
                            max_market_cap=10_000_000,  # 10M to match main app and allow more coins
                            top_n=50,
                            sort_by="runner_potential"
                        ))
                        
                        # Filter out invalid Kraken pairs before displaying
                        valid_results = []
                        invalid_count = 0
                        
                        for r in runners:
                            symbol = r.symbol.upper()
                            # Try common Kraken pair formats
                            possible_pairs = [
                                f"{symbol}/USD",
                                f"{symbol}USD",
                                f"{symbol}/USDT",
                                f"{symbol}USDT"
                            ]
                            
                            # Check if any format is valid on Kraken
                            is_valid = False
                            valid_pair = None
                            for pair in possible_pairs:
                                try:
                                    test_info = kraken_client.get_ticker_info(pair)
                                    if test_info and 'c' in test_info:
                                        is_valid = True
                                        valid_pair = pair
                                        break
                                except Exception:
                                    continue
                            
                            if is_valid and valid_pair:
                                valid_results.append({
                                    "Ticker": valid_pair,
                                    "Price": r.price_usd,
                                    "Change": r.change_24h,
                                    "Score": r.runner_potential_score
                                })
                            else:
                                invalid_count += 1
                        
                        st.session_state.scan_results = valid_results
                        
                        if invalid_count > 0:
                            st.info(f"ℹ️ Filtered out {invalid_count} coins not available on Kraken (e.g., {runners[0].symbol.upper() if runners else 'N/A'})")
                    
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
    sub_penny_discovery=None
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
        display_trade_setup(kraken_client, crypto_config)


def display_trade_setup(kraken_client: KrakenClient, crypto_config):
    """
    Display the trade execution form with AI analysis
    """
    st.markdown("### ⚡ Execute Trade")
    
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
            if st.button("🚀 Execute Trade", use_container_width=True, type="primary"):
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
            
            # Place the order
            success, result = kraken_client.place_order(
                pair=analysis['pair'],
                side=order_side,
                order_type=OrderType.MARKET,
                volume=quantity
            )
            
            if success:
                st.success(f"✅ Trade executed successfully!")
                st.json(result)
                
                # Send Discord notification if configured
                try:
                    alert = TradingAlert(
                        ticker=analysis['pair'],
                        alert_type=AlertType.TRADE_EXECUTED,
                        message=f"{analysis['direction']} order executed at ${analysis['current_price']:.4f}",
                        priority=AlertPriority.MEDIUM
                    )
                    send_discord_alert(alert)
                except Exception as e:
                    logger.warning(f"Failed to send Discord alert: {e}")
                    
            else:
                st.error(f"❌ Trade failed: {result}")
                
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

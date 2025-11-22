"""
Auto-Trader Tab
Automated trading system with strategy management and monitoring

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple
import os
import time
from datetime import datetime
import pandas as pd
import json

from ui.tabs.common_imports import (
    get_advanced_scanner,
    get_ai_scanner,
    get_ml_scanner,
    get_ticker_manager,
    ComprehensiveAnalyzer,
    get_trading_mode_manager
)

# Import the exact types needed by scan_opportunities method
from services.advanced_opportunity_scanner import ScanFilters, ScanType



def _run_stock_analysis(tickers: List[str], styles: List[str]):
    with st.spinner(f"Analyzing {len(tickers)} tickers across {len(styles)} strategies..."):
        analysis_results = []
        total_steps = len(tickers) * len(styles)
        current_step = 0
        progress_bar = st.progress(0)
        
        for ticker in tickers:
            for style in styles:
                current_step += 1
                progress_bar.progress(current_step / total_steps)
                
                try:
                    result = ComprehensiveAnalyzer.analyze_stock(ticker, style)
                    if result:
                        # Logic to map result to UI format
                        signal = "HOLD"
                        if result.confidence_score >= 70: signal = "BUY"
                        elif result.confidence_score <= 30: signal = "SELL"
                        
                        analysis_results.append({
                            "Ticker": ticker,
                            "Style": style,
                            "Price": result.price,
                            "Signal": signal,
                            "Confidence": result.confidence_score,
                            "RSI": result.rsi,
                            "Trend": result.trend,
                            "Target": result.price * 1.04, # Mockup target
                            "IV_Rank": result.iv_rank,
                            "Recommendation": result.recommendation
                        })
                except Exception as e:
                    logger.error(f"Analysis error for {ticker} ({style}): {e}")
        
        st.session_state.stock_analysis_results = analysis_results
        st.success(f"Analysis Complete! Generated {len(analysis_results)} insights.")


def render_tab():
    """Main render function called from app.py"""
    st.header("🤖 Auto-Trader & Smart Scanner")
    
    # Navigation Tabs
    tab_smart, tab_bot = st.tabs(["🔍 Smart Scanner & Trade", "⚙️ Background Bot Config"])
    
    with tab_smart:
        render_smart_scanner_ui()
        
    with tab_bot:
        render_background_bot_ui()

def render_smart_scanner_ui():
    st.markdown("### 🔍 Smart Stock Scanner & Analyzer")
    st.markdown("**Workflow:** Scan → Bulk Select → Analyze All → Pick Best → Execute")

    # --- 1. Scan for Opportunities ---
    st.markdown("### 1️⃣ SCAN FOR OPPORTUNITIES")
    st.caption("Choose a scanner or manually enter tickers to get started")
    
    scanner_map = {
        "✍️ Manual Selection": "manual",
        "⭐ My Watchlist": "watchlist",
        "🚀 Top Gainers": "top_gainers",
        "📈 Options Plays": "options",
        "💰 Penny Stocks (<$5)": "penny_stocks",
        "🔥 Buzzing Stocks": "buzzing",
        "💥 Breakouts": "breakouts"
    }

    scanner_col1, scanner_col2 = st.columns([2, 1])
    with scanner_col1:
        if 'stock_scanner_type' not in st.session_state:
            st.session_state.stock_scanner_type = list(scanner_map.keys())[0]
        
        scan_type_display = st.selectbox(
            "Scanner Type",
            options=list(scanner_map.keys()),
            index=list(scanner_map.keys()).index(st.session_state.stock_scanner_type) if st.session_state.stock_scanner_type in scanner_map.keys() else 0,
            key='stock_scanner_type_selectbox'
        )
        st.session_state.stock_scanner_type = scan_type_display
        scan_type = scanner_map[scan_type_display]

    with scanner_col2:
        st.write("")
        st.write("")
        
        if scan_type == "manual":
            button_label = "✍️ Add Manually"
            button_type = "secondary"
        else:
            button_label = "🚀 Scan"
            button_type = "primary"
        
        if st.button(button_label, width='stretch', type=button_type, key="stock_scan_btn"):
            if scan_type == "manual":
                st.session_state.stock_manual_mode = True
                st.info("💡 Use the text input below to manually add tickers")
            else:
                st.session_state.stock_manual_mode = False
                st.session_state.stock_scan_results = []
                st.session_state.stock_selected_tickers = []
                
                with st.spinner(f"Scanning for {scan_type_display}..."):
                    try:
                        scanner = get_advanced_scanner()
                        results = []
                        
                        if scan_type == "watchlist":
                            tm = get_ticker_manager()
                            if tm:
                                watchlist = tm.get_all_tickers()
                                tickers = [item['ticker'] for item in watchlist]
                                for ticker in tickers:
                                    # Simple fetch for watchlist
                                    try:
                                        analysis = ComprehensiveAnalyzer.analyze_stock(ticker)
                                        if analysis:
                                            results.append({
                                                "Ticker": ticker,
                                                "Price": analysis.price,
                                                "Change": analysis.change_pct,
                                                "Score": 50  # Default score
                                            })
                                    except Exception:
                                        pass
                        else:
                            # Map to AdvancedOpportunityScanner types
                            adv_scan_type = {
                                "top_gainers": ScanType.MOMENTUM, # Approximate
                                "options": ScanType.OPTIONS,
                                "penny_stocks": ScanType.PENNY_STOCKS,
                                "buzzing": ScanType.BUZZING,
                                "breakouts": ScanType.BREAKOUTS
                            }.get(scan_type, ScanType.ALL)
                            
                            # Run scanner
                            filters = ScanFilters() # Default filters
                            if scan_type == "top_gainers":
                                filters.min_change_pct = 3.0
                            
                            if scanner:
                                opps = scanner.scan_opportunities(scan_type=adv_scan_type, filters=filters, top_n=20)
                                
                                for opp in opps:
                                    results.append({
                                        "Ticker": opp.ticker,
                                        "Price": opp.price,
                                        "Change": opp.change_pct,
                                        "Score": opp.score
                                    })
                        
                        st.session_state.stock_scan_results = results
                        if results:
                            st.success(f"Found {len(results)} tickers!")
                        else:
                            st.warning("No results found.")
                            
                    except Exception as e:
                        st.error(f"Scan failed: {e}")
                        logger.error(f"Stock scan error: {e}")

    # --- Manual Selection ---
    if st.session_state.get('stock_manual_mode', False) or scan_type == "manual":
        st.markdown("---")
        st.markdown("#### ✍️ Manual Selection")
        
        manual_col1, manual_col2 = st.columns([3, 1])
        with manual_col1:
            manual_input = st.text_area(
                "Enter symbols (comma separated)",
                value="",
                height=100,
                key='stock_manual_input'
            )
        with manual_col2:
            st.write("")
            st.write("")
            if st.button("✅ Add", width='stretch'):
                tickers = [t.strip().upper() for t in manual_input.split(',') if t.strip()]
                current_results = st.session_state.get('stock_scan_results', [])
                
                for ticker in tickers:
                    # Fetch basic data
                    try:
                        analysis = ComprehensiveAnalyzer.analyze_stock(ticker)
                        if analysis:
                            # Check if already exists
                            if not any(r['Ticker'] == ticker for r in current_results):
                                current_results.append({
                                    "Ticker": ticker,
                                    "Price": analysis.price,
                                    "Change": analysis.change_pct,
                                    "Score": 50
                                })
                    except:
                        pass
                
                st.session_state.stock_scan_results = current_results
                st.success(f"Added {len(tickers)} tickers")

    # --- 2. Select Tickers ---
    if st.session_state.get('stock_scan_results'):
        st.markdown("---")
        st.markdown("### 2️⃣ SELECT TICKERS FOR ANALYSIS")
        
        df = pd.DataFrame(st.session_state.stock_scan_results)
        
        # Selection buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("✅ SELECT ALL", use_container_width=True, key="sel_all_stock"):
                st.session_state.stock_selected_tickers = df['Ticker'].tolist()
        with btn_col2:
            if st.button("❌ CLEAR ALL", use_container_width=True, key="clear_all_stock"):
                st.session_state.stock_selected_tickers = []
        with btn_col3:
            st.metric("Selected", len(st.session_state.get('stock_selected_tickers', [])))

        # Display DataFrame with selection
        if 'stock_selected_tickers' not in st.session_state:
            st.session_state.stock_selected_tickers = []

        selected = st.multiselect(
            "Select tickers to analyze",
            options=df['Ticker'].tolist(),
            default=st.session_state.stock_selected_tickers,
            key='stock_ticker_multiselect'
        )
        # Sync multiselect with session state
        if selected != st.session_state.stock_selected_tickers:
            st.session_state.stock_selected_tickers = selected
        
        st.dataframe(df, use_container_width=True)

    # --- 3. Analyze Selected ---
    selected_tickers = st.session_state.get('stock_selected_tickers', [])
    if selected_tickers:
        st.markdown("---")
        st.markdown("### 3️⃣ ANALYZE SELECTED TICKERS")
        
        st.info(f"Ready to analyze **{len(selected_tickers)}** tickers")
        
        # Analysis Mode Selection
        if 'stock_analysis_mode' not in st.session_state:
            st.session_state.stock_analysis_mode = 'standard'

        mode_col1, mode_col2, mode_col3 = st.columns(3)
        with mode_col1:
            if st.button("🔬 STANDARD", type="primary" if st.session_state.stock_analysis_mode == 'standard' else "secondary", use_container_width=True, key="mode_std"):
                st.session_state.stock_analysis_mode = 'standard'
                st.rerun()
        with mode_col2:
            if st.button("🎯 MULTI-CONFIG", type="primary" if st.session_state.stock_analysis_mode == 'multi' else "secondary", use_container_width=True, key="mode_multi"):
                st.session_state.stock_analysis_mode = 'multi'
                st.rerun()
        with mode_col3:
            if st.button("🚀 ULTIMATE", type="primary" if st.session_state.stock_analysis_mode == 'ultimate' else "secondary", use_container_width=True, key="mode_ult"):
                st.session_state.stock_analysis_mode = 'ultimate'
                st.rerun()

        # Configuration & Execution
        if st.session_state.stock_analysis_mode == 'standard':
            col_style, col_btn = st.columns([2, 1])
            with col_style:
                trading_style = st.selectbox(
                    "Trading Style",
                    options=["SCALP", "DAY_TRADE", "SWING_TRADE", "OPTIONS", "BUY_HOLD"],
                    format_func=lambda x: x.replace('_', ' ').title(),
                    key="std_trading_style"
                )
            
            with col_btn:
                st.write("")
                st.write("")
                if st.button("🚀 Analyze All", type="primary", width='stretch', key="analyze_stock_btn"):
                    _run_stock_analysis(selected_tickers, [trading_style])

        elif st.session_state.stock_analysis_mode == 'multi':
            st.caption("Test multiple trading styles simultaneously")
            styles = st.multiselect(
                "Styles to Test",
                options=["SCALP", "DAY_TRADE", "SWING_TRADE", "OPTIONS", "BUY_HOLD"],
                default=["DAY_TRADE", "SWING_TRADE"],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="multi_styles"
            )
            
            if st.button("🚀 Run Multi-Config Analysis", type="primary", key="analyze_multi_btn"):
                if styles:
                    _run_stock_analysis(selected_tickers, styles)
                else:
                    st.error("Select at least one style")

        elif st.session_state.stock_analysis_mode == 'ultimate':
            st.caption("Run ALL analysis types + AI Validation")
            st.info("🔥 Will analyze: Scalp, Day Trade, Swing, Options, Buy & Hold for ALL tickers")
            
            if st.button("🔥 START ULTIMATE ANALYSIS", type="primary", key="analyze_ult_btn"):
                all_styles = ["SCALP", "DAY_TRADE", "SWING_TRADE", "OPTIONS", "BUY_HOLD"]
                _run_stock_analysis(selected_tickers, all_styles)
        
    # --- 4. Results & Execute ---
    if st.session_state.get('stock_analysis_results'):
        st.markdown("---")
        st.markdown("### 4️⃣ RANKED OPPORTUNITIES")
        
        results = st.session_state.stock_analysis_results
        # Sort by confidence
        results.sort(key=lambda x: x['Confidence'], reverse=True)
        
        for res in results:
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 2])
                
                c1.markdown(f"### {res['Ticker']}")
                c1.caption(f"Strategy: {res['Style']}")
                
                signal_color = "green" if res['Signal'] == "BUY" else "red" if res['Signal'] == "SELL" else "grey"
                c2.markdown(f"**Signal:** :{signal_color}[{res['Signal']}]")
                c2.metric("Confidence", f"{res['Confidence']}%")
                
                c3.metric("Price", f"${res['Price']:.2f}")
                
                # Show Style-specific metrics
                if res['Style'] == "OPTIONS":
                     c4.metric("IV Rank", f"{res.get('IV_Rank', 'N/A')}")
                else:
                     c4.metric("Target", f"${res['Target']:.2f}")
                
                with c5:
                    st.write("")
                    if st.button("⚡ Use Setup", key=f"use_setup_{res['Ticker']}_{res['Style']}"):
                        st.session_state['stock_prefill_symbol'] = res['Ticker']
                        st.session_state['ex_symbol'] = res['Ticker']
                        
                        # Determine correct action based on strategy and recommendation
                        action_val = "buy"
                        
                        if res['Style'] == "OPTIONS":
                            # Check recommendation for "Sell" or "Credit" or "Write"
                            rec_text = res.get('Recommendation', '').lower()
                            if any(x in rec_text for x in ['sell', 'write', 'credit', 'short']):
                                action_val = "sell_to_open"
                            else:
                                action_val = "buy_to_open"
                        elif res['Style'] == "BUY_HOLD":
                            # Buy & Hold usually implies long-only or exit
                            if res['Signal'] == "BUY":
                                action_val = "buy"
                            elif res['Signal'] == "SELL":
                                action_val = "sell"
                        else:
                            # Active Trading (Scalp, Day, Swing, Warrior, etc.)
                            # SELL signal implies Short Entry
                            if res['Signal'] == "BUY":
                                action_val = "buy"
                            elif res['Signal'] == "SELL":
                                action_val = "sell_short"
                        
                        st.session_state['stock_prefill_action'] = action_val
                        st.session_state['ex_action'] = action_val
                        
                        st.session_state['stock_prefill_qty'] = 10
                        st.session_state['ex_qty'] = 10
                        
                        st.success(f"Populated Order Form with {action_val}!")
                
                # Show Recommendation text in expander
                with st.expander("📋 Analysis Details"):
                    st.text(res.get('Recommendation', 'No details available'))
                
                st.divider()



    # --- 5. Execution (Simple Form) ---
    st.markdown("### ⚡ EXECUTE TRADE")
    
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        # Initialize session state default if not present
        if 'ex_symbol' not in st.session_state:
            st.session_state.ex_symbol = st.session_state.get('stock_prefill_symbol', '')
        symbol = st.text_input("Symbol", key="ex_symbol")
        
    with ex_col2:
        action_options = ["buy", "sell", "sell_short", "buy_to_cover", "buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"]
        # Initialize session state default if not present
        if 'ex_action' not in st.session_state:
             # Check prefill or default to 'buy'
             prefill = st.session_state.get('stock_prefill_action', 'buy')
             if prefill in action_options:
                 st.session_state.ex_action = prefill
             else:
                 # Fallback if prefill is not in options (e.g. sell_short in paper?)
                 st.session_state.ex_action = 'buy'
        
        action = st.selectbox("Action", action_options, key="ex_action")
        
    with ex_col3:
        if 'ex_qty' not in st.session_state:
            st.session_state.ex_qty = st.session_state.get('stock_prefill_qty', 10)
        qty = st.number_input("Quantity", min_value=1, key="ex_qty")
    
    if st.button("🚀 Submit Order", type="primary", key="ex_submit"):
        if 'tradier_client' in st.session_state and st.session_state.tradier_client:
            with st.spinner("Placing order..."):
                try:
                    # Use tradier client to place order
                    client = st.session_state.tradier_client
                    side = action
                    
                    # Map detailed actions to simple side if needed, or use as is if client supports it
                    # Tradier client usually expects 'buy', 'sell', 'buy_to_open', etc.
                    
                    order_response = client.place_equity_order(symbol, side, qty, "market") # Market order for simplicity
                    if order_response:
                        st.success(f"Order Submitted: {order_response}")
                    else:
                        st.error("Order failed")
                except Exception as e:
                    st.error(f"Order failed: {e}")
        else:
            st.warning("Tradier client not connected. Please connect in Tradier tab.")
            st.info(f"Simulated Order: {action.upper()} {qty} {symbol}")


def render_background_bot_ui():
    """Background Bot Configuration"""
    st.header("Auto-Trader")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("🤖 Automated Trading Bot")
    st.write("Set up automated trading that monitors your watchlist and executes high-confidence signals.")
    
    st.warning("⚠️ **IMPORTANT**: Start with Paper Trading mode to test before using real money!")
    
    # Initialize auto-trader in session state
    if 'auto_trader' not in st.session_state:
        st.session_state.auto_trader = None
    
    # ========================================================================
    # BACKGROUND TRADER CONFIGURATION MANAGER - DYNAMIC STRATEGY SYSTEM
    # ========================================================================
    
    st.divider()
    st.subheader("⚙️ Dynamic Strategy Configuration")
    st.write("Select a strategy, modify its settings, and save to its specific config file. The background trader will automatically use your selection.")
    
    # Helper functions for .env file management
    def update_env_file_for_paper_trading():
        """Update .env file to set paper trading mode"""
        try:
            env_file = '.env'
            if not os.path.exists(env_file):
                logger.warning(f".env file not found at {env_file}")
                return False
            
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace paper trading settings
            content = content.replace('IS_PAPER_TRADING=False', 'IS_PAPER_TRADING=True')
            content = content.replace('PAPER_TRADING_MODE=False', 'PAPER_TRADING_MODE=True')
            
            # Ensure settings exist if they don't
            if 'IS_PAPER_TRADING=' not in content:
                content += '\nIS_PAPER_TRADING=True\n'
            if 'PAPER_TRADING_MODE=' not in content:
                content += '\nPAPER_TRADING_MODE=True\n'
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            logger.info("✅ Updated .env file for paper trading")
            return True
        except Exception as e:
            logger.error(f"Error updating .env file for paper trading: {e}")
            return False
    
    def update_env_file_for_live_trading():
        """Update .env file to set live trading mode"""
        try:
            env_file = '.env'
            if not os.path.exists(env_file):
                logger.warning(f".env file not found at {env_file}")
                return False
            
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace live trading settings
            content = content.replace('IS_PAPER_TRADING=True', 'IS_PAPER_TRADING=False')
            content = content.replace('PAPER_TRADING_MODE=True', 'PAPER_TRADING_MODE=False')
            
            # Ensure settings exist if they don't
            if 'IS_PAPER_TRADING=' not in content:
                content += '\nIS_PAPER_TRADING=False\n'
            if 'PAPER_TRADING_MODE=' not in content:
                content += '\nPAPER_TRADING_MODE=False\n'
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            logger.info("✅ Updated .env file for live trading")
            return True
        except Exception as e:
            logger.error(f"Error updating .env file for live trading: {e}")
            return False
    
    # Load active strategy selector
    def load_active_strategy():
        """Load the currently active strategy from active_strategy.json"""
        import json  # Local import to avoid closure issues
        try:
            with open('active_strategy.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Create default if not exists
            default_strategy = {
                "active_strategy": "GENERAL_TRADING",
                "config_file": "config_background_trader.py",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "available_strategies": {
                    "WARRIOR_SCALPING": {
                        "name": "Warrior Scalping",
                        "config_file": "config_warrior_scalping.py",
                        "description": "Gap & Go strategy (9:30-10:00 AM, $2-$20 stocks, 2-20% gaps)",
                        "trading_mode": "WARRIOR_SCALPING"
                    },
                    "GENERAL_TRADING": {
                        "name": "General Trading",
                        "config_file": "config_background_trader.py",
                        "description": "Standard scalping, stocks, or options trading",
                        "trading_mode": "SCALPING"
                    },
                    "OPTIONS_PREMIUM": {
                        "name": "Options Premium Selling",
                        "config_file": "config_options_premium.py",
                        "description": "Wheel strategy, credit spreads, iron condors",
                        "trading_mode": "OPTIONS"
                    },
                    "SWING_TRADING": {
                        "name": "Swing Trading",
                        "config_file": "config_swing_trader.py",
                        "description": "Medium-term positions (1-5 days)",
                        "trading_mode": "SWING_TRADE"
                    }
                }
            }
            with open('active_strategy.json', 'w') as f:
                json.dump(default_strategy, f, indent=2)
            return default_strategy
    
    def save_active_strategy(strategy_key):
        """Save the active strategy selection"""
        import json  # Local import to avoid closure issues
        try:
            logger.info(f"💾 Attempting to save active strategy: {strategy_key}")
            strategy_config = load_active_strategy()
            
            if strategy_key not in strategy_config['available_strategies']:
                logger.error("❌ Strategy key '{}' not found in available strategies", str(strategy_key))
                st.error(f"Strategy '{strategy_key}' not found!")
                return False
            
            # Update strategy config
            strategy_config['active_strategy'] = strategy_key
            strategy_config['config_file'] = strategy_config['available_strategies'][strategy_key]['config_file']
            strategy_config['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info("📝 Updating active_strategy.json: strategy={}, config_file={strategy_config['config_file']}", str(strategy_key))
            
            # Write to file with explicit flush and error handling
            file_path = 'active_strategy.json'
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(strategy_config, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Verify the write worked
                import time
                time.sleep(0.1)  # Brief pause to ensure file system sync
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    verification = json.load(f)
                
                if verification.get('active_strategy') == strategy_key:
                    logger.info(f"✅ Successfully saved and verified active strategy: {strategy_key}")
                    logger.info(f"   Config file: {verification.get('config_file')}")
                    return True
                else:
                    logger.error("❌ Verification failed! Saved '{strategy_key}' but file shows '{}'", str(verification.get('active_strategy')))
                    st.error(f"⚠️ Saved {strategy_key} but verification failed! File may be locked.")
                    return False
                    
            except PermissionError as pe:
                logger.error(f"❌ Permission denied writing to {file_path}: {pe}")
                st.error(f"⚠️ Permission denied! Make sure {file_path} is not open in another program.")
                return False
            except Exception as write_error:
                logger.error("❌ Error writing to file: {}", str(write_error), exc_info=True)
                st.error(f"Error writing to file: {write_error}")
                return False
                
        except Exception as e:
            logger.error("❌ Error saving active strategy: {}", str(e), exc_info=True)
            st.error(f"Error saving active strategy: {e}")
        return False
    
    # Helper functions for config file management
    def load_config_file(config_filename, force_reload=True):
        """Load settings from any config file dynamically (with broker-specific support)"""
        try:
            # Use broker-specific config loader
            from utils.config_loader import load_config_module
            cfg = load_config_module(config_filename, force_reload=force_reload)
            
            return {
                'trading_mode': getattr(cfg, 'TRADING_MODE', 'SCALPING'),
                'scan_interval': getattr(cfg, 'SCAN_INTERVAL_MINUTES', 15),
                'min_confidence': getattr(cfg, 'MIN_CONFIDENCE', 70),
                'max_daily_orders': getattr(cfg, 'MAX_DAILY_ORDERS', 10),
                'max_position_size_pct': getattr(cfg, 'MAX_POSITION_SIZE_PCT', 15.0),
                'use_bracket_orders': getattr(cfg, 'USE_BRACKET_ORDERS', True),
                'scalping_take_profit_pct': getattr(cfg, 'SCALPING_TAKE_PROFIT_PCT', 2.0),
                'scalping_stop_loss_pct': getattr(cfg, 'SCALPING_STOP_LOSS_PCT', 1.0),
                'risk_per_trade_pct': getattr(cfg, 'RISK_PER_TRADE_PCT', 0.02),
                'max_daily_loss_pct': getattr(cfg, 'MAX_DAILY_LOSS_PCT', 0.04),
                'use_smart_scanner': getattr(cfg, 'USE_SMART_SCANNER', False),
                'watchlist': getattr(cfg, 'WATCHLIST', ['SPY', 'QQQ', 'AAPL']),
                'allow_short_selling': getattr(cfg, 'ALLOW_SHORT_SELLING', False),
                'use_settled_funds_only': getattr(cfg, 'USE_SETTLED_FUNDS_ONLY', True),
                # Capital Management
                'total_capital': getattr(cfg, 'TOTAL_CAPITAL', 10000.0),
                'reserve_cash_pct': getattr(cfg, 'RESERVE_CASH_PCT', 10.0),
                'max_capital_utilization_pct': getattr(cfg, 'MAX_CAPITAL_UTILIZATION_PCT', 80.0),
                # AI-Powered Hybrid Mode (NEW)
                'use_ml_enhanced_scanner': getattr(cfg, 'USE_ML_ENHANCED_SCANNER', True),
                'use_ai_validation': getattr(cfg, 'USE_AI_VALIDATION', True),
                'min_ensemble_score': getattr(cfg, 'MIN_ENSEMBLE_SCORE', 70.0),
                'min_ai_validation_confidence': getattr(cfg, 'MIN_AI_VALIDATION_CONFIDENCE', 0.7),
                # Fractional Shares (IBKR Only)
                'use_fractional_shares': getattr(cfg, 'USE_FRACTIONAL_SHARES', False),
                'fractional_price_threshold': getattr(cfg, 'FRACTIONAL_PRICE_THRESHOLD', 100.0),
                'fractional_min_amount': getattr(cfg, 'FRACTIONAL_MIN_AMOUNT', 50.0),
                'fractional_max_amount': getattr(cfg, 'FRACTIONAL_MAX_AMOUNT', 1000.0),
                # Extra fields for special strategies
                'config_filename': config_filename,
            }
        except Exception as e:
            st.error(f"Error loading {config_filename}: {e}")
            return None
    
    def save_config_to_file(config_dict, config_filename):
        """Save settings to any config file dynamically"""
        try:
            # Determine which template to use based on filename
            module_name = config_filename.replace('.py', '')
            # Read the template
            config_content = f'''"""
Configuration for Background Auto-Trader
Customize your trading bot settings here
Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

# ==============================================================================
# TRADING CONFIGURATION
# ==============================================================================

# Trading Mode: "SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"
TRADING_MODE = "{config_dict['trading_mode']}"

# Scan Interval (minutes)
SCAN_INTERVAL_MINUTES = {config_dict['scan_interval']}

# Minimum Confidence % (only execute signals above this)
MIN_CONFIDENCE = {config_dict['min_confidence']}

# ==============================================================================
# CAPITAL MANAGEMENT
# ==============================================================================

# Total capital allocated to auto-trading
TOTAL_CAPITAL = {config_dict['total_capital']}  # ${config_dict['total_capital']:,.0f}

# Reserve cash percentage (kept aside, not used for trading)
RESERVE_CASH_PCT = {config_dict['reserve_cash_pct']}  # {config_dict['reserve_cash_pct']}% = ${config_dict['total_capital'] * config_dict['reserve_cash_pct'] / 100:,.0f} reserved

# Maximum capital utilization (% of usable capital that can be deployed)
MAX_CAPITAL_UTILIZATION_PCT = {config_dict['max_capital_utilization_pct']}  # Max {config_dict['max_capital_utilization_pct']}% of usable capital in positions

# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================

MAX_DAILY_ORDERS = {config_dict['max_daily_orders']}
MAX_POSITION_SIZE_PCT = {config_dict['max_position_size_pct']}  # Max % per position
RISK_PER_TRADE_PCT = {config_dict['risk_per_trade_pct']}  # {config_dict['risk_per_trade_pct'] * 100:.1f}% risk per trade
MAX_DAILY_LOSS_PCT = {config_dict['max_daily_loss_pct']}  # {config_dict['max_daily_loss_pct'] * 100:.1f}% max daily loss

# Bracket Orders (Stop-Loss & Take-Profit)
USE_BRACKET_ORDERS = {config_dict['use_bracket_orders']}
SCALPING_TAKE_PROFIT_PCT = {config_dict['scalping_take_profit_pct']}
SCALPING_STOP_LOSS_PCT = {config_dict['scalping_stop_loss_pct']}

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = {config_dict['use_settled_funds_only']}
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2

# ==============================================================================
# TICKER SELECTION
# ==============================================================================

# Use Smart Scanner (finds best tickers automatically)
USE_SMART_SCANNER = {config_dict['use_smart_scanner']}

# Your Custom Watchlist (used only if USE_SMART_SCANNER = False)
WATCHLIST = {config_dict['watchlist']}

# ==============================================================================
# AI-POWERED HYBRID MODE (1-2 KNOCKOUT COMBO) 🥊
# ==============================================================================

# Enable ML-Enhanced Scanner for triple validation (40% ML + 35% LLM + 25% Quant)
USE_ML_ENHANCED_SCANNER = {config_dict.get('use_ml_enhanced_scanner', True)}  # RECOMMENDED: Superior trade quality

# Enable AI Pre-Trade Validation (final risk check before execution)
USE_AI_VALIDATION = {config_dict.get('use_ai_validation', True)}  # RECOMMENDED: Blocks high-risk trades

# Minimum ensemble score for ML-Enhanced Scanner (0-100)
MIN_ENSEMBLE_SCORE = {config_dict.get('min_ensemble_score', 70.0)}  # Only trades passing all 3 systems with 70%+ score

# Minimum AI validation confidence (0-1.0)
MIN_AI_VALIDATION_CONFIDENCE = {config_dict.get('min_ai_validation_confidence', 0.7)}  # AI must be 70%+ confident to approve

# NOTE: When both are enabled, you get the 1-2 KNOCKOUT COMBO:
#   PUNCH 1: ML-Enhanced Scanner filters trades (triple validation)
#   PUNCH 2: AI Validator performs final risk check
#   Result: Only the highest quality, lowest risk trades execute!

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Short Selling (ONLY works in paper trading)
ALLOW_SHORT_SELLING = {config_dict['allow_short_selling']}

# Multi-Agent System
USE_AGENT_SYSTEM = False

# ==============================================================================
# FRACTIONAL SHARES (IBKR ONLY) 📊
# ==============================================================================

# Enable fractional share trading for expensive stocks
USE_FRACTIONAL_SHARES = {config_dict.get('use_fractional_shares', False)}

# Auto-use fractional shares for stocks above this price
FRACTIONAL_PRICE_THRESHOLD = {config_dict.get('fractional_price_threshold', 100.0)}  # ${config_dict.get('fractional_price_threshold', 100.0):.0f}

# Dollar amount limits for fractional trades
FRACTIONAL_MIN_AMOUNT = {config_dict.get('fractional_min_amount', 50.0)}  # Min ${config_dict.get('fractional_min_amount', 50.0):.2f} per trade
FRACTIONAL_MAX_AMOUNT = {config_dict.get('fractional_max_amount', 1000.0)}  # Max ${config_dict.get('fractional_max_amount', 1000.0):.2f} per trade

# NOTE: Fractional shares only work with Interactive Brokers (IBKR)
# - Automatically uses fractional shares for stocks above threshold
# - Allows precise dollar-based position sizing (e.g., $250 in NVDA)
# - Better diversification with limited capital
'''
            
            with open(config_filename, 'w', encoding='utf-8') as f:
                f.write(config_content)
            return True
        except Exception as e:
            st.error(f"Error saving config: {e}")
            return False

    # ========================================================================
    # TRADING ENVIRONMENT & CONTROLS
    # ========================================================================
    
    st.divider()
    st.subheader("🎮 Control Center")
    
    col_env, col_scan, col_bg = st.columns(3)
    
    # --- Environment Control ---
    with col_env:
        st.markdown("#### 🌍 Environment")
        is_paper = os.getenv('IS_PAPER_TRADING', 'True').lower() == 'true'
        
        if is_paper:
            st.success("✅ **PAPER TRADING ACTIVE**")
            st.caption("Safe mode. No real money used.")
            if st.button("🔴 Switch to LIVE TRADING"):
                if update_env_file_for_live_trading():
                    st.success("Switched to LIVE mode! Restarting...")
                    time.sleep(1)
                    st.rerun()
        else:
            st.error("⚠️ **LIVE TRADING ACTIVE**")
            st.caption("Real money at risk!")
            if st.button("🟢 Switch to PAPER TRADING"):
                if update_env_file_for_paper_trading():
                    st.success("Switched to PAPER mode! Restarting...")
                    time.sleep(1)
                    st.rerun()

    # --- Manual Scanner ---
    with col_scan:
        st.markdown("#### 🔍 Manual Scanner")
        st.caption("Run a quick scan to see current opportunities.")
        
        if st.button("🚀 Run Stock Scanner"):
            with st.spinner("Scanning market..."):
                try:
                    # Import here to avoid circular deps
                    from services.top_trades_scanner import TopTradesScanner
                    import pandas as pd
                    
                    scanner = TopTradesScanner()
                    # Use a default universe or the one from config
                    results = scanner.scan_top_options_trades(top_n=10)
                    
                    if results:
                        st.session_state.last_scan_results = results
                        st.success(f"Found {len(results)} opportunities!")
                    else:
                        st.warning("No opportunities found.")
                except Exception as e:
                    st.error(f"Scan failed: {e}")
        
        if 'last_scan_results' in st.session_state and st.session_state.last_scan_results:
            if st.button("🗑️ Clear Results"):
                del st.session_state.last_scan_results
                st.rerun()

    # --- Background Bot Control ---
    with col_bg:
        st.markdown("#### 🤖 Background Bot")
        
        # Check if running
        import psutil
        bg_pid = None
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'python' in proc.info['name'] and 'run_autotrader_background.py' in ' '.join(cmdline):
                    bg_pid = proc.info['pid']
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        if bg_pid:
            st.success(f"✅ **Running** (PID: {bg_pid})")
            if st.button("🛑 Stop Background Bot"):
                try:
                    import subprocess
                    subprocess.Popen("stop_autotrader.bat", shell=True)
                    st.info("Stopping...")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to stop: {e}")
        else:
            st.warning("🔴 **Stopped**")
            if st.button("🟢 Start Background Bot"):
                try:
                    import subprocess
                    subprocess.Popen("start_autotrader_background.bat", shell=True)
                    st.success("Starting...")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")

    # Display Scan Results if available
    if 'last_scan_results' in st.session_state and st.session_state.last_scan_results:
        st.divider()
        st.subheader("📊 Scan Results")
        
        # Convert to DataFrame for display
        import pandas as pd
        results_data = []
        for trade in st.session_state.last_scan_results:
            results_data.append({
                "Ticker": trade.ticker,
                "Score": trade.score,
                "Price": trade.price,
                "Strategy": trade.strategy if hasattr(trade, 'strategy') else 'N/A',
                "Confidence": f"{trade.confidence:.1f}%" if isinstance(trade.confidence, float) else str(trade.confidence)
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)

    # ========================================================================
    # STRATEGY SELECTOR
    # ========================================================================
    
    st.markdown("""
    ### 📋 Configuration Workflow
    
    <div style="background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c); padding: 2px; border-radius: 10px; margin: 10px 0;">
        <div style="background: white; padding: 20px; border-radius: 8px;">
            <h4 style="margin: 0 0 10px 0;">Simple 3-Step Process:</h4>
            <p style="margin: 5px 0;"><strong>1️⃣ SELECT</strong> → Choose which strategy to configure (dropdown loads its settings)</p>
            <p style="margin: 5px 0;"><strong>2️⃣ EDIT</strong> → Modify settings in tabs below (changes are temporary)</p>
            <p style="margin: 5px 0;"><strong>3️⃣ SAVE</strong> → Write changes to config file (permanent)</p>
            <hr style="margin: 10px 0;">
            <p style="margin: 5px 0; color: #ff6b6b;"><strong>⚠️ To Use Saved Settings:</strong> Click "Activate" (if not active), then restart background trader</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Load strategy config
    active_strategy_data = load_active_strategy()
    available_strategies = active_strategy_data['available_strategies']
    current_active_strategy = active_strategy_data['active_strategy']
    
    # Create strategy selector with better visual hierarchy
    st.subheader("🎯 Step 1: Select Strategy to Configure")
    
    col_strategy, col_status, col_action = st.columns([3, 1, 1])
    
    with col_strategy:
        strategy_options = {k: v['name'] for k, v in available_strategies.items()}
        strategy_descriptions = {k: v['description'] for k, v in available_strategies.items()}
        
        selected_strategy = st.selectbox(
            "Choose which strategy's settings to edit:",
            options=list(strategy_options.keys()),
            index=list(strategy_options.keys()).index(current_active_strategy) if current_active_strategy in strategy_options else 0,
            format_func=lambda x: f"{strategy_options[x]}",
            help="Selecting a strategy will LOAD its current settings below for editing",
            key="strategy_selector",
            label_visibility="collapsed"
        )
    
    with col_status:
        st.write("")  # Spacing
        if selected_strategy == current_active_strategy:
            st.success("✅ **ACTIVE**")
        else:
            st.warning("📝 **Editing**")
    
    with col_action:
        st.write("")  # Spacing
        if selected_strategy != current_active_strategy:
            if st.button("🎯 Activate", help="Make this the active strategy for background trader"):
                if save_active_strategy(selected_strategy):
                    st.success("✅ Activated!")
                    # Show what will be loaded
                    config_file = available_strategies[selected_strategy]['config_file']
                    st.info(f"📁 Background trader will load: `{config_file}`")
                    st.info(f"🎯 Trading Mode: {available_strategies[selected_strategy].get('trading_mode', 'UNKNOWN')}")
                    st.success("🔄 **Config change detected!** Background trader will auto-restart within 60 seconds.")
                    st.info("💡 If using `start_autotrader_auto_restart.bat`, it will restart automatically. Otherwise, manually restart.")
                    
                    # Verify file was actually updated
                    try:
                        import json
                        with open('active_strategy.json', 'r') as f:
                            verify = json.load(f)
                        if verify.get('active_strategy') == selected_strategy:
                            st.success(f"✅ Verified: File updated correctly to `{verify.get('active_strategy')}`")
                        else:
                            st.error(f"❌ Mismatch! File shows `{verify.get('active_strategy')}` but expected `{selected_strategy}`")
                    except Exception as e:
                        st.warning(f"⚠️ Could not verify file: {e}")
                    
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Failed to save! Check logs for details.")
    
    # Show active strategy info with better formatting
    st.markdown("---")
    
    # CRITICAL: Show what background trader will ACTUALLY load
    try:
        import json
        with open('active_strategy.json', 'r') as f:
            file_content = json.load(f)
        file_strategy = file_content.get('active_strategy', 'UNKNOWN')
        file_config = file_content.get('config_file', 'UNKNOWN')
        
        st.markdown("### 🔍 Background Trader Configuration")
        if file_strategy == current_active_strategy:
            st.success(f"✅ **Active Strategy:** `{file_strategy}` → Config: `{file_config}`")
        else:
            st.error(f"⚠️ **MISMATCH DETECTED!**")
            st.error(f"- Streamlit shows: `{current_active_strategy}`")
            st.error(f"- File contains: `{file_strategy}` → Config: `{file_config}`")
            st.warning("🚨 Background trader will use what's in the FILE, not what Streamlit shows!")
            st.info("💡 Click '🎯 Activate' to sync Streamlit with the file")
    except Exception as e:
        st.warning(f"⚠️ Could not read active_strategy.json: {e}")
    selected_config_file = available_strategies[selected_strategy]['config_file']
    
    st.info(f"""
    **📝 Now Editing:** `{strategy_options[selected_strategy]}`  
    **📁 Config File:** `{selected_config_file}`  
    **📖 Description:** {strategy_descriptions[selected_strategy]}
    
    ℹ️ **Settings below are loaded from this config file. Changes are NOT saved until you click "💾 Save Configuration" at the bottom.**
    """)
    
    st.divider()
    
    st.subheader("🎯 Step 2: Edit Settings Below")
    st.caption("Make your changes in the tabs below. Settings are loaded from the selected strategy's config file.")
    
    # Load config for selected strategy
    current_config = load_config_file(selected_config_file)
    
    if current_config:
        st.success(f"✅ Loaded configuration from `{selected_config_file}`")
        
        # Use stateful navigation instead of st.tabs() to prevent reruns
        if 'config_tab' not in st.session_state:
            st.session_state.config_tab = "📊 Strategy & Tickers"
        
        # Tab selector using radio buttons (no rerun on selection)
        config_tab_selector = st.radio(
            "Configuration Section",
            options=["📊 Strategy & Tickers", "⚖️ Risk & AI Settings", "💾 Step 3: Save"],
            horizontal=True,
            key="config_tab_selector",
            label_visibility="collapsed"
        )
        
        # Update session state if changed
        if config_tab_selector != st.session_state.config_tab:
            st.session_state.config_tab = config_tab_selector
        
        # Render the selected tab
        cfg_tab1_active = st.session_state.config_tab == "📊 Strategy & Tickers"
        cfg_tab2_active = st.session_state.config_tab == "⚖️ Risk & AI Settings"
        cfg_tab3_active = st.session_state.config_tab == "💾 Step 3: Save"
        
        # Initialize session state for config values if not already set
        config_key = f"config_{selected_strategy}"
        if config_key not in st.session_state:
            st.session_state[config_key] = {}
        
        # Initialize all variables from session state (if set) or current_config
        # This ensures widget values persist across tab switches
        trading_mode = st.session_state[config_key].get('trading_mode', current_config.get('trading_mode', 'SCALPING'))
        scan_interval = st.session_state[config_key].get('scan_interval', current_config.get('scan_interval', 15))
        min_confidence = st.session_state[config_key].get('min_confidence', current_config.get('min_confidence', 70))
        use_bracket_orders = st.session_state[config_key].get('use_bracket_orders', current_config.get('use_bracket_orders', True))
        use_smart_scanner = st.session_state[config_key].get('use_smart_scanner', current_config.get('use_smart_scanner', True))
        watchlist_str = st.session_state[config_key].get('watchlist_str', ", ".join(current_config.get('watchlist', [])))
        total_capital = st.session_state[config_key].get('total_capital', current_config.get('total_capital', 10000.0))
        reserve_cash_pct = st.session_state[config_key].get('reserve_cash_pct', current_config.get('reserve_cash_pct', 10.0))
        max_capital_utilization_pct = st.session_state[config_key].get('max_capital_utilization_pct', current_config.get('max_capital_utilization_pct', 80.0))
        max_daily_orders = st.session_state[config_key].get('max_daily_orders', current_config.get('max_daily_orders', 5))
        max_position_size_pct = st.session_state[config_key].get('max_position_size_pct', current_config.get('max_position_size_pct', 10.0))
        risk_per_trade_pct = st.session_state[config_key].get('risk_per_trade_pct', current_config.get('risk_per_trade_pct', 0.02))
        max_daily_loss_pct = st.session_state[config_key].get('max_daily_loss_pct', current_config.get('max_daily_loss_pct', 0.05))
        scalping_take_profit_pct = st.session_state[config_key].get('scalping_take_profit_pct', current_config.get('scalping_take_profit_pct', 2.0))
        scalping_stop_loss_pct = st.session_state[config_key].get('scalping_stop_loss_pct', current_config.get('scalping_stop_loss_pct', 1.0))
        use_settled_funds_only = st.session_state[config_key].get('use_settled_funds_only', current_config.get('use_settled_funds_only', False))
        allow_short_selling = st.session_state[config_key].get('allow_short_selling', current_config.get('allow_short_selling', False))
        use_ml_enhanced_scanner = st.session_state[config_key].get('use_ml_enhanced_scanner', current_config.get('use_ml_enhanced_scanner', True))
        use_ai_validation = st.session_state[config_key].get('use_ai_validation', current_config.get('use_ai_validation', True))
        min_ensemble_score = st.session_state[config_key].get('min_ensemble_score', current_config.get('min_ensemble_score', 70))
        min_ai_validation_confidence = st.session_state[config_key].get('min_ai_validation_confidence', current_config.get('min_ai_validation_confidence', 0.7))
        
        if cfg_tab1_active:
            st.subheader("Strategy Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                trading_mode = st.selectbox(
                    "Trading Mode",
                    options=["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"],
                    index=["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"].index(trading_mode) if trading_mode in ["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"] else 0,
                    help="SCALPING: Fast intraday | WARRIOR_SCALPING: Gap & Go (9:30-10:00 AM) | STOCKS: Swing trades | OPTIONS: Options trading",
                    key=f"{config_key}_trading_mode"
                )
                st.session_state[config_key]['trading_mode'] = trading_mode
                
                scan_interval = st.slider(
                    "Scan Interval (minutes)",
                    min_value=5,
                    max_value=60,
                    value=int(scan_interval),
                    step=5,
                    help="How often to scan for new opportunities",
                    key=f"{config_key}_scan_interval"
                )
                st.session_state[config_key]['scan_interval'] = scan_interval
            
            with col2:
                min_confidence = st.slider(
                    "Minimum Confidence %",
                    min_value=60,
                    max_value=95,
                    value=int(min_confidence),
                    step=5,
                    help="Only execute signals above this confidence level",
                    key=f"{config_key}_min_confidence"
                )
                st.session_state[config_key]['min_confidence'] = min_confidence
                
                use_bracket_orders = st.checkbox(
                    "Use Bracket Orders (Stop-Loss + Take-Profit)",
                    value=use_bracket_orders,
                    help="Automatically set protective orders",
                    key=f"{config_key}_use_bracket_orders"
                )
                st.session_state[config_key]['use_bracket_orders'] = use_bracket_orders
            
            st.divider()
            st.subheader("Ticker Selection")
            
            use_smart_scanner = st.checkbox(
                "🧠 Use Smart Scanner (Auto-discover best tickers)",
                value=use_smart_scanner,
                help="When enabled, ignores watchlist and automatically finds opportunities",
                key=f"{config_key}_use_smart_scanner"
            )
            st.session_state[config_key]['use_smart_scanner'] = use_smart_scanner
            
            # Get checked tickers from the watchlist section
            def get_selected_tickers_from_ui():
                """Get tickers that are checked in the main watchlist"""
                selected = []
                try:
                    # Use cached ticker manager from session state
                    ticker_mgr = st.session_state.ticker_manager
                    # Use cached ticker data if available
                    if st.session_state.ticker_cache and 'all_tickers' in st.session_state.ticker_cache:
                        all_tickers = st.session_state.ticker_cache['all_tickers']
                    else:
                        all_tickers = ticker_mgr.get_all_tickers()
                    if all_tickers:
                        for t in all_tickers:
                            ticker = t['ticker']
                            checkbox_key = f"auto_trade_{ticker}"
                            if checkbox_key in st.session_state and st.session_state[checkbox_key]:
                                selected.append(ticker)
                except Exception:
                    pass
                return selected
            
            # Add sync button
            col_sync1, col_sync2 = st.columns([3, 1])
            with col_sync1:
                st.write("**Quick Actions:**")
            with col_sync2:
                if st.button("📋 Copy Checked Tickers", help="Copy tickers you checked in the Watchlist section below"):
                    checked_tickers = get_selected_tickers_from_ui()
                    if checked_tickers:
                        # Update both session state keys to ensure text area updates
                        st.session_state['synced_watchlist'] = ", ".join(checked_tickers)
                        st.session_state['watchlist_text_area'] = ", ".join(checked_tickers)
                        st.success(f"✅ Copied {len(checked_tickers)} tickers!")
                        st.info(f"📋 **Tickers ready to save:** {', '.join(checked_tickers[:10])}{'...' if len(checked_tickers) > 10 else ''}")
                        st.rerun()
                    else:
                        st.warning("⚠️ No tickers checked in Watchlist section below. Scroll down and check some first!")
            
            # Use synced watchlist if available, otherwise use session state or current config
            if 'synced_watchlist' in st.session_state:
                default_watchlist = st.session_state['synced_watchlist']
                # Clear the synced state so it doesn't persist forever
                if st.session_state.get('clear_sync', False):
                    del st.session_state['synced_watchlist']
                    st.session_state['clear_sync'] = False
            else:
                default_watchlist = watchlist_str
            
            if not use_smart_scanner:
                st.info("💡 Smart Scanner disabled - will use your custom watchlist below")
                watchlist_str = st.text_area(
                    "Custom Watchlist (comma-separated)",
                    value=default_watchlist,
                    help="Enter tickers separated by commas. Example: TSLA, NVDA, AMD, AAPL\nTip: Use '📋 Copy Checked Tickers' to auto-fill from your checked tickers below!",
                    height=100,
                    key=f"{config_key}_watchlist_text_area"
                )
                st.session_state[config_key]['watchlist_str'] = watchlist_str
            else:
                st.warning("⚠️ Smart Scanner enabled - watchlist below will be IGNORED")
                watchlist_str = st.text_area(
                    "Custom Watchlist (not used when Smart Scanner enabled)",
                    value=default_watchlist,
                    help="These tickers are ignored while Smart Scanner is enabled",
                    height=100,
                    disabled=True,
                    key=f"{config_key}_watchlist_text_area_disabled"
                )
                st.session_state[config_key]['watchlist_str'] = watchlist_str
        
        elif cfg_tab2_active:
            st.subheader("💰 Capital Management")
            st.markdown("_Control how much capital the bot can use for trading_")
            
            col1, col2 = st.columns(2)
            
            with col1:
                total_capital = st.number_input(
                    "Total Capital Allocated to Bot",
                    min_value=100.0,
                    max_value=1000000.0,
                    value=float(total_capital),
                    step=100.0,
                    help="💵 Total account balance or capital allocated for auto-trading",
                    key=f"{config_key}_total_capital"
                )
                st.session_state[config_key]['total_capital'] = total_capital
                
                reserve_cash_pct = st.slider(
                    "Reserve Cash %",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(reserve_cash_pct),
                    step=5.0,
                    help="💰 Percentage kept aside, not used for trading (emergency cash)",
                    key=f"{config_key}_reserve_cash_pct"
                )
                st.session_state[config_key]['reserve_cash_pct'] = reserve_cash_pct
                
                st.info(f"**Usable Capital:** ${total_capital * (1 - reserve_cash_pct/100):,.2f}")
            
            with col2:
                max_capital_utilization_pct = st.slider(
                    "Max Capital Utilization %",
                    min_value=20.0,
                    max_value=100.0,
                    value=float(max_capital_utilization_pct),
                    step=5.0,
                    help="📊 Maximum % of usable capital that can be deployed in positions",
                    key=f"{config_key}_max_capital_utilization_pct"
                )
                st.session_state[config_key]['max_capital_utilization_pct'] = max_capital_utilization_pct
                
                usable_capital = total_capital * (1 - reserve_cash_pct/100)
                max_deployed = usable_capital * (max_capital_utilization_pct/100)
                st.info(f"**Max Deployed:** ${max_deployed:,.2f}")
                st.info(f"**Always Available:** ${usable_capital - max_deployed:,.2f}")
            
            st.divider()
            st.subheader("⚖️ Risk Management")
            
            col3, col4 = st.columns(2)
            
            with col3:
                max_daily_orders = st.number_input(
                    "Max Daily Orders",
                    min_value=1,
                    max_value=50,
                    value=int(max_daily_orders),
                    help="Maximum number of trades per day",
                    key=f"{config_key}_max_daily_orders"
                )
                st.session_state[config_key]['max_daily_orders'] = max_daily_orders
                
                max_position_size_pct = st.slider(
                    "Max Position Size %",
                    min_value=1.0,
                    max_value=50.0,
                    value=float(max_position_size_pct),
                    step=1.0,
                    help="Maximum % of total capital per single trade",
                    key=f"{config_key}_max_position_size_pct"
                )
                st.session_state[config_key]['max_position_size_pct'] = max_position_size_pct
                
                risk_per_trade_pct = st.slider(
                    "Risk Per Trade %",
                    min_value=0.5,
                    max_value=5.0,
                    value=float(risk_per_trade_pct * 100),
                    step=0.5,
                    help="Risk % of account per trade",
                    key=f"{config_key}_risk_per_trade_pct"
                ) / 100.0
                st.session_state[config_key]['risk_per_trade_pct'] = risk_per_trade_pct
            
            with col4:
                max_daily_loss_pct = st.slider(
                    "Max Daily Loss %",
                    min_value=1.0,
                    max_value=10.0,
                    value=float(max_daily_loss_pct * 100),
                    step=0.5,
                    help="Stop trading if down this % in a day",
                    key=f"{config_key}_max_daily_loss_pct"
                ) / 100.0
                st.session_state[config_key]['max_daily_loss_pct'] = max_daily_loss_pct
                
                scalping_take_profit_pct = st.slider(
                    "Take-Profit % (Scalping)",
                    min_value=0.5,
                    max_value=10.0,
                    value=float(scalping_take_profit_pct),
                    step=0.5,
                    help="Target profit % for scalping mode",
                    key=f"{config_key}_scalping_take_profit_pct"
                )
                st.session_state[config_key]['scalping_take_profit_pct'] = scalping_take_profit_pct
                
                scalping_stop_loss_pct = st.slider(
                    "Stop-Loss % (Scalping)",
                    min_value=0.25,
                    max_value=5.0,
                    value=float(scalping_stop_loss_pct),
                    step=0.25,
                    help="Stop loss % for scalping mode",
                    key=f"{config_key}_scalping_stop_loss_pct"
                )
                st.session_state[config_key]['scalping_stop_loss_pct'] = scalping_stop_loss_pct
            
            st.divider()
            st.subheader("Advanced Options")
            
            col3, col4 = st.columns(2)
            
            with col3:
                use_settled_funds_only = st.checkbox(
                    "PDT-Safe: Use Settled Funds Only",
                    value=use_settled_funds_only,
                    help="Avoids Pattern Day Trader restrictions",
                    key=f"{config_key}_use_settled_funds_only"
                )
                st.session_state[config_key]['use_settled_funds_only'] = use_settled_funds_only
            
            with col4:
                allow_short_selling = st.checkbox(
                    "Allow Short Selling (Paper Only)",
                    value=allow_short_selling,
                    help="⚠️ Advanced: Enable short selling in paper trading",
                    key=f"{config_key}_allow_short_selling"
                )
                st.session_state[config_key]['allow_short_selling'] = allow_short_selling
            
            st.divider()
            st.subheader("🥊 AI-Powered Hybrid Mode (1-2 KNOCKOUT COMBO)")
            st.markdown("""
            **The ultimate trade quality system** - Only the best trades survive double validation!
            
            - **PUNCH 1**: ML-Enhanced Scanner (40% ML + 35% LLM + 25% Quant)
            - **PUNCH 2**: AI Pre-Trade Validation (final risk check)
            - **Result**: Maximum trade quality + risk control
            """)
            
            col_ai1, col_ai2 = st.columns(2)
            
            with col_ai1:
                use_ml_enhanced_scanner = st.checkbox(
                    "🧠 Enable ML-Enhanced Scanner (PUNCH 1)",
                    value=use_ml_enhanced_scanner,
                    help="Triple validation: 40% ML + 35% LLM + 25% Quantitative analysis",
                    key=f"{config_key}_use_ml_enhanced_scanner"
                )
                st.session_state[config_key]['use_ml_enhanced_scanner'] = use_ml_enhanced_scanner
                
                if use_ml_enhanced_scanner:
                    min_ensemble_score = st.slider(
                        "Min Ensemble Score %",
                        min_value=50,
                        max_value=95,
                        value=int(min_ensemble_score),
                        step=5,
                        help="Minimum combined score from ML+LLM+Quant (70%+ recommended)",
                        key=f"{config_key}_min_ensemble_score"
                    )
                    st.session_state[config_key]['min_ensemble_score'] = min_ensemble_score
                else:
                    min_ensemble_score = 70.0
                    st.session_state[config_key]['min_ensemble_score'] = min_ensemble_score
            
            with col_ai2:
                use_ai_validation = st.checkbox(
                    "🛡️ Enable AI Pre-Trade Validation (PUNCH 2)",
                    value=use_ai_validation,
                    help="LLM validates risk/reward, portfolio fit, and red flags before execution",
                    key=f"{config_key}_use_ai_validation"
                )
                st.session_state[config_key]['use_ai_validation'] = use_ai_validation
                
                if use_ai_validation:
                    min_ai_validation_confidence = st.slider(
                        "Min AI Validation Confidence",
                        min_value=0.5,
                        max_value=0.95,
                        value=float(min_ai_validation_confidence),
                        step=0.05,
                        format="%.2f",
                        help="Minimum confidence for AI to approve trade (0.7+ recommended)",
                        key=f"{config_key}_min_ai_validation_confidence"
                    )
                    st.session_state[config_key]['min_ai_validation_confidence'] = min_ai_validation_confidence
                else:
                    min_ai_validation_confidence = 0.7
                    st.session_state[config_key]['min_ai_validation_confidence'] = min_ai_validation_confidence
            
            if use_ml_enhanced_scanner and use_ai_validation:
                st.success("🥊 **KNOCKOUT COMBO ACTIVE!** Maximum trade quality & risk control enabled.")
            elif use_ml_enhanced_scanner:
                st.info("🧠 ML-Enhanced Scanner active. Enable AI Validation for full knockout combo!")
            elif use_ai_validation:
                st.info("🛡️ AI Validation active. Enable ML-Enhanced Scanner for full knockout combo!")
            else:
                st.warning("⚠️ AI features disabled. Enable for superior trade quality!")
            
            st.divider()
            st.subheader("📊 Fractional Shares (IBKR Only)")
            st.markdown("""
            **Trade expensive stocks with smaller capital** - Automatically use fractional shares for stocks above your threshold.
            
            - 📊 **Auto-Detection**: Automatically use fractional for stocks above threshold
            - 💰 **Dollar-Based**: Set exact dollar amounts per stock (e.g., $250 in NVDA)
            - 🎯 **Better Diversification**: Spread capital across more positions
            - ⚠️ **IBKR Only**: Fractional shares only work with Interactive Brokers
            """)
            
            # Initialize fractional share settings if not in session state
            use_fractional_shares = st.session_state[config_key].get('use_fractional_shares', current_config.get('use_fractional_shares', False))
            fractional_price_threshold = st.session_state[config_key].get('fractional_price_threshold', current_config.get('fractional_price_threshold', 100.0))
            fractional_min_amount = st.session_state[config_key].get('fractional_min_amount', current_config.get('fractional_min_amount', 50.0))
            fractional_max_amount = st.session_state[config_key].get('fractional_max_amount', current_config.get('fractional_max_amount', 1000.0))
            
            col_frac1, col_frac2 = st.columns(2)
            
            with col_frac1:
                use_fractional_shares = st.checkbox(
                    "✅ Enable Fractional Shares (IBKR Only)",
                    value=use_fractional_shares,
                    help="Automatically use fractional shares for expensive stocks (IBKR only)",
                    key=f"{config_key}_use_fractional_shares"
                )
                st.session_state[config_key]['use_fractional_shares'] = use_fractional_shares
                
                if use_fractional_shares:
                    fractional_price_threshold = st.slider(
                        "Auto-Enable Above Price",
                        min_value=50.0,
                        max_value=500.0,
                        value=float(fractional_price_threshold),
                        step=10.0,
                        format="$%.0f",
                        help="Automatically use fractional shares for stocks above this price",
                        key=f"{config_key}_fractional_price_threshold"
                    )
                    st.session_state[config_key]['fractional_price_threshold'] = fractional_price_threshold
            
            with col_frac2:
                if use_fractional_shares:
                    fractional_min_amount = st.number_input(
                        "Min Dollar Amount",
                        min_value=10.0,
                        max_value=1000.0,
                        value=float(fractional_min_amount),
                        step=10.0,
                        format="%.2f",
                        help="Minimum dollar amount per fractional trade",
                        key=f"{config_key}_fractional_min_amount"
                    )
                    st.session_state[config_key]['fractional_min_amount'] = fractional_min_amount
                    
                    fractional_max_amount = st.number_input(
                        "Max Dollar Amount",
                        min_value=50.0,
                        max_value=10000.0,
                        value=float(fractional_max_amount),
                        step=50.0,
                        format="%.2f",
                        help="Maximum dollar amount per fractional trade",
                        key=f"{config_key}_fractional_max_amount"
                    )
                    st.session_state[config_key]['fractional_max_amount'] = fractional_max_amount
            
            if use_fractional_shares:
                # Example calculation
                st.info(f"""
                📊 **Example:** 
                - Stock at **${fractional_price_threshold:.0f}** → Fractional used → Buy **${fractional_max_amount:.0f}** worth = **{fractional_max_amount/fractional_price_threshold:.2f} shares**
                - Stock at **${fractional_price_threshold - 10:.0f}** → Below threshold → Regular whole share sizing
                """)
                
                # Link to detailed configuration
                if st.button("⚙️ Advanced Fractional Share Settings", key="open_fractional_config"):
                    st.info("Navigate to 'Fractional Shares' tab in the sidebar for advanced per-ticker configuration")
        
        elif cfg_tab3_active:
            st.subheader("🎯 Step 3: Save Configuration")
            
            st.markdown(f"""
            ### 💾 Ready to Save?
            
            You are editing: **`{strategy_options[selected_strategy]}`**  
            Config file: **`{selected_config_file}`**
            
            **What happens when you click "Save":**
            1. ✅ Your changes are written to **`{selected_config_file}`**
            2. ✅ The config file is permanently updated
            3. ⚠️ Changes take effect only after you **restart the background trader**
            
            **To apply saved changes:**
            ```powershell
            # Stop trader
            .\\stop_autotrader.bat
            
            # Start trader (loads the updated config)
            .\\start_autotrader_background.bat
            ```
            
            💡 **Tip:** You can edit multiple strategies, save them all, then activate the one you want to use.
            """)
            
            st.divider()
            
            # Show what will be saved
            with st.expander("👁️ Preview Configuration"):
                # Read values from session state for preview
                preview_config_dict = st.session_state.get(config_key, {})
                preview_trading_mode = preview_config_dict.get('trading_mode', trading_mode)
                preview_scan_interval = preview_config_dict.get('scan_interval', scan_interval)
                preview_min_confidence = preview_config_dict.get('min_confidence', min_confidence)
                preview_use_smart_scanner = preview_config_dict.get('use_smart_scanner', use_smart_scanner)
                preview_watchlist_str = preview_config_dict.get('watchlist_str', watchlist_str)
                preview_total_capital = preview_config_dict.get('total_capital', total_capital)
                preview_reserve_cash_pct = preview_config_dict.get('reserve_cash_pct', reserve_cash_pct)
                preview_max_capital_utilization_pct = preview_config_dict.get('max_capital_utilization_pct', max_capital_utilization_pct)
                preview_max_daily_orders = preview_config_dict.get('max_daily_orders', max_daily_orders)
                preview_max_position_size_pct = preview_config_dict.get('max_position_size_pct', max_position_size_pct)
                preview_risk_per_trade_pct = preview_config_dict.get('risk_per_trade_pct', risk_per_trade_pct)
                preview_max_daily_loss_pct = preview_config_dict.get('max_daily_loss_pct', max_daily_loss_pct)
                preview_use_bracket_orders = preview_config_dict.get('use_bracket_orders', use_bracket_orders)
                preview_scalping_take_profit_pct = preview_config_dict.get('scalping_take_profit_pct', scalping_take_profit_pct)
                preview_scalping_stop_loss_pct = preview_config_dict.get('scalping_stop_loss_pct', scalping_stop_loss_pct)
                preview_use_ml_enhanced_scanner = preview_config_dict.get('use_ml_enhanced_scanner', use_ml_enhanced_scanner)
                preview_use_ai_validation = preview_config_dict.get('use_ai_validation', use_ai_validation)
                preview_min_ensemble_score = preview_config_dict.get('min_ensemble_score', min_ensemble_score)
                preview_min_ai_validation_confidence = preview_config_dict.get('min_ai_validation_confidence', min_ai_validation_confidence)
                
                preview_config = {
                    'Trading Mode': preview_trading_mode,
                    'Scan Interval': f"{preview_scan_interval} minutes",
                    'Min Confidence': f"{preview_min_confidence}%",
                    'Smart Scanner': "Enabled" if preview_use_smart_scanner else "Disabled",
                    'Watchlist': preview_watchlist_str if not preview_use_smart_scanner else "(Using Smart Scanner)",
                    '--- Capital Management ---': '---',
                    'Total Capital': f"${preview_total_capital:,.2f}",
                    'Reserve Cash': f"{preview_reserve_cash_pct}% (${preview_total_capital * preview_reserve_cash_pct / 100:,.2f})",
                    'Usable Capital': f"${preview_total_capital * (1 - preview_reserve_cash_pct/100):,.2f}",
                    'Max Capital Utilization': f"{preview_max_capital_utilization_pct}%",
                    'Max Deployed Capital': f"${preview_total_capital * (1 - preview_reserve_cash_pct/100) * preview_max_capital_utilization_pct / 100:,.2f}",
                    '--- Risk Management ---': '---',
                    'Max Daily Orders': preview_max_daily_orders,
                    'Max Position Size': f"{preview_max_position_size_pct}% (${preview_total_capital * preview_max_position_size_pct / 100:,.2f})",
                    'Risk Per Trade': f"{preview_risk_per_trade_pct * 100:.1f}%",
                    'Max Daily Loss': f"{preview_max_daily_loss_pct * 100:.1f}%",
                    'Use Bracket Orders': "Yes" if preview_use_bracket_orders else "No",
                    'Take-Profit': f"{preview_scalping_take_profit_pct}%",
                    'Stop-Loss': f"{preview_scalping_stop_loss_pct}%",
                    '--- AI-Powered Hybrid Mode (1-2 KNOCKOUT COMBO) ---': '🥊',
                    'ML-Enhanced Scanner (PUNCH 1)': "✅ Enabled" if preview_use_ml_enhanced_scanner else "❌ Disabled",
                    'Min Ensemble Score': f"{preview_min_ensemble_score}%" if preview_use_ml_enhanced_scanner else "N/A",
                    'AI Pre-Trade Validation (PUNCH 2)': "✅ Enabled" if preview_use_ai_validation else "❌ Disabled",
                    'Min AI Validation Confidence': f"{preview_min_ai_validation_confidence:.2f}" if preview_use_ai_validation else "N/A",
                    'Knockout Combo Status': "🥊 ACTIVE - Maximum Quality!" if (preview_use_ml_enhanced_scanner and preview_use_ai_validation) else "⚠️ Partial" if (preview_use_ml_enhanced_scanner or preview_use_ai_validation) else "❌ Disabled",
                }
                st.json(preview_config)
            
            st.divider()
            
            # Save button
            if st.button("💾 Save Configuration to File", type="primary", width="stretch"):
                # Read values from session state (most up-to-date) or fall back to initialized variables
                saved_config = st.session_state.get(config_key, {})
                
                # Get values from session state if available, otherwise use initialized variables
                trading_mode_save = saved_config.get('trading_mode', trading_mode)
                scan_interval_save = saved_config.get('scan_interval', scan_interval)
                min_confidence_save = saved_config.get('min_confidence', min_confidence)
                use_bracket_orders_save = saved_config.get('use_bracket_orders', use_bracket_orders)
                use_smart_scanner_save = saved_config.get('use_smart_scanner', use_smart_scanner)
                watchlist_str_save = saved_config.get('watchlist_str', watchlist_str)
                total_capital_save = saved_config.get('total_capital', total_capital)
                reserve_cash_pct_save = saved_config.get('reserve_cash_pct', reserve_cash_pct)
                max_capital_utilization_pct_save = saved_config.get('max_capital_utilization_pct', max_capital_utilization_pct)
                max_daily_orders_save = saved_config.get('max_daily_orders', max_daily_orders)
                max_position_size_pct_save = saved_config.get('max_position_size_pct', max_position_size_pct)
                risk_per_trade_pct_save = saved_config.get('risk_per_trade_pct', risk_per_trade_pct)
                max_daily_loss_pct_save = saved_config.get('max_daily_loss_pct', max_daily_loss_pct)
                scalping_take_profit_pct_save = saved_config.get('scalping_take_profit_pct', scalping_take_profit_pct)
                scalping_stop_loss_pct_save = saved_config.get('scalping_stop_loss_pct', scalping_stop_loss_pct)
                use_settled_funds_only_save = saved_config.get('use_settled_funds_only', use_settled_funds_only)
                allow_short_selling_save = saved_config.get('allow_short_selling', allow_short_selling)
                use_ml_enhanced_scanner_save = saved_config.get('use_ml_enhanced_scanner', use_ml_enhanced_scanner)
                use_ai_validation_save = saved_config.get('use_ai_validation', use_ai_validation)
                min_ensemble_score_save = saved_config.get('min_ensemble_score', min_ensemble_score)
                min_ai_validation_confidence_save = saved_config.get('min_ai_validation_confidence', min_ai_validation_confidence)
                
                # Parse watchlist
                if not use_smart_scanner_save:
                    watchlist_tickers = [t.strip().upper() for t in watchlist_str_save.split(',') if t.strip()]
                    if not watchlist_tickers:
                        st.error("❌ Watchlist cannot be empty when Smart Scanner is disabled!")
                        st.stop()
                else:
                    watchlist_tickers = [t.strip().upper() for t in watchlist_str_save.split(',') if t.strip()]
                    if not watchlist_tickers:
                        watchlist_tickers = ['SPY', 'QQQ', 'AAPL']  # Fallback
                
                # Prepare config dict
                new_config = {
                    'trading_mode': trading_mode_save,
                    'scan_interval': scan_interval_save,
                    'min_confidence': min_confidence_save,
                    'max_daily_orders': max_daily_orders_save,
                    'max_position_size_pct': max_position_size_pct_save,
                    'use_bracket_orders': use_bracket_orders_save,
                    'scalping_take_profit_pct': scalping_take_profit_pct_save,
                    'scalping_stop_loss_pct': scalping_stop_loss_pct_save,
                    'risk_per_trade_pct': risk_per_trade_pct_save,
                    'max_daily_loss_pct': max_daily_loss_pct_save,
                    'use_smart_scanner': use_smart_scanner_save,
                    'watchlist': watchlist_tickers,
                    'allow_short_selling': allow_short_selling_save,
                    'use_settled_funds_only': use_settled_funds_only_save,
                    # Capital Management (NEW)
                    'total_capital': total_capital_save,
                    'reserve_cash_pct': reserve_cash_pct_save,
                    'max_capital_utilization_pct': max_capital_utilization_pct_save,
                    # AI-Powered Hybrid Mode (NEW)
                    'use_ml_enhanced_scanner': use_ml_enhanced_scanner_save,
                    'use_ai_validation': use_ai_validation_save,
                    'min_ensemble_score': min_ensemble_score_save,
                    'min_ai_validation_confidence': min_ai_validation_confidence_save,
                }
                
                # Save to file
                if save_config_to_file(new_config, selected_config_file):
                    st.success(f"✅ Configuration saved successfully to `{selected_config_file}`!")
                    
                    st.warning("""
                    **⚠️ RESTART REQUIRED**
                    
                    To apply these changes, restart the background trader:
                    
                    **Windows PowerShell:**
                    ```powershell
                    .\\stop_autotrader.bat
                    .\\start_autotrader_background.bat
                    ```
                    
                    **Or manually:**
                    ```powershell
                    Stop-Process -Name pythonw -Force
                    Start-Process pythonw -ArgumentList "run_autotrader_background.py" -WorkingDirectory "C:\\Users\\seaso\\Sentient Trader"
                    ```
                    
                    **Verify new settings in logs:**
                    ```powershell
                    Get-Content logs\\autotrader_background.log -Tail 20
                    ```
                    """)
                    
                    # Clear synced watchlist after saving
                    if 'synced_watchlist' in st.session_state:
                        del st.session_state['synced_watchlist']
                    
                    # Force page reload to show updated config
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to save configuration. Check file permissions.")
    else:
        st.warning(f"""
        ⚠️ Configuration file not found!
        
        The file `{selected_config_file}` doesn't exist yet.
        
        **To create it:**
        1. Use the configuration below to set your preferences
        2. Click "Save Configuration" 
        3. File will be created automatically
        
        **Alternatively:**
        - Select a different strategy that has an existing config file
        - Or copy an existing config file and rename it to `{selected_config_file}`
        """)
        
        # Show default form for creating new config
        st.info("📝 Using default settings. Customize below and save to create the config file.")
        
        trading_mode = st.selectbox("Trading Mode", ["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"], index=0, help="SCALPING: Fast intraday | WARRIOR_SCALPING: Gap & Go (9:30-10:00 AM)")
        scan_interval = st.slider("Scan Interval (minutes)", 5, 60, 15, 5)
        min_confidence = st.slider("Min Confidence %", 60, 95, 75, 5)
        use_smart_scanner = st.checkbox("Use Smart Scanner", value=True)
        watchlist_str = st.text_area("Watchlist", value="SPY, QQQ, AAPL, TSLA, NVDA")
        
        if st.button("💾 Create Configuration File"):
            watchlist_tickers = [t.strip().upper() for t in watchlist_str.split(',') if t.strip()]
            new_config = {
                'trading_mode': trading_mode,
                'scan_interval': scan_interval,
                'min_confidence': min_confidence,
                'max_daily_orders': 10,
                'max_position_size_pct': 20.0,
                'use_bracket_orders': True,
                'scalping_take_profit_pct': 2.0,
                'scalping_stop_loss_pct': 1.0,
                'risk_per_trade_pct': 0.02,
                'max_daily_loss_pct': 0.04,
                'use_smart_scanner': use_smart_scanner,
                'watchlist': watchlist_tickers,
                'allow_short_selling': False,
                'use_settled_funds_only': True,
            }
            if save_config_to_file(new_config, selected_config_file):
                st.success("✅ Configuration file created!")
                st.rerun()
    
    st.divider()
    
    # ========================================================================
    # END BACKGROUND TRADER CONFIGURATION MANAGER
    # ========================================================================
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    
    with col_cfg1:
        scan_interval = st.number_input(
            "Scan Interval (minutes)",
            min_value=5,
            max_value=60,
            value=15,
            help="How often to scan for new signals"
        )
        min_confidence = st.slider(
            "Min Confidence %",
            min_value=60,
            max_value=95,
            value=75,
            help="Only execute signals above this confidence"
        )
    
    with col_cfg2:
        max_daily_orders = st.number_input(
            "Max Daily Orders",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum orders per day"
        )
        use_bracket_orders = st.checkbox(
            "Use Bracket Orders",
            value=True,
            help="Automatically set stop-loss and take-profit"
        )
    
    with col_cfg3:
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            options=["LOW", "MEDIUM", "HIGH"],
            index=1
        )
        paper_trading = st.checkbox(
            "Paper Trading Mode",
            value=True,
            help="HIGHLY RECOMMENDED: Test with paper trading first"
        )
        allow_short_selling = st.checkbox(
            "Allow Short Selling (Advanced)",
            value=False,
            help="⚠️ Enable short selling for SELL signals. NOT recommended for scalping or cash accounts. Only for margin accounts and advanced strategies.",
            disabled=not paper_trading
        )
        test_mode = st.checkbox(
            "🧪 Test Mode (Bypass Market Hours)",
            value=False,
            help="⚠️ TESTING ONLY: Allows trading when market is closed. Use this to test your scalping setup while the market is closed. Make sure you're in Paper Trading mode!"
        )
    
    if test_mode:
        st.warning("""
        🧪 **Test Mode Enabled**
        
        - ✅ Market hours check is **DISABLED** - you can test even when the market is closed
        - ⚠️ **FOR TESTING ONLY** - Only use this when testing your setup
        - ✅ Make sure **Paper Trading** is enabled to avoid real trades
        - 📝 The scalper will run and scan for signals even outside market hours
        """)
    
    # Smart Scanner option
    st.divider()
    use_smart_scanner = st.checkbox(
        "🧠 Use Smart Scanner (Advanced)",
        value=True,  # Default to True - automatically finds opportunities
        help="IGNORES your ticker selections and automatically finds the best tickers using the Advanced Scanner. Leave unchecked to only scan YOUR selected tickers."
    )
    
    if use_smart_scanner:
        st.warning("""
        ⚠️ **Smart Scanner Mode:**
        - **IGNORES** your ticker checkboxes below
        - Automatically scans 24-33 curated tickers based on strategy
        - Uses Advanced Scanner to find top opportunities
        - Updates dynamically each scan cycle
        
        **Strategy Mapping:**
        - SCALPING → Scans 24 high-volume tickers, returns top 10
        - STOCKS → Scans 33 swing trade candidates, returns top 15
        - OPTIONS → Scans 24 high IV tickers, returns top 15
        - ALL → Scans 24 mixed tickers, returns top 20
        
        💡 **Recommended if:** You want automated ticker discovery
        ❌ **Not recommended if:** You want to control which tickers to trade
        """)
    else:
        st.success("✅ Using YOUR selected tickers below (manual control)")
    
    # Trading mode selection
    st.subheader("📈 Trading Mode")
    col_mode1, col_mode2 = st.columns(2)
    
    with col_mode1:
        trading_mode = st.selectbox(
            "Strategy Type",
            options=["STOCKS", "OPTIONS", "SCALPING", "WARRIOR_SCALPING", "ALL"],
            index=2,  # Default to SCALPING
            help="SCALPING: Fast intraday trades | WARRIOR_SCALPING: Gap & Go strategy (9:30-10:00 AM)"
        )
    
    with col_mode2:
        if trading_mode == "SCALPING":
            scalp_take_profit = st.number_input(
                "Scalp Take Profit %",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Target profit percentage for scalping"
            )
            scalp_stop_loss = st.number_input(
                "Scalp Stop Loss %",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                help="Stop loss percentage for scalping"
            )
        else:
            scalp_take_profit = 2.0
            scalp_stop_loss = 1.0
    
    st.divider()
    
    # Watchlist selection
    st.subheader("📋 Watchlist")
    st.write("Select tickers to monitor for automated trading:")
    
    # Get tickers from database with caching
    try:
        # Use cached ticker manager from session state
        ticker_mgr = st.session_state.ticker_manager
        # Use cached ticker data if available
        if st.session_state.ticker_cache and 'all_tickers' in st.session_state.ticker_cache:
            all_tickers = st.session_state.ticker_cache['all_tickers']
        else:
            all_tickers = ticker_mgr.get_all_tickers()
            # Cache the result
            st.session_state.ticker_cache['all_tickers'] = all_tickers
            st.session_state.ticker_cache_timestamp = datetime.now()
        ticker_symbols = [t['ticker'] for t in all_tickers] if all_tickers else []
    except Exception:
        ticker_symbols = []
    
    if ticker_symbols:
        st.write("**Enable/Disable Auto-Trading Per Ticker:**")
        
        # Show checkboxes for each ticker
        selected_tickers = []
        cols_per_row = 4
        ticker_rows = [ticker_symbols[i:i+cols_per_row] for i in range(0, len(ticker_symbols), cols_per_row)]
        
        # Track if we need to show migration warning
        needs_migration = False
        
        for row in ticker_rows:
            cols = st.columns(cols_per_row)
            for idx, ticker in enumerate(row):
                with cols[idx]:
                    # Get current auto-trade status
                    ticker_data = ticker_mgr.get_ticker(ticker)
                    current_enabled = ticker_data.get('auto_trade_enabled', False) if ticker_data else False
                    
                    # Checkbox for enabling auto-trade
                    enabled = st.checkbox(
                        f"✅ {ticker}",
                        value=current_enabled,
                        key=f"auto_trade_{ticker}",
                        help=f"Enable auto-trading for {ticker}"
                    )
                    
                    if enabled:
                        selected_tickers.append(ticker)
                        # Update database if changed
                        if enabled != current_enabled:
                            success = ticker_mgr.set_auto_trade(ticker, enabled, trading_mode)
                            if not success:
                                needs_migration = True
        
        if needs_migration:
            st.error("⚠️ **Database Migration Required**")
            st.warning("The auto-trade columns are missing from your database. Please run this SQL in your Supabase SQL Editor:")
            st.code("""
ALTER TABLE saved_tickers 
ADD COLUMN IF NOT EXISTS auto_trade_enabled BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS auto_trade_strategy TEXT;
            """, language="sql")
            st.info("📁 Full migration script available at: `migrations/add_auto_trade_columns.sql`")
        
        if not selected_tickers:
            st.info("👆 Check the boxes above to enable auto-trading for specific tickers")
    else:
        st.warning("No tickers in your watchlist. Add some in the '⭐ My Tickers' tab first!")
        selected_tickers = []
    
    st.divider()
    
    # Control buttons
    st.subheader("🎮 Controls")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🚀 Start Auto-Trader", type="primary", disabled=len(selected_tickers) == 0):
            if not st.session_state.tradier_client:
                st.error("❌ Tradier not connected! Go to 🏦 Tradier Account tab to connect.")
            else:
                try:
                    from services.auto_trader import create_auto_trader, AutoTraderConfig
                    from services.ai_trading_signals import create_ai_signal_generator
                    
                    # Create config
                    config = AutoTraderConfig(
                        enabled=True,
                        scan_interval_minutes=scan_interval,
                        min_confidence=min_confidence,
                        max_daily_orders=max_daily_orders,
                        use_bracket_orders=use_bracket_orders,
                       
                        risk_tolerance=risk_tolerance,
                        paper_trading=paper_trading,
                        trading_mode=trading_mode,
                        scalping_take_profit_pct=scalp_take_profit,
                        scalping_stop_loss_pct=scalp_stop_loss,
                        allow_short_selling=allow_short_selling if paper_trading else False,
                        test_mode=test_mode
                    )
                    
                    # Create signal generator
                    signal_gen = create_ai_signal_generator()
                    
                    # Create and start auto-trader
                    auto_trader = create_auto_trader(
                        broker_client=st.session_state.tradier_client,
                        signal_generator=signal_gen,
                        watchlist=selected_tickers,
                        config=config,
                        use_smart_scanner=use_smart_scanner
                    )
                    
                    auto_trader.start()
                    st.session_state.auto_trader = auto_trader
                    
                    st.success("✅ Auto-Trader started successfully!")
                    if test_mode:
                        st.warning("🧪 Test Mode enabled: Market hours check is bypassed. You can test while the market is closed.")
                    if use_smart_scanner:
                        st.info(f"🧠 Smart Scanner enabled: Will dynamically find top tickers for {trading_mode} strategy each scan")
                    else:
                        st.info(f"Monitoring {len(selected_tickers)} tickers: {', '.join(selected_tickers)}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to start Auto-Trader: {e}")
                    logger.error("Auto-trader start error: {}", str(e), exc_info=True)
    
    with col_btn2:
        if st.button("🛑 Stop Auto-Trader", disabled=st.session_state.auto_trader is None):
            if st.session_state.auto_trader:
                st.session_state.auto_trader.stop()
                st.session_state.auto_trader = None
                st.success("Auto-Trader stopped")
                st.rerun()
    
    with col_btn3:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    st.divider()
    
    # ========================================================================
    # AUTO-TRADER TABS
    # ========================================================================
    
    # Use stateful navigation instead of st.tabs() to prevent reruns
    if 'autotrader_tab' not in st.session_state:
        st.session_state.autotrader_tab = "📊 Status & Control"
    
    # Tab selector using radio buttons
    autotrader_tab_selector = st.radio(
        "Select Section:",
        ["⚙️ Settings", "📊 Status & Control", "🔔 Entry Monitors", "📈 Active Positions", "📓 Trade History"],
        key="autotrader_tab_radio",
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Update session state when tab is selected
    if autotrader_tab_selector != st.session_state.autotrader_tab:
        st.session_state.autotrader_tab = autotrader_tab_selector
        st.rerun()
    
    st.divider()
    
    # Display content based on selected tab
    if st.session_state.autotrader_tab == "⚙️ Settings":
        st.subheader("⚙️ Auto-Trader Settings")
        st.info("Configuration settings are managed above in the 'Dynamic Strategy Configuration' section.")
    
    elif st.session_state.autotrader_tab == "🔔 Entry Monitors":
        st.subheader("🔔 Stock Entry Monitors")
        
        # Custom Stock Analysis Section
        with st.expander("🎯 Analyze Custom Stock", expanded=True):
            st.write("Analyze any stock ticker for optimal entry timing using AI.")
            
            from ui.stock_ai_entry_ui import display_stock_ai_entry_analysis
            
            # Get broker client
            broker_client = None
            if st.session_state.get('ibkr_client'):
                broker_client = st.session_state.ibkr_client
            elif st.session_state.get('tradier_client'):
                broker_client = st.session_state.tradier_client
            elif st.session_state.get('broker_client'):
                broker_client = st.session_state.broker_client
            
            if broker_client:
                display_stock_ai_entry_analysis(broker_client)
            else:
                st.warning("⚠️ Please connect to a broker (Tradier or IBKR) to use AI entry analysis.")
        
        st.divider()
        
        # Monitored Stocks Section
        st.markdown("### 📊 Monitored Stocks - Waiting for Optimal Timing")
        try:
            from ui.stock_entry_monitors_ui import display_stock_entry_monitors
            display_stock_entry_monitors()
        except Exception as e:
            st.error(f"Error loading entry monitors: {e}")
            logger.error("Stock entry monitors error: {}", str(e), exc_info=True)
    
    elif st.session_state.autotrader_tab == "📈 Active Positions":
        st.subheader("📈 Active Positions")
        # Display position exit monitor data
        if st.session_state.auto_trader and hasattr(st.session_state.auto_trader, '_position_monitor') and st.session_state.auto_trader._position_monitor:
            monitor = st.session_state.auto_trader._position_monitor
            positions = monitor.get_monitored_positions()
            
            if positions:
                st.success(f"Monitoring {len(positions)} position(s)")
                for pos in positions:
                    # Check if fractional
                    is_fractional = (pos.quantity % 1 != 0)
                    qty_label = f"{pos.quantity:.4f} 📊" if is_fractional else str(int(pos.quantity))
                    
                    with st.expander(f"**{pos.symbol}** - ${pos.current_price:.2f} {('📊' if is_fractional else '')}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Quantity", qty_label)
                            st.metric("Entry", f"${pos.entry_price:.2f}")
                            if is_fractional:
                                st.caption(f"💰 Cost: ${pos.quantity * pos.entry_price:.2f}")
                        with col2:
                            unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                            unrealized_pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
                            st.metric("Unrealized P&L", f"${unrealized_pnl:.2f}", f"{unrealized_pnl_pct:+.2f}%")
                            if is_fractional:
                                st.caption(f"📊 Value: ${pos.quantity * pos.current_price:.2f}")
                        with col3:
                            if pos.stop_loss:
                                st.metric("Stop Loss", f"${pos.stop_loss:.2f}")
                            if pos.take_profit:
                                st.metric("Take Profit", f"${pos.take_profit:.2f}")
            else:
                st.info("No active positions being monitored.")
        else:
            st.info("Position monitoring not enabled or auto-trader not running.")
    
    elif st.session_state.autotrader_tab == "📓 Trade History":
        st.subheader("📓 Trade History")
        if st.session_state.auto_trader:
            history = st.session_state.auto_trader.get_execution_history()
            
            if history:
                st.write(f"**Total Executions:** {len(history)}")
                
                for idx, execution in enumerate(reversed(history[-10:]), 1):  # Show last 10
                    with st.expander(f"{idx}. {execution['symbol']} - {execution['signal']} ({execution['timestamp']})"):
                        col_ex1, col_ex2, col_ex3 = st.columns(3)
                        
                        with col_ex1:
                            st.write(f"**Confidence:** {execution['confidence']:.1f}%")
                            st.write(f"**Quantity:** {execution['quantity']}")
                        
                        with col_ex2:
                            st.write(f"**Entry:** ${execution['entry_price']:.2f}")
                            st.write(f"**Target:** ${execution['target_price']:.2f}")
                        
                        with col_ex3:
                            st.write(f"**Stop Loss:** ${execution['stop_loss']:.2f}")
                            
                            profit_pct = ((execution['target_price'] - execution['entry_price']) / execution['entry_price'] * 100) if execution['entry_price'] else 0
                            st.write(f"**Potential:** {profit_pct:+.1f}%")
            else:
                st.info("No executions yet. The bot will execute when it finds high-confidence signals.")
        else:
            st.info("Auto-Trader is not running.")
    
    else:  # Status & Control tab (default)
        st.subheader("📊 Auto-Trader Status")
    
    # Check for background auto-trader (keep this for Status & Control tab)
    if st.session_state.autotrader_tab == "📊 Status & Control":
        def check_background_trader():
            """Check if background auto-trader is running"""
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] in ['pythonw.exe', 'python.exe']:
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any('run_autotrader_background' in str(cmd) for cmd in cmdline):
                                return True, proc.info['pid']
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except ImportError:
                pass  # psutil not installed
            return False, None
        
        bg_running, bg_pid = check_background_trader()
    
        if bg_running:
            st.info(f"""
            🟢 **Background Auto-Trader Detected**
            
            A background auto-trader is currently running (PID: {bg_pid})
            
            ⚠️ **IMPORTANT**: Don't start another auto-trader here to avoid duplicate trades!
            
            📊 **Monitor it via:**
            - Logs: `logs/autotrader_background.log`
            - State: `data/trade_state.json`
            - Command: `Get-Content logs\\autotrader_background.log -Tail 50 -Wait`
            
            🛑 **To stop it:** Run `stop_autotrader.bat` or kill process {bg_pid}
            """)
        
        # Status display
        st.subheader("📊 Status")
        
        if st.session_state.auto_trader:
            status = st.session_state.auto_trader.get_status()
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                status_icon = "🟢" if status['is_running'] else "🔴"
                st.metric("Status", f"{status_icon} {'Running' if status['is_running'] else 'Stopped'}")
            
            with col_stat2:
                st.metric("Daily Orders", f"{status['daily_orders']}/{status['max_daily_orders']}")
            
            with col_stat3:
                st.metric("Watchlist Size", status['watchlist_size'])
            
            with col_stat4:
                if status['config'].get('test_mode', False):
                    hours_status = "🧪 Test Mode"
                else:
                    hours_status = "✅ Yes" if status['in_trading_hours'] else "❌ No"
                st.metric("Trading Hours", hours_status)
            
            # Short positions display (if enabled)
            if status.get('short_positions', 0) > 0:
                st.divider()
                st.subheader("📉 Active Short Positions")
                short_details = status.get('short_positions_details', [])
                for short_pos in short_details:
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.write(f"**{short_pos['symbol']}**")
                    with col_s2:
                        st.write(f"Qty: {short_pos['quantity']}")
                    with col_s3:
                        st.write(f"Entry: ${short_pos['entry_price']:.2f}")
                    with col_s4:
                        st.write(f"Time: {short_pos['entry_time'][:16]}")
            
            # Configuration display
            with st.expander("⚙️ Current Configuration"):
                st.json(status['config'])
            
            # Execution history (moved to separate tab - keeping for backward compatibility)
            # Now primarily shown in "📓 Trade History" tab
        else:
            st.info("Auto-Trader is not running. Configure settings above and click 'Start Auto-Trader'.")
    
    # Help section
    with st.expander("❓ How It Works"):
        st.markdown("""
### Automated Trading Process

1. **Monitoring**: The bot scans your watchlist every X minutes
2. **Analysis**: Generates AI signals using comprehensive analysis
3. **Filtering**: Only executes signals above your confidence threshold
4. **Execution**: Places bracket orders with stop-loss and take-profit
5. **Safety**: Respects daily limits and trading hours

### Safety Features

- ✅ **Trading Hours**: Only trades during market hours (9:30 AM - 3:30 PM ET)
- ✅ **Daily Limits**: Stops after max daily orders reached
- ✅ **Confidence Filter**: Only executes high-confidence signals
- ✅ **Bracket Orders**: Automatic stop-loss protection
- ✅ **Paper Trading**: Test mode before using real money
- ✅ **Position Checks**: Won't add to existing positions
- ✅ **Short Selling**: Supports shorting in paper trading mode

### Short Selling (Advanced - Disabled by Default)

⚠️ **NOT recommended for scalping or cash account strategies!**

When enabled (paper trading only), SELL signals can open short positions:
- **Requires**: Margin account with sufficient equity
- **Best for**: Advanced swing trading or hedge strategies
- **Not for**: Scalping, day trading with cash accounts
- **BUY signals**: Opens long positions or covers shorts
- **SELL signals**: Closes long positions or opens shorts

**For scalping**: Keep this DISABLED. Only sell stocks you own!

### Trading Modes

**STOCKS**: Standard stock trading with AI signals
**OPTIONS**: Options strategies (coming soon)
**SCALPING**: Fast intraday trades with tight stops
- Default: 2% profit target, 1% stop loss
- Orders close same day
- Scan interval: 5-15 minutes recommended
- Best for: High-volume, liquid stocks

**WARRIOR_SCALPING**: Gap & Go strategy (Ross Cameron's approach)
- Focus: 9:30-10:00 AM momentum window
- Filters: $2-$20 price, 4-10% gap, 2-3x volume
- Setups: Gap & Go, Micro Pullback, Red-to-Green, Bull Flag
- Targets: 2% profit, 1% stop loss
- Best for: Premarket gappers with morning momentum

**ALL**: Combines all strategies

### Best Practices

1. **Start with Paper Trading** - Test for at least a week
2. **Monitor Daily** - Check execution history regularly
3. **Start Small** - Use low max daily orders (5-10)
4. **High Confidence** - Keep min confidence at 75%+
5. **Diversify** - Monitor 5-10 different tickers
6. **Review Results** - Analyze what works and adjust
7. **Scalping Tips**: Use 5-10 min intervals, liquid stocks only

### Risk Warning

⚠️ Automated trading carries significant risk. Past performance doesn't guarantee future results. 
Always start with paper trading and only risk capital you can afford to lose.
        """)
    
    # ========================================================================
    # MULTI-CONFIGURATION BULK ANALYSIS
    # ========================================================================
    
    st.divider()
    st.header("🎯 Multi-Configuration Analysis")
    st.write("Test multiple configurations on your watchlist to find optimal trading setups before automating.")
    
    # Get ticker manager and AI assistant
    tm = st.session_state.get('ticker_manager')
    entry_assistant = st.session_state.get('stock_ai_entry_assistant')
    
    if tm and entry_assistant:
        # Get all tickers from watchlist
        try:
            all_tickers = tm.get_all_tickers()
            
            if all_tickers:
                ticker_list = [t['ticker'] for t in all_tickers]
                
                # Import multi-config UI
                from ui.bulk_ai_entry_analysis_ui import display_multi_config_bulk_analysis
                
                st.info("💡 Use this to test different position sizes, risk levels, and trading styles across your watchlist. Save the best configurations to populate AI entry actions for filtering.")
                
                # Display multi-config analysis
                display_multi_config_bulk_analysis(ticker_list, entry_assistant, tm)
            else:
                st.info("No tickers in your watchlist. Add some tickers in the Watchlist tab first.")
        except Exception as e:
            logger.error(f"Error loading tickers for multi-config: {e}")
            st.error(f"Failed to load tickers: {str(e)}")
    else:
        st.warning("⚠️ Multi-config analysis requires:")
        debug_info = []
        
        if not tm:
            debug_info.append("❌ Ticker Manager not initialized")
        else:
            debug_info.append("✅ Ticker Manager ready")
        
        if not entry_assistant:
            debug_info.append("❌ AI Entry Assistant not initialized - needs broker + LLM")
        else:
            debug_info.append("✅ AI Entry Assistant ready")
        
        for info in debug_info:
            st.write(info)
        
        if not entry_assistant:
            st.info("💡 Connect a broker (IBKR or Tradier) and configure LLM to enable AI analysis")


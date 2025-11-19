"""
UI Component for Bulk AI Entry Analysis

Provides a user interface for running AI entry analysis on multiple stock tickers at once.
Includes multi-configuration testing similar to crypto trading.
"""

import streamlit as st
import pandas as pd
from loguru import logger
from typing import List, Dict, Optional
from datetime import datetime

from services.ai_stock_entry_assistant import AIStockEntryAssistant, get_ai_stock_entry_assistant
from services.ticker_manager import TickerManager
from services.hybrid_llm_manager import get_llm_for_bulk_operations

def display_bulk_ai_entry_analysis(tickers: List[str], entry_assistant: AIStockEntryAssistant, ticker_manager: TickerManager):
    """
    Renders the UI for bulk AI entry analysis.

    Args:
        tickers (List[str]): A list of ticker symbols to analyze.
        entry_assistant (AIStockEntryAssistant): The AI assistant instance for analysis.
        ticker_manager (TickerManager): The ticker manager for saving results.
    """
    st.markdown("#### 🧠 Bulk AI Entry Analysis")
    
    # Speed mode selector
    col_title, col_speed = st.columns([3, 1])
    with col_title:
        st.write("Analyze multiple tickers at once to find the best entry opportunities.")
    with col_speed:
        use_fast_mode = st.toggle(
            "⚡ Fast Mode",
            value=True,
            help="Use fast cloud API instead of local Ollama (76s → ~3s per analysis)",
            key="bulk_fast_mode"
        )

    if not tickers:
        st.info("Add tickers to your watchlist to use this feature.")
        return

    # --- Selection UI ---
    with st.container():
        st.write("**1. Select Tickers to Analyze**")
        
        # Create a DataFrame for selection
        df = pd.DataFrame({"Ticker": tickers})
        df["Select"] = True  # Default to all selected
        
        edited_df = st.data_editor(
            df,
            key="bulk_ai_ticker_selection",
            width='stretch',
            hide_index=True,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", default=True),
                "Ticker": st.column_config.TextColumn("Ticker", disabled=True)
            }
        )
        
        selected_tickers = edited_df[edited_df["Select"]]["Ticker"].tolist()
        
        st.caption(f"{len(selected_tickers)} of {len(tickers)} tickers selected.")

    # --- Analysis Execution ---
    st.write("**2. Run Analysis**")
    
    # Create fast assistant if fast mode is enabled
    fast_assistant = None
    if use_fast_mode:
        with st.spinner("Initializing fast cloud LLM..."):
            try:
                fast_llm = get_llm_for_bulk_operations()
                if fast_llm and entry_assistant.broker_client:
                    fast_assistant = get_ai_stock_entry_assistant(
                        broker_client=entry_assistant.broker_client,
                        llm_analyzer=fast_llm
                    )
                    st.success("✅ Fast mode enabled - using cloud API")
                else:
                    st.warning("⚠️ Fast mode unavailable, using local Ollama")
            except Exception as e:
                logger.warning(f"Could not initialize fast mode: {e}")
                st.warning(f"⚠️ Fast mode failed: {e}. Using local Ollama.")
    
    # Use fast assistant if available, otherwise use default
    active_assistant = fast_assistant if fast_assistant else entry_assistant
    
    if st.button(f"🤖 Analyze {len(selected_tickers)} Selected Tickers", type="primary", disabled=not selected_tickers):
        st.session_state.bulk_ai_analysis_results = []
        progress_bar = st.progress(0, text="Starting bulk analysis...")
        results = []

        for i, ticker in enumerate(selected_tickers):
            progress_text = f"Analyzing {ticker} ({i + 1}/{len(selected_tickers)})..."
            progress_bar.progress((i + 1) / len(selected_tickers), text=progress_text)
            
            try:
                analysis = active_assistant.analyze_entry(
                    symbol=ticker,
                    side="BUY",  # Default to BUY for bulk analysis
                    position_size=1000, # Standardized for comparison
                    risk_pct=2.0,
                    take_profit_pct=6.0
                )
                
                result_data = {
                    "Ticker": ticker,
                    "Action": analysis.action,
                    "Confidence": analysis.confidence,
                    "Urgency": analysis.urgency,
                    "Reasoning": analysis.reasoning,
                    "Technical": analysis.technical_score,
                    "Timing": analysis.timing_score,
                    "Risk": analysis.risk_score
                }
                results.append(result_data)

                # Save to DB
                analysis_to_save = {
                    'confidence': analysis.confidence,
                    'action': analysis.action,
                    'reasons': [analysis.reasoning] if analysis.reasoning else [],
                    'targets': {
                        'suggested_entry': analysis.suggested_entry,
                        'suggested_stop': analysis.suggested_stop,
                        'suggested_target': analysis.suggested_target
                    }
                }
                ticker_manager.update_ai_entry_analysis(ticker, analysis_to_save)

            except Exception as e:
                logger.error(f"Bulk analysis for {ticker} failed: {e}")
                results.append({"Ticker": ticker, "Action": "ERROR", "Confidence": 0, "Reasoning": str(e)})

        st.session_state.bulk_ai_analysis_results = results
        progress_bar.empty()
        st.toast("✅ Bulk analysis complete!", icon="🎉")

    # --- Results Display ---
    if 'bulk_ai_analysis_results' in st.session_state and st.session_state.bulk_ai_analysis_results:
        st.write("**3. Analysis Results**")
        
        results_df = pd.DataFrame(st.session_state.bulk_ai_analysis_results)
        results_df = results_df.sort_values(by="Confidence", ascending=False)

        # Action-based color coding for the table
        def get_action_badge(action):
            if action == "ENTER_NOW":
                return f"🟢 {action}"
            elif action in ["WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT", "PLACE_LIMIT_ORDER"]:
                return f"🟡 {action}"
            elif action == "DO_NOT_ENTER":
                return f"🔴 {action}"
            else:
                return f"⚫️ {action}"

        results_df["Action"] = results_df["Action"].apply(get_action_badge)

        st.dataframe(
            results_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
                "Technical": st.column_config.NumberColumn("Tech", format="%.0f"),
                "Timing": st.column_config.NumberColumn("Time", format="%.0f"),
                "Risk": st.column_config.NumberColumn("Risk", format="%.0f"),
            },
            column_order=["Ticker", "Action", "Confidence", "Urgency", "Technical", "Timing", "Risk", "Reasoning"]
        )


def analyze_multi_config_bulk(tickers: List[str], entry_assistant: AIStockEntryAssistant, ticker_manager: TickerManager, 
                              position_sizes: List[float], risk_levels: List[float], trading_styles: List[str]) -> List[Dict]:
    """
    Analyze multiple tickers with multiple configurations (position size, risk, style).
    Similar to crypto multi-config analysis.
    
    Args:
        tickers: List of ticker symbols
        entry_assistant: AI Entry Assistant instance
        ticker_manager: Ticker manager for saving results
        position_sizes: List of position sizes in USD
        risk_levels: List of risk percentages (e.g., [1.0, 2.0, 3.0])
        trading_styles: List of trading styles (e.g., ['SWING', 'DAY_TRADE', 'SCALP'])
    
    Returns:
        List of analysis results with configuration details
    """
    all_results = []
    
    for ticker in tickers:
        for pos_size in position_sizes:
            for risk_pct in risk_levels:
                for style in trading_styles:
                    try:
                        # Calculate take profit based on style
                        if style == 'SCALP':
                            tp_pct = risk_pct * 1.5  # 1.5:1 R:R for scalping
                        elif style == 'DAY_TRADE':
                            tp_pct = risk_pct * 2.0  # 2:1 R:R for day trading
                        else:  # SWING
                            tp_pct = risk_pct * 3.0  # 3:1 R:R for swing trading
                        
                        # Run AI analysis
                        analysis = entry_assistant.analyze_entry(
                            symbol=ticker,
                            side="BUY",
                            position_size=pos_size,
                            risk_pct=risk_pct,
                            take_profit_pct=tp_pct
                        )
                        
                        # Calculate composite score
                        composite_score = analysis.confidence
                        
                        # Add configuration details
                        result = {
                            'ticker': ticker,
                            'position_size': pos_size,
                            'risk_pct': risk_pct,
                            'tp_pct': tp_pct,
                            'style': style,
                            'action': analysis.action,
                            'confidence': analysis.confidence,
                            'composite_score': composite_score,
                            'urgency': analysis.urgency,
                            'reasoning': analysis.reasoning,
                            'technical_score': analysis.technical_score,
                            'trend_score': analysis.trend_score,
                            'timing_score': analysis.timing_score,
                            'risk_score': analysis.risk_score,
                            'suggested_entry': analysis.suggested_entry,
                            'suggested_stop': analysis.suggested_stop,
                            'suggested_target': analysis.suggested_target,
                            'current_price': analysis.current_price,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        all_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Multi-config analysis failed for {ticker} (${pos_size}, {risk_pct}%, {style}): {e}")
                        continue
    
    return all_results


def display_multi_config_bulk_analysis(tickers: List[str], entry_assistant: AIStockEntryAssistant, ticker_manager: TickerManager):
    """
    Display UI for multi-configuration bulk analysis (like crypto).
    Tests multiple configurations per ticker and shows best setups.
    """
    st.markdown("#### 🎯 Multi-Configuration Bulk Analysis")
    st.write("Test ALL configurations for each ticker and find the optimal setup.")
    
    # Initialize shortlist in session state
    if 'config_shortlist' not in st.session_state:
        st.session_state.config_shortlist = []
    
    # Display Shortlist section at the top
    if st.session_state.config_shortlist:
        with st.expander(f"⭐ **YOUR SHORTLIST** ({len(st.session_state.config_shortlist)} configs)", expanded=True):
            st.markdown("**Saved configurations for quick access**")
            st.caption("These are your hand-picked setups. Monitor these for entry signals!")
            
            for idx, config in enumerate(st.session_state.config_shortlist):
                action_emoji = "🟢" if config['action'] == 'ENTER_NOW' else "🟡" if 'WAIT' in config['action'] else "🔴"
                
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"{action_emoji} **{config['ticker']}** - {config['style']} | ${config['position_size']:,.0f} | Score: {config['composite_score']:.1f}% | {config['action']}")
                    st.caption(f"Entry: ${config['suggested_entry']:.2f} | Stop: ${config['suggested_stop']:.2f} | Target: ${config['suggested_target']:.2f} | Added: {config.get('shortlisted_at', 'N/A')}")
                
                with col2:
                    if st.button("🗑️ Remove", key=f"remove_shortlist_{idx}"):
                        st.session_state.config_shortlist.pop(idx)
                        st.rerun()
            
            # Bulk actions
            col_export, col_clear = st.columns(2)
            
            with col_export:
                if st.button("📥 Export Shortlist to CSV"):
                    df_shortlist = pd.DataFrame(st.session_state.config_shortlist)
                    csv = df_shortlist.to_csv(index=False)
                    st.download_button(
                        label="Download Shortlist CSV",
                        data=csv,
                        file_name=f"config_shortlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col_clear:
                if st.button("🗑️ Clear All Shortlist"):
                    if st.session_state.get('confirm_clear_shortlist'):
                        st.session_state.config_shortlist = []
                        st.session_state.confirm_clear_shortlist = False
                        st.success("✅ Shortlist cleared!")
                        st.rerun()
                    else:
                        st.session_state.confirm_clear_shortlist = True
                        st.warning("⚠️ Click again to confirm")
            
            st.divider()
    
    if not tickers:
        st.info("Add tickers to your watchlist to use this feature.")
        return
    
    # Fast mode toggle for multi-config
    col_config_title, col_config_speed = st.columns([3, 1])
    with col_config_title:
        st.write("Configure and run comprehensive analysis on multiple tickers")
    with col_config_speed:
        use_fast_mode_mc = st.toggle(
            "⚡ Fast Mode",
            value=True,
            help="Use fast cloud API instead of local Ollama",
            key="mc_fast_mode"
        )
    
    # Configuration Panel
    with st.expander("⚙️ Configuration Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Trading Styles to Test**")
            test_swing = st.checkbox("📈 Swing Trading (3:1 R:R)", value=True, key="mc_swing")
            test_day = st.checkbox("⚡ Day Trading (2:1 R:R)", value=True, key="mc_day")
            test_scalp = st.checkbox("🔥 Scalping (1.5:1 R:R)", value=False, key="mc_scalp")
        
        with col2:
            st.write("**Position Sizing & Risk**")
            
            # Position sizes
            default_pos_sizes = [1000, 2000, 5000]
            pos_size_input = st.text_input(
                "Position Sizes (USD, comma-separated)",
                value="1000,2000,5000",
                key="mc_pos_sizes"
            )
            
            # Risk levels
            default_risk_levels = [1.0, 2.0, 3.0]
            risk_input = st.text_input(
                "Risk Levels (%, comma-separated)",
                value="1.0,2.0,3.0",
                key="mc_risks"
            )
        
        # Ticker selection
        st.write("**Tickers to Analyze**")
        
        # Option to use watchlist or custom tickers
        ticker_source = st.radio(
            "Ticker Source",
            ["📋 From Watchlist", "✏️ Custom Input"],
            key="mc_ticker_source",
            horizontal=True
        )
        
        if ticker_source == "📋 From Watchlist":
            if tickers:
                # Helper functions for selection callbacks
                def select_all_watchlist():
                    st.session_state.mc_watchlist_selection = tickers
                
                def clear_watchlist():
                    st.session_state.mc_watchlist_selection = []

                # Initialize selection in session state if not present
                if 'mc_watchlist_selection' not in st.session_state:
                    # Default to first 5 or all if less than 5
                    st.session_state.mc_watchlist_selection = tickers[:5] if len(tickers) >= 5 else tickers

                # Selection buttons similar to crypto UI
                col_btn1, col_btn2, _ = st.columns([1, 1, 2])
                with col_btn1:
                    st.button("✅ Select All", on_click=select_all_watchlist, key="mc_btn_select_all", use_container_width=True)
                with col_btn2:
                    st.button("❌ Clear All", on_click=clear_watchlist, key="mc_btn_clear_all", use_container_width=True)

                # Searchable Dropdown (Multiselect)
                selected_tickers = st.multiselect(
                    "Select Tickers",
                    options=tickers,
                    key="mc_watchlist_selection",
                    help="Start typing to search your watchlist or select from the dropdown"
                )
                
                if selected_tickers:
                    st.caption(f"Will analyze {len(selected_tickers)} tickers: {', '.join(selected_tickers[:10])}{' ...' if len(selected_tickers) > 10 else ''}")
                else:
                    st.warning("⚠️ Please select at least one ticker from the dropdown.")
            else:
                st.warning("No tickers in watchlist. Add some or use Custom Input.")
                selected_tickers = []
        else:
            # Custom ticker input
            custom_tickers = st.text_input(
                "Enter ticker symbols (comma-separated)",
                value="AAPL,TSLA,NVDA",
                key="mc_custom_tickers",
                help="Example: AAPL,TSLA,NVDA,MSFT"
            )
            if custom_tickers:
                selected_tickers = [t.strip().upper() for t in custom_tickers.split(',') if t.strip()]
                st.caption(f"Will analyze: {', '.join(selected_tickers)}")
            else:
                st.warning("Please enter at least one ticker symbol.")
                selected_tickers = []
    
    # Build configuration lists
    trading_styles = []
    if test_swing:
        trading_styles.append('SWING')
    if test_day:
        trading_styles.append('DAY_TRADE')
    if test_scalp:
        trading_styles.append('SCALP')
    
    try:
        position_sizes = [float(x.strip()) for x in pos_size_input.split(',')]
        risk_levels = [float(x.strip()) for x in risk_input.split(',')]
    except ValueError:
        st.error("Invalid input format. Please use comma-separated numbers.")
        return
    
    if not trading_styles:
        st.warning("Please select at least one trading style.")
        return
    
    if not selected_tickers:
        st.warning("Please select tickers from watchlist or enter custom tickers.")
        return
    
    # Calculate total configurations
    total_configs = len(selected_tickers) * len(position_sizes) * len(risk_levels) * len(trading_styles)
    
    st.info(f"📊 Will test **{total_configs} configurations** ({len(selected_tickers)} tickers × {len(position_sizes)} position sizes × {len(risk_levels)} risk levels × {len(trading_styles)} styles)")
    
    # Initialize fast assistant if fast mode enabled
    fast_assistant_mc = None
    if use_fast_mode_mc:
        with st.spinner("Initializing fast cloud LLM..."):
            try:
                fast_llm = get_llm_for_bulk_operations()
                if fast_llm and entry_assistant.broker_client:
                    fast_assistant_mc = get_ai_stock_entry_assistant(
                        broker_client=entry_assistant.broker_client,
                        llm_analyzer=fast_llm
                    )
                    st.success("✅ Fast mode enabled - multi-config will use cloud API")
                else:
                    st.warning("⚠️ Fast mode unavailable, using local Ollama")
            except Exception as e:
                logger.warning(f"Could not initialize fast mode: {e}")
                st.warning(f"⚠️ Fast mode failed: {e}. Using local Ollama.")
    
    # Use fast assistant if available
    active_assistant_mc = fast_assistant_mc if fast_assistant_mc else entry_assistant
    
    # Run Analysis Button
    if st.button(f"🚀 Analyze All Configurations ({total_configs} total)", type="primary", key="mc_analyze"):
        with st.spinner(f"Running {total_configs} AI analyses..."):
            progress_bar = st.progress(0, text="Starting multi-config analysis...")
            
            # Track progress
            completed = 0
            
            # Run analysis with progress updates
            all_results = []
            
            for ticker_idx, ticker in enumerate(selected_tickers):
                for pos_size in position_sizes:
                    for risk_pct in risk_levels:
                        for style in trading_styles:
                            completed += 1
                            progress_pct = completed / total_configs
                            progress_bar.progress(progress_pct, text=f"Analyzing {ticker} ({completed}/{total_configs})...")
                            
                            try:
                                # Calculate take profit based on style
                                if style == 'SCALP':
                                    tp_pct = risk_pct * 1.5
                                elif style == 'DAY_TRADE':
                                    tp_pct = risk_pct * 2.0
                                else:  # SWING
                                    tp_pct = risk_pct * 3.0
                                
                                # Run AI analysis
                                analysis = active_assistant_mc.analyze_entry(
                                    symbol=ticker,
                                    side="BUY",
                                    position_size=pos_size,
                                    risk_pct=risk_pct,
                                    take_profit_pct=tp_pct
                                )
                                
                                # Calculate composite score
                                composite_score = analysis.confidence
                                
                                result = {
                                    'ticker': ticker,
                                    'position_size': pos_size,
                                    'risk_pct': risk_pct,
                                    'tp_pct': tp_pct,
                                    'style': style,
                                    'action': analysis.action,
                                    'confidence': analysis.confidence,
                                    'composite_score': composite_score,
                                    'urgency': analysis.urgency,
                                    'reasoning': analysis.reasoning,
                                    'technical_score': analysis.technical_score,
                                    'trend_score': analysis.trend_score,
                                    'timing_score': analysis.timing_score,
                                    'risk_score': analysis.risk_score,
                                    'suggested_entry': analysis.suggested_entry,
                                    'suggested_stop': analysis.suggested_stop,
                                    'suggested_target': analysis.suggested_target,
                                    'current_price': analysis.current_price
                                }
                                
                                all_results.append(result)
                                
                            except Exception as e:
                                logger.error(f"Config analysis failed for {ticker}: {e}")
                                continue
            
            progress_bar.empty()
            st.session_state.multi_config_results = all_results
            st.toast("✅ Multi-config analysis complete!", icon="🎉")
    
    # Display Results
    if 'multi_config_results' in st.session_state:
        results = st.session_state.multi_config_results
        
        # Handle both list and DataFrame types
        if isinstance(results, pd.DataFrame):
            if results.empty:
                return
            results = results.to_dict('records')
        elif not results:  # Empty list or None
            return
        
        st.markdown("---")
        st.subheader("📊 Multi-Configuration Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Configs", len(results))
        with col2:
            enter_now_count = sum(1 for r in results if r['action'] == 'ENTER_NOW')
            st.metric("ENTER NOW", enter_now_count, delta=f"{enter_now_count/len(results)*100:.0f}%")
        with col3:
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        with col4:
            best_score = max(r['composite_score'] for r in results)
            st.metric("Best Score", f"{best_score:.1f}%")
        
        # Best configuration per ticker
        st.markdown("### 🏆 Best Configuration Per Ticker")
        
        # Group by ticker and find best config
        ticker_best = {}
        for result in results:
            ticker = result['ticker']
            if ticker not in ticker_best or result['composite_score'] > ticker_best[ticker]['composite_score']:
                ticker_best[ticker] = result
        
        # Sort by score
        sorted_best = sorted(ticker_best.values(), key=lambda x: x['composite_score'], reverse=True)
        
        for result in sorted_best:
            action_emoji = "🟢" if result['action'] == 'ENTER_NOW' else "🟡" if 'WAIT' in result['action'] else "🔴"
            
            with st.expander(f"{action_emoji} **{result['ticker']}** - {result['style']} | Score: {result['composite_score']:.1f}% | {result['action']}"):
                # Configuration details
                conf_col1, conf_col2, conf_col3 = st.columns(3)
                
                with conf_col1:
                    st.metric("Position Size", f"${result['position_size']:,.0f}")
                    st.metric("Risk", f"{result['risk_pct']}%")
                with conf_col2:
                    st.metric("Trading Style", result['style'])
                    st.metric("Take Profit", f"{result['tp_pct']:.1f}%")
                with conf_col3:
                    st.metric("AI Confidence", f"{result['confidence']:.1f}%")
                    st.metric("Urgency", result['urgency'])
                
                # Pricing
                st.write("**💰 Pricing:**")
                price_col1, price_col2, price_col3, price_col4 = st.columns(4)
                with price_col1:
                    st.write(f"Current: **${result['current_price']:.2f}**")
                with price_col2:
                    st.write(f"Entry: **${result['suggested_entry']:.2f}**")
                with price_col3:
                    st.write(f"Stop: **${result['suggested_stop']:.2f}**")
                with price_col4:
                    st.write(f"Target: **${result['suggested_target']:.2f}**")
                
                # Scores
                st.write("**📊 Analysis Scores:**")
                score_col1, score_col2, score_col3, score_col4 = st.columns(4)
                with score_col1:
                    st.write(f"Technical: **{result['technical_score']:.0f}**/100")
                with score_col2:
                    st.write(f"Trend: **{result['trend_score']:.0f}**/100")
                with score_col3:
                    st.write(f"Timing: **{result['timing_score']:.0f}**/100")
                with score_col4:
                    st.write(f"Risk: **{result['risk_score']:.0f}**/100")
                
                # AI Reasoning
                st.write("**🤖 AI Analysis:**")
                st.info(result['reasoning'])
                
                # Action buttons
                btn_col1, btn_col2 = st.columns(2)
                
                with btn_col1:
                    # Check if already shortlisted
                    is_shortlisted = any(
                        s['ticker'] == result['ticker'] and 
                        s['style'] == result['style'] and 
                        s['position_size'] == result['position_size']
                        for s in st.session_state.config_shortlist
                    )
                    
                    if not is_shortlisted:
                        if st.button(f"⭐ Add to Shortlist", key=f"shortlist_{result['ticker']}_{result['style']}_{result['position_size']}"):
                            # Add to shortlist with timestamp
                            shortlist_entry = result.copy()
                            shortlist_entry['shortlisted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            st.session_state.config_shortlist.append(shortlist_entry)
                            
                            # Send Discord notification
                            try:
                                from src.integrations.discord_webhook import send_discord_alert
                                from models.alerts import TradingAlert, AlertType, AlertPriority
                                
                                alert = TradingAlert(
                                    ticker=result['ticker'],
                                    alert_type=AlertType.BUY_SIGNAL if 'BUY' in result['action'] or 'ENTER' in result['action'] else AlertType.REVIEW_REQUIRED,
                                    message=f"⭐ Configuration Shortlisted: {result['style']} | {result['action']}",
                                    priority=AlertPriority.HIGH if result['action'] == 'ENTER_NOW' else AlertPriority.MEDIUM,
                                    details={
                                        'entry_price': result['suggested_entry'],
                                        'target_price': result['suggested_target'],
                                        'stop_loss': result['suggested_stop'],
                                        'position_size': f"${result['position_size']:,.0f}",
                                        'risk_reward': f"{result['tp_pct'] / result['risk_pct']:.2f}" if result['risk_pct'] > 0 else "N/A",
                                        'reasoning': f"{result['style']} setup | Confidence: {result['confidence']:.1f}% | {result['reasoning'][:500]}"
                                    }
                                )
                                send_discord_alert(alert)
                                logger.info(f"✅ Discord alert sent for shortlisted config: {result['ticker']}")
                            except Exception as e:
                                logger.debug(f"Discord notification not available or failed: {e}")
                            
                            st.success(f"⭐ Added {result['ticker']} to shortlist!")
                            st.rerun()
                    else:
                        st.success("✅ Already in shortlist")
                
                with btn_col2:
                    # Save to database button
                    if st.button(f"💾 Save to DB", key=f"save_{result['ticker']}"):
                        analysis_to_save = {
                            'confidence': result['confidence'],
                            'action': result['action'],
                            'reasons': [result['reasoning']],
                            'targets': {
                                'suggested_entry': result['suggested_entry'],
                                'suggested_stop': result['suggested_stop'],
                                'suggested_target': result['suggested_target'],
                                'position_size': result['position_size'],
                                'risk_pct': result['risk_pct'],
                                'style': result['style']
                            }
                        }
                        if ticker_manager.update_ai_entry_analysis(result['ticker'], analysis_to_save):
                            st.success(f"✅ Saved to database!")
                        else:
                            st.error(f"❌ Failed to save")
        
        # Full comparison table
        st.markdown("### 📋 Full Configuration Comparison")
        
        # Filter controls
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            action_filter = st.selectbox(
                "Filter by Action",
                ["All", "ENTER_NOW", "WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT", "DO_NOT_ENTER"],
                key="mc_action_filter"
            )
        with filter_col2:
            min_confidence = st.slider("Min Confidence %", 0, 100, 0, key="mc_min_conf")
        with filter_col3:
            style_filter = st.multiselect(
                "Filter by Style",
                ["SWING", "DAY_TRADE", "SCALP"],
                default=["SWING", "DAY_TRADE", "SCALP"],
                key="mc_style_filter"
            )
        
        # Apply filters
        filtered_results = results
        if action_filter != "All":
            filtered_results = [r for r in filtered_results if r['action'] == action_filter]
        if min_confidence > 0:
            filtered_results = [r for r in filtered_results if r['confidence'] >= min_confidence]
        if style_filter:
            filtered_results = [r for r in filtered_results if r['style'] in style_filter]
        
        st.caption(f"Showing {len(filtered_results)} of {len(results)} configurations")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(filtered_results)
        
        if not df.empty:
            # Sort by composite score
            df = df.sort_values('composite_score', ascending=False)
            
            # Format columns for display
            display_df = df[[
                'ticker', 'action', 'confidence', 'style', 'position_size', 
                'risk_pct', 'tp_pct', 'urgency', 'reasoning'
            ]].copy()
            
            display_df.columns = [
                'Ticker', 'Action', 'Confidence', 'Style', 'Position',
                'Risk %', 'TP %', 'Urgency', 'Reasoning'
            ]
            
            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Position": st.column_config.NumberColumn("Position", format="$%d"),
                    "Risk %": st.column_config.NumberColumn("Risk %", format="%.1f%%"),
                    "TP %": st.column_config.NumberColumn("TP %", format="%.1f%%"),
                    "Reasoning": st.column_config.TextColumn("AI Reasoning", width="large"),
                }
            )
            
            # Export option
            if st.button("📥 Export Results to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"multi_config_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No configurations match your filters.")

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
                pass  # pass  # pass  # pass  # pass  # pass  # pass  # pass  # pass  # pass  # logger.warning(f"Could not initialize fast mode: {e}")
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


def display_multi_config_bulk_analysis(tickers: List[str], entry_assistant: AIStockEntryAssistant, ticker_manager: TickerManager):
    """
    Renders the UI for multi-configuration bulk AI entry analysis.
    Tests multiple position sizes, risk levels, and trading styles to find optimal setups.

    Args:
        tickers (List[str]): A list of ticker symbols to analyze.
        entry_assistant (AIStockEntryAssistant): The AI assistant instance for analysis.
        ticker_manager (TickerManager): The ticker manager for saving results.
    """
    st.markdown("#### 🎯 Multi-Configuration Bulk Analysis")
    st.caption("Test different position sizes, risk levels, and trading styles to find optimal setups")
    
    if not tickers:
        st.info("Add tickers to your watchlist to use this feature.")
        return
    
    # --- Configuration UI ---
    st.write("**1. Configure Test Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Trading Styles**")
        use_swing = st.checkbox("Swing Trading (3:1 R:R)", value=True, key="multi_config_swing")
        use_day = st.checkbox("Day Trading (2:1 R:R)", value=True, key="multi_config_day")
        use_scalp = st.checkbox("Scalping (1.5:1 R:R)", value=False, key="multi_config_scalp")
        
        selected_styles = []
        if use_swing:
            selected_styles.append("SWING")
        if use_day:
            selected_styles.append("DAY_TRADE")
        if use_scalp:
            selected_styles.append("SCALP")
        
        if not selected_styles:
            st.warning("⚠️ Select at least one trading style")
    
    with col2:
        st.write("**Position Sizes (USD)**")
        position_sizes_input = st.text_input(
            "Comma-separated values",
            value="1000,2000,5000",
            help="e.g., 1000,2000,5000",
            key="multi_config_positions"
        )
        
        st.write("**Risk Levels (%)**")
        risk_levels_input = st.text_input(
            "Comma-separated values",
            value="1.0,2.0,3.0",
            help="e.g., 1.0,2.0,3.0",
            key="multi_config_risks"
        )
    
    # Parse position sizes and risk levels
    try:
        position_sizes = [float(x.strip()) for x in position_sizes_input.split(",") if x.strip()]
        risk_levels = [float(x.strip()) for x in risk_levels_input.split(",") if x.strip()]
    except ValueError:
        st.error("⚠️ Invalid format. Use comma-separated numbers (e.g., 1000,2000,5000)")
        return
    
    if not position_sizes or not risk_levels:
        st.error("⚠️ Position sizes and risk levels cannot be empty")
        return
    
    # Ticker selection
    st.write("**2. Select Tickers**")
    max_possible = min(20, len(tickers))
    default_tickers = min(5, len(tickers))
    if max_possible <= 1:
        # Avoid creating a slider with min==max; show fixed value instead
        st.write("Number of tickers to analyze:", max_possible)
        max_tickers = max_possible
    else:
        max_tickers = st.slider(
            "Number of tickers to analyze",
            min_value=1,
            max_value=max_possible,
            value=default_tickers,
            key="multi_config_max_tickers"
        )
    
    selected_tickers = tickers[:max_tickers]
    
    # Calculate total configurations
    total_configs = len(selected_tickers) * len(position_sizes) * len(risk_levels) * len(selected_styles)
    
    if total_configs == 0:
        st.warning("⚠️ No configurations to test. Select at least one trading style.")
        return
    
    st.info(f"🔬 Ready to test **{total_configs} configurations** across **{len(selected_tickers)} tickers**")
    st.caption(f"Test Matrix: {len(position_sizes)} positions × {len(risk_levels)} risks × {len(selected_styles)} styles = {len(position_sizes) * len(risk_levels) * len(selected_styles)} configs per ticker")
    
    # --- Analysis Execution ---
    st.write("**3. Run Analysis**")
    
    # Fast mode toggle
    use_fast_mode = st.toggle(
        "⚡ Fast Mode",
        value=True,
        help="Use fast cloud API instead of local Ollama",
        key="multi_config_fast_mode"
    )
    
    # Create fast assistant if fast mode is enabled
    fast_assistant = None
    if use_fast_mode:
        try:
            fast_llm = get_llm_for_bulk_operations()
            if fast_llm and entry_assistant.broker_client:
                fast_assistant = get_ai_stock_entry_assistant(
                    broker_client=entry_assistant.broker_client,
                    llm_analyzer=fast_llm
                )
        except Exception as e:
            logger.warning(f"Could not initialize fast mode: {e}")
    
    active_assistant = fast_assistant if fast_assistant else entry_assistant
    
    if st.button(f"🚀 Analyze All Configurations", type="primary"):
        st.session_state.multi_config_stock_results = []
        progress_bar = st.progress(0, text="Starting multi-config analysis...")
        status_text = st.empty()
        
        all_results = []
        combo_idx = 0
        
        # Risk/Reward ratios by style
        style_rr_ratios = {
            "SWING": 3.0,
            "DAY_TRADE": 2.0,
            "SCALP": 1.5
        }
        
        try:
            for ticker in selected_tickers:
                for position_size in position_sizes:
                    for risk_pct in risk_levels:
                        for style in selected_styles:
                            combo_idx += 1
                            progress = combo_idx / total_configs
                            progress_bar.progress(progress)
                            
                            # Calculate take profit based on style
                            take_profit_pct = risk_pct * style_rr_ratios.get(style, 2.0)
                            
                            status_text.text(f"Analyzing {ticker} - ${position_size:.0f}, {risk_pct}% risk, {style} ({combo_idx}/{total_configs})...")
                            
                            try:
                                analysis = active_assistant.analyze_entry(
                                    symbol=ticker,
                                    side="BUY",
                                    position_size=position_size,
                                    risk_pct=risk_pct,
                                    take_profit_pct=take_profit_pct
                                )
                                
                                result_data = {
                                    "Ticker": ticker,
                                    "Position Size": position_size,
                                    "Risk %": risk_pct,
                                    "Take Profit %": take_profit_pct,
                                    "Style": style,
                                    "Action": analysis.action,
                                    "Confidence": analysis.confidence,
                                    "Urgency": analysis.urgency,
                                    "Reasoning": analysis.reasoning,
                                    "Technical": analysis.technical_score,
                                    "Timing": analysis.timing_score,
                                    "Risk Score": analysis.risk_score,
                                    "Entry Price": analysis.suggested_entry,
                                    "Stop Price": analysis.suggested_stop,
                                    "Target Price": analysis.suggested_target
                                }
                                all_results.append(result_data)
                                
                            except Exception as e:
                                logger.error(f"Multi-config analysis for {ticker} failed: {e}")
                                import traceback
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                all_results.append({
                                    "Ticker": ticker,
                                    "Position Size": position_size,
                                    "Risk %": risk_pct,
                                    "Take Profit %": take_profit_pct,
                                    "Style": style,
                                    "Action": "ERROR",
                                    "Confidence": 0,
                                    "Reasoning": str(e),
                                    "Urgency": "LOW",
                                    "Technical": 0,
                                    "Timing": 0,
                                    "Risk Score": 0,
                                    "Entry Price": 0,
                                    "Stop Price": 0,
                                    "Target Price": 0
                                })
            
            progress_bar.empty()
            status_text.empty()
            
            if all_results:
                st.session_state.multi_config_stock_results = all_results
                st.toast("✅ Multi-config analysis complete!", icon="🎉")
            else:
                st.error("No results generated. Check logs for errors.")
                
        except Exception as e:
            logger.error("Multi-config bulk analysis error: {}", str(e), exc_info=True)
            st.error(f"Analysis failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # --- Results Display ---
    if 'multi_config_stock_results' in st.session_state and st.session_state.multi_config_stock_results:
        st.write("**4. Analysis Results**")
        
        results_df = pd.DataFrame(st.session_state.multi_config_stock_results)
        
        # Summary metrics
        st.markdown("##### 📊 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_configs_tested = len(results_df)
        enter_now_count = len(results_df[results_df["Action"] == "ENTER_NOW"])
        avg_confidence = results_df["Confidence"].mean()
        best_confidence = results_df["Confidence"].max()
        
        col1.metric("Total Configs Tested", total_configs_tested)
        col2.metric("ENTER NOW", f"{enter_now_count} ({enter_now_count/total_configs_tested*100:.1f}%)" if total_configs_tested > 0 else "0")
        col3.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        col4.metric("Best Confidence", f"{best_confidence:.1f}%")
        
        # Best configuration per ticker
        st.markdown("##### 🏆 Best Configuration Per Ticker")
        st.caption("Ranked by AI confidence score")
        
        best_per_ticker = results_df.loc[results_df.groupby("Ticker")["Confidence"].idxmax()]
        best_per_ticker = best_per_ticker.sort_values("Confidence", ascending=False)
        
        for idx, row in best_per_ticker.iterrows():
            with st.expander(f"🎯 {row['Ticker']} - ${row['Position Size']:.0f}, {row['Risk %']}% risk, {row['Style']} (Confidence: {row['Confidence']:.1f}%)"):
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.metric("Position Size", f"${row['Position Size']:.0f}")
                    st.metric("Risk %", f"{row['Risk %']:.1f}%")
                
                with col_info2:
                    st.metric("Take Profit %", f"{row['Take Profit %']:.1f}%")
                    st.metric("Trading Style", row['Style'])
                
                with col_info3:
                    st.metric("AI Confidence", f"{row['Confidence']:.1f}%")
                    action_badge = "🟢 ENTER_NOW" if row['Action'] == "ENTER_NOW" else \
                                  "🟡 WAIT" if "WAIT" in row['Action'] else \
                                  "🔴 DO_NOT_ENTER" if row['Action'] == "DO_NOT_ENTER" else f"⚫️ {row['Action']}"
                    st.write(f"**Action:** {action_badge}")
                
                if pd.notna(row.get('Entry Price')):
                    st.markdown("**Pricing:**")
                    pricing_col1, pricing_col2, pricing_col3 = st.columns(3)
                    pricing_col1.metric("Entry", f"${row['Entry Price']:.2f}")
                    pricing_col2.metric("Stop", f"${row['Stop Price']:.2f}")
                    pricing_col3.metric("Target", f"${row['Target Price']:.2f}")
                
                st.markdown("**Scores:**")
                scores_col1, scores_col2, scores_col3, scores_col4 = st.columns(4)
                scores_col1.metric("Technical", f"{row['Technical']:.0f}")
                scores_col2.metric("Timing", f"{row['Timing']:.0f}")
                scores_col3.metric("Risk", f"{row['Risk Score']:.0f}")
                scores_col4.metric("Urgency", row.get('Urgency', 'N/A'))
                
                if pd.notna(row.get('Reasoning')):
                    st.markdown("**AI Analysis:**")
                    st.write(row['Reasoning'])
                
                # Save button
                if st.button(f"💾 Save Best Config for {row['Ticker']}", key=f"save_best_{row['Ticker']}"):
                    try:
                        analysis_to_save = {
                            'confidence': row['Confidence'],
                            'action': row['Action'],
                            'reasons': [row['Reasoning']] if pd.notna(row.get('Reasoning')) else [],
                            'targets': {
                                'suggested_entry': row.get('Entry Price'),
                                'suggested_stop': row.get('Stop Price'),
                                'suggested_target': row.get('Target Price'),
                                'position_size': row['Position Size'],
                                'risk_pct': row['Risk %'],
                                'style': row['Style']
                            }
                        }
                        ticker_manager.update_ai_entry_analysis(row['Ticker'], analysis_to_save)
                        st.success(f"✅ Saved best configuration for {row['Ticker']}!")
                    except Exception as e:
                        logger.error("Error saving config for {row['Ticker']}: {}", str(e))
                        st.error(f"Failed to save: {str(e)}")
        
        # Full comparison table
        st.markdown("##### 📋 Full Comparison Table")
        
        # Filters
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            action_filter = st.multiselect(
                "Filter by Action",
                options=results_df["Action"].unique(),
                default=results_df["Action"].unique(),
                key="multi_config_action_filter"
            )
        
        with filter_col2:
            min_confidence = st.slider(
                "Min Confidence %",
                min_value=0,
                max_value=100,
                value=0,
                key="multi_config_min_confidence"
            )
        
        with filter_col3:
            style_filter = st.multiselect(
                "Filter by Style",
                options=results_df["Style"].unique(),
                default=results_df["Style"].unique(),
                key="multi_config_style_filter"
            )
        
        # Apply filters
        filtered_df = results_df[
            (results_df["Action"].isin(action_filter)) &
            (results_df["Confidence"] >= min_confidence) &
            (results_df["Style"].isin(style_filter))
        ].copy()
        
        if len(filtered_df) == 0:
            st.warning("No configurations match your filters.")
        else:
            # Format action column
            def get_action_badge(action):
                if action == "ENTER_NOW":
                    return f"🟢 {action}"
                elif action in ["WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT", "PLACE_LIMIT_ORDER"]:
                    return f"🟡 {action}"
                elif action == "DO_NOT_ENTER":
                    return f"🔴 {action}"
                else:
                    return f"⚫️ {action}"
            
            filtered_df["Action"] = filtered_df["Action"].apply(get_action_badge)
            
            # Sort by confidence
            filtered_df = filtered_df.sort_values("Confidence", ascending=False)
            
            st.dataframe(
                filtered_df,
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
                },
                column_order=["Ticker", "Position Size", "Risk %", "Style", "Action", "Confidence", "Urgency", "Technical", "Timing", "Risk Score", "Reasoning"]
            )
            
            # Export to CSV
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Export to CSV",
                data=csv,
                file_name=f"multi_config_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="multi_config_export_csv"
            )
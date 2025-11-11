"""
UI Component for Bulk AI Entry Analysis

Provides a user interface for running AI entry analysis on multiple stock tickers at once.
"""

import streamlit as st
import pandas as pd
from loguru import logger
from typing import List, Dict

from services.ai_stock_entry_assistant import AIStockEntryAssistant
from services.ticker_manager import TickerManager

def display_bulk_ai_entry_analysis(tickers: List[str], entry_assistant: AIStockEntryAssistant, ticker_manager: TickerManager):
    """
    Renders the UI for bulk AI entry analysis.

    Args:
        tickers (List[str]): A list of ticker symbols to analyze.
        entry_assistant (AIStockEntryAssistant): The AI assistant instance for analysis.
        ticker_manager (TickerManager): The ticker manager for saving results.
    """
    st.markdown("#### 🧠 Bulk AI Entry Analysis")
    st.write("Analyze multiple tickers at once to find the best entry opportunities.")

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
            use_container_width=True,
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
    if st.button(f"🤖 Analyze {len(selected_tickers)} Selected Tickers", type="primary", disabled=not selected_tickers):
        st.session_state.bulk_ai_analysis_results = []
        progress_bar = st.progress(0, text="Starting bulk analysis...")
        results = []

        for i, ticker in enumerate(selected_tickers):
            progress_text = f"Analyzing {ticker} ({i + 1}/{len(selected_tickers)})..."
            progress_bar.progress((i + 1) / len(selected_tickers), text=progress_text)
            
            try:
                analysis = entry_assistant.analyze_entry(
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
            use_container_width=True,
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

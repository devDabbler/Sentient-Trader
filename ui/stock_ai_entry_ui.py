"""
UI Component for Stock AI Entry Analysis
Provides instant AI-powered entry timing analysis for stock tickers
"""

import streamlit as st
from loguru import logger
from datetime import datetime
from typing import Optional

from services.ticker_manager import TickerManager


def display_stock_ai_entry_analysis(broker_client, llm_analyzer=None):
    """
    Display AI entry analysis form for stocks
    
    Args:
        broker_client: Tradier/IBKR/BrokerAdapter client
        llm_analyzer: LLM strategy analyzer for AI decisions
    """
    st.markdown("### 🤖 AI Entry Analysis")
    st.write("Get instant AI-powered analysis on whether now is a good time to enter a stock trade.")
    
    # Input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker_input = st.text_input(
            "Stock Ticker Symbol",
            value=st.session_state.get('stock_ai_entry_ticker', ''),
            key="stock_ai_entry_ticker",
            help="Enter any stock ticker (e.g., AAPL, TSLA, NVDA)",
            placeholder="e.g., AAPL"
        ).upper().strip()
    
    with col2:
        direction = st.radio(
            "Direction",
            options=["BUY", "SELL"],
            horizontal=True,
            key="stock_ai_direction"
        )
    
    # Position sizing
    st.markdown("#### Position Settings")
    pos_col1, pos_col2, pos_col3 = st.columns(3)
    
    with pos_col1:
        position_size = st.number_input(
            "Position Size (USD)",
            min_value=100.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
            key="stock_ai_position_size"
        )
    
    with pos_col2:
        risk_pct = st.number_input(
            "Risk % (Stop Loss)",
            min_value=0.5,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key="stock_ai_risk_pct"
        )
    
    with pos_col3:
        take_profit_pct = st.number_input(
            "Take Profit %",
            min_value=1.0,
            max_value=50.0,
            value=6.0,
            step=1.0,
            key="stock_ai_target_pct"
        )
    
    # AI Analysis button
    analysis_col1, analysis_col2 = st.columns([2, 1])
    
    with analysis_col1:
        if st.button("🤖 Analyze Entry Timing", width='stretch', type="primary", disabled=not ticker_input):
            if not ticker_input:
                st.warning("⚠️ Please enter a ticker symbol")
            else:
                with st.spinner(f"🤖 AI analyzing entry timing for {ticker_input}..."):
                    try:
                        # Initialize AI Stock Entry Assistant
                        from services.ai_stock_entry_assistant import get_ai_stock_entry_assistant
                        from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                        
                        if 'stock_ai_entry_assistant' not in st.session_state:
                            if llm_analyzer is None:
                                llm_analyzer = LLMStrategyAnalyzer()
                            
                            entry_assistant = get_ai_stock_entry_assistant(
                                broker_client=broker_client,
                                llm_analyzer=llm_analyzer,
                                check_interval_seconds=60,
                                enable_auto_entry=False  # Manual approval required
                            )
                            st.session_state.stock_ai_entry_assistant = entry_assistant
                            
                            # Start monitoring if not running
                            if not entry_assistant.is_running:
                                entry_assistant.start_monitoring()
                        else:
                            entry_assistant = st.session_state.stock_ai_entry_assistant
                        
                        # Get AI entry analysis
                        entry_analysis = entry_assistant.analyze_entry(
                            symbol=ticker_input,
                            side=direction,
                            position_size=position_size,
                            risk_pct=risk_pct,
                            take_profit_pct=take_profit_pct
                        )
                        
                        # Store in session state
                        st.session_state.stock_entry_analysis = entry_analysis
                        st.session_state.stock_entry_params = {
                            'ticker': ticker_input,
                            'direction': direction,
                            'current_price': entry_analysis.current_price,
                            'stop_loss': entry_analysis.suggested_stop or (entry_analysis.current_price * (1 - risk_pct / 100)),
                            'take_profit': entry_analysis.suggested_target or (entry_analysis.current_price * (1 + take_profit_pct / 100)),
                            'position_size': position_size,
                            'risk_pct': risk_pct,
                            'take_profit_pct': take_profit_pct,
                            'risk_reward_ratio': entry_analysis.risk_reward_ratio if entry_analysis.risk_reward_ratio > 0 else (take_profit_pct / risk_pct)
                        }
                        
                        logger.info(f"🤖 AI Entry Analysis: {entry_analysis.action} (Confidence: {entry_analysis.confidence:.1f}%)")

                        # Save analysis to database
                        try:
                            ticker_manager = TickerManager()
                            analysis_to_save = {
                                'confidence': entry_analysis.confidence,
                                'action': entry_analysis.action,
                                'reasons': [entry_analysis.reasoning] if entry_analysis.reasoning else [],
                                'targets': {
                                    'suggested_entry': entry_analysis.suggested_entry,
                                    'suggested_stop': entry_analysis.suggested_stop,
                                    'suggested_target': entry_analysis.suggested_target
                                }
                            }
                            ticker_manager.update_ai_entry_analysis(ticker_input, analysis_to_save)
                            st.toast(f"💾 AI analysis for {ticker_input} saved!", icon="✅")
                        except Exception as db_error:
                            logger.error(f"Failed to save AI entry analysis for {ticker_input}: {db_error}")
                            st.warning("Could not save analysis results to the database.")
                        
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
                        logger.error(f"Stock AI entry analysis error: {e}", exc_info=True)
    
    with analysis_col2:
        def clear_analysis():
            if 'stock_entry_analysis' in st.session_state:
                del st.session_state.stock_entry_analysis
            if 'stock_entry_params' in st.session_state:
                del st.session_state.stock_entry_params
        
        st.button("🔄 Clear", width='stretch', on_click=clear_analysis)
    
    # Display AI entry analysis if available
    if 'stock_entry_analysis' in st.session_state:
        entry_analysis = st.session_state.stock_entry_analysis
        
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
            wait_price_text = f" to ${entry_analysis.wait_for_price:.2f}" if entry_analysis.wait_for_price else ""
            wait_rsi_text = f" (RSI < {entry_analysis.wait_for_rsi:.0f})" if entry_analysis.wait_for_rsi else ""
            st.warning(f"⏳ **AI says WAIT FOR PULLBACK**{wait_price_text}{wait_rsi_text}")
        elif entry_analysis.action == "WAIT_FOR_BREAKOUT":
            st.warning(f"⏳ **AI says WAIT FOR BREAKOUT** - Consolidating, wait for confirmation")
        elif entry_analysis.action == "PLACE_LIMIT_ORDER":
            st.info(f"📝 **AI suggests LIMIT ORDER** at ${entry_analysis.suggested_entry:.2f}")
        else:  # DO_NOT_ENTER
            st.error("❌ **AI says DO NOT ENTER** - Poor setup, avoid this trade")
    
    # Display trade setup if available
    if 'stock_entry_params' in st.session_state:
        params = st.session_state.stock_entry_params
        
        st.markdown("---")
        st.markdown("#### 📈 Trade Setup")
        
        metric_cols = st.columns(4)
        metric_cols[0].metric("Current Price", f"${params['current_price']:.2f}")
        metric_cols[1].metric("Stop Loss", f"${params['stop_loss']:.2f}")
        metric_cols[2].metric("Take Profit", f"${params['take_profit']:.2f}")
        metric_cols[3].metric("R:R Ratio", f"{params['risk_reward_ratio']:.2f}")
        
        # Monitor & Alert button if AI says to wait
        has_entry_analysis = 'stock_entry_analysis' in st.session_state
        entry_analysis = st.session_state.get('stock_entry_analysis')
        
        if has_entry_analysis and entry_analysis and entry_analysis.action in ["WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT"]:
            st.markdown("---")
            monitor_col1, monitor_col2 = st.columns([2, 1])
            
            with monitor_col1:
                if st.button("🔔 Monitor & Alert Me", width='stretch', type="primary"):
                    try:
                        entry_assistant = st.session_state.get('stock_ai_entry_assistant')
                        if entry_assistant:
                            opp_id = entry_assistant.monitor_entry_opportunity(
                                symbol=params['ticker'],
                                side=params['direction'],
                                position_size=params['position_size'],
                                risk_pct=params['risk_pct'],
                                take_profit_pct=params['take_profit_pct'],
                                analysis=entry_analysis,
                                auto_execute=False
                            )
                            st.success(f"✅ Monitoring {params['ticker']} for entry opportunity!")
                            st.info(f"📊 Will alert when conditions improve (Opportunity ID: {opp_id})")
                            logger.info(f"🔔 User set up monitoring for {params['ticker']}")
                        else:
                            st.error("AI Entry Assistant not initialized")
                    except Exception as e:
                        st.error(f"Failed to set up monitoring: {e}")
                        logger.error(f"Monitor setup error: {e}", exc_info=True)
            
            with monitor_col2:
                st.info("💡 You can view all monitored stocks in the Auto-Trader → Entry Monitors tab")


def display_compact_stock_ai_entry(broker_client, llm_analyzer=None, default_ticker: str = ""):
    """
    Display compact version of AI entry analysis (for My Tickers tab)
    
    Args:
        broker_client: Tradier/IBKR/BrokerAdapter client
        llm_analyzer: LLM strategy analyzer for AI decisions
        default_ticker: Pre-fill ticker field with this value
    """
    with st.expander("🤖 AI Entry Analysis", expanded=False):
        st.write("Get instant AI-powered analysis on whether now is a good time to enter this stock.")
        
        # Simpler input form
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            ticker = st.text_input(
                "Ticker",
                value=default_ticker,
                key=f"compact_ai_ticker_{default_ticker}",
                placeholder="AAPL"
            ).upper().strip()
        
        with quick_col2:
            direction = st.selectbox(
                "Side",
                options=["BUY", "SELL"],
                key=f"compact_ai_direction_{default_ticker}"
            )
        
        with quick_col3:
            position = st.number_input(
                "Position ($)",
                min_value=100,
                max_value=50000,
                value=1000,
                step=100,
                key=f"compact_ai_position_{default_ticker}"
            )
        
        if st.button("🤖 Analyze", width='stretch', type="primary", key=f"compact_ai_btn_{default_ticker}"):
            if not ticker:
                st.warning("Enter a ticker symbol")
            else:
                with st.spinner(f"Analyzing {ticker}..."):
                    try:
                        from services.ai_stock_entry_assistant import get_ai_stock_entry_assistant
                        from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                        
                        if 'stock_ai_entry_assistant' not in st.session_state:
                            if llm_analyzer is None:
                                llm_analyzer = LLMStrategyAnalyzer()
                            
                            entry_assistant = get_ai_stock_entry_assistant(
                                broker_client=broker_client,
                                llm_analyzer=llm_analyzer,
                                check_interval_seconds=60,
                                enable_auto_entry=False
                            )
                            st.session_state.stock_ai_entry_assistant = entry_assistant
                            if not entry_assistant.is_running:
                                entry_assistant.start_monitoring()
                        else:
                            entry_assistant = st.session_state.stock_ai_entry_assistant
                        
                        analysis = entry_assistant.analyze_entry(
                            symbol=ticker,
                            side=direction,
                            position_size=position,
                            risk_pct=2.0,
                            take_profit_pct=6.0
                        )
                        
                        # Display compact results
                        if analysis.confidence >= 85:
                            st.success(f"✅ **{analysis.action}** ({analysis.confidence:.0f}% confidence)")
                        elif analysis.confidence >= 70:
                            st.warning(f"⚠️ **{analysis.action}** ({analysis.confidence:.0f}% confidence)")
                        else:
                            st.error(f"❌ **{analysis.action}** ({analysis.confidence:.0f}% confidence)")
                        
                        st.write(f"**Reasoning:** {analysis.reasoning}")

                        # Save analysis to database
                        try:
                            ticker_manager = TickerManager()
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
                            st.toast(f"💾 AI analysis for {ticker} saved!", icon="✅")
                        except Exception as db_error:
                            logger.error(f"Failed to save compact AI entry analysis for {ticker}: {db_error}")
                            st.warning("Could not save analysis results.")
                        
                        result_cols = st.columns(3)
                        result_cols[0].metric("Technical", f"{analysis.technical_score:.0f}/100")
                        result_cols[1].metric("Timing", f"{analysis.timing_score:.0f}/100")
                        result_cols[2].metric("Risk", f"{analysis.risk_score:.0f}/100")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                        logger.error(f"Compact AI analysis error: {e}", exc_info=True)

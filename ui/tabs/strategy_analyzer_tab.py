"""
Strategy Analyzer Tab
Backtest and analyze trading strategies

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime, timedelta

# Import with fallbacks
try:
    from services.llm_strategy_analyzer import LLMStrategyAnalyzer
except ImportError:
    logger.debug("LLMStrategyAnalyzer not available")
    LLMStrategyAnalyzer = None

try:
    from utils.option_alpha_ocr import extract_bot_config_from_screenshot
except ImportError:
    logger.debug("OCR utilities not available")
    def extract_bot_config_from_screenshot(image_bytes):
        return None, "OCR not available"

def render_tab():
    """Main render function called from app.py"""
    st.header("Strategy Analyzer")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("🤖 Strategy Analyzer")
    st.write("Analyze Option Alpha bot configs using an LLM provider. Choose provider, model and optionally provide an API key to run analysis.")

    col1, col2 = st.columns(2)

    with col1:
        provider = st.selectbox("LLM Provider", options=["openai", "anthropic", "google", "openrouter"], index=3, key='tab12_llm_provider_select')
        model = st.text_input("Model (leave blank for default)", value=os.getenv("AI_ANALYZER_MODEL", ""))
        api_key_input = st.text_input("API Key (optional, will override env var)", value="", type="password")
        run_btn = st.button("🔎 Run Analysis", type="primary")

    with col2:
        st.subheader("Sample Bot Configuration")
        # Provide a default empty config or upload option
        uploaded_file = st.file_uploader("Upload bot screenshot (optional)", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            sample_config, error_msg = extract_bot_config_from_screenshot(image_bytes)
            if error_msg:
                st.warning(error_msg)
        else:
            sample_config = {"bot_name": "Example Bot", "strategy": "Sample strategy config"}
        st.json(sample_config)

    if run_btn:
        logger.info("--- 'Run Analysis' button clicked ---")
        with st.spinner("Running strategy analysis..."):
            try:
                analyzer = LLMStrategyAnalyzer(provider=provider, model=(model or None), api_key=(api_key_input or None))
            except Exception as e:
                st.error(f"Failed to initialize analyzer: {e}")
            else:
                try:
                    analysis = analyzer.analyze_bot_strategy(sample_config)

                    # Save for quick sidebar summary and future access
                    st.session_state.strategy_analysis = analysis

                    # Prominent top-level card
                    st.markdown("---")
                    st.subheader(f"Analysis for: {getattr(analysis, 'bot_name', 'Bot')}")

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Overall Rating", getattr(analysis, 'overall_rating', 'N/A'))
                    with m2:
                        rs = getattr(analysis, 'risk_score', None)
                        st.metric("Risk Score", f"{rs:.2f}" if isinstance(rs, (int, float)) else rs)
                    with m3:
                        cf = getattr(analysis, 'confidence', None)
                        st.metric("Confidence", f"{cf:.2f}" if isinstance(cf, (int, float)) else cf)

                    # Summary and quick actions
                    summary_col, actions_col = st.columns([3, 1])
                    with summary_col:
                        summary_text = getattr(analysis, 'summary', '') or ''
                        st.write(summary_text)
                    with actions_col:
                        if st.button("🔝 Back to Top"):
                            st.rerun()
                        # Provide a copyable summary text area
                        if st.button("📋 Copy Summary"):
                            st.text_area("Summary (select and copy)", value=summary_text, height=160)

                    # Collapsible detailed sections for readability
                    with st.expander("Strengths", expanded=True):
                        strengths = getattr(analysis, 'strengths', []) or []
                        if strengths:
                            for s in strengths:
                                st.write(f"• {s}")
                        else:
                            st.write("No strengths found.")

                    with st.expander("Weaknesses", expanded=True):
                        weaknesses = getattr(analysis, 'weaknesses', []) or []
                        if weaknesses:
                            for w in weaknesses:
                                st.write(f"• {w}")
                        else:
                            st.write("No weaknesses found.")

                    with st.expander("Recommendations", expanded=True):
                        recommendations = getattr(analysis, 'recommendations', []) or []
                        if recommendations:
                            for r in recommendations:
                                st.write(f"• {r}")
                        else:
                            st.write("No recommendations returned.")

                    # If user triggered quick access from sidebar, show a note and focus
                    if st.session_state.get('goto_strategy_analyzer'):
                        st.success("Opened Strategy Analyzer — scroll down for details below.")
                        # clear the flag
                        st.session_state.goto_strategy_analyzer = False

                    # Add "Apply Strategy" functionality
                    st.divider()
                    st.subheader("🎯 Apply Favorable Strategy Plays")
                    st.write("Use the analyzed strategy to generate trading signals with recommended parameters.")
                    
                    # Extract recommended strategies from analysis
                    recommendations = getattr(analysis, 'recommendations', []) or []
                    if recommendations:
                        st.write("**AI Recommended Strategies:**")
                        for idx, rec in enumerate(recommendations[:3]):
                            with st.expander(f"Recommendation {idx + 1}"):
                                st.write(rec)
                    
                    col_apply1, col_apply2 = st.columns(2)
                    
                    with col_apply1:
                        # Prefill ticker from analysis if available
                        apply_ticker = st.text_input(
                            "Ticker Symbol",
                            value=getattr(analysis, 'ticker', st.session_state.get('selected_ticker', 'SPX')),
                            help="Enter the ticker for this strategy"
                        )
                        
                        # Strategy selection from recommendations
                        strategy_options = st.session_state.config.allowed_strategies
                        apply_strategy = st.selectbox(
                            "Select Strategy",
                            options=strategy_options,
                            help="Choose the option strategy from analysis"
                        )
                    
                    with col_apply2:
                        # Risk and confidence from analysis
                        risk_score = getattr(analysis, 'risk_score', 5.0)
                        if isinstance(risk_score, (int, float)):
                            estimated_risk = float(risk_score) * 40  # Convert to dollar amount
                        else:
                            estimated_risk = 200.0
                        
                        confidence = getattr(analysis, 'confidence', 0.75)
                        if isinstance(confidence, (int, float)):
                            ai_confidence = float(confidence)
                        else:
                            ai_confidence = 0.75
                        
                        st.metric("Estimated Risk", f"${estimated_risk:.0f}")
                        st.metric("AI Confidence", f"{ai_confidence:.2f}")
                    
                    if st.button("📊 Load into Signal Generator", type="primary", width="stretch"):
                        # Store strategy parameters in session state for Signal Generator tab
                        st.session_state.selected_ticker = apply_ticker.upper()
                        st.session_state.selected_strategy = apply_strategy
                        st.session_state.example_trade = {
                            'strike': 50.0,  # Default, user will adjust
                            'qty': 2,
                            'estimated_risk': estimated_risk,
                            'llm_score': ai_confidence,
                            'iv_rank': 50.0,  # Default
                            'expiry': (datetime.now() + timedelta(days=30)).date(),
                            'dte': 30
                        }
                        st.success(f"✅ Strategy loaded! Go to 'Generate Signal' tab to configure and send.")
                        st.info(f"Ticker: {apply_ticker} | Strategy: {apply_strategy} | Risk: ${estimated_risk:.0f} | Confidence: {ai_confidence:.2f}")

                except Exception as e:
                    st.error(f"Analysis failed: {e}")


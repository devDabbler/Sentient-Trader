"""
Advanced Scanner Tab
Advanced stock scanner with AI scoring and filters

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from .common_imports import (
    ScanType, 
    get_advanced_scanner,
    get_ai_scanner,
    get_ml_scanner,
    ComprehensiveAnalyzer,
    PENNY_THRESHOLDS
)

# Import ScanFilters directly
try:
    from services.advanced_opportunity_scanner import ScanFilters
except ImportError:
    logger.warning("ScanFilters not available, using fallback")
    # Fallback ScanFilters class
    class ScanFilters:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Import StrategyAdvisor if available
try:
    from analyzers.strategy import StrategyAdvisor
except ImportError:
    logger.debug("StrategyAdvisor not available")
    StrategyAdvisor = None

# Helper functions for filter presets
def _apply_filter_preset(filters, preset_name):
    """Apply a filter preset to the filters object"""
    # Implement preset logic based on preset_name
    logger.debug(f"Applying filter preset: {preset_name}")
    pass

def _apply_secondary_filter(filters, secondary_name):
    """Apply secondary filter to the filters object"""
    # Implement secondary filter logic
    logger.debug(f"Applying secondary filter: {secondary_name}")
    pass

def render_tab():
    """Main render function called from app.py"""
    st.header("Advanced Scanner")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("🚀 Advanced Opportunity Scanner")
    st.write("**All-in-one scanner** with AI/ML analysis, powerful filters, reverse split detection, and merger candidate identification!")
    
    # Use cached scanners from session state (only initialized once, reused on reruns)
    scanner = st.session_state.advanced_scanner
    ai_scanner = st.session_state.ai_scanner
    ml_scanner = st.session_state.ml_scanner
    
    # Analysis mode selector
    analysis_mode = st.radio(
        "🔬 Analysis Mode:",
        options=["⚡ Quick Scan (Fast)", "🧠 AI+ML Enhanced (Comprehensive)"],
        horizontal=True,
        help="Quick Scan uses technical analysis only. AI+ML Enhanced adds AI confidence ratings and ML predictions for maximum accuracy."
    )
    
    use_ai_ml = analysis_mode == "🧠 AI+ML Enhanced (Comprehensive)"
    
    if use_ai_ml:
        with st.expander("ℹ️ What does AI+ML Enhanced include?", expanded=False):
            st.markdown("""
            **AI+ML Enhanced Mode** combines three powerful analysis systems:
            - **🤖 AI Confidence Analysis**: LLM-powered reasoning and risk assessment
            - **🧠 ML Predictions**: 158 alpha factors from Qlib (if installed)
            - **📊 Technical Analysis**: All standard indicators plus reverse splits and merger detection
            
            This provides the **highest confidence** trading signals by requiring agreement across multiple systems.
            """)
    
    st.divider()
    
    # Scan configuration
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Scan Type")
        # Use session state to prevent unnecessary reruns on dropdown change
        if 'scan_type_display' not in st.session_state:
            st.session_state.scan_type_display = "🎯 All Opportunities"
        
        scan_type_display = st.selectbox(
            "What to scan for:",
            options=[
                "🎯 All Opportunities",
                "📈 Options Plays", 
                "💰 Penny Stocks (<$5)",
                "💥 Breakouts",
                "🚀 Momentum Plays",
                "🔥 Buzzing Stocks",
                "🌶️ Hottest Stocks"
            ],
            key="scan_type_display",
            help="Select the type of opportunities to find"
        )
        
        scan_type_map = {
            "🎯 All Opportunities": ScanType.ALL,
            "📈 Options Plays": ScanType.OPTIONS,
            "💰 Penny Stocks (<$5)": ScanType.PENNY_STOCKS,
            "💥 Breakouts": ScanType.BREAKOUTS,
            "🚀 Momentum Plays": ScanType.MOMENTUM,
            "🔥 Buzzing Stocks": ScanType.BUZZING,
            "🌶️ Hottest Stocks": ScanType.HOTTEST_STOCKS
        }
        scan_type = scan_type_map[scan_type_display]
        
        # Trading style selector
        st.subheader("📈 Trading Style")
        # Use session state to prevent unnecessary reruns on dropdown change
        if 'trading_style_display' not in st.session_state:
            st.session_state.trading_style_display = "📈 Swing Trading (days-weeks)"
        
        trading_style_display = st.selectbox(
            "Strategy recommendations for:",
            options=[
                "📊 Options Trading",
                "⚡ Scalping (seconds-minutes)",
                "🎯 Day Trading (intraday)",
                "📈 Swing Trading (days-weeks)",
                "💎 Buy & Hold (long-term)"
            ],
            key="trading_style_display",
            help="Choose your preferred trading style for strategy recommendations"
        )
        
        trading_style_map = {
            "📊 Options Trading": "OPTIONS",
            "⚡ Scalping (seconds-minutes)": "SCALP",
            "🎯 Day Trading (intraday)": "DAY_TRADE",
            "📈 Swing Trading (days-weeks)": "SWING_TRADE",
            "💎 Buy & Hold (long-term)": "BUY_HOLD"
        }
        trading_style = trading_style_map[trading_style_display]
        
        num_results = st.slider("Number of results", 5, 50, 20, 5, key="num_results_slider")
        
        # Performance control for buzzing and hottest stocks scans
        max_tickers_to_scan = None
        if scan_type in [ScanType.BUZZING, ScanType.HOTTEST_STOCKS]:
            max_tickers_to_scan = st.slider(
                "Max tickers to scan (performance)", 
                min_value=50, 
                max_value=300, 
                value=150, 
                step=25,
                key="max_tickers_slider",
                help="Limit the number of tickers to scan for faster results. More tickers = wider net but slower scan."
            )
    
    with col2:
        st.subheader("🎚️ Quick Filters")
        use_extended_universe = st.checkbox("Use Extended Universe (200+ tickers)", value=True, 
                                           key="use_extended_universe_cb",
                                           help="Includes obscure plays and emerging stocks")
        
        # Initialize quick_filter variable
        quick_filter = "None - Show All"  # Default value
        
        # Strategy-Based Hybrid Approach
        use_hybrid_approach = st.checkbox("🧬 Use Strategy-Based Hybrid Approach", value=False,
                                        key="use_hybrid_approach_cb",
                                        help="Use proven strategy combinations for balanced risk and opportunity")
        
        if use_hybrid_approach:
            st.info("💡 **Strategy-Based Mode**: Uses proven filter combinations with AI analysis and personalized strategy recommendations")
            
            # Strategy selection
            st.markdown("### 🎯 Choose Your Strategy")
            strategy_choice = st.radio(
                "Select a proven strategy combination:",
                options=[
                    "🎯 Quality Momentum (Recommended)",
                    "📈 Aggressive Growth", 
                    "⚡ Conservative Income",
                    "🔥 High-Volatility Plays",
                    "🔧 Custom Combination"
                ],
                key="strategy_choice"
            )
            
            # Strategy-specific configurations
            if strategy_choice == "🎯 Quality Momentum (Recommended)":
                st.success("**Best for balanced risk and opportunity**")
                st.markdown("""
                **Combines:**
                - High Confidence Only (≥70) - Foundation filter
                - Volume Surge (>2x avg) - Confirmation of interest  
                - Power Zone Stocks OR EMA Reclaim Setups - Technical validation
                
                **Why it works:** Risk management through confidence scores + growth potential through momentum + clear entry/exit points
                """)
                
                primary_approach = "High Confidence Only (Score ≥70)"
                secondary_approach = ["Volume Surge (>2x avg)", "Power Zone Stocks Only"]
                technical_choice = st.radio("Technical Confirmation:", ["Power Zone Stocks Only", "EMA Reclaim Setups"], key="quality_tech")
                if technical_choice == "EMA Reclaim Setups":
                    secondary_approach = ["Volume Surge (>2x avg)", "EMA Reclaim Setups"]
            
            elif strategy_choice == "📈 Aggressive Growth":
                st.warning("**Higher risk, higher reward - for experienced traders**")
                st.markdown("""
                **Combines:**
                - Penny Stocks ($1-$5) - Price range for upside potential
                - Volume Surge (>2x avg) - Liquidity filter
                - High Confidence (≥70) - Quality control
                
                **Why it works:** Penny stock upside + volume ensures you can enter/exit + confidence filter reduces risk
                """)
                
                primary_approach = "Penny Stocks ($1-$5)"
                secondary_approach = ["Volume Surge (>2x avg)", "High Confidence Only (Score ≥70)"]
            
            elif strategy_choice == "⚡ Conservative Income":
                st.info("**Lower risk, steady returns**")
                st.markdown("""
                **Combines:**
                - High Confidence Only (≥70) - Quality foundation
                - EMA Reclaim Setups - High-probability entries
                - RSI Oversold (<30) - Mean reversion opportunities
                
                **Why it works:** Conservative screening + technical confirmation + oversold bounce potential
                """)
                
                primary_approach = "High Confidence Only (Score ≥70)"
                secondary_approach = ["EMA Reclaim Setups", "RSI Oversold (<30)"]
            
            elif strategy_choice == "🔥 High-Volatility Plays":
                st.error("**Highest risk, highest reward - for aggressive traders only**")
                st.markdown("""
                **Combines:**
                - Ultra-Low Price (<$1) - Maximum upside potential
                - Volume Surge (>2x avg) - Confirmation of interest
                - Strong Momentum (>5% change) - Already moving stocks
                
                **Why it works:** Maximum upside + volume confirmation + momentum continuation
                """)
                
                primary_approach = "Ultra-Low Price (<$1)"
                secondary_approach = ["Volume Surge (>2x avg)", "Strong Momentum (>5% change)"]
            
            else:  # Custom Combination
                st.markdown("**Build your own strategy combination**")
                
                col_custom1, col_custom2 = st.columns(2)
                
                with col_custom1:
                    st.markdown("**Primary Filter:**")
                    primary_approach = st.selectbox(
                        "Main Filter:",
                        options=[
                            "High Confidence Only (Score ≥70)",
                            "Ultra-Low Price (<$1)",
                            "Penny Stocks ($1-$5)",
                            "Volume Surge (>2x avg)",
                            "Strong Momentum (>5% change)",
                            "Power Zone Stocks Only",
                            "EMA Reclaim Setups"
                        ],
                        key="custom_primary"
                    )
                
                with col_custom2:
                    st.markdown("**Additional Filters:**")
                    secondary_approach = st.multiselect(
                        "Secondary Filters:",
                        options=[
                            "High Confidence Only (Score ≥70)",
                            "Volume Surge (>2x avg)",
                            "Strong Momentum (>5% change)",
                            "Power Zone Stocks Only",
                            "EMA Reclaim Setups",
                            "RSI Oversold (<30)",
                            "RSI Overbought (>70)",
                            "High IV Rank (>60)",
                            "Low IV Rank (<40)"
                        ],
                        default=["Volume Surge (>2x avg)"],
                        key="custom_secondary"
                    )
            
            # Strategy recommendation integration
            st.divider()
            include_strategy_recs = st.checkbox("🎯 Include Strategy Recommendations", value=True,
                                              help="Get specific trading strategy recommendations for each found opportunity")
            
            if include_strategy_recs:
                st.markdown("**Your Trading Profile:**")
                strategy_col1, strategy_col2 = st.columns(2)
                
                with strategy_col1:
                    user_experience_hybrid = st.selectbox(
                        "Experience Level",
                        options=["Beginner", "Intermediate", "Advanced"],
                        key='hybrid_experience'
                    )
                    
                    risk_tolerance_hybrid = st.selectbox(
                        "Risk Tolerance",
                        options=["Conservative", "Moderate", "Aggressive"],
                        key='hybrid_risk'
                    )
                
                with strategy_col2:
                    capital_available_hybrid = st.number_input(
                        "Available Capital ($)",
                        min_value=500,
                        max_value=1000000,
                        value=5000,
                        step=500,
                        key='hybrid_capital'
                    )
                    
                    outlook_hybrid = st.selectbox(
                        "Market Outlook",
                        options=["Bullish", "Bearish", "Neutral"],
                        key='hybrid_outlook'
                    )
        else:
            # Original single filter approach
            quick_filter = st.selectbox(
                "Filter Preset:",
                options=[
                    "None - Show All",
                    "High Confidence Only (Score ≥70)",
                    "Ultra-Low Price (<$1)",
                    "Penny Stocks ($1-$5)",
                    "Volume Surge (>2x avg)",
                    "Strong Momentum (>5% change)",
                    "Power Zone Stocks Only",
                    "EMA Reclaim Setups"
                ],
                key='scanner_quick_filter'
            )
    
    # Advanced Filters (Expandable)
    with st.expander("🔧 Advanced Filters", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        
        with fcol1:
            st.markdown("**Price Filters**")
            min_price = st.number_input("Min Price ($)", min_value=0.0, value=None, step=0.1, key="adv_min_price")
            max_price = st.number_input("Max Price ($)", min_value=0.0, value=None, step=1.0, key="adv_max_price")
            
            st.markdown("**Volume Filters**")
            min_volume = st.number_input("Min Volume", min_value=0, value=None, step=100000, key="adv_min_vol")
            min_volume_ratio = st.number_input("Min Volume Ratio (x avg)", min_value=0.0, value=None, step=0.5, key="adv_vol_ratio")
        
        with fcol2:
            st.markdown("**Momentum Filters**")
            min_change = st.number_input("Min Change %", value=None, step=1.0, key="adv_min_change")
            max_change = st.number_input("Max Change %", value=None, step=1.0, key="adv_max_change")
            
            st.markdown("**Score Filters**")
            min_score = st.slider("Min Score", 0, 100, 50, 5, key="adv_min_score")
            min_confidence = st.number_input("Min Confidence Score", min_value=0, max_value=100, value=None, step=5, key="adv_min_conf")
        
        with fcol3:
            st.markdown("**Technical Filters**")
            require_power_zone = st.checkbox("Require Power Zone (8>21 EMA)", key="adv_power")
            require_reclaim = st.checkbox("Require EMA Reclaim", key="adv_reclaim")
            require_alignment = st.checkbox("Require Timeframe Alignment", key="adv_align")
            
            st.markdown("**Entropy Filters** 🔬")
            require_low_entropy = st.checkbox("Require Low Entropy (< 50)", key="adv_low_entropy", 
                                             help="Only show structured markets, ideal for day trading")
            max_entropy = st.number_input("Max Entropy", min_value=0, max_value=100, value=None, step=5, key="adv_max_entropy",
                                         help="Filter out high-noise markets above this threshold")
            
            st.markdown("**RSI Filters**")
            rsi_range = st.slider("RSI Range", 0, 100, (0, 100), key="adv_rsi")
    
    # Build filters object
    filters = ScanFilters(
        min_price=min_price,
        max_price=max_price,
        min_volume=min_volume,
        min_volume_ratio=min_volume_ratio,
        min_change_pct=min_change,
        max_change_pct=max_change,
        min_score=min_score,
        min_confidence_score=min_confidence,
        require_power_zone=require_power_zone,
        require_ema_reclaim=require_reclaim,
        require_timeframe_alignment=require_alignment,
        min_rsi=rsi_range[0] if rsi_range[0] > 0 else None,
        max_rsi=rsi_range[1] if rsi_range[1] < 100 else None,
        require_low_entropy=require_low_entropy,
        max_entropy=max_entropy
    )
    
    # Apply filter presets (hybrid or single)
    min_buzz_score = 30.0  # Default
    
    if use_hybrid_approach:
        # Apply hybrid approach filters
        st.session_state.hybrid_approach_active = True
        # Store values in different session state keys to avoid widget conflicts
        st.session_state.hybrid_primary_value = primary_approach
        st.session_state.hybrid_secondary_value = secondary_approach
        st.session_state.include_strategy_recs_value = include_strategy_recs
        st.session_state.strategy_choice_value = strategy_choice
        
        # Apply primary approach
        _apply_filter_preset(filters, primary_approach)
        
        # Apply secondary approaches
        for secondary in secondary_approach:
            _apply_secondary_filter(filters, secondary)
        
        # Set buzz score for hybrid
        min_buzz_score = 50.0  # Higher threshold for hybrid approach
        
    else:
        # Apply single filter preset (original logic)
        st.session_state.hybrid_approach_active = False
        
        if quick_filter == "None - Show All":
            filters.min_score = 0.0  # Show everything
            min_buzz_score = 10.0  # Very low threshold for buzzing scan
        elif quick_filter == "High Confidence Only (Score ≥70)":
            filters.min_score = 70.0
            min_buzz_score = 60.0
        elif quick_filter == "Ultra-Low Price (<$1)":
            filters.max_price = PENNY_THRESHOLDS.ULTRA_LOW_PRICE
        elif quick_filter == "Penny Stocks ($1-$5)":
            filters.min_price = PENNY_THRESHOLDS.ULTRA_LOW_PRICE
            filters.max_price = PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE
        elif quick_filter == "Volume Surge (>2x avg)":
            filters.min_volume_ratio = 2.0
            min_buzz_score = 40.0  # Higher threshold for volume surge
        elif quick_filter == "Strong Momentum (>5% change)":
            filters.min_change_pct = 5.0
            min_buzz_score = 40.0
        elif quick_filter == "Power Zone Stocks Only":
            filters.require_power_zone = True
        elif quick_filter == "EMA Reclaim Setups":
            filters.require_ema_reclaim = True
    
    # Scan button
    st.divider()
    scan_col1, scan_col2 = st.columns([1, 3])
    with scan_col1:
        scan_button = st.button("🔍 Scan Markets", type="primary", width="stretch", key="advanced_scan_button")
    with scan_col2:
        if scan_type == ScanType.BUZZING:
            st.info(f"💡 **Buzzing scan** detects unusual volume, volatility, price action + **Reddit/news sentiment** (min score: {min_buzz_score:.0f})")
        else:
            if use_hybrid_approach:
                strategy_name = st.session_state.get('strategy_choice_value', 'Custom Strategy')
                st.info(f"💡 Scanning for **{scan_type_display}** using **{strategy_name}** strategy")
            else:
                st.info(f"💡 Scanning for **{scan_type_display}** with {quick_filter}")
    
    # Execute scan
    if scan_button:
        with st.status("🔍 Scanning markets...", expanded=True) as status:
            try:
                st.write(f"Analyzing {scan_type_display}...")
                
                if use_ai_ml:
                    # AI+ML Enhanced Mode
                    st.write("🧠 Running ML analysis with 158 alpha factors...")
                    st.write("🤖 Generating AI confidence ratings...")
                    st.write("⚡ Calculating technical indicators, reverse splits, and merger candidates...")
                    
                    # Use ML scanner for Options or Penny Stocks scans
                    if scan_type in [ScanType.OPTIONS, ScanType.PENNY_STOCKS]:
                        if scan_type == ScanType.OPTIONS:
                            opportunities = st.session_state.ml_scanner.scan_top_options_with_ml(
                                top_n=num_results,
                                min_ensemble_score=filters.min_score if filters.min_score else 60.0
                            )
                        else:  # Penny stocks
                            opportunities = st.session_state.ml_scanner.scan_top_penny_stocks_with_ml(
                                top_n=num_results,
                                min_ensemble_score=filters.min_score if filters.min_score else 50.0
                            )
                        
                        # Convert to OpportunityResult format if needed
                        # ML scanner returns different format, wrap in simple display
                        st.session_state.adv_scan_ai_results = opportunities
                        st.session_state.adv_scan_mode = "AI+ML"
                    else:
                        # For other scan types, use standard scanner with AI enabled
                        if scan_type == ScanType.BUZZING:
                            opportunities = scanner.scan_buzzing_stocks(
                                top_n=num_results,
                                trading_style=trading_style,
                                min_buzz_score=min_buzz_score,
                                max_tickers_to_scan=max_tickers_to_scan
                            )
                        else:
                            opportunities = scanner.scan_opportunities(
                                scan_type=scan_type,
                                top_n=num_results,
                                trading_style=trading_style,
                                filters=filters,
                                use_extended_universe=use_extended_universe
                            )
                        st.session_state.adv_scan_results = opportunities
                        st.session_state.adv_scan_mode = "Standard"
                else:
                    # Quick Scan Mode
                    if scan_type == ScanType.BUZZING:
                        opportunities = scanner.scan_buzzing_stocks(
                            top_n=num_results,
                            trading_style=trading_style,
                            min_buzz_score=min_buzz_score,
                            max_tickers_to_scan=max_tickers_to_scan
                        )
                    else:
                        opportunities = scanner.scan_opportunities(
                            scan_type=scan_type,
                            top_n=num_results,
                            trading_style=trading_style,
                            filters=filters,
                            use_extended_universe=use_extended_universe
                        )
                    st.session_state.adv_scan_results = opportunities
                    st.session_state.adv_scan_mode = "Standard"
                
                # Store scan type
                st.session_state.adv_scan_type = scan_type_display
                
                # Get count
                result_count = len(opportunities) if hasattr(opportunities, '__len__') else len(st.session_state.get('adv_scan_results', []))
                
                status.update(label=f"✅ Found {result_count} opportunities!", state="complete")
                if use_ai_ml:
                    st.success(f"✅ AI+ML Scan complete! Found {result_count} quality {scan_type_display}")
                else:
                    st.success(f"✅ Scan complete! Found {result_count} {scan_type_display}")
                
            except Exception as e:
                status.update(label="❌ Scan failed", state="error")
                st.error(f"Error during scan: {str(e)}")
                logger.error("Advanced scan error: {}", str(e), exc_info=True)
    
    # Display AI+ML results (if available)
    if 'adv_scan_ai_results' in st.session_state and st.session_state.adv_scan_ai_results:
        ai_results = st.session_state.adv_scan_ai_results
        
        st.divider()
        st.subheader(f"🧠 AI+ML Enhanced Results: {st.session_state.adv_scan_type}")
        
        st.info(f"📡 **Real-time AI+ML analysis** - Found {len(ai_results)} quality opportunities with high confidence")
        
        for i, trade in enumerate(ai_results, 1):
            # Determine emoji based on rating
            if trade.ai_rating >= 8.0:
                emoji = "🟢"
            elif trade.ai_rating >= 6.5:
                emoji = "🟡"
            else:
                emoji = "🟠"
            
            # Build header
            if hasattr(trade, 'combined_score'):
                header = f"{emoji} #{i} **{trade.ticker}** - Ensemble: {trade.combined_score:.1f}/100 | ML: {trade.ml_prediction_score:.1f} | AI: {trade.ai_rating:.1f}/10"
            else:
                header = f"{emoji} #{i} **{trade.ticker}** - AI: {trade.ai_rating:.1f}/10 | Score: {trade.score:.1f}/100"
            
            with st.expander(header, expanded=(i==1)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💵 Price", f"${trade.price:.2f}", f"{trade.change_pct:+.1f}%")
                    st.metric("📊 Volume", f"{trade.volume_ratio:.1f}x", "Above Avg" if trade.volume_ratio > 1.5 else "Normal")
                
                with col2:
                    st.metric("🎯 AI Rating", f"{trade.ai_rating:.1f}/10", trade.ai_confidence)
                    st.metric("⚠️ Risk", trade.risk_level)
                
                with col3:
                    if hasattr(trade, 'ml_prediction_score'):
                        st.metric("🧠 ML Score", f"{trade.ml_prediction_score:.1f}/100")
                    # Add to My Tickers button
                    if st.button(f"⭐ Add to My Tickers", key=f"add_ai_{trade.ticker}_{i}"):
                        # Use cached ticker manager from session state
                        st.session_state.ticker_manager.add_ticker(trade.ticker, "AI+ML Scanner")
                        # Invalidate ticker cache to force refresh
                        if 'ticker_cache' in st.session_state:
                            st.session_state.ticker_cache = {}
                        st.success(f"✅ Added {trade.ticker} to My Tickers!")
                
                st.divider()
                
                if trade.ai_reasoning:
                    st.markdown("**🤖 AI Analysis**")
                    st.info(trade.ai_reasoning)
                
                if trade.ai_risks:
                    st.markdown("**⚠️ Risk Assessment**")
                    st.warning(trade.ai_risks)
    
    # Display standard results
    elif 'adv_scan_results' in st.session_state and st.session_state.adv_scan_results:
        opportunities = st.session_state.adv_scan_results
        scan_summary = scanner.get_scan_summary(opportunities)
        
        st.divider()
        
        # Hybrid approach summary
        if st.session_state.get('hybrid_approach_active', False):
            strategy_name = st.session_state.get('strategy_choice_value', 'Custom Strategy')
            st.subheader(f"🧬 {strategy_name} Results: {st.session_state.adv_scan_type}")
            st.info(f"**Strategy:** {strategy_name} | **Primary:** {st.session_state.get('hybrid_primary_value', 'N/A')} | **Secondary:** {', '.join(st.session_state.get('hybrid_secondary_value', []))}")
        else:
            st.subheader(f"📊 Results: {st.session_state.adv_scan_type}")
        
        # Summary metrics
        mcol1, mcol2, mcol3, mcol4, mcol5, mcol6, mcol7 = st.columns(7)
        with mcol1:
            st.metric("Total", scan_summary['total'])
        with mcol2:
            st.metric("Avg Score", f"{scan_summary['avg_score']:.1f}")
        with mcol3:
            st.metric("High Confidence", scan_summary['high_confidence'])
        with mcol4:
            st.metric("Breakouts", scan_summary['breakouts'])
        with mcol5:
            st.metric("Buzzing", scan_summary['buzzing'])
        with mcol6:
            reverse_split_count = scan_summary.get('reverse_split_stocks', 0)
            st.metric("⚠️ Rev Splits", reverse_split_count)
        with mcol7:
            merger_count = scan_summary.get('merger_candidates', 0)
            st.metric("🔄 Mergers", merger_count)
        
        # Results table
        st.markdown("### 📋 Top Opportunities")
        
        for i, opp in enumerate(opportunities, 1):
            with st.expander(f"#{i} {opp.ticker} - Score: {opp.score:.1f} | ${opp.price:.2f} ({opp.change_pct:+.1f}%)", expanded=(i <= 3)):
                rcol1, rcol2, rcol3 = st.columns([2, 2, 1])
                
                with rcol1:
                    st.markdown(f"**{opp.ticker}**")
                    st.write(f"💰 **Price:** ${opp.price:.2f} ({opp.change_pct:+.1f}%)")
                    st.write(f"📊 **Volume:** {opp.volume:,} ({opp.volume_ratio:.1f}x avg)")
                    if opp.market_cap:
                        st.write(f"💼 **Market Cap:** ${opp.market_cap:.1f}M")
                    if opp.sector:
                        st.write(f"🏢 **Sector:** {opp.sector}")
                
                with rcol2:
                    st.write(f"🎯 **Score:** {opp.score:.1f}/100")
                    st.write(f"✅ **Confidence:** {opp.confidence}")
                    st.write(f"⚠️ **Risk:** {opp.risk_level}")
                    if opp.entropy is not None:
                        entropy_emoji = "✅" if opp.entropy < 50 else "⚠️" if opp.entropy < 70 else "❌"
                        st.write(f"🔬 **Entropy:** {entropy_emoji} {opp.entropy:.0f}/100")
                        if opp.entropy_state:
                            st.caption(f"State: {opp.entropy_state}")
                    if opp.trend:
                        st.write(f"📈 **Trend:** {opp.trend}")
                    if opp.rsi:
                        st.write(f"📉 **RSI:** {opp.rsi:.1f}")
                
                with rcol3:
                    if opp.is_breakout:
                        st.success("💥 BREAKOUT")
                    if opp.is_buzzing:
                        st.warning(f"🔥 BUZZING\n{opp.buzz_score:.0f}")
                    if opp.is_merger_candidate:
                        st.info(f"🔄 MERGER\n{opp.merger_score:.0f}")
                
                st.markdown(f"**Reason:** {opp.reason}")
                
                # Reverse split warning (prominent for penny stocks)
                if opp.reverse_split_warning:
                    st.error(f"⚠️ **{opp.reverse_split_warning}**")
                    if opp.reverse_splits:
                        split_history = ", ".join([f"{s['ratio_str']} on {s['date']}" for s in opp.reverse_splits[:3]])
                        st.caption(f"Split History: {split_history}")
                
                if opp.breakout_signals:
                    st.info(f"🎯 **Breakout Signals:** {', '.join(opp.breakout_signals)}")
                
                if opp.buzz_reasons:
                    st.warning(f"🔥 **Buzz Reasons:** {', '.join(opp.buzz_reasons)}")
                
                if opp.is_merger_candidate and opp.merger_signals:
                    st.info(f"🔄 **Merger Signals:** {', '.join(opp.merger_signals)}")
                
                # Hybrid approach: Add strategy recommendations
                if st.session_state.get('hybrid_approach_active', False) and st.session_state.get('include_strategy_recs_value', False):
                    st.markdown("---")
                    st.markdown("**🎯 Strategy Recommendations**")
                    
                    # Get analysis for strategy recommendations (only if StrategyAdvisor is available)
                    if StrategyAdvisor is None:
                        st.info("Strategy recommendations require StrategyAdvisor module")
                    else:
                        try:
                            analysis = ComprehensiveAnalyzer.analyze_stock(opp.ticker, "SWING_TRADE")
                            if analysis:
                                recommendations = StrategyAdvisor.get_recommendations(
                                    analysis=analysis,
                                    user_experience=st.session_state.get('hybrid_experience', 'Intermediate'),
                                    risk_tolerance=st.session_state.get('hybrid_risk', 'Moderate'),
                                    capital_available=st.session_state.get('hybrid_capital', 5000),
                                    outlook=st.session_state.get('hybrid_outlook', 'Neutral')
                                )
                                
                                if recommendations:
                                    # Show top 2 recommendations
                                    for j, rec in enumerate(recommendations[:2], 1):
                                        confidence_pct = int(rec.confidence * 100)
                                        st.markdown(f"**{j}. {rec.strategy_name}** ({confidence_pct}% match)")
                                        st.caption(f"Risk: {rec.risk_level} | Best for: {', '.join(rec.best_conditions[:2])}")
                                        
                                        if st.button(f"Use Strategy", key=f"use_strategy_{opp.ticker}_{j}"):
                                            st.session_state.selected_strategy = rec.action
                                            st.session_state.selected_ticker = opp.ticker
                                            st.success(f"✅ Selected {rec.strategy_name} for {opp.ticker}")
                                else:
                                    st.info("No specific strategies recommended for this stock")
                        except Exception as e:
                            st.warning(f"Could not generate strategy recommendations: {str(e)}")
                
                # Action buttons
                acol1, acol2 = st.columns(2)
                with acol1:
                    if st.button(f"📊 Full Analysis", key=f"analyze_{opp.ticker}_{i}"):
                        st.session_state.active_main_tab = "🔍 Stock Intelligence"
                        st.session_state.analyze_ticker = opp.ticker
                        st.session_state.trigger_analysis = True
                        st.rerun()
                with acol2:
                    if st.button(f"⭐ Add to My Tickers", key=f"add_{opp.ticker}_{i}"):
                        # Use cached ticker manager from session state
                        st.session_state.ticker_manager.add_ticker(opp.ticker, "Advanced Scanner")
                        # Invalidate ticker cache to force refresh
                        if 'ticker_cache' in st.session_state:
                            st.session_state.ticker_cache = {}
                        st.success(f"✅ Added {opp.ticker} to My Tickers!")
        
        # Export option
        st.divider()
        export_col1, export_col2 = st.columns([1, 3])
        with export_col1:
            if st.button("📥 Export to CSV"):
                df = pd.DataFrame([{
                    'Ticker': o.ticker,
                    'Score': o.score,
                    'Price': o.price,
                    'Change %': o.change_pct,
                    'Volume': o.volume,
                    'Volume Ratio': o.volume_ratio,
                    'Confidence': o.confidence,
                    'Risk': o.risk_level,
                    'Entropy': o.entropy if o.entropy is not None else '',
                    'Entropy State': o.entropy_state if o.entropy_state else '',
                    'Trend': o.trend,
                    'RSI': o.rsi,
                    'Breakout': o.is_breakout,
                    'Buzzing': o.is_buzzing,
                    'Reverse Split Warning': o.reverse_split_warning if o.reverse_split_warning else '',
                    'Reverse Splits Count': len(o.reverse_splits),
                    'Merger Candidate': o.is_merger_candidate,
                    'Merger Score': o.merger_score,
                    'Reason': o.reason
                } for o in opportunities])
                
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    f"advanced_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )
        with export_col2:
            st.caption(f"💡 Export {len(opportunities)} opportunities to CSV for further analysis")
    
    else:
        st.info("👆 Configure your scan settings and click 'Scan Markets' to find opportunities")


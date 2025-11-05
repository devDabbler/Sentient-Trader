"""
Streamlit UI components for Reddit strategy selection and AI validation.
"""

import streamlit as st
from loguru import logger
from typing import Optional, Dict, List

from models.reddit_strategies import (
    get_all_reddit_strategies,
    get_reddit_strategy,
    get_strategies_by_experience,
    RedditStrategy,
    StrategyType
)
from services.reddit_strategy_validator import (
    RedditStrategyValidator,
    StrategyValidation,
    format_validation_report
)
from models.analysis import StockAnalysis



def render_reddit_strategy_selector(
    user_experience: str = "Intermediate",
    show_all: bool = False
) -> Optional[RedditStrategy]:
    """
    Render a UI component for selecting Reddit strategies.
    
    Args:
        user_experience: User's experience level to filter strategies
        show_all: If True, show all strategies regardless of experience
    
    Returns:
        Selected RedditStrategy or None
    """
    st.markdown("### 🎯 Reddit Community Strategies")
    st.caption("Professional strategies sourced from successful Reddit traders")
    
    # Get available strategies
    if show_all:
        strategies = get_all_reddit_strategies()
    else:
        strategies = get_strategies_by_experience(user_experience)
    
    if not strategies:
        st.warning(f"No strategies available for {user_experience} level traders.")
        return None
    
    # Create strategy selection
    strategy_names = [s.name for s in strategies]
    strategy_dict = {s.name: s for s in strategies}
    
    selected_name = st.selectbox(
        "Select a Strategy",
        options=strategy_names,
        help="Choose a Reddit-sourced strategy to review"
    )
    
    if not selected_name:
        return None
    
    selected_strategy = strategy_dict[selected_name]
    
    # Display strategy overview
    with st.expander("📋 Strategy Overview", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Source:** {selected_strategy.source}")
            st.markdown(f"**Type:** {selected_strategy.strategy_type.value}")
            st.markdown(f"**Experience Level:** {selected_strategy.experience_level}")
            st.markdown(f"**Risk Level:** {selected_strategy.risk_level}")
        
        with col2:
            st.markdown(f"**Capital Required:** {selected_strategy.capital_requirement}")
            if selected_strategy.typical_win_rate:
                st.markdown(f"**Win Rate:** {selected_strategy.typical_win_rate}")
            st.markdown(f"**Products:** {', '.join(selected_strategy.suitable_products[:3])}")
        
        st.markdown("---")
        st.markdown(f"**Description:**\n{selected_strategy.description}")
        
        st.markdown(f"**Philosophy:**\n{selected_strategy.philosophy}")
    
    # Display key metrics
    with st.expander("📊 Key Performance Metrics", expanded=False):
        metrics_col = st.columns(len(selected_strategy.key_metrics))
        for i, (key, value) in enumerate(selected_strategy.key_metrics.items()):
            with metrics_col[i]:
                st.metric(key.replace("_", " ").title(), value)
    
    # Display warnings
    if selected_strategy.warnings:
        with st.expander("⚠️ Important Warnings", expanded=True):
            for warning in selected_strategy.warnings:
                st.warning(warning)
    
    return selected_strategy


def render_strategy_details(strategy: RedditStrategy):
    """
    Render detailed information about a Reddit strategy.
    
    Args:
        strategy: The RedditStrategy to display
    """
    st.markdown("### 📖 Strategy Details")
    
    # Parameters
    with st.expander("⚙️ Strategy Parameters", expanded=True):
        for param in strategy.parameters:
            st.markdown(f"**{param.name}:** `{param.value}`")
            st.caption(param.description)
            if param.validation_rule:
                st.code(f"Validation: {param.validation_rule}", language="python")
            st.markdown("---")
    
    # Setup Rules
    with st.expander("📋 Setup Rules (Step-by-Step)", expanded=True):
        for rule in sorted(strategy.setup_rules, key=lambda r: r.priority):
            st.markdown(f"**{rule.priority}. {rule.condition}**")
            st.info(f"**Action:** {rule.action}")
            if rule.notes:
                st.caption(f"📝 {rule.notes}")
            st.markdown("---")
    
    # Risk Management
    with st.expander("🛡️ Risk Management Rules", expanded=True):
        for rule in strategy.risk_management:
            mandatory_badge = "🔴 MANDATORY" if rule.mandatory else "🟡 OPTIONAL"
            st.markdown(f"**{rule.rule_type.upper()}** {mandatory_badge}")
            st.markdown(f"Value: `{rule.value}`")
            st.caption(rule.description)
            st.markdown("---")
    
    # Validation Checklist
    with st.expander("✅ Pre-Trade Validation Checklist", expanded=False):
        st.markdown("**Before entering this trade, verify:**")
        for i, item in enumerate(strategy.validation_checklist, 1):
            st.checkbox(item, key=f"checklist_{strategy.strategy_id}_{i}")
    
    # Red Flags
    with st.expander("🚩 Red Flags to Avoid", expanded=False):
        st.markdown("**DO NOT trade if any of these apply:**")
        for flag in strategy.red_flags:
            st.error(f"🚩 {flag}")
    
    # Additional Notes
    if strategy.notes:
        with st.expander("📝 Additional Notes", expanded=False):
            st.info(strategy.notes)


def render_ai_validation_interface(
    strategy: RedditStrategy,
    ticker: Optional[str] = None,
    analysis: Optional[StockAnalysis] = None
) -> Optional[StrategyValidation]:
    """
    Render AI validation interface for a Reddit strategy.
    
    Args:
        strategy: The RedditStrategy to validate
        ticker: Optional ticker symbol to validate against
        analysis: Optional StockAnalysis for the ticker
    
    Returns:
        StrategyValidation result or None
    """
    st.markdown("### 🤖 AI Strategy Validation")
    st.caption("Get AI-powered analysis of strategy viability for your specific situation")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        validation_ticker = st.text_input(
            "Ticker Symbol",
            value=ticker or "",
            placeholder="SPY",
            help="Enter the ticker you want to apply this strategy to"
        ).upper()
    
    with col2:
        include_market_context = st.checkbox(
            "Include Market Context",
            value=True,
            help="Include broader market conditions in validation"
        )
    
    # Additional context
    market_context = None
    if include_market_context:
        with st.expander("📊 Additional Market Context (Optional)", expanded=False):
            vix_level = st.number_input("VIX Level", min_value=0.0, max_value=100.0, value=20.0)
            market_sentiment = st.selectbox(
                "Market Sentiment",
                ["Bullish", "Neutral", "Bearish", "Panic", "Euphoric"]
            )
            upcoming_events = st.text_area(
                "Upcoming Events/Catalysts",
                placeholder="e.g., Fed meeting next week, earnings season, etc."
            )
            
            market_context = {
                "vix": vix_level,
                "sentiment": market_sentiment,
                "upcoming_events": upcoming_events
            }
    
    # Validation button
    if st.button("🚀 Validate Strategy with AI", type="primary", disabled=not validation_ticker):
        if not validation_ticker:
            st.error("Please enter a ticker symbol")
            return None
        
        with st.spinner(f"AI is analyzing {strategy.name} for {validation_ticker}..."):
            try:
                # Initialize validator
                validator = RedditStrategyValidator()
                
                # Run validation
                validation = validator.validate_strategy(
                    strategy=strategy,
                    ticker=validation_ticker,
                    analysis=analysis,
                    market_context=market_context
                )
                
                # Display results
                render_validation_results(validation)
                
                return validation
                
            except Exception as e:
                st.error(f"Validation failed: {str(e)}")
                logger.error(f"Strategy validation error: {e}", exc_info=True)
                return None
    
    return None


def render_validation_results(validation: StrategyValidation):
    """
    Render the results of AI strategy validation.
    
    Args:
        validation: The StrategyValidation result to display
    """
    st.markdown("---")
    st.markdown("### 📊 Validation Results")
    
    # Overall verdict
    if validation.is_viable:
        st.success(f"✅ **VIABLE** - {validation.market_alignment} Market Alignment")
    else:
        st.error(f"❌ **NOT VIABLE** - {validation.market_alignment} Market Alignment")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Viability Score", f"{validation.viability_score:.1%}")
    with col2:
        st.metric("Confidence", f"{validation.confidence:.1%}")
    with col3:
        st.metric("Alignment", validation.market_alignment)
    
    # Reasoning
    with st.expander("🧠 AI Reasoning", expanded=True):
        st.markdown(validation.reasoning)
    
    # Strengths
    if validation.strengths:
        with st.expander("✅ Strengths", expanded=True):
            for strength in validation.strengths:
                st.success(f"✅ {strength}")
    
    # Concerns
    if validation.concerns:
        with st.expander("⚠️ Concerns", expanded=True):
            for concern in validation.concerns:
                st.warning(f"⚠️ {concern}")
    
    # Missing Conditions
    if validation.missing_conditions:
        with st.expander("❌ Missing Conditions", expanded=True):
            st.markdown("**The following required conditions are NOT currently met:**")
            for condition in validation.missing_conditions:
                st.error(f"❌ {condition}")
    
    # Red Flags
    if validation.red_flags_detected:
        with st.expander("🚩 Red Flags Detected", expanded=True):
            st.markdown("**⚠️ WARNING: The following red flags were detected:**")
            for flag in validation.red_flags_detected:
                st.error(f"🚩 {flag}")
    
    # Recommendations
    if validation.recommendations:
        with st.expander("💡 Recommendations", expanded=True):
            for rec in validation.recommendations:
                st.info(f"💡 {rec}")
    
    # Download report
    report_text = format_validation_report(validation)
    st.download_button(
        label="📥 Download Full Report",
        data=report_text,
        file_name=f"strategy_validation_{validation.strategy_name.replace(' ', '_')}.txt",
        mime="text/plain"
    )


def render_strategy_comparison_ui(
    strategies: List[RedditStrategy],
    ticker: str,
    analysis: Optional[StockAnalysis] = None
):
    """
    Render UI for comparing multiple strategies.
    
    Args:
        strategies: List of RedditStrategy objects to compare
        ticker: Ticker symbol to validate against
        analysis: Optional StockAnalysis for the ticker
    """
    st.markdown("### 🔄 Strategy Comparison")
    st.caption(f"Compare multiple strategies for {ticker}")
    
    if st.button("🚀 Compare All Strategies", type="primary"):
        with st.spinner("AI is comparing strategies..."):
            try:
                validator = RedditStrategyValidator()
                validations = validator.batch_validate_strategies(
                    strategies=strategies,
                    ticker=ticker,
                    analysis=analysis
                )
                
                # Display comparison table
                st.markdown("#### 📊 Comparison Results")
                
                comparison_data = []
                for val in validations:
                    comparison_data.append({
                        "Strategy": val.strategy_name,
                        "Viable": "✅" if val.is_viable else "❌",
                        "Score": f"{val.viability_score:.1%}",
                        "Alignment": val.market_alignment,
                        "Confidence": f"{val.confidence:.1%}",
                        "Red Flags": len(val.red_flags_detected)
                    })
                
                st.dataframe(comparison_data, width="stretch")
                
                # Show detailed results for each
                st.markdown("---")
                st.markdown("#### 📋 Detailed Results")
                
                for val in validations:
                    with st.expander(f"{val.strategy_name} - {val.market_alignment}", expanded=False):
                        render_validation_results(val)
                
                # Summary insights
                comparison_summary = validator.compare_strategies(validations)
                
                st.markdown("---")
                st.markdown("#### 💡 Summary Insights")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Strategies", comparison_summary["total_strategies"])
                with col2:
                    st.metric("Viable Strategies", comparison_summary["viable_count"])
                with col3:
                    st.metric("Avg Score", f"{comparison_summary['average_viability_score']:.1%}")
                
                if comparison_summary.get("best_strategy"):
                    st.success(f"🏆 **Best Strategy:** {comparison_summary['best_strategy']} (Score: {comparison_summary['best_score']:.1%})")
                
            except Exception as e:
                st.error(f"Comparison failed: {str(e)}")
                logger.error(f"Strategy comparison error: {e}", exc_info=True)


def render_reddit_strategy_tab():
    """
    Render a complete tab for Reddit strategy selection and validation.
    This can be integrated into the main app.
    """
    st.header("🎯 Reddit Community Strategies")
    st.markdown("""
    Explore and validate professional option strategies sourced from successful Reddit traders.
    These strategies come with detailed playbooks, risk management rules, and AI-powered validation.
    """)
    
    # User profile
    with st.expander("👤 Your Trading Profile", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            user_experience = st.selectbox(
                "Experience Level",
                ["Beginner", "Intermediate", "Advanced", "Professional"],
                index=1
            )
        with col2:
            show_all_strategies = st.checkbox(
                "Show all strategies (ignore experience filter)",
                value=False
            )
    
    st.markdown("---")
    
    # Strategy selection
    selected_strategy = render_reddit_strategy_selector(
        user_experience=user_experience,
        show_all=show_all_strategies
    )
    
    if selected_strategy:
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📖 Strategy Details", "🤖 AI Validation", "📚 Learning"])
        
        with tab1:
            render_strategy_details(selected_strategy)
        
        with tab2:
            render_ai_validation_interface(selected_strategy)
        
        with tab3:
            st.markdown("### 📚 Understanding This Strategy")
            st.markdown(f"**{selected_strategy.name}**")
            st.markdown(selected_strategy.description)
            
            st.markdown("#### 🎯 When to Use")
            st.markdown("**Required Conditions:**")
            for cond in selected_strategy.required_conditions:
                st.markdown(f"- ✅ {cond.value}")
            
            st.markdown("**Avoid When:**")
            for cond in selected_strategy.unsuitable_conditions:
                st.markdown(f"- ❌ {cond.value}")
            
            st.markdown("#### 💰 Capital & Risk")
            st.info(f"""
            - **Capital Required:** {selected_strategy.capital_requirement}
            - **Risk Level:** {selected_strategy.risk_level}
            - **Experience Level:** {selected_strategy.experience_level}
            """)
            
            if selected_strategy.notes:
                st.markdown("#### 📝 Important Notes")
                st.warning(selected_strategy.notes)

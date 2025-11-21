"""
Strategy Advisor Tab
AI-powered strategy recommendations

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple

def render_tab():
    """Main render function called from app.py"""
    st.header("Strategy Advisor")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("🎯 Intelligent Strategy Advisor")
    st.write("Get personalized strategy recommendations based on comprehensive analysis.")
    
    # Check if we have a current analysis to work with
    current_analysis = st.session_state.get('current_analysis', None)
    
    if current_analysis:
        st.success(f"📊 **Current Analysis Available:** {current_analysis.ticker} @ ${current_analysis.price:.2f}")
        
        # Quick analysis summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price Change", f"{current_analysis.change_pct:+.2f}%")
        with col2:
            st.metric("RSI", f"{current_analysis.rsi:.1f}")
        with col3:
            st.metric("IV Rank", f"{current_analysis.iv_rank:.1f}%")
        with col4:
            st.metric("Trend", current_analysis.trend)
        
        # Generate strategy recommendations based on current analysis
        st.subheader("🎯 AI-Powered Strategy Recommendations")
        st.write("Based on your current analysis and market conditions:")
        
        # Get user preferences
        col1, col2 = st.columns(2)
        with col1:
            user_experience = st.selectbox(
                "Your Experience Level",
                ["Beginner", "Intermediate", "Advanced"],
                index=1,
                key="advisor_exp"
            )
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["Low", "Moderate", "High"],
                index=1,
                key="advisor_risk"
            )
        with col2:
            capital_available = st.number_input(
                "Available Capital ($)",
                min_value=100,
                max_value=1000000,
                value=5000,
                step=100,
                key="advisor_capital"
            )
            market_outlook = st.selectbox(
                "Market Outlook",
                ["Bullish", "Bearish", "Neutral"],
                index=2,
                key="advisor_outlook"
            )
        
        # Generate recommendations
        if st.button("🔍 Generate Strategy Recommendations", type="primary"):
            with st.spinner("Analyzing market conditions and generating recommendations..."):
                try:
                    recommendations = StrategyAdvisor.get_recommendations(
                        analysis=current_analysis,
                        user_experience=user_experience,
                        risk_tolerance=risk_tolerance,
                        capital_available=capital_available,
                        outlook=market_outlook
                    )
                    
                    if recommendations:
                        st.success(f"✅ Generated {len(recommendations)} strategy recommendations!")
                        
                        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                            with st.expander(f"#{i} {rec.name} (Score: {int(rec.score * 100))}/100)", expanded=i==1):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Risk Level", rec.risk_level)
                                    st.metric("Max Loss", rec.max_loss)
                                with col2:
                                    st.metric("Max Gain", rec.max_gain)
                                    st.metric("Win Rate", rec.win_rate)
                                with col3:
                                    st.metric("Capital Req", rec.capital_req)
                                    st.metric("Experience", rec.experience)
                                
                                st.write(f"**Description:** {rec.description}")
                                st.write(f"**Best For:** {', '.join(rec.best_for)}")
                                
                                if rec.reasoning:
                                    st.write("**Why This Strategy:**")
                                    for reason in rec.reasoning_list:
                                        st.write(f"• {reason}")
                                
                                if rec.setup_steps:
                                    st.write("**Setup Steps:**")
                                    for j, step in enumerate(rec.setup_steps, 1):
                                        st.write(f"{j}. {step}")
                                
                                if rec.warnings:
                                    st.write("**⚠️ Warnings:**")
                                    for warning in rec.warnings:
                                        st.warning(warning)
                                
                                # Action buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"Use This Strategy", key=f"use_rec_{i}"):
                                        st.session_state.selected_strategy = rec.name
                                        st.session_state.selected_ticker = current_analysis.ticker
                                        st.success(f"✅ Strategy '{rec.name}' selected for {current_analysis.ticker}")
                                with col2:
                                    if st.button(f"View Details", key=f"details_rec_{i}"):
                                        st.info("Navigate to 'Generate Signal' tab to configure this strategy")
                    else:
                        st.warning("No suitable strategies found for current market conditions. Try adjusting your preferences.")
                        
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
    else:
        st.info("💡 **No stock analysis available.** Go to the Dashboard tab to analyze a stock first, then return here for strategy recommendations.")
        if st.button("Go to Dashboard"):
            st.info("Navigate to the 'Dashboard' tab above to analyze a stock")
    
    st.divider()
    
    # Add educational section about filtered investment approaches
    with st.expander("📚 Understanding Filtered Investment Approaches", expanded=False):
        st.markdown("""
        ### What are Filtered Investment Approaches?
        
        Filtered investment approaches are pre-configured scanning strategies that help you find specific types of trading opportunities based on your risk tolerance, market conditions, and investment goals. Each approach applies different criteria to filter stocks and identify the most relevant opportunities for your trading style.
        
        ### Available Investment Approaches:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 High Confidence Only (Score ≥70)**
            - **What it does**: Shows only stocks with high AI confidence scores
            - **Best for**: Conservative traders, beginners, reliable setups
            - **Risk level**: Low to Medium
            - **Why use**: Reduces false signals, focuses on quality setups
            
            **💰 Ultra-Low Price (<$1)**
            - **What it does**: Finds stocks trading under $1 per share
            - **Best for**: High-risk, high-reward traders, penny stock enthusiasts
            - **Risk level**: Very High
            - **Why use**: Maximum upside potential, but requires careful risk management
            
            **💵 Penny Stocks ($1-$5)**
            - **What it does**: Targets stocks between $1-$5 per share
            - **Best for**: Growth-focused traders, small-cap investors
            - **Risk level**: High
            - **Why use**: Classic penny stock range with moderate risk/reward
            
            **📈 Volume Surge (>2x avg)**
            - **What it does**: Identifies stocks with unusually high trading volume
            - **Best for**: Momentum traders, breakout specialists
            - **Risk level**: Medium to High
            - **Why use**: High volume often precedes significant price movements
            """)
        
        with col2:
            st.markdown("""
            **🚀 Strong Momentum (>5% change)**
            - **What it does**: Finds stocks with significant price movements
            - **Best for**: Trend followers, momentum traders
            - **Risk level**: Medium to High
            - **Why use**: Captures stocks already in motion with strong directional bias
            
            **⚡ Power Zone Stocks Only**
            - **What it does**: Filters for stocks in EMA 8>21 power zone
            - **Best for**: Technical traders, trend followers
            - **Risk level**: Medium
            - **Why use**: EMA power zones indicate strong uptrend momentum
            
            **🔄 EMA Reclaim Setups**
            - **What it does**: Finds stocks that have reclaimed key EMA levels
            - **Best for**: Mean reversion traders, technical analysts
            - **Risk level**: Low to Medium
            - **Why use**: High-probability entry points with defined risk levels
            """)
        
        st.markdown("""
        ### ⚠️ Important Risk Considerations:
        
        - **Penny Stocks & Ultra-Low Price**: These stocks are highly volatile and can experience rapid price swings. Many penny stocks have low liquidity and may be difficult to exit quickly.
        
        - **Volume Surge & Momentum**: While high volume and momentum can indicate strong moves, they can also signal the end of a trend. Always use proper risk management.
        
        - **Technical Setups**: Power zones and EMA reclaims are based on historical patterns and may not always predict future performance.
        
        - **Diversification**: Don't put all your capital into one approach. Consider spreading risk across different strategies and timeframes.
        
        ### 💡 Pro Tips:
        
        1. **Start Conservative**: Begin with "High Confidence Only" to understand the platform
        2. **Combine Approaches**: Use multiple filters together for more targeted results
        3. **Risk Management**: Never risk more than you can afford to lose
        4. **Research First**: Always do your own due diligence before trading
        5. **Paper Trade**: Test strategies with paper trading before using real money
        """)
    
    # Check if we have analysis (optional for traditional strategies)
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        st.success(f"Using analysis for: **{analysis.ticker}** (${analysis.price}, {analysis.change_pct:+.2f}%)")
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Trading Profile")
            
            # Use stateful navigation instead of st.tabs() to prevent reruns
            if 'profile_comparison_tab' not in st.session_state:
                st.session_state.profile_comparison_tab = "Single Profile"
            
            # Tab selector using radio buttons (no rerun on selection)
            profile_tab_selector = st.radio(
                "Input Mode",
                options=["Single Profile", "Compare Scenarios"],
                horizontal=True,
                key="profile_comparison_tab_selector",
                label_visibility="collapsed"
            )
            
            # Update session state if changed
            if profile_tab_selector != st.session_state.profile_comparison_tab:
                st.session_state.profile_comparison_tab = profile_tab_selector
            
            # Render the selected tab
            profile_tab_active = st.session_state.profile_comparison_tab == "Single Profile"
            comparison_tab_active = st.session_state.profile_comparison_tab == "Compare Scenarios"

            if profile_tab_active:
                user_experience = st.selectbox(
                    "Experience Level",
                    options=["Beginner", "Intermediate", "Advanced"],
                    key='user_experience_select',
                    help="Affects which strategies are recommended"
                )

                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    options=["Conservative", "Moderate", "Aggressive"],
                    key='risk_tolerance_select',
                    help="Conservative = Lower risk strategies, Aggressive = Higher risk/reward"
                )

                capital_available = st.number_input(
                    "Available Capital ($)",
                    min_value=100,
                    max_value=1000000,
                    value=500,
                    step=100,
                    help="Total capital you're willing to risk on this trade"
                )
                
                # Add position sizing controls
                st.subheader("Position Sizing")
                max_position_pct = st.slider(
                    "Max % of Capital per Trade",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Maximum percentage of capital to risk on a single trade"
                )
                
                max_position_amount = capital_available * (max_position_pct / 100)
                st.info(f"💰 Max position size: ${max_position_amount:,.0f}")
                
                # Risk calculator
                st.subheader("Risk Calculator")
                risk_per_trade = st.number_input(
                    "Risk per Trade ($)",
                    min_value=10.0,
                    max_value=float(max_position_amount),
                    value=float(min(100, max_position_amount)),
                    step=10.0,
                    help="Maximum amount you're willing to lose on this single trade"
                )
                
                risk_percentage = (risk_per_trade / capital_available) * 100
                st.metric("Risk as % of Capital", f"{risk_percentage:.1f}%")
                
                if risk_percentage > 5:
                    st.warning("⚠️ Risk is high (>5% of capital). Consider reducing position size.")
                elif risk_percentage > 2:
                    st.info("ℹ️ Moderate risk level (2-5% of capital).")
                else:
                    st.success("✅ Conservative risk level (<2% of capital).")

            elif comparison_tab_active:
                st.subheader("Compare Different Scenarios")
                
                # Scenario 1
                st.write("**Scenario 1 (Conservative)**")
                col1a, col1b, col1c = st.columns(3)
                with col1a:
                    exp1 = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], key="exp1")
                with col1b:
                    risk1 = st.selectbox("Risk", ["Conservative", "Moderate", "Aggressive"], key="risk1")
                with col1c:
                    cap1 = st.number_input("Capital ($)", 100, 1000000, 500, 100, key="cap1")
                
                # Scenario 2
                st.write("**Scenario 2 (Moderate)**")
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    exp2 = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], key="exp2")
                with col2b:
                    risk2 = st.selectbox("Risk", ["Conservative", "Moderate", "Aggressive"], key="risk2")
                with col2c:
                    cap2 = st.number_input("Capital ($)", 100, 1000000, 2000, 100, key="cap2")
                
                # Scenario 3
                st.write("**Scenario 3 (Aggressive)**")
                col3a, col3b, col3c = st.columns(3)
                with col3a:
                    exp3 = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], key="exp3")
                with col3b:
                    risk3 = st.selectbox("Risk", ["Conservative", "Moderate", "Aggressive"], key="risk3")
                with col3c:
                    cap3 = st.number_input("Capital ($)", 100, 1000000, 10000, 100, key="cap3")
                
                # Store scenarios for comparison
                scenarios = [
                    {"name": "Conservative", "exp": exp1, "risk": risk1, "cap": cap1},
                    {"name": "Moderate", "exp": exp2, "risk": risk2, "cap": cap2},
                    {"name": "Aggressive", "exp": exp3, "risk": risk3, "cap": cap3}
                ]

        with col2:
            st.subheader("Your Market View")

            outlook = st.selectbox(
                "Market Outlook for this Stock",
                options=["Bullish", "Bearish", "Neutral"],
                key='outlook_select',
                help="What direction do you expect?"
            )

            st.write("**Current Analysis Summary:**")
            st.write(f"• Trend: {analysis.trend}")
            st.write(f"• RSI: {analysis.rsi} {'(Oversold)' if analysis.rsi < 30 else '(Overbought)' if analysis.rsi > 70 else '(Neutral)'}")
            st.write(f"• MACD: {analysis.macd_signal}")
            st.write(f"• IV Rank: {analysis.iv_rank}%")
            st.write(f"• Sentiment: {('Positive' if analysis.sentiment_score > 0.2 else 'Negative' if analysis.sentiment_score < -0.2 else 'Neutral')}")

        # Generate recommendations based on selected tab
        if st.button("🚀 Generate Strategy Recommendations", type="primary", width="stretch"):
            with st.spinner("Analyzing optimal strategies..."):
                # Check which tab is active by looking at the current tab selection
                # For now, we'll generate both single and comparison views
                
                # Single profile recommendations
                single_recommendations = StrategyAdvisor.get_recommendations(
                    analysis=analysis,
                    user_experience=user_experience,
                    risk_tolerance=risk_tolerance,
                    capital_available=capital_available,
                    outlook=outlook
                )
                
                # Comparison recommendations
                comparison_results = []
                for scenario in scenarios:
                    scenario_recs = StrategyAdvisor.get_recommendations(
                        analysis=analysis,
                        user_experience=scenario["exp"],
                        risk_tolerance=scenario["risk"],
                        capital_available=scenario["cap"],
                        outlook=outlook
                    )
                    comparison_results.append({
                        "scenario": scenario,
                        "recommendations": scenario_recs
                    })
                
                # Display results
                if single_recommendations:
                    st.subheader(f"📋 Recommended Strategies for {analysis.ticker}")
                    
                    # Show single profile results
                    st.write("**Your Profile Results:**")
                    for idx, rec in enumerate(single_recommendations, 1):
                        confidence_pct = int(rec.confidence * 100)
                        badge = "🟢 High" if confidence_pct >= 70 else "🟡 Moderate" if confidence_pct >= 50 else "🟠 Low"

                        with st.container():
                            cols = st.columns([1, 3, 1])
                            with cols[0]:
                                st.markdown(f"**#{idx}**")
                                st.write(f"**{badge}**")
                                st.progress(confidence_pct / 100)

                            with cols[1]:
                                st.markdown(f"### {rec.strategy_name}")
                                st.write(f"**Match:** {confidence_pct}% • **Risk:** {rec.risk_level} • **Level:** {rec.experience_level}")
                                st.write("**Why this strategy?**")
                                st.write(rec.reasoning)

                                st.write("**When to use / Best conditions:**")
                                for condition in rec.best_conditions:
                                    st.caption(f"• {condition}")

                                # Optional examples and notes if present
                                if hasattr(rec, 'examples') and rec.examples:
                                    st.write("**Examples:**")
                                    for ex in rec.examples:
                                        st.caption(f"• {ex}")

                                if hasattr(rec, 'notes') and rec.notes:
                                    st.info(rec.notes)

                            with cols[2]:
                                st.metric("Confidence", f"{confidence_pct}%")
                                st.write("")
                                st.write("**Risk/Reward**")
                                st.write(f"• Max Loss: {rec.max_loss}")
                                st.write(f"• Max Gain: {rec.max_gain}")

                                if st.button(f"Select", key=f"use_strategy_{idx}"):
                                    st.session_state.selected_strategy = rec.action
                                    st.session_state.selected_ticker = analysis.ticker
                                    st.success(f"✅ Strategy selected! Go to 'Generate Signal' tab.")
                                # Load Example Trade button - populates Generate Signal form with suggested defaults
                                    if st.button(f"Load Example Trade", key=f"strategy_load_example_{idx}"):
                                    # Derive suggested values from examples or defaults
                                        suggested_qty = 2
                                    suggested_iv = int(st.session_state.current_analysis.iv_rank if st.session_state.current_analysis else 48)
                                    suggested_dte = 30
                                    suggested_expiry = (datetime.now() + timedelta(days=suggested_dte)).date()
                                    # Use the current analysis price as a basis for suggested strike if available
                                    base_price = getattr(analysis, 'price', None) or (st.session_state.current_analysis.price if st.session_state.current_analysis else 10)
                                    suggested_strike = round(float(base_price) * 1.0, 2)
                                    # Set session state fields used by Generate Signal tab
                                    st.session_state.selected_strategy = rec.action
                                    st.session_state.selected_ticker = analysis.ticker
                                    st.session_state.example_trade = {
                                        'expiry': suggested_expiry,
                                        'strike': suggested_strike,
                                        'qty': suggested_qty,
                                        'iv_rank': suggested_iv,
                                        'estimated_risk': 200.0,
                                        'llm_score': float(rec.confidence)
                                    }
                                    st.success("✅ Example trade loaded. Go to 'Generate Signal' to review and send.")
                                if st.button(f"Details", key=f"details_strategy_{idx}"):
                                    # Expand a modal-like view by showing an expander with full details
                                    with st.expander(f"Details - {rec.strategy_name}", expanded=True):
                                        st.write(rec.reasoning)
                                        st.write("**Best Conditions:**")
                                        for condition in rec.best_conditions:
                                            st.write(f"• {condition}")
                                        st.write("**Risk/Reward:**")
                                        st.write(f"• Max Loss: {rec.max_loss}")
                                        st.write(f"• Max Gain: {rec.max_gain}")
                                        if hasattr(rec, 'examples') and rec.examples:
                                            st.write("**Examples:**")
                                            for ex in rec.examples:
                                                st.write(f"• {ex}")
                                        if hasattr(rec, 'notes') and rec.notes:
                                            st.write("**Notes:**")
                                            st.write(rec.notes)
                                st.write("\n")
                else:
                    st.warning("No suitable strategies found. Try adjusting your parameters.")
                
                # Display comparison results
                st.divider()
                st.subheader("📊 Strategy Comparison Across Different Scenarios")
                
                if comparison_results:
                    # Create a comparison table
                    comparison_data = []
                    for result in comparison_results:
                        scenario = result["scenario"]
                        recommendations = result["recommendations"]
                        
                        if recommendations:
                            top_rec = recommendations[0]  # Get the top recommendation
                            comparison_data.append({
                                "Scenario": scenario["name"],
                                "Experience": scenario["exp"],
                                "Risk": scenario["risk"],
                                "Capital": f"${scenario['cap']:,}",
                                "Top Strategy": top_rec.strategy_name,
                                "Confidence": f"{int(top_rec.confidence * 100))}%",
                                "Risk Level": top_rec.risk_level,
                                "Max Loss": top_rec.max_loss,
                                "Max Gain": top_rec.max_gain
                            })
                    
                    if comparison_data:
                        # Display as a table
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, width="stretch")
                        
                        # Show detailed comparison for each scenario
                        st.subheader("🔍 Detailed Strategy Analysis by Scenario")
                        
                        for i, result in enumerate(comparison_results):
                            scenario = result["scenario"]
                            recommendations = result["recommendations"]
                            
                            with st.expander(f"Scenario {i+1}: {scenario['name']} ({scenario['exp']} + {scenario['risk']} + ${scenario['cap']:,})", expanded=False):
                                if recommendations:
                                    for idx, rec in enumerate(recommendations[:3], 1):  # Show top 3
                                        confidence_pct = int(rec.confidence * 100)
                                        badge = "🟢 High" if confidence_pct >= 70 else "🟡 Moderate" if confidence_pct >= 50 else "🟠 Low"
                                        
                                        st.write(f"**#{idx} {rec.strategy_name}** - {badge} ({confidence_pct}%)")
                                        st.write(f"Risk: {rec.risk_level} | Max Loss: {rec.max_loss} | Max Gain: {rec.max_gain}")
                                        st.write(f"*{rec.reasoning}*")
                                        st.write("---")
                                else:
                                    st.write("No suitable strategies found for this scenario.")
                    else:
                        st.warning("No strategies found for any scenario. Try adjusting parameters.")
    else:
        # No analysis available - show message
        st.info("💡 **Traditional Strategy Recommendations** require stock analysis. Analyze a stock in the Dashboard tab first, or use Advanced Strategies below.")
    
    # Custom Template Strategies Section
    st.divider()
    # Enhanced Custom Strategy Templates with Analysis Integration
    st.subheader("📚 Your Custom Strategy Templates")
    st.caption("Strategies you've saved in the Strategy Templates tab")
    
    try:
        from models.option_strategy_templates import template_manager
        
        custom_templates = template_manager.get_all_templates()
        
        if custom_templates:
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                template_exp_filter = st.selectbox(
                    "Experience Level",
                    ["All", "Beginner", "Intermediate", "Advanced", "Professional"],
                    key="template_exp_filter"
                )
            with col2:
                template_direction_filter = st.selectbox(
                    "Direction",
                    ["All", "Bullish", "Bearish", "Neutral", "Volatility"],
                    key="template_direction_filter"
                )
            with col3:
                template_oa_filter = st.checkbox(
                    "Option Alpha Compatible Only",
                    value=True,
                    key="template_oa_filter"
                )
            
            # Apply filters
            filtered_templates = custom_templates
            if template_exp_filter != "All":
                filtered_templates = [t for t in filtered_templates if t.experience_level == template_exp_filter]
            if template_direction_filter != "All":
                filtered_templates = [t for t in filtered_templates if template_direction_filter.upper() in t.direction.upper()]
            if template_oa_filter:
                filtered_templates = [t for t in filtered_templates if t.option_alpha_compatible]
            
            if filtered_templates:
                st.write(f"**{len(filtered_templates)} template(s) available**")
                
                # If we have current analysis, show compatibility scores
                if current_analysis:
                    st.info("🎯 **Analysis-Based Recommendations:** Templates are scored based on current market conditions")
                    
                    # Score templates based on current analysis
                    scored_templates = []
                    for template in filtered_templates:
                        score = 0
                        reasoning = []
                        
                        # IV Rank compatibility
                        if template.ideal_iv_rank == "High (>60)" and current_analysis.iv_rank > 60:
                            score += 30
                            reasoning.append(f"✅ High IV Rank ({current_analysis.iv_rank}%) - perfect for premium selling")
                        elif template.ideal_iv_rank == "Low (<30)" and current_analysis.iv_rank < 30:
                            score += 30
                            reasoning.append(f"✅ Low IV Rank ({current_analysis.iv_rank}%) - good for option buying")
                        elif template.ideal_iv_rank == "Medium (30-60)" and 30 <= current_analysis.iv_rank <= 60:
                            score += 25
                            reasoning.append(f"✅ Medium IV Rank ({current_analysis.iv_rank}%) - balanced conditions")
                        
                        # RSI compatibility
                        if template.direction == "BULLISH" and current_analysis.rsi < 30:
                            score += 20
                            reasoning.append(f"✅ Oversold RSI ({current_analysis.rsi:.1f}) - bullish opportunity")
                        elif template.direction == "BEARISH" and current_analysis.rsi > 70:
                            score += 20
                            reasoning.append(f"✅ Overbought RSI ({current_analysis.rsi:.1f}) - bearish opportunity")
                        elif template.direction == "NEUTRAL" and 30 <= current_analysis.rsi <= 70:
                            score += 15
                            reasoning.append(f"✅ Neutral RSI ({current_analysis.rsi:.1f}) - good for neutral strategies")
                        
                        # Trend compatibility
                        if template.direction == "BULLISH" and current_analysis.trend == "Uptrend":
                            score += 25
                            reasoning.append("✅ Uptrending stock - bullish strategies favorable")
                        elif template.direction == "BEARISH" and current_analysis.trend == "Downtrend":
                            score += 25
                            reasoning.append("✅ Downtrending stock - bearish strategies favorable")
                        elif template.direction == "NEUTRAL" and current_analysis.trend == "Sideways":
                            score += 20
                            reasoning.append("✅ Sideways movement - neutral strategies ideal")
                        
                        # Price movement compatibility
                        if template.direction == "BULLISH" and current_analysis.change_pct > 0:
                            score += 10
                            reasoning.append(f"✅ Positive price movement ({current_analysis.change_pct:+.1f}%)")
                        elif template.direction == "BEARISH" and current_analysis.change_pct < 0:
                            score += 10
                            reasoning.append(f"✅ Negative price movement ({current_analysis.change_pct:+.1f}%)")
                        
                        scored_templates.append((template, score, reasoning))
                    
                    # Sort by score
                    scored_templates.sort(key=lambda x: x[1], reverse=True)
                    
                    for template, score, reasoning in scored_templates:
                        with st.expander(f"📋 {template.name} (Compatibility: {score}/100) ({template.experience_level} | {template.risk_level} Risk)", expanded=score>70):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Direction", template.direction)
                                st.metric("Risk Level", template.risk_level)
                            with col2:
                                st.metric("Capital Required", template.capital_requirement)
                                if template.typical_win_rate:
                                    st.metric("Win Rate", template.typical_win_rate)
                            with col3:
                                st.metric("IV Rank", template.ideal_iv_rank)
                                st.metric("Type", template.strategy_type)
                            
                            st.markdown(f"**Description:** {template.description}")
                            st.markdown(f"**Max Loss:** {template.max_loss}")
                            st.markdown(f"**Max Gain:** {template.max_gain}")
                            
                            if reasoning:
                                st.write("**Why This Strategy Works Now:**")
                                for reason in reasoning:
                                    st.write(f"• {reason}")
                            
                            if template.setup_steps:
                                st.write("**Setup Steps:**")
                                for i, step in enumerate(template.setup_steps, 1):
                                    st.write(f"{i}. {step}")
                            
                            if template.warnings:
                                st.write("**⚠️ Warnings:**")
                                for warning in template.warnings:
                                    st.warning(warning)
                            
                            if template.option_alpha_compatible:
                                st.success(f"✅ Option Alpha Compatible - Action: `{template.option_alpha_action}`")
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Use This Template", key=f"use_template_{template.strategy_id}"):
                                    st.session_state.selected_template = template.strategy_id
                                    st.session_state.selected_strategy = template.name
                                    st.session_state.selected_ticker = current_analysis.ticker
                                    st.success(f"✅ Template '{template.name}' selected for {current_analysis.ticker}")
                            with col2:
                                if st.button(f"View Full Details", key=f"details_template_{template.strategy_id}"):
                                    st.info("Navigate to 'Generate Signal' tab to configure this strategy")
                else:
                    # No current analysis - show basic template list
                    for template in filtered_templates:
                        with st.expander(f"📋 {template.name} ({template.experience_level} | {template.risk_level} Risk)"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Direction", template.direction)
                                st.metric("Risk Level", template.risk_level)
                            with col2:
                                st.metric("Capital Required", template.capital_requirement)
                                if template.typical_win_rate:
                                    st.metric("Win Rate", template.typical_win_rate)
                            with col3:
                                st.metric("IV Rank", template.ideal_iv_rank)
                                st.metric("Type", template.strategy_type)
                            
                            st.markdown(f"**Description:** {template.description}")
                            st.markdown(f"**Max Loss:** {template.max_loss}")
                            st.markdown(f"**Max Gain:** {template.max_gain}")
                            
                            if template.option_alpha_compatible:
                                st.success(f"✅ Option Alpha Compatible - Action: `{template.option_alpha_action}`")
                            
                            if st.button(f"Use This Template", key=f"use_template_{template.strategy_id}"):
                                st.session_state.selected_template = template.strategy_id
                                st.success(f"✅ Template selected! Configure in Generate Signal tab.")
            else:
                st.info("No templates match your filters. Adjust filters or add templates in the Strategy Templates tab.")
        else:
            st.info("💡 No custom templates yet. Create your first template in the **Strategy Templates** tab!")
            if st.button("Go to Strategy Templates"):
                st.info("Navigate to the 'Strategy Templates' tab above to create templates")
    
    except Exception as e:
        st.error(f"Error loading custom templates: {e}")
    
    # Strategy Testing and Comparison Section
    if current_analysis:
        st.divider()
        st.subheader("🧪 Strategy Testing & Comparison")
        st.caption("Test and compare different strategies for the current analysis")
        
        # Quick strategy comparison
        if st.button("🔍 Compare All Strategies", type="secondary"):
            with st.spinner("Analyzing all strategies for current market conditions..."):
                try:
                    from models.option_strategy_templates import template_manager
                    
                    all_templates = template_manager.get_all_templates()
                    comparison_results = []
                    
                    for template in all_templates:
                        score = 0
                        reasoning = []
                        
                        # IV Rank compatibility
                        if template.ideal_iv_rank == "High (>60)" and current_analysis.iv_rank > 60:
                            score += 30
                            reasoning.append(f"High IV Rank ({current_analysis.iv_rank}%) - perfect for premium selling")
                        elif template.ideal_iv_rank == "Low (<30)" and current_analysis.iv_rank < 30:
                            score += 30
                            reasoning.append(f"Low IV Rank ({current_analysis.iv_rank}%) - good for option buying")
                        elif template.ideal_iv_rank == "Medium (30-60)" and 30 <= current_analysis.iv_rank <= 60:
                            score += 25
                            reasoning.append(f"Medium IV Rank ({current_analysis.iv_rank}%) - balanced conditions")
                        
                        # RSI compatibility
                        if template.direction == "BULLISH" and current_analysis.rsi < 30:
                            score += 20
                            reasoning.append(f"Oversold RSI ({current_analysis.rsi:.1f}) - bullish opportunity")
                        elif template.direction == "BEARISH" and current_analysis.rsi > 70:
                            score += 20
                            reasoning.append(f"Overbought RSI ({current_analysis.rsi:.1f}) - bearish opportunity")
                        elif template.direction == "NEUTRAL" and 30 <= current_analysis.rsi <= 70:
                            score += 15
                            reasoning.append(f"Neutral RSI ({current_analysis.rsi:.1f}) - good for neutral strategies")
                        
                        # Trend compatibility
                        if template.direction == "BULLISH" and current_analysis.trend == "Uptrend":
                            score += 25
                            reasoning.append("Uptrending stock - bullish strategies favorable")
                        elif template.direction == "BEARISH" and current_analysis.trend == "Downtrend":
                            score += 25
                            reasoning.append("Downtrending stock - bearish strategies favorable")
                        elif template.direction == "NEUTRAL" and current_analysis.trend == "Sideways":
                            score += 20
                            reasoning.append("Sideways movement - neutral strategies ideal")
                        
                        comparison_results.append({
                            'template': template,
                            'score': score,
                            'reasoning': reasoning
                        })
                    
                    # Sort by score
                    comparison_results.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Display results
                    st.success(f"✅ Analyzed {len(comparison_results))} strategies")
                    
                    # Create a comparison table
                    # pandas already imported at module level
                    comparison_data = []
                    for result in comparison_results[:10]:  # Top 10
                        template = result['template']
                        comparison_data.append({
                            'Strategy': template.name,
                            'Direction': template.direction,
                            'Risk': template.risk_level,
                            'Score': result['score'],
                            'IV Match': template.ideal_iv_rank,
                            'Experience': template.experience_level,
                            'Capital Req': template.capital_requirement
                        })
                    
                    df = pd.DataFrame(comparison_data)
                    st.dataframe(df, width="stretch")
                    
                    # Show top 3 strategies in detail
                    st.subheader("🏆 Top 3 Recommended Strategies")
                    for i, result in enumerate(comparison_results[:3], 1):
                        template = result['template']
                        with st.expander(f"#{i} {template.name} (Score: {result['score']}/100)", expanded=i==1):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Direction", template.direction)
                                st.metric("Risk Level", template.risk_level)
                            with col2:
                                st.metric("Capital Required", template.capital_requirement)
                                if template.typical_win_rate:
                                    st.metric("Win Rate", template.typical_win_rate)
                            with col3:
                                st.metric("IV Rank", template.ideal_iv_rank)
                                st.metric("Type", template.strategy_type)
                            
                            st.markdown(f"**Description:** {template.description}")
                            
                            if result['reasoning']:
                                st.write("**Why This Strategy Works Now:**")
                                for reason in result['reasoning']:
                                    st.write(f"• {reason}")
                            
                            if template.setup_steps:
                                st.write("**Setup Steps:**")
                                for j, step in enumerate(template.setup_steps, 1):
                                    st.write(f"{j}. {step}")
                            
                            # Action buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button(f"Use This Strategy", key=f"use_compare_{template.strategy_id}"):
                                    st.session_state.selected_template = template.strategy_id
                                    st.session_state.selected_strategy = template.name
                                    st.session_state.selected_ticker = current_analysis.ticker
                                    st.success(f"✅ Strategy '{template.name}' selected for {current_analysis.ticker}")
                            with col2:
                                if st.button(f"Test Strategy", key=f"test_compare_{template.strategy_id}"):
                                    st.info("Navigate to 'Generate Signal' tab to test this strategy")
                    
                except Exception as e:
                    st.error(f"Error comparing strategies: {e}")
    
    # Advanced Strategies Section (works independently)
    st.divider()
    st.subheader("🚀 Advanced Professional Strategies")
    st.caption("Professional-grade strategies with AI validation")
    
    # Import advanced strategy modules
    try:
        from models.reddit_strategies import get_all_custom_strategies, get_custom_strategy
        from services.reddit_strategy_validator import StrategyValidator
        from analyzers.strategy import StrategyAdvisor as AdvancedAdvisor
        
        # Get available strategies
        user_exp_advanced = st.selectbox(
            "Your Experience Level for Advanced Strategies",
            ["Beginner", "Intermediate", "Advanced", "Professional"],
            index=1,
            key="advanced_exp_level"
        )
        
        advanced_strategies = AdvancedAdvisor.get_custom_strategies(user_exp_advanced)
        
        if not advanced_strategies:
            st.info(f"ℹ️ No advanced strategies available for {user_exp_advanced} level. Try selecting a higher experience level.")
        else:
            # Strategy selection
            strategy_names = [s.name for s in advanced_strategies]
            selected_name = st.selectbox(
                "Select Advanced Strategy",
                strategy_names,
                key="advanced_strategy_select"
            )
            
            # Get selected strategy
            selected_strategy = next(s for s in advanced_strategies if s.name == selected_name)
            
            # Display strategy overview
            with st.expander("📋 Strategy Overview", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Source", selected_strategy.source)
                    st.metric("Experience", selected_strategy.experience_level)
                with col2:
                    st.metric("Risk Level", selected_strategy.risk_level)
                    st.metric("Capital Required", selected_strategy.capital_requirement)
                with col3:
                    if selected_strategy.typical_win_rate:
                        st.metric("Win Rate", selected_strategy.typical_win_rate)
                    st.metric("Products", len(selected_strategy.suitable_products))
                
                st.markdown(f"**Description:** {selected_strategy.description}")
                st.markdown(f"**Philosophy:** {selected_strategy.philosophy}")
            
            # Key metrics
            with st.expander("📊 Performance Metrics"):
                metric_cols = st.columns(len(selected_strategy.key_metrics))
                for i, (key, value) in enumerate(selected_strategy.key_metrics.items()):
                    with metric_cols[i]:
                        st.metric(key.replace("_", " ").title(), value)
            
            # Setup rules
            with st.expander("📖 Step-by-Step Playbook"):
                for rule in sorted(selected_strategy.setup_rules, key=lambda r: r.priority):
                    st.markdown(f"**{rule.priority}. {rule.condition}**")
                    st.info(rule.action)
                    if rule.notes:
                        st.caption(f"📝 {rule.notes}")
                    st.markdown("---")
            
            # Risk management
            with st.expander("🛡️ Risk Management"):
                for rule in selected_strategy.risk_management:
                    mandatory = "🔴 MANDATORY" if rule.mandatory else "🟡 Optional"
                    st.markdown(f"**{rule.rule_type.upper()}** {mandatory}")
                    st.write(f"Value: `{rule.value}`")
                    st.caption(rule.description)
                    st.markdown("---")
            
            # Warnings
            if selected_strategy.warnings:
                with st.expander("⚠️ Important Warnings", expanded=True):
                    for warning in selected_strategy.warnings:
                        st.warning(warning)
            
            # AI Validation Section
            st.markdown("---")
            st.markdown("### 🤖 AI Strategy Validation")
            st.caption("Validate this strategy for a specific ticker with AI analysis")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                ticker_input = st.text_input(
                    "Ticker to Validate",
                    value=st.session_state.current_analysis.ticker if st.session_state.current_analysis else "",
                    placeholder="SPY",
                    key="advanced_strat_ticker"
                )
            with col2:
                include_context = st.checkbox("Include Market Context", value=True, key="include_market_ctx")
            
            market_context = None
            if include_context:
                with st.expander("📊 Market Context (Optional)"):
                    ctx_col1, ctx_col2 = st.columns(2)
                    with ctx_col1:
                        vix = st.number_input("VIX Level", 0.0, 100.0, 20.0, key="vix_input")
                        sentiment = st.selectbox(
                            "Market Sentiment",
                            ["Bullish", "Neutral", "Bearish", "Panic", "Euphoric"],
                            key="sentiment_input"
                        )
                    with ctx_col2:
                        events = st.text_area(
                            "Upcoming Events",
                            placeholder="Fed meeting, earnings season, etc.",
                            key="events_input"
                        )
                    market_context = {
                        "vix": vix,
                        "sentiment": sentiment,
                        "upcoming_events": events
                    }
            
            if st.button("🚀 Validate Strategy with AI", type="primary", key="validate_advanced_btn"):
                if not ticker_input:
                    st.error("Please enter a ticker symbol")
                else:
                    with st.spinner(f"AI analyzing {selected_strategy.name} for {ticker_input}..."):
                        try:
                            # Get stock analysis if available
                            analysis_for_validation = None
                            if st.session_state.current_analysis and st.session_state.current_analysis.ticker == ticker_input:
                                analysis_for_validation = st.session_state.current_analysis
                            else:
                                # Try to get fresh analysis
                                try:
                                    analysis_for_validation = ComprehensiveAnalyzer.analyze_stock(ticker_input, "SWING_TRADE")
                                except Exception as e:
                                    st.warning(f"Could not get fresh analysis: {e}")
                            
                            # Run validation
                            validator = StrategyValidator()
                            validation = validator.validate_strategy(
                                strategy=selected_strategy,
                                ticker=ticker_input,
                                analysis=analysis_for_validation,
                                market_context=market_context
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("### 📊 Validation Results")
                            
                            # Overall verdict
                            if validation.is_viable:
                                st.success(f"✅ **VIABLE** - {validation.market_alignment} Market Alignment")
                            else:
                                st.error(f"❌ **NOT VIABLE** - {validation.market_alignment} Market Alignment")
                            
                            # Metrics
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Viability Score", f"{validation.viability_score:.1%}")
                            with metric_col2:
                                st.metric("Confidence", f"{validation.confidence:.1%}")
                            with metric_col3:
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
                            
                            # Missing conditions
                            if validation.missing_conditions:
                                with st.expander("❌ Missing Conditions", expanded=True):
                                    st.markdown("**The following required conditions are NOT currently met:**")
                                    for condition in validation.missing_conditions:
                                        st.error(f"❌ {condition}")
                            
                            # Red flags
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
                        
                        except Exception as e:
                            st.error(f"Validation failed: {str(e)}")
                            with st.expander("Error Details"):
                                import traceback
                                st.code(traceback.format_exc())
    
    except ImportError as e:
        st.error(f"Advanced strategies module not available: {e}")
        st.info("Make sure the following files exist:\n- models/reddit_strategies.py\n- services/reddit_strategy_validator.py")
    
    # Strategy Explanation Section
    st.divider()
    st.subheader("📚 Understanding Option Strategies")
    
    with st.expander("🔍 Learn About Different Option Strategies", expanded=False):
        st.markdown("""
        ### For Beginners (Low Risk):
        - **Long Call/Put**: Buy options with limited risk (premium paid)
        - **Covered Call**: Sell calls against stock you own
        - **Cash-Secured Put**: Sell puts with cash backing (like your NOK example)
        
        ### For Intermediate (Medium Risk):
        - **Credit Spreads**: Sell one option, buy another to limit risk
        - **Debit Spreads**: Buy one option, sell another to reduce cost
        - **Iron Condors**: Range-bound strategies for sideways markets
        
        ### For Advanced (Higher Risk):
        - **Straddles/Strangles**: Profit from big moves in either direction
        - **Calendar Spreads**: Time-based strategies
        - **Wheel Strategy**: Systematic put selling and call writing
        """)
        
        st.markdown("""
        ### Risk Management Tips:
        1. **Start Small**: Use only 1-2% of capital per trade initially
        2. **Define Risk**: Always know your maximum loss before entering
        3. **Diversify**: Don't put all capital in one strategy or stock
        4. **Learn Gradually**: Master one strategy before trying others
        5. **Use Stops**: Set mental or actual stop losses
        """)
    
    # Quick Reference Section
    st.divider()
    st.subheader("📋 Quick Reference: Investment Approaches")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Conservative Approaches**
        - High Confidence Only
        - EMA Reclaim Setups
        - Power Zone Stocks
        """)
    
    with col2:
        st.markdown("""
        **⚡ Active Trading**
        - Volume Surge
        - Strong Momentum
        - Power Zone Stocks
        """)
    
    with col3:
        st.markdown("""
        **💰 High Risk/Reward**
        - Ultra-Low Price
        - Penny Stocks
        - Volume Surge + Momentum
        """)
    
    st.info("💡 **Tip**: Use the Advanced Scanner tab to apply these approaches to find specific opportunities in the market!")
    
    # Hybrid Approach Explanation
    st.divider()
    st.subheader("🧬 Hybrid Approach: Holistic Stock Assessment")
    
    st.markdown("""
    ### What is the Hybrid Approach?
    
    The **Hybrid Approach** in the Advanced Scanner combines multiple investment approaches with AI analysis and strategy recommendations to provide a **comprehensive, holistic assessment** of the best stocks to invest in and the optimal strategies to use.
    
    ### Key Features:
    
    **🔗 Multi-Filter Combination:**
    - **Primary Approach**: Choose your main investment focus (e.g., "High Confidence Only")
    - **Secondary Filters**: Add additional criteria (e.g., "Volume Surge" + "Power Zone")
    - **Smart Filtering**: Combines all criteria to find stocks that meet multiple conditions
    
    **🎯 Strategy Integration:**
    - **Personalized Recommendations**: Get specific trading strategies for each found opportunity
    - **Risk-Adjusted**: Strategies are tailored to your experience level and risk tolerance
    - **Capital-Aware**: Recommendations consider your available capital
    - **Market Outlook**: Strategies align with your market expectations
    
    **🤖 AI-Enhanced Analysis:**
    - **Comprehensive Scoring**: Combines technical, fundamental, and sentiment analysis
    - **Confidence Ratings**: Each opportunity gets an AI confidence score
    - **Risk Assessment**: Detailed risk analysis for each recommendation
    
    ### Example Hybrid Scenarios:
    
    **Conservative Hybrid:**
    - Primary: "High Confidence Only (Score ≥70)"
    - Secondary: "EMA Reclaim Setups" + "RSI Oversold (<30)"
    - Result: High-quality, low-risk setups with strong technical confirmation
    
    **Momentum Hybrid:**
    - Primary: "Strong Momentum (>5% change)"
    - Secondary: "Volume Surge (>2x avg)" + "Power Zone Stocks Only"
    - Result: High-momentum stocks with strong volume and technical confirmation
    
    **Penny Stock Hybrid:**
    - Primary: "Penny Stocks ($1-$5)"
    - Secondary: "Volume Surge (>2x avg)" + "High Confidence Only"
    - Result: Quality penny stocks with strong volume and AI confidence
    
    ### Benefits of Hybrid Approach:
    
    1. **Higher Quality Results**: Multiple filters reduce false signals
    2. **Personalized Strategies**: Get specific trading recommendations for each stock
    3. **Risk Management**: Built-in risk assessment and strategy matching
    4. **Comprehensive Analysis**: Combines technical, fundamental, and AI analysis
    5. **Actionable Insights**: Not just what to buy, but how to trade it
    
    ### How to Use:
    
    1. **Enable Hybrid Mode**: Check "🧬 Use Hybrid Approach" in the Advanced Scanner
    2. **Set Primary Filter**: Choose your main investment approach
    3. **Add Secondary Filters**: Select additional criteria to combine
    4. **Configure Strategy Preferences**: Set your experience, risk tolerance, and capital
    5. **Run Scan**: Get comprehensive results with strategy recommendations
    6. **Review Results**: Each opportunity shows both analysis and recommended strategies
    7. **Take Action**: Use the recommended strategies or get full analysis
    
    This hybrid approach gives you the **most comprehensive and actionable** stock analysis available, combining the best of technical analysis, AI insights, and personalized strategy recommendations.
    """)


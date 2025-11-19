"""
UI for displaying monitored entry opportunities
Shows all active monitors waiting for entry conditions
"""

import streamlit as st
from loguru import logger
from datetime import datetime, timedelta


def display_entry_monitors():
    """Display all monitored entry opportunities"""
    
    st.markdown("## 🔔 Monitored Entry Opportunities")
    st.write("AI is watching these coins for optimal entry timing. You'll get Discord alerts when conditions are met.")
    
    # Get AI Entry Assistant from session state
    if 'ai_entry_assistant' not in st.session_state:
        st.info("👋 **No monitors active yet.**\n\nUse the **Quick Trade** tab to analyze coins. When AI says 'WAIT', click '🔔 Monitor & Alert' to add them here.")
        return
    
    entry_assistant = st.session_state.ai_entry_assistant
    opportunities = entry_assistant.get_monitored_opportunities()
    
    if not opportunities:
        st.info("👋 **No monitors active yet.**\n\nUse the **Quick Trade** tab to analyze coins. When AI says 'WAIT', click '🔔 Monitor & Alert' to add them here.")
        return
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Monitors", len(opportunities))
    
    # Count by action type
    wait_pullback = sum(1 for opp in opportunities if opp.original_analysis.action == "WAIT_FOR_PULLBACK")
    wait_breakout = sum(1 for opp in opportunities if opp.original_analysis.action == "WAIT_FOR_BREAKOUT")
    col2.metric("Waiting for Pullback", wait_pullback)
    col3.metric("Waiting for Breakout", wait_breakout)
    
    st.markdown("---")
    
    # Display each monitored opportunity
    for opp in opportunities:
        with st.expander(f"**{opp.pair}** - {opp.original_analysis.action}", expanded=True):
            # Header with status
            status_col1, status_col2 = st.columns([3, 1])
            
            with status_col1:
                st.markdown(f"### {opp.pair}")
                st.markdown(f"**Direction:** {opp.side}")
                
                # Calculate monitoring duration
                duration = datetime.now() - opp.created_time
                hours = int(duration.total_seconds() // 3600)
                minutes = int((duration.total_seconds() % 3600) // 60)
                st.caption(f"⏱️ Monitoring for: {hours}h {minutes}m")
            
            with status_col2:
                # Confidence badge
                confidence = opp.original_analysis.confidence
                if confidence >= 70:
                    st.success(f"🟢 {confidence:.0f}%")
                elif confidence >= 50:
                    st.warning(f"🟡 {confidence:.0f}%")
                else:
                    st.error(f"🔴 {confidence:.0f}%")
            
            # Target conditions
            st.markdown("#### 🎯 Target Conditions")
            
            cond_col1, cond_col2, cond_col3 = st.columns(3)
            
            with cond_col1:
                if opp.target_price:
                    current_price = opp.current_price if opp.current_price > 0 else opp.original_analysis.current_price
                    pct_to_target = ((opp.target_price - current_price) / current_price) * 100
                    
                    st.metric(
                        "Target Price",
                        f"${opp.target_price:,.6f}",
                        delta=f"{pct_to_target:+.2f}% to go",
                        delta_color="normal" if pct_to_target < 0 else "inverse"
                    )
                    st.caption(f"Current: ${current_price:,.6f}")
                else:
                    st.metric("Target Price", "Breakout")
            
            with cond_col2:
                if opp.target_rsi:
                    st.metric("Target RSI", f"< {opp.target_rsi:.0f}")
                    st.caption("(Reset from overbought)")
                else:
                    st.metric("Target RSI", "Any")
            
            with cond_col3:
                st.metric("Position Size", f"${opp.position_size:,.0f}")
                st.caption(f"Risk: {opp.risk_pct:.1f}% | TP: {opp.take_profit_pct:.1f}%")
            
            # AI reasoning
            st.markdown("#### 💡 AI Analysis")
            st.info(f"**Reasoning:** {opp.original_analysis.reasoning}")
            
            # Scores
            score_col1, score_col2, score_col3, score_col4 = st.columns(4)
            score_col1.metric("Technical", f"{opp.original_analysis.technical_score:.0f}/100")
            score_col2.metric("Trend", f"{opp.original_analysis.trend_score:.0f}/100")
            score_col3.metric("Timing", f"{opp.original_analysis.timing_score:.0f}/100")
            score_col4.metric("Risk", f"{opp.original_analysis.risk_score:.0f}/100")
            
            # Action buttons
            st.markdown("---")
            btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])
            
            with btn_col1:
                if st.button(f"🔄 Re-Analyze {opp.pair}", key=f"reanalyze_{opp.pair}", width='stretch'):
                    st.info(f"Navigate to Quick Trade tab and analyze {opp.pair} again to get updated recommendation")
            
            with btn_col2:
                if st.button("📊 Quick Trade", key=f"goto_qt_{opp.pair}", width='stretch'):
                    # Store pair for quick access
                    st.session_state['crypto_custom_pair'] = opp.pair
                    st.info(f"Switch to 'Quick Trade' tab to execute {opp.pair}")
            
            with btn_col3:
                if st.button("🗑️ Remove", key=f"remove_{opp.pair}", width='stretch', type="secondary"):
                    # Find the opportunity ID
                    opp_id = None
                    for oid, opportunity in entry_assistant.opportunities.items():
                        if opportunity.pair == opp.pair:
                            opp_id = oid
                            break
                    
                    if opp_id:
                        entry_assistant.remove_opportunity(opp_id)
                        st.success(f"✅ Removed {opp.pair} from monitoring")
                        st.rerun()
                    else:
                        st.error("Failed to remove monitor")
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    ### ℹ️ How Monitoring Works
    
    **AI checks every 60 seconds:**
    - Current price vs target
    - RSI levels (if specified)
    - Trend still valid?
    - Volume still present?
    
    **When conditions are met:**
    - 🔔 Discord alert sent
    - Return to app → Quick Trade tab
    - Re-analyze to confirm
    - Execute if AI approves
    
    **Tips:**
    - Keep Discord open for alerts
    - Don't manually watch prices
    - Trust AI's patience
    - Execute promptly when alerted
    """)


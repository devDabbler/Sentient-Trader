"""
UI for displaying monitored stock entry opportunities
"""

import streamlit as st
from loguru import logger
from datetime import datetime


def display_stock_entry_monitors():
    """Display all monitored stock entry opportunities"""
    
    st.markdown("## 🔔 Stock Entry Monitors")
    st.write("AI is watching these stocks for optimal entry timing in your auto-trader.")
    
    # Get AI Entry Assistant from auto-trader
    if 'auto_trader' not in st.session_state or not st.session_state.auto_trader:
        st.info("👋 **Auto-trader not running.**\n\nStart auto-trader with `use_ai_entry_timing=True` to enable stock entry monitoring.")
        return
    
    auto_trader = st.session_state.auto_trader
    if not hasattr(auto_trader, '_stock_entry_assistant') or not auto_trader._stock_entry_assistant:
        st.info("👋 **AI entry timing not enabled.**\n\nSet `use_ai_entry_timing=True` in your auto-trader config to enable this feature.")
        return
    
    entry_assistant = auto_trader._stock_entry_assistant
    opportunities = entry_assistant.get_monitored_opportunities()
    
    if not opportunities:
        st.info("👋 **No stocks being monitored yet.**\n\nWhen auto-trader finds signals but AI says 'WAIT', they'll appear here.")
        return
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Active Monitors", len(opportunities))
    
    wait_pullback = sum(1 for opp in opportunities if opp.original_analysis.action == "WAIT_FOR_PULLBACK")
    wait_breakout = sum(1 for opp in opportunities if opp.original_analysis.action == "WAIT_FOR_BREAKOUT")
    col2.metric("Waiting for Pullback", wait_pullback)
    col3.metric("Waiting for Breakout", wait_breakout)
    
    st.markdown("---")
    
    # Display each opportunity
    for opp in opportunities:
        with st.expander(f"**${opp.symbol}** - {opp.original_analysis.action}", expanded=True):
            status_col1, status_col2 = st.columns([3, 1])
            
            with status_col1:
                st.markdown(f"### ${opp.symbol}")
                st.markdown(f"**Side:** {opp.side} ({'Long' if opp.side == 'BUY' else 'Short'})")
                
                # Market hours check
                now = datetime.now()
                market_hours = 9.5 <= now.hour + now.minute/60 <= 16
                if market_hours:
                    st.success("🟢 Market Open")
                else:
                    st.warning("🟡 Market Closed")
                
                # Monitoring duration
                duration = datetime.now() - opp.created_time
                hours = int(duration.total_seconds() // 3600)
                minutes = int((duration.total_seconds() % 3600) // 60)
                st.caption(f"⏱️ Monitoring: {hours}h {minutes}m")
            
            with status_col2:
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
                    current = opp.current_price if opp.current_price > 0 else opp.original_analysis.current_price
                    pct_to_target = ((opp.target_price - current) / current) * 100
                    
                    st.metric(
                        "Target Price",
                        f"${opp.target_price:,.2f}",
                        delta=f"{pct_to_target:+.2f}% to go"
                    )
                    st.caption(f"Current: ${current:,.2f}")
                else:
                    st.metric("Target Price", "Breakout")
            
            with cond_col2:
                if opp.target_rsi:
                    st.metric("Target RSI", f"< {opp.target_rsi:.0f}")
                else:
                    st.metric("Target RSI", "Any")
            
            with cond_col3:
                st.metric("Position Size", f"${opp.position_size:,.0f}")
                st.caption(f"Risk: {opp.risk_pct:.1f}% | TP: {opp.take_profit_pct:.1f}%")
            
            # AI reasoning
            st.markdown("#### 💡 AI Analysis")
            st.info(f"**Reasoning:** {opp.original_analysis.reasoning}")
            
            # Scores
            score_cols = st.columns(4)
            score_cols[0].metric("Technical", f"{opp.original_analysis.technical_score:.0f}/100")
            score_cols[1].metric("Trend", f"{opp.original_analysis.trend_score:.0f}/100")
            score_cols[2].metric("Timing", f"{opp.original_analysis.timing_score:.0f}/100")
            score_cols[3].metric("Risk", f"{opp.original_analysis.risk_score:.0f}/100")
            
            # Action buttons
            st.markdown("---")
            btn_cols = st.columns([2, 1, 1])
            
            with btn_cols[0]:
                if st.button(f"🔄 Re-Analyze ${opp.symbol}", key=f"reanalyze_{opp.symbol}"):
                    st.info("Re-analysis will happen on next auto-trader scan cycle")
            
            with btn_cols[1]:
                if st.button(f"🗑️ Remove", key=f"remove_{opp.symbol}", type="secondary"):
                    # Find opportunity ID
                    opp_id = None
                    for oid, opportunity in entry_assistant.opportunities.items():
                        if opportunity.symbol == opp.symbol:
                            opp_id = oid
                            break
                    
                    if opp_id:
                        entry_assistant.remove_opportunity(opp_id)
                        st.success(f"✅ Removed ${opp.symbol}")
                        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### ℹ️ How It Works
    
    **AI checks every 60 seconds:**
    - Price vs target
    - RSI levels
    - Trend validity
    - Volume presence
    
    **When conditions met:**
    - 🔔 Discord alert
    - If auto-entry enabled → Executes immediately
    - If manual → Wait for your approval
    """)


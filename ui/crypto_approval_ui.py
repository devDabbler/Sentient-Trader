"""
UI for viewing pending Discord trade approvals
Shows all trades awaiting user approval via Discord
"""

import streamlit as st
from loguru import logger
from datetime import datetime, timedelta
from typing import Optional


def display_pending_approvals():
    """Display all pending trade approvals"""
    
    st.markdown("## ⏳ Pending Discord Trade Approvals")
    st.write("These trades are waiting for your approval via Discord. Reply with APPROVE or REJECT in Discord.")
    
    # Get Discord approval manager
    try:
        from services.discord_trade_approval import get_discord_approval_manager
        
        approval_manager = get_discord_approval_manager()
        
        if not approval_manager or not approval_manager.enabled:
            st.warning("⚠️ **Discord Approval Manager not configured.**\n\nTo enable Discord trade approvals:\n1. Add DISCORD_BOT_TOKEN to .env\n2. Add DISCORD_CHANNEL_IDS to .env\n3. Restart the application")
            return
        
        if not approval_manager.is_running():
            st.warning("⚠️ **Discord bot not running.**\n\nBot is starting... please wait a moment and refresh.")
            
            # Try to start it
            if st.button("🔄 Start Discord Bot"):
                with st.spinner("Starting Discord bot..."):
                    success = approval_manager.start()
                    if success:
                        st.success("✅ Discord bot started!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to start Discord bot. Check logs for details.")
            return
        
        # Get pending approvals
        pending = approval_manager.get_pending_approvals()
        
        if not pending:
            st.info("✅ **No pending approvals.**\n\nWhen AI detects entry opportunities, you'll receive approval requests in Discord.\n\n**To approve/reject:**\n• Go to Discord\n• Reply with `APPROVE` or `REJECT`")
            return
        
        # Cleanup expired
        expired_count = approval_manager.cleanup_expired()
        if expired_count > 0:
            st.warning(f"⏰ Removed {expired_count} expired approval request(s)")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Pending Approvals", len(pending))
        
        approved_count = sum(1 for a in pending.values() if a.approved)
        rejected_count = sum(1 for a in pending.values() if a.rejected)
        awaiting_count = sum(1 for a in pending.values() if not a.approved and not a.rejected)
        
        col2.metric("Awaiting Response", awaiting_count)
        col3.metric("Approved/Rejected", f"{approved_count}/{rejected_count}")
        
        st.markdown("---")
        
        # Display each pending approval
        for approval_id, approval in sorted(pending.items(), key=lambda x: x[1].created_time, reverse=True):
            # Determine status and color
            if approval.approved:
                status_emoji = "✅"
                status_text = "APPROVED"
                status_color = "green"
            elif approval.rejected:
                status_emoji = "❌"
                status_text = "REJECTED"
                status_color = "red"
            else:
                # Check if expired
                if approval.is_expired():
                    status_emoji = "⏰"
                    status_text = "EXPIRED"
                    status_color = "orange"
                else:
                    status_emoji = "⏳"
                    status_text = "AWAITING"
                    status_color = "blue"
            
            with st.expander(f"**{approval.pair}** - {status_emoji} {status_text}", expanded=(status_text == "AWAITING")):
                # Header with status
                status_col1, status_col2, status_col3 = st.columns([2, 1, 1])
                
                with status_col1:
                    st.markdown(f"### {approval.pair}")
                    st.markdown(f"**Direction:** {approval.side}")
                    st.markdown(f"**Strategy:** {approval.strategy}")
                
                with status_col2:
                    # Confidence badge
                    confidence = approval.confidence
                    if confidence >= 80:
                        st.success(f"🟢 {confidence:.0f}%")
                    elif confidence >= 60:
                        st.warning(f"🟡 {confidence:.0f}%")
                    else:
                        st.error(f"🔴 {confidence:.0f}%")
                
                with status_col3:
                    # Time since request
                    elapsed = datetime.now() - approval.created_time
                    hours = int(elapsed.total_seconds() // 3600)
                    minutes = int((elapsed.total_seconds() % 3600) // 60)
                    
                    if hours > 0:
                        st.caption(f"⏱️ {hours}h {minutes}m ago")
                    else:
                        st.caption(f"⏱️ {minutes}m ago")
                    
                    # Expiry
                    if not approval.approved and not approval.rejected:
                        remaining = approval.expires_minutes - (elapsed.total_seconds() / 60)
                        if remaining > 0:
                            st.caption(f"⏳ {int(remaining)}min left")
                        else:
                            st.caption("⏰ EXPIRED")
                
                # Trade details
                st.markdown("#### 💰 Trade Details")
                
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.metric("Entry Price", f"${approval.entry_price:,.6f}")
                    st.caption(f"Position: ${approval.position_size:,.2f}")
                
                with detail_col2:
                    risk_pct = abs((approval.entry_price - approval.stop_loss) / approval.entry_price) * 100
                    st.metric("Stop Loss", f"${approval.stop_loss:,.6f}")
                    st.caption(f"Risk: {risk_pct:.2f}%")
                
                with detail_col3:
                    reward_pct = abs((approval.take_profit - approval.entry_price) / approval.entry_price) * 100
                    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                    st.metric("Take Profit", f"${approval.take_profit:,.6f}")
                    st.caption(f"R:R = {rr_ratio:.2f}:1")
                
                # AI reasoning
                st.markdown("#### 💡 AI Analysis")
                st.info(f"**Reasoning:** {approval.reasoning}")
                
                # Execution result (if executed)
                if approval.executed and approval.execution_result:
                    st.markdown("#### ✅ Execution Result")
                    st.success(f"Trade executed successfully!")
                    
                    if 'order_id' in approval.execution_result:
                        st.caption(f"Order ID: {approval.execution_result['order_id']}")
                    
                    if 'fill_price' in approval.execution_result:
                        st.caption(f"Fill Price: ${approval.execution_result['fill_price']:,.6f}")
                
                # Action buttons (only show for awaiting)
                if status_text == "AWAITING":
                    st.markdown("---")
                    st.markdown("#### 📲 How to Approve/Reject")
                    
                    st.info(
                        f"**Go to Discord and reply with:**\n"
                        f"• `APPROVE` or `YES` → Execute this trade ✅\n"
                        f"• `REJECT` or `NO` → Cancel this trade ❌\n\n"
                        f"Discord Channel: <#{approval_manager.channel_id}>\n"
                        f"Approval ID: `{approval_id}`"
                    )
                    
                    # Manual buttons (fallback if Discord not working)
                    st.markdown("**Or use buttons below (if Discord unavailable):**")
                    btn_col1, btn_col2 = st.columns(2)
                    
                    with btn_col1:
                        if st.button(f"✅ Approve {approval.pair}", key=f"approve_{approval_id}", type="primary"):
                            # Simulate Discord approval
                            approval.approved = True
                            if approval_manager.bot and approval_manager.bot.approval_callback:
                                approval_manager.bot.approval_callback(approval_id, True)
                            st.success(f"✅ Approved! Executing trade...")
                            st.rerun()
                    
                    with btn_col2:
                        if st.button(f"❌ Reject {approval.pair}", key=f"reject_{approval_id}", type="secondary"):
                            # Simulate Discord rejection
                            approval.rejected = True
                            if approval_manager.bot and approval_manager.bot.approval_callback:
                                approval_manager.bot.approval_callback(approval_id, False)
                            st.warning(f"❌ Rejected. Trade cancelled.")
                            st.rerun()
        
        # Footer info
        st.markdown("---")
        st.markdown("""
        ### ℹ️ How Discord Approvals Work
        
        **When AI finds an entry opportunity:**
        1. 🔔 Discord bot sends approval request to your channel
        2. 📊 Shows trade details, AI confidence, risk/reward
        3. ⏳ Waits for your response (60 minutes timeout)
        
        **To approve/reject:**
        1. 📱 Open Discord on phone or desktop
        2. 💬 Reply in the channel with:
           - `APPROVE`, `YES`, `GO`, or `EXECUTE` → Trade executes ✅
           - `REJECT`, `NO`, or `CANCEL` → Trade cancelled ❌
        3. 🤖 Bot executes approved trades automatically
        4. 📈 Positions added to AI Position Manager for monitoring
        
        **Tips:**
        - Keep Discord open with notifications enabled
        - Respond quickly to get best entry prices
        - Review AI reasoning before approving
        - Check risk:reward ratio (aim for 2:1+)
        - Monitor executed trades in AI Position Monitor tab
        """)
    
    except ImportError as e:
        st.error(f"❌ Discord approval system not available: {e}")
        st.info("Make sure discord_trade_approval.py is in the services folder")
    
    except Exception as e:
        logger.error("Error displaying pending approvals: {}", str(e), exc_info=True)
        st.error(f"❌ Error loading approvals: {str(e)}")


def display_approval_status_widget():
    """Compact widget showing approval status (for sidebar or quick view)"""
    try:
        from services.discord_trade_approval import get_discord_approval_manager
        
        approval_manager = get_discord_approval_manager()
        
        if not approval_manager or not approval_manager.enabled:
            return
        
        pending = approval_manager.get_pending_approvals()
        awaiting = sum(1 for a in pending.values() if not a.approved and not a.rejected and not a.is_expired())
        
        if awaiting > 0:
            st.warning(f"⏳ **{awaiting} trade(s) awaiting approval in Discord!**")
            if st.button("📋 View Approvals"):
                st.session_state['switch_to_approvals'] = True
                st.rerun()
    
    except Exception as e:
        logger.debug(f"Error in approval status widget: {e}")

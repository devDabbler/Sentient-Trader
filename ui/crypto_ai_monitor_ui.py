"""
AI Crypto Position Monitor UI
Real-time dashboard for AI-managed crypto positions
"""

import streamlit as st
from loguru import logger
from datetime import datetime
from typing import Optional
import pandas as pd
import os


def display_crypto_ai_monitors():
    """
    Alias function for backward compatibility
    """
    return display_ai_position_monitor()


def display_ai_position_monitor():
    """
    Display AI Position Monitor dashboard
    Shows all active positions being managed by AI
    """
    st.markdown("### 🤖 AI Position Monitor")
    st.markdown("Real-time AI-powered position management and intelligent exit decisions")
    
    # Add Portfolio Analysis section (always visible)
    display_portfolio_analysis()
    
    st.markdown("---")
    
    # Check if AI manager exists
    if 'ai_position_manager' not in st.session_state:
        st.info("ℹ️ No active AI position monitoring. Execute a trade to activate AI monitoring.")
        
        # 🆕 ENHANCED: Offer to load existing Kraken positions
        st.markdown("---")
        st.markdown("#### 🔄 Load Existing Positions")
        st.info("💡 **Tip:** You can load your existing Kraken positions into AI monitoring")
        
        if st.button("🔍 Scan & Load Kraken Positions", width='stretch', type="primary"):
            with st.spinner("Scanning your Kraken account for open positions..."):
                try:
                    # Get Kraken client from environment variables
                    kraken_key = os.getenv('KRAKEN_API_KEY')
                    kraken_secret = os.getenv('KRAKEN_API_SECRET')
                    
                    if not kraken_key or not kraken_secret:
                        st.error("❌ Kraken API credentials not found. Please set KRAKEN_API_KEY and KRAKEN_API_SECRET in your .env file.")
                    else:
                        # Use cached Kraken client (same as app.py)
                        from clients.kraken_client import KrakenClient
                        
                        # Try to get from session state first, otherwise create new
                        if 'kraken_client' in st.session_state:
                            kraken_client = st.session_state.kraken_client
                        else:
                            kraken_client = KrakenClient(api_key=kraken_key, api_secret=kraken_secret)
                            success, message = kraken_client.validate_connection()
                            if not success:
                                st.error(f"❌ Failed to connect to Kraken: {message}")
                                return
                            st.session_state.kraken_client = kraken_client
                        
                        # Fetch open positions with accurate entry prices
                        positions = kraken_client.get_open_positions(calculate_real_cost=True)
                        
                        if not positions:
                            st.warning("⚠️ No open positions found in your Kraken account")
                        else:
                            # Initialize AI Position Manager
                            from services.ai_crypto_position_manager import get_ai_position_manager
                            from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                            import time
                            
                            llm_analyzer = LLMStrategyAnalyzer()
                            ai_manager = get_ai_position_manager(
                                kraken_client=kraken_client,
                                llm_analyzer=llm_analyzer,
                                check_interval_seconds=60,
                                enable_ai_decisions=True,
                                enable_trailing_stops=True,
                                enable_breakeven_moves=True,
                                enable_partial_exits=True,
                                require_manual_approval=True  # SAFETY: Always require approval
                            )
                            st.session_state.ai_position_manager = ai_manager
                            
                            # Add each position to AI monitoring
                            loaded_count = 0
                            for pos in positions:
                                trade_id = f"{pos.pair}_imported_{int(time.time())}_{loaded_count}"
                                
                                # Calculate stop loss and take profit (5% and 10% from current)
                                stop_loss = pos.current_price * 0.95
                                take_profit = pos.current_price * 1.10
                                
                                success = ai_manager.add_position(
                                    trade_id=trade_id,
                                    pair=pos.pair,
                                    side='BUY',  # Kraken spot is always long
                                    volume=pos.volume,
                                    entry_price=pos.entry_price if pos.entry_price > 0 else pos.current_price,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    strategy='Imported',
                                    entry_order_id=f"imported_{loaded_count}"
                                )
                                
                                if success:
                                    loaded_count += 1
                            
                            # Start monitoring loop
                            if not ai_manager.is_running:
                                ai_manager.start_monitoring_loop()
                                logger.info("🤖 AI Position Manager monitoring loop started")
                            
                            st.success(f"✅ Loaded {loaded_count} position(s) into AI monitoring!")
                            st.balloons()
                            st.rerun()
                            
                except Exception as e:
                    st.error(f"Failed to load positions: {e}")
                    logger.error("Position load error: {}", str(e), exc_info=True)
        
        # Show configuration options
        st.markdown("---")
        with st.expander("⚙️ AI Monitor Configuration"):
            st.markdown("""
            **AI Position Manager Features:**
            - ✅ Real-time position monitoring (every 60 seconds)
            - ✅ AI-powered exit decisions using LLM analysis
            - ✅ Dynamic stop loss adjustments (trailing stops)
            - ✅ Automatic breakeven protection (after +3% profit)
            - ✅ Intelligent partial profit taking
            - ✅ Multi-factor trend analysis
            - ✅ 24/7 monitoring for crypto markets
            
            **AI will automatically:**
            - Hold longer when trend is strengthening
            - Exit sooner when trend weakens or reverses
            - Tighten stops to protect profits
            - Extend targets in strong trends
            - Take partial profits at resistance levels
            """)
        return
    
    ai_manager = st.session_state.ai_position_manager
    
    # Get current status
    try:
        status = ai_manager.get_status()
    except Exception as e:
        st.error(f"Error getting AI manager status: {e}")
        logger.error("AI manager status error: {}", str(e), exc_info=True)
        return
    
    # Status header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if status['is_running']:
            st.metric("Status", "🟢 ACTIVE", delta="Monitoring")
        else:
            st.metric("Status", "🔴 STOPPED", delta="Inactive")
    
    with col2:
        st.metric("Active Positions", status['active_positions'])
    
    with col3:
        st.metric("AI Adjustments", status['statistics']['total_ai_adjustments'])
    
    with col4:
        st.metric("Trailing Stops", status['statistics']['trailing_stop_activations'])
    
    # Statistics
    st.markdown("#### 📊 AI Performance Statistics")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Breakeven Moves", status['statistics']['breakeven_moves'])
    
    with stats_col2:
        st.metric("Partial Exits", status['statistics']['partial_exits_taken'])
    
    with stats_col3:
        st.metric("AI Exits", status['statistics']['ai_exit_signals'])
    
    with stats_col4:
        check_interval = status['config']['check_interval']
        st.metric("Check Interval", f"{check_interval}s")
    
    st.markdown("---")
    
    # 🚨 SAFETY: Show pending approvals first
    if hasattr(ai_manager, 'pending_approvals') and ai_manager.pending_approvals:
        st.markdown("#### 🚨 Pending Trade Approvals")
        st.warning(f"⚠️ **{len(ai_manager.pending_approvals)} trade(s) awaiting your approval!** AI will NOT execute without your explicit confirmation.")
        
        for approval_id, approval_data in list(ai_manager.pending_approvals.items()):
            decision = approval_data['decision']
            pair = approval_data['pair']
            timestamp = approval_data['timestamp']
            
            with st.expander(f"🚨 {pair} - {decision.action} (Requested {timestamp.strftime('%H:%M:%S')})", expanded=True):
                st.markdown(f"**Action:** {decision.action}")
                st.markdown(f"**Confidence:** {decision.confidence:.0f}%")
                st.markdown(f"**Reasoning:** {decision.reasoning}")
                st.markdown(f"**Urgency:** {decision.urgency}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"✅ Approve & Execute", key=f"approve_{approval_id}", type="primary"):
                        try:
                            # Execute with approval skip
                            success = ai_manager.execute_decision(
                                trade_id=approval_data['trade_id'],
                                decision=decision,
                                skip_approval=True  # User approved it
                            )
                            if success:
                                # Remove from pending
                                del ai_manager.pending_approvals[approval_id]
                                st.success(f"✅ Trade executed successfully!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.error("❌ Failed to execute trade")
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with col2:
                    if st.button(f"❌ Reject", key=f"reject_{approval_id}"):
                        # Remove from pending without executing
                        del ai_manager.pending_approvals[approval_id]
                        st.info("Trade rejected - no action taken")
                        st.rerun()
        
        st.markdown("---")
    
    # Active positions
    if status['active_positions'] > 0:
        st.markdown("#### 🎯 Active Positions")
        
        positions_data = []
        for trade_id, pos in status['positions'].items():
            if pos['status'] == 'ACTIVE':
                # Calculate colors based on P&L
                pnl = pos['pnl_pct']
                pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                
                # Time formatting
                hold_time = pos['hold_time_minutes']
                if hold_time < 60:
                    time_str = f"{hold_time:.0f}m"
                elif hold_time > 0:
                    time_str = f"{hold_time/60:.1f}h"
                else:
                    time_str = "0m"
                
                positions_data.append({
                    'Pair': pos['pair'],
                    'Side': pos['side'],
                    'Entry': f"${pos['entry_price']:,.6f}",
                    'Current': f"${pos['current_price']:,.6f}",
                    'P&L%': f"{pnl_color} {pnl:+.2f}%",
                    'Stop': f"${pos['stop_loss']:,.6f}",
                    'Target': f"${pos['take_profit']:,.6f}",
                    'Hold Time': time_str,
                    'AI Action': pos['last_ai_action'],
                    'AI Confidence': f"{pos['last_ai_confidence']:.0f}%",
                    'BE': "✅" if pos['moved_to_breakeven'] else "❌",
                    'Partial': "✅" if pos['partial_exit_taken'] else "❌"
                })
        
        if positions_data:
            df = pd.DataFrame(positions_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            # Position details
            st.markdown("#### 🔍 Position Details")
            
            for trade_id, pos in status['positions'].items():
                if pos['status'] != 'ACTIVE':
                    continue
                
                with st.expander(f"{pos['pair']} - {pos['side']} ({pos['pnl_pct']:+.2f}%)"):
                    detail_col1, detail_col2, detail_col3 = st.columns(3)
                    
                    with detail_col1:
                        st.markdown("**Position Info:**")
                        st.write(f"- Entry: ${pos['entry_price']:,.6f}")
                        st.write(f"- Current: ${pos['current_price']:,.6f}")
                        st.write(f"- P&L: {pos['pnl_pct']:+.2f}%")
                        st.write(f"- Hold Time: {pos['hold_time_minutes']:.1f} min")
                    
                    with detail_col2:
                        st.markdown("**Risk Management:**")
                        st.write(f"- Stop Loss: ${pos['stop_loss']:,.6f}")
                        st.write(f"- Take Profit: ${pos['take_profit']:,.6f}")
                        st.write(f"- Breakeven: {'✅ Yes' if pos['moved_to_breakeven'] else '❌ No'}")
                        st.write(f"- Partial Exit: {'✅ Yes' if pos['partial_exit_taken'] else '❌ No'}")
                    
                    with detail_col3:
                        st.markdown("**AI Analysis:**")
                        st.write(f"- Last Action: {pos['last_ai_action']}")
                        st.write(f"- Confidence: {pos['last_ai_confidence']:.0f}%")
                        st.write(f"- Status: {pos['status']}")
                    
                    # Manual controls
                    st.markdown("**Manual Controls:**")
                    manual_col1, manual_col2 = st.columns(2)
                    
                    with manual_col1:
                        if st.button(f"🚨 Close Position", key=f"close_{trade_id}"):
                            try:
                                from services.ai_crypto_position_manager import AITradeDecision, PositionAction
                                
                                decision = AITradeDecision(
                                    action=PositionAction.CLOSE_NOW.value,
                                    confidence=100.0,
                                    reasoning="Manual close requested by user",
                                    urgency="HIGH"
                                )
                                
                                success = ai_manager.execute_decision(trade_id, decision)
                                if success:
                                    st.success(f"✅ Position {pos['pair']} closed successfully")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to close position")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    
                    with manual_col2:
                        if st.button(f"🔒 Move to Breakeven", key=f"be_{trade_id}", 
                                   disabled=pos['moved_to_breakeven']):
                            try:
                                from services.ai_crypto_position_manager import AITradeDecision, PositionAction
                                
                                # Calculate breakeven price
                                fee_buffer = pos['entry_price'] * 0.002
                                be_price = pos['entry_price'] + fee_buffer if pos['side'] == 'BUY' else pos['entry_price'] - fee_buffer
                                
                                decision = AITradeDecision(
                                    action=PositionAction.MOVE_TO_BREAKEVEN.value,
                                    confidence=100.0,
                                    reasoning="Manual breakeven move requested",
                                    urgency="MEDIUM",
                                    new_stop=be_price
                                )
                                
                                success = ai_manager.execute_decision(trade_id, decision)
                                if success:
                                    st.success(f"✅ Stop moved to breakeven")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to move stop")
                            except Exception as e:
                                st.error(f"Error: {e}")
        
        else:
            st.info("No active positions")
    
    else:
        st.info("📭 No active positions being monitored")
    
    st.markdown("---")
    
    # Configuration
    with st.expander("⚙️ AI Monitor Configuration"):
        config = status['config']
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("**AI Features:**")
            st.write(f"- AI Decisions: {'✅ Enabled' if config['ai_enabled'] else '❌ Disabled'}")
            st.write(f"- Trailing Stops: {'✅ Enabled' if config['trailing_stops'] else '❌ Disabled'}")
            st.write(f"- Breakeven Moves: {'✅ Enabled' if config['breakeven_moves'] else '❌ Disabled'}")
            st.write(f"- Partial Exits: {'✅ Enabled' if config['partial_exits'] else '❌ Disabled'}")
        
        with config_col2:
            st.markdown("**Settings:**")
            st.write(f"- Check Interval: {config['check_interval']}s")
            st.write(f"- Min AI Confidence: {config['min_ai_confidence']}%")
            st.write(f"- Max Positions: {ai_manager.max_positions}")
        
        st.markdown("**How AI Makes Decisions:**")
        st.markdown("""
        The AI analyzes multiple factors every check cycle:
        1. **Technical Indicators**: RSI, MACD, EMA crossovers, volume
        2. **Trend Strength**: Momentum building or weakening?
        3. **Support/Resistance**: Proximity to key levels
        4. **Risk/Reward**: Current R:R still favorable?
        5. **Time Factor**: Position duration and time-based momentum
        6. **Position Progress**: Near entry or near target?
        
        **AI Actions:**
        - **HOLD**: Trend intact, continue monitoring
        - **TIGHTEN_STOP**: Protect profits in uncertain conditions
        - **EXTEND_TARGET**: Strong trend continuation signals
        - **TAKE_PARTIAL**: Lock in profits at resistance (50% at +5%, 75% at +10%)
        - **CLOSE_NOW**: Clear reversal or target reached
        """)
    
    # Control buttons
    st.markdown("---")
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("🔄 Refresh Status", width='stretch'):
            st.rerun()
    
    with control_col2:
        if status['is_running']:
            if st.button("⏸️ Stop Monitoring", width='stretch'):
                ai_manager.stop()
                st.success("AI monitoring stopped")
                st.rerun()
        else:
            if st.button("▶️ Start Monitoring", width='stretch'):
                ai_manager.start_monitoring_loop()
                st.success("AI monitoring started")
                st.rerun()
    
    with control_col3:
        if st.button("🗑️ Clear All", width='stretch'):
            if st.session_state.get('confirm_clear', False):
                # Stop monitoring
                ai_manager.stop()
                # Clear session state
                del st.session_state.ai_position_manager
                st.success("All positions cleared")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("⚠️ Click again to confirm clearing all positions")


def display_ai_monitor_sidebar():
    """
    Display compact AI monitor in sidebar
    """
    if 'ai_position_manager' not in st.session_state:
        return
    
    ai_manager = st.session_state.ai_position_manager
    status = ai_manager.get_status()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Monitor")
    
    # Status indicator
    if status['is_running']:
        st.sidebar.success("🟢 Active")
    else:
        st.sidebar.error("🔴 Stopped")
    
    # Quick stats
    st.sidebar.metric("Positions", status['active_positions'])
    
    if status['active_positions'] > 0:
        # Show quick P&L summary
        total_pnl = 0
        active_count = 0
        for pos in status['positions'].values():
            if pos['status'] == 'ACTIVE':
                total_pnl += pos['pnl_pct']
                active_count += 1
        
        # Use actual active count to avoid division by zero
        avg_pnl = total_pnl / active_count if active_count > 0 else 0
        
        pnl_color = "normal" if avg_pnl >= 0 else "inverse"
        st.sidebar.metric("Avg P&L", f"{avg_pnl:+.2f}%", delta=None, delta_color=pnl_color)


def display_portfolio_analysis():
    """
    Display comprehensive portfolio analysis with AI recommendations
    """
    st.markdown("#### 📊 Portfolio Analysis & AI Recommendations")
    
    # Get Kraken client
    kraken_key = os.getenv('KRAKEN_API_KEY')
    kraken_secret = os.getenv('KRAKEN_API_SECRET')
    
    if not kraken_key or not kraken_secret:
        st.warning("⚠️ Kraken credentials not found. Set KRAKEN_API_KEY and KRAKEN_API_SECRET in .env")
        return
    
    # Get or create Kraken client
    if 'kraken_client' in st.session_state:
        kraken_client = st.session_state.kraken_client
    else:
        try:
            from clients.kraken_client import KrakenClient
            kraken_client = KrakenClient(api_key=kraken_key, api_secret=kraken_secret)
            success, message = kraken_client.validate_connection()
            if not success:
                st.error(f"❌ Failed to connect to Kraken: {message}")
                return
            st.session_state.kraken_client = kraken_client
        except Exception as e:
            st.error(f"❌ Error initializing Kraken client: {e}")
            return
    
    # Fetch portfolio analysis button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh Analysis", width='stretch'):
            # Clear cached data
            if 'portfolio_analysis' in st.session_state:
                del st.session_state.portfolio_analysis
    
    # Get portfolio analysis (cached in session state)
    if 'portfolio_analysis' not in st.session_state or st.button("📈 Analyze Portfolio", width='stretch', type="primary"):
        with st.spinner("Analyzing your Kraken portfolio..."):
            try:
                # 1. Fetch raw positions from client
                from dataclasses import asdict
                positions_raw = kraken_client.get_open_positions(calculate_real_cost=True)
                positions_dict = [asdict(p) for p in positions_raw]

                # 2. Get AI manager instance to access analysis method
                from services.ai_crypto_position_manager import get_ai_position_manager
                from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                
                llm_analyzer = LLMStrategyAnalyzer() # This can be a dummy for analysis
                ai_manager = get_ai_position_manager(kraken_client=kraken_client, llm_analyzer=llm_analyzer)

                # 3. Run new, corrected analysis function
                analysis = ai_manager.analyze_portfolio(positions_dict)
                st.session_state.portfolio_analysis = analysis

            except Exception as e:
                st.error(f"❌ Error analyzing portfolio: {e}")
                logger.error("Portfolio analysis error: {}", str(e), exc_info=True)
                return
    
    if 'portfolio_analysis' not in st.session_state:
        st.info("👆 Click 'Analyze Portfolio' to view your portfolio metrics and AI recommendations")
        return
    
    analysis = st.session_state.portfolio_analysis
    
    if 'error' in analysis:
        st.error(f"❌ Analysis error: {analysis['error']}")
        return
    
    if not analysis.get('positions') and not analysis.get('staked_assets'):
        st.info("📭 No positions found in your Kraken account")
        return

    summary = analysis.get('summary', {})
    positions = analysis.get('positions', [])
    num_positions = len(positions)

    # Portfolio Summary Metrics
    st.markdown("##### 💰 Portfolio Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Tradeable Value",
            f"${summary.get('total_value', 0):,.2f}",
            delta=None
        )

    with col2:
        pnl_color = "normal" if summary.get('total_pnl', 0) >= 0 else "inverse"
        st.metric(
            "P&L",
            f"${summary.get('total_pnl', 0):,.2f}",
            delta=f"{summary.get('total_pnl_pct', 0):+.2f}%",
            delta_color=pnl_color
        )

    with col3:
        st.metric(
            "Staked Value",
            f"${analysis.get('total_staked_value', 0):,.2f}",
            delta=f"{len(analysis.get('staked_assets', []))} asset(s)"
        )

    with col4:
        st.metric(
            "Combined Total",
            f"${analysis.get('combined_value', summary.get('total_value', 0)):,.2f}",
            delta="Tradeable + Staked"
        )

    with col5:
        num_winners = len(summary.get('top_gainers', []))
        num_losers = len(summary.get('top_losers', []))
        win_rate = (num_winners / num_positions * 100) if num_positions > 0 else 0
        st.metric(
            "Win Rate",
            f"{win_rate:.0f}%",
            delta=f"{num_winners}W / {num_losers}L" if num_positions > 0 else "No positions"
        )
    
    # AI Recommendations
    if analysis['recommendations']:
        st.markdown("---")
        st.markdown("##### 🤖 AI Recommendations")
        
        for rec in analysis['recommendations']:
            priority_emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
            action_emoji = {
                'TAKE_PROFIT': '💰',
                'CUT_LOSS': '✂️',
                'REBALANCE': '⚖️'
            }.get(rec['action'], '💡')
            
            st.info(f"{priority_emoji} {action_emoji} **{rec['pair']}**: {rec['reason']}")
    
    # Detailed Position Breakdown
    if analysis['positions']:
        st.markdown("---")
        st.markdown("##### 📈 Position Details")
        
        # Create DataFrame for positions
        positions_df = pd.DataFrame(analysis['positions'])
        positions_df = positions_df.sort_values('pnl_pct', ascending=False)
        
        # Format columns
        display_df = pd.DataFrame({
            'Pair': positions_df['pair'],
            'Volume': positions_df['volume'].apply(lambda x: f"{x:,.6f}"),
            'Entry': positions_df['entry_price'].apply(lambda x: f"${x:,.6f}"),
            'Current': positions_df['current_price'].apply(lambda x: f"${x:,.6f}"),
            'P&L %': positions_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%"),
            'P&L $': positions_df['pnl'].apply(lambda x: f"${x:+,.2f}"),
            'Value': positions_df['current_value'].apply(lambda x: f"${x:,.2f}"),
            'Allocation': positions_df['allocation_pct'].apply(lambda x: f"{x:.1f}%")
        })
        
        st.dataframe(display_df, width='stretch', hide_index=True)
        
        # Top Winners and Losers
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis['winners']:
                st.markdown("###### 🏆 Top Winners")
                for i, winner in enumerate(analysis['winners'][:3], 1):
                    st.success(f"{i}. **{winner['pair']}**: +{winner['pnl_pct']:.2f}% (${winner['pnl']:+,.2f})")
        
        with col2:
            if analysis['losers']:
                st.markdown("###### 📉 Top Losers")
                for i, loser in enumerate(analysis['losers'][:3], 1):
                    st.error(f"{i}. **{loser['pair']}**: {loser['pnl_pct']:.2f}% (${loser['pnl']:,.2f})")
    
    # Staked Assets Section
    if analysis.get('staked_assets'):
        st.markdown("---")
        st.markdown("##### 🔒 Staked Holdings (Read-Only)")
        st.info("💡 **Note:** Staked assets cannot be traded directly. Unstake them on Kraken.com to trade.")
        
        # Create DataFrame for staked assets
        staked_df = pd.DataFrame(analysis['staked_assets'])
        
        if not staked_df.empty:
            # Format for display
            display_staked = pd.DataFrame({
                'Asset': staked_df['currency'],
                'Amount': staked_df['balance'].apply(lambda x: f"{x:,.6f}"),
                'Type': staked_df['type'],
                'Price': staked_df['current_price'].apply(lambda x: f"${x:,.6f}" if x > 0 else "N/A"),
                'Value': staked_df['value_usd'].apply(lambda x: f"${x:,.2f}" if x > 0 else "N/A")
            })
            
            st.dataframe(display_staked, width='stretch', hide_index=True)
            
            # Show instructions for unstaking
            with st.expander("ℹ️ How to unstake your crypto"):
                st.markdown("""
                **To trade your staked crypto:**
                
                1. **Log into Kraken.com**
                2. **Navigate to:** Earn → Staking
                3. **Find your asset** (e.g., ADA)
                4. **Click "Unstake"**
                5. **Wait for unstaking period** (usually instant or a few days)
                6. **Return here** - asset will show in tradeable positions
                
                **Staking Types:**
                - **Flex Staking (.F)**: Instant unstaking, lower APY
                - **Locked Staking (.S)**: Fixed period, higher APY
                - **Parachain (.P)**: DOT/KSM parachains, specific lock periods
                
                **Note:** You'll stop earning staking rewards once unstaked.
                """)

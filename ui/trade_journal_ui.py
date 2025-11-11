"""
Trade Journal UI
Unified journal viewer for stocks and crypto with AI decision tracking
"""

import streamlit as st
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional


def display_trade_journal():
    """
    Display unified trade journal with statistics and analysis
    """
    st.markdown("### 📓 Trade Journal - Learn from Every Trade")
    st.markdown("Automatic tracking of all trades across stocks and crypto with AI decision analysis")
    
    try:
        from services.unified_trade_journal import get_unified_journal, TradeType
        journal = get_unified_journal()
    except Exception as e:
        st.error(f"Error loading journal: {e}")
        logger.error(f"Journal loading error: {e}", exc_info=True)
        return
    
    # Filters
    st.markdown("#### 🔍 Filters")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        trade_type_filter = st.selectbox(
            "Trade Type",
            ["All", "Crypto", "Stocks", "Options"],
            key="journal_trade_type"
        )
        trade_type_map = {
            "All": None,
            "Crypto": TradeType.CRYPTO.value,
            "Stocks": TradeType.STOCK.value,
            "Options": TradeType.OPTION.value
        }
        trade_type = trade_type_map[trade_type_filter]
    
    with filter_col2:
        status_filter = st.selectbox(
            "Status",
            ["All", "Open", "Closed"],
            key="journal_status"
        )
        status = None if status_filter == "All" else status_filter.upper()
    
    with filter_col3:
        date_range = st.selectbox(
            "Date Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All Time"],
            key="journal_date_range"
        )
        
        if date_range == "Last 7 days":
            start_date = datetime.now() - timedelta(days=7)
        elif date_range == "Last 30 days":
            start_date = datetime.now() - timedelta(days=30)
        elif date_range == "Last 90 days":
            start_date = datetime.now() - timedelta(days=90)
        else:
            start_date = None
    
    with filter_col4:
        ai_only = st.checkbox("AI-Managed Only", value=False, key="journal_ai_only")
    
    st.markdown("---")
    
    # Get statistics
    stats = journal.get_statistics(
        start_date=start_date,
        trade_type=trade_type
    )
    
    # Display statistics
    st.markdown("#### 📊 Performance Statistics")
    
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    
    with stats_col1:
        st.metric("Total Trades", stats.total_trades)
        st.metric("Open Trades", stats.open_trades)
    
    with stats_col2:
        win_rate_color = "normal" if stats.win_rate >= 0.5 else "inverse"
        st.metric("Win Rate", f"{stats.win_rate*100:.1f}%", delta=None, delta_color=win_rate_color)
        st.metric("Wins/Losses", f"{stats.winning_trades}W / {stats.losing_trades}L")
    
    with stats_col3:
        pnl_color = "normal" if stats.total_pnl >= 0 else "inverse"
        st.metric("Total P&L", f"${stats.total_pnl:,.2f}", delta=None, delta_color=pnl_color)
        st.metric("Expectancy", f"${stats.expectancy:.2f}")
    
    with stats_col4:
        st.metric("Avg Win", f"${stats.avg_win:.2f}")
        st.metric("Avg Loss", f"${stats.avg_loss:.2f}")
    
    with stats_col5:
        st.metric("Profit Factor", f"{stats.profit_factor:.2f}x")
        st.metric("Avg R-Multiple", f"{stats.avg_r_multiple:.2f}R")
    
    # AI Performance
    if stats.ai_managed_trades > 0:
        st.markdown("#### 🤖 AI Performance")
        ai_col1, ai_col2, ai_col3, ai_col4, ai_col5 = st.columns(5)
        
        with ai_col1:
            st.metric("AI Trades", stats.ai_managed_trades)
        
        with ai_col2:
            ai_wr_color = "normal" if stats.ai_win_rate >= 0.5 else "inverse"
            st.metric("AI Win Rate", f"{stats.ai_win_rate*100:.1f}%", delta=None, delta_color=ai_wr_color)
        
        with ai_col3:
            st.metric("Avg Adjustments", f"{stats.ai_avg_adjustments:.1f}")
        
        with ai_col4:
            st.metric("Breakeven Moves", stats.breakeven_moves)
        
        with ai_col5:
            st.metric("Partial Exits", stats.partial_exits)
    
    # Hold Time Analysis
    st.markdown("#### ⏱️ Hold Time Analysis")
    time_col1, time_col2, time_col3 = st.columns(3)
    
    with time_col1:
        avg_hold = stats.avg_hold_time_minutes
        if avg_hold < 60:
            st.metric("Avg Hold Time", f"{avg_hold:.0f}m")
        else:
            st.metric("Avg Hold Time", f"{avg_hold/60:.1f}h")
    
    with time_col2:
        avg_win_hold = stats.avg_win_hold_time
        if avg_win_hold < 60:
            st.metric("Avg Win Hold", f"{avg_win_hold:.0f}m")
        else:
            st.metric("Avg Win Hold", f"{avg_win_hold/60:.1f}h")
    
    with time_col3:
        avg_loss_hold = stats.avg_loss_hold_time
        if avg_loss_hold < 60:
            st.metric("Avg Loss Hold", f"{avg_loss_hold:.0f}m")
        else:
            st.metric("Avg Loss Hold", f"{avg_loss_hold/60:.1f}h")
    
    # Strategy Performance
    if stats.strategy_stats:
        st.markdown("#### 📈 Strategy Performance")
        
        strategy_data = []
        for strategy, strat_stats in stats.strategy_stats.items():
            strategy_data.append({
                'Strategy': strategy,
                'Trades': strat_stats['trades'],
                'Wins': strat_stats['wins'],
                'Win Rate': f"{strat_stats['win_rate']*100:.1f}%",
                'Total P&L': f"${strat_stats['total_pnl']:,.2f}",
                'Avg P&L': f"${strat_stats['avg_pnl']:,.2f}"
            })
        
        if strategy_data:
            df_strategy = pd.DataFrame(strategy_data)
            st.dataframe(df_strategy, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Get trades
    trades = journal.get_trades(
        trade_type=trade_type,
        status=status,
        start_date=start_date,
        ai_managed_only=ai_only,
        limit=100
    )
    
    if not trades:
        st.info("📭 No trades found matching filters")
        return
    
    st.markdown(f"#### 📋 Trade Log ({len(trades)} trades)")
    
    # Convert to DataFrame for display
    trade_records = []
    for trade in trades:
        # Calculate hold time string
        if trade.hold_time_minutes:
            if trade.hold_time_minutes < 60:
                hold_time_str = f"{trade.hold_time_minutes}m"
            else:
                hold_time_str = f"{trade.hold_time_minutes/60:.1f}h"
        else:
            # For open trades, calculate current hold time
            if trade.status == "OPEN":
                hold_minutes = (datetime.now() - trade.entry_time).total_seconds() / 60
                hold_time_str = f"{hold_minutes:.0f}m" if hold_minutes < 60 else f"{hold_minutes/60:.1f}h"
            else:
                hold_time_str = "N/A"
        
        # Status indicator
        if trade.status == "CLOSED":
            if trade.pnl_pct and trade.pnl_pct > 0:
                status_icon = "✅"
            elif trade.pnl_pct and trade.pnl_pct < 0:
                status_icon = "❌"
            else:
                status_icon = "⚪"
        else:
            status_icon = "🔵"
        
        trade_records.append({
            'Status': status_icon,
            'Type': trade.trade_type,
            'Symbol': trade.symbol,
            'Side': trade.side,
            'Entry': f"${trade.entry_price:.4f}",
            'Exit': f"${trade.exit_price:.4f}" if trade.exit_price else "-",
            'P&L': f"${trade.realized_pnl:+,.2f}" if trade.realized_pnl else "-",
            'P&L %': f"{trade.pnl_pct:+.2f}%" if trade.pnl_pct else "-",
            'R': f"{trade.r_multiple:.2f}R" if trade.r_multiple else "-",
            'Hold': hold_time_str,
            'Strategy': trade.strategy,
            'AI': "🤖" if trade.ai_managed else "",
            'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M')
        })
    
    df_trades = pd.DataFrame(trade_records)
    st.dataframe(df_trades, use_container_width=True, hide_index=True)
    
    # Trade details
    st.markdown("#### 🔍 Trade Details")
    
    selected_trade_idx = st.selectbox(
        "Select trade to view details",
        range(len(trades)),
        format_func=lambda i: f"{trades[i].symbol} - {trades[i].side} @ ${trades[i].entry_price:.4f} ({trades[i].entry_time.strftime('%Y-%m-%d %H:%M')})",
        key="journal_selected_trade"
    )
    
    if selected_trade_idx is not None:
        trade = trades[selected_trade_idx]
        
        with st.expander(f"📊 {trade.symbol} - {trade.side} Trade Details", expanded=True):
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            
            with detail_col1:
                st.markdown("**Entry Info:**")
                st.write(f"- Time: {trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"- Price: ${trade.entry_price:.4f}")
                st.write(f"- Quantity: {trade.quantity:.6f}")
                st.write(f"- Position Size: ${trade.position_size_usd:.2f}")
                st.write(f"- Strategy: {trade.strategy}")
                st.write(f"- Broker: {trade.broker or 'N/A'}")
            
            with detail_col2:
                st.markdown("**Risk Management:**")
                st.write(f"- Stop Loss: ${trade.stop_loss:.4f}")
                st.write(f"- Take Profit: ${trade.take_profit:.4f}")
                st.write(f"- Risk: {trade.risk_pct:.2f}%")
                st.write(f"- Reward: {trade.reward_pct:.2f}%")
                st.write(f"- R:R Ratio: {trade.risk_reward_ratio:.2f}:1")
            
            with detail_col3:
                if trade.status == "CLOSED":
                    st.markdown("**Exit Info:**")
                    st.write(f"- Time: {trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else 'N/A'}")
                    st.write(f"- Price: ${trade.exit_price:.4f}" if trade.exit_price else "- Price: N/A")
                    st.write(f"- Reason: {trade.exit_reason or 'N/A'}")
                    st.write(f"- P&L: ${trade.realized_pnl:+,.2f}" if trade.realized_pnl else "- P&L: N/A")
                    st.write(f"- P&L %: {trade.pnl_pct:+.2f}%" if trade.pnl_pct else "- P&L %: N/A")
                    st.write(f"- R-Multiple: {trade.r_multiple:.2f}R" if trade.r_multiple else "- R-Multiple: N/A")
                else:
                    st.markdown("**Current Status:**")
                    st.write(f"- Status: {trade.status}")
                    st.write(f"- Position: OPEN")
            
            # AI Management
            if trade.ai_managed:
                st.markdown("**🤖 AI Management:**")
                ai_info_col1, ai_info_col2 = st.columns(2)
                
                with ai_info_col1:
                    st.write(f"- AI Adjustments: {trade.ai_adjustments_count}")
                    st.write(f"- Breakeven Move: {'✅ Yes' if trade.moved_to_breakeven else '❌ No'}")
                    st.write(f"- Trailing Stop: {'✅ Yes' if trade.trailing_stop_activated else '❌ No'}")
                
                with ai_info_col2:
                    st.write(f"- Partial Exit: {'✅ Yes' if trade.partial_exit_taken else '❌ No'}")
                    st.write(f"- Max Favorable: {trade.max_favorable_pct:+.2f}%")
                    st.write(f"- Max Adverse: {trade.max_adverse_pct:+.2f}%")
                
                # AI Decisions
                if trade.ai_adjustments_count > 0:
                    st.markdown("**🤖 AI Decision History:**")
                    
                    ai_decisions = journal.get_ai_decisions(trade.trade_id)
                    
                    if ai_decisions:
                        decision_data = []
                        for decision in ai_decisions:
                            decision_data.append({
                                'Time': decision.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'Action': decision.action,
                                'Confidence': f"{decision.confidence:.0f}%",
                                'Reasoning': decision.reasoning,
                                'Tech': f"{decision.technical_score:.0f}",
                                'Trend': f"{decision.trend_score:.0f}",
                                'Risk': f"{decision.risk_score:.0f}"
                            })
                        
                        df_decisions = pd.DataFrame(decision_data)
                        st.dataframe(df_decisions, use_container_width=True, hide_index=True)
                    else:
                        st.info("No AI decisions logged")
            
            # Market Conditions
            if trade.rsi_entry or trade.macd_entry:
                st.markdown("**📊 Market Conditions:**")
                
                market_col1, market_col2 = st.columns(2)
                
                with market_col1:
                    st.write("**Entry:**")
                    if trade.rsi_entry:
                        st.write(f"- RSI: {trade.rsi_entry:.2f}")
                    if trade.macd_entry:
                        st.write(f"- MACD: {trade.macd_entry:.4f}")
                    if trade.volume_change_entry:
                        st.write(f"- Volume Change: {trade.volume_change_entry:+.1f}%")
                    if trade.trend_entry:
                        st.write(f"- Trend: {trade.trend_entry}")
                
                with market_col2:
                    if trade.status == "CLOSED":
                        st.write("**Exit:**")
                        if trade.rsi_exit:
                            st.write(f"- RSI: {trade.rsi_exit:.2f}")
                        if trade.macd_exit:
                            st.write(f"- MACD: {trade.macd_exit:.4f}")
                        if trade.volume_change_exit:
                            st.write(f"- Volume Change: {trade.volume_change_exit:+.1f}%")
                        if trade.trend_exit:
                            st.write(f"- Trend: {trade.trend_exit}")
            
            # Notes
            if trade.notes:
                st.markdown("**📝 Notes:**")
                st.write(trade.notes)
            
            # Tags
            if trade.tags:
                st.markdown("**🏷️ Tags:**")
                st.write(", ".join(trade.tags))
    
    st.markdown("---")
    
    # Export options
    st.markdown("#### 📥 Export Options")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("Export to CSV", use_container_width=True):
            try:
                output_path = f"data/exports/trade_journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                journal.export_to_csv(output_path, start_date=start_date)
                st.success(f"✅ Exported to {output_path}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with export_col2:
        if st.button("View Analysis Tips", use_container_width=True):
            st.info("""
            **📚 Journal Analysis Tips:**
            
            1. **Look for Patterns:**
               - Which strategies have the best win rate?
               - Do you hold winners/losers too long?
               - What market conditions favor your setups?
            
            2. **AI Performance:**
               - Are AI-managed trades performing better?
               - When does AI add the most value?
               - Review AI reasoning for losing trades
            
            3. **Time Analysis:**
               - What time of day are you most profitable?
               - Do longer hold times improve results?
            
            4. **Risk Management:**
               - Are you maintaining proper R:R ratios?
               - Is your expectancy positive?
               - What's your profit factor trend?
            
            5. **Continuous Improvement:**
               - Document what worked and what didn't
               - Adjust strategies based on data
               - Set specific goals for next period
            """)


def display_journal_sidebar():
    """
    Display compact journal stats in sidebar
    """
    try:
        from services.unified_trade_journal import get_unified_journal
        journal = get_unified_journal()
        
        stats = journal.get_statistics(
            start_date=datetime.now() - timedelta(days=30)
        )
        
        if stats.total_trades > 0:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📓 Last 30 Days")
            st.sidebar.metric("Trades", stats.total_trades)
            st.sidebar.metric("Win Rate", f"{stats.win_rate*100:.1f}%")
            st.sidebar.metric("P&L", f"${stats.total_pnl:,.2f}")
    except Exception as e:
        logger.debug(f"Could not display journal sidebar: {e}")


"""
Discord Alerts Tab for Streamlit UI
Displays Discord trading alerts and allows filtering, analysis, and export
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from discord_alert_listener import create_discord_manager, DiscordAlertManager
from discord_config import DiscordConfig
import json

logger = logging.getLogger(__name__)


def init_discord_session_state():
    """Initialize Discord-related session state"""
    if 'discord_manager' not in st.session_state:
        st.session_state.discord_manager = None
    if 'discord_bot_running' not in st.session_state:
        st.session_state.discord_bot_running = False
    if 'discord_config' not in st.session_state:
        st.session_state.discord_config = DiscordConfig()


def render_discord_config_section():
    """Render Discord configuration section"""
    st.header("⚙️ Discord Bot Configuration")
    
    config = st.session_state.discord_config
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Bot token input
        bot_token = st.text_input(
            "Discord Bot Token",
            value=config.bot_token or "",
            type="password",
            help="Get this from Discord Developer Portal"
        )
        
        if bot_token:
            config.bot_token = bot_token
    
    with col2:
        st.metric(
            "Bot Status",
            "🟢 Running" if st.session_state.discord_bot_running else "🔴 Stopped"
        )
    
    # Channel management
    st.subheader("📢 Monitored Channels")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        channel_id = st.text_input(
            "Channel ID",
            placeholder="1427896857274617939",
            help="Right-click channel > Copy Channel ID (Developer Mode required)"
        )
    
    with col2:
        is_premium = st.checkbox("Premium Channel", value=False)
    
    if st.button("➕ Add Channel"):
        if channel_id:
            config.add_channel(channel_id, is_premium=is_premium)
            config.save_config()
            st.success(f"Added channel: {channel_id}")
            st.rerun()
    
    # Display current channels
    if config.channels:
        channels_df = pd.DataFrame([
            {
                'Channel ID': ch.channel_id,
                'Name': ch.channel_name,
                'Premium': '🔒' if ch.is_premium else '📢',
                'Enabled': ch.enabled
            }
            for ch in config.channels.values()
        ])
        
        st.dataframe(channels_df, use_container_width=True)
        
        # Remove channel
        channel_to_remove = st.selectbox(
            "Remove Channel",
            options=[''] + list(config.channels.keys()),
            format_func=lambda x: f"{x} - {config.channels[x].channel_name}" if x else "Select..."
        )
        
        if st.button("🗑️ Remove Channel") and channel_to_remove:
            config.remove_channel(channel_to_remove)
            config.save_config()
            st.success(f"Removed channel: {channel_to_remove}")
            st.rerun()
    
    # Bot controls
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Bot", disabled=st.session_state.discord_bot_running):
            if config.is_configured():
                try:
                    manager = create_discord_manager()
                    if manager:
                        manager.start()
                        st.session_state.discord_manager = manager
                        st.session_state.discord_bot_running = True
                        st.success("Discord bot started!")
                        st.rerun()
                    else:
                        st.error("Failed to create Discord manager")
                except Exception as e:
                    st.error(f"Error starting bot: {e}")
            else:
                st.error("Please configure bot token and channels first")
    
    with col2:
        if st.button("⏹️ Stop Bot", disabled=not st.session_state.discord_bot_running):
            if st.session_state.discord_manager:
                st.session_state.discord_manager.stop()
                st.session_state.discord_manager = None
                st.session_state.discord_bot_running = False
                st.success("Discord bot stopped")
                st.rerun()
    
    with col3:
        if st.button("💾 Save Config"):
            config.save_config()
            st.success("Configuration saved!")


def render_alerts_dashboard(discord_mgr: DiscordAlertManager):
    """Render alerts dashboard"""
    st.header("📊 Discord Trading Alerts")
    
    # Get alerts
    alerts = discord_mgr.get_alerts(limit=100)
    
    if not alerts:
        st.info("No alerts received yet. Waiting for messages in monitored channels...")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(alerts)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alerts))
    
    with col2:
        unique_symbols = df['symbol'].nunique()
        st.metric("Unique Symbols", unique_symbols)
    
    with col3:
        entry_count = len(df[df['alert_type'] == 'ENTRY'])
        st.metric("Entry Signals", entry_count)
    
    with col4:
        runner_count = len(df[df['alert_type'] == 'RUNNER'])
        st.metric("Runner Alerts", runner_count)
    
    st.divider()
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol_filter = st.multiselect(
            "Symbol",
            options=['All'] + sorted(df['symbol'].unique().tolist()),
            default=['All']
        )
    
    with col2:
        type_filter = st.multiselect(
            "Alert Type",
            options=['All'] + sorted(df['alert_type'].unique().tolist()),
            default=['All']
        )
    
    with col3:
        channel_filter = st.multiselect(
            "Channel",
            options=['All', 'Premium Only', 'Free Only'],
            default=['All']
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if 'All' not in symbol_filter:
        filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]
    
    if 'All' not in type_filter:
        filtered_df = filtered_df[filtered_df['alert_type'].isin(type_filter)]
    
    if 'Premium Only' in channel_filter:
        filtered_df = filtered_df[filtered_df['premium_channel'] == True]
    elif 'Free Only' in channel_filter:
        filtered_df = filtered_df[filtered_df['premium_channel'] == False]
    
    st.subheader(f"📋 Alerts ({len(filtered_df)} results)")
    
    # Display alerts
    for idx, alert in filtered_df.iterrows():
        with st.expander(
            f"{'🔒' if alert['premium_channel'] else '📢'} "
            f"{alert['symbol']} - {alert['alert_type']} "
            f"{'@ $' + str(alert['price']) if alert['price'] else ''}"
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Symbol:** {alert['symbol']}")
                st.markdown(f"**Type:** {alert['alert_type']}")
                st.markdown(f"**Channel:** {alert['channel_name']}")
                st.markdown(f"**Author:** {alert['author']}")
                st.markdown(f"**Time:** {alert['timestamp']}")
                
                if alert['reasoning']:
                    st.markdown(f"**Message:** {alert['reasoning']}")
            
            with col2:
                if alert['price']:
                    st.metric("Price", f"${alert['price']:.2f}")
                if alert['target']:
                    st.metric("Target", f"${alert['target']:.2f}")
                if alert['stop_loss']:
                    st.metric("Stop Loss", f"${alert['stop_loss']:.2f}")
                if alert['confidence']:
                    st.metric("Confidence", alert['confidence'])
    
    # Export section
    st.divider()
    st.subheader("💾 Export Alerts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                f"discord_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("📋 Export to JSON"):
            json_str = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                "⬇️ Download JSON",
                json_str,
                f"discord_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )
    
    with col3:
        if st.button("🗑️ Clear All Alerts"):
            if st.session_state.discord_manager:
                st.session_state.discord_manager.bot.alerts.clear()
                st.success("Alerts cleared!")
                st.rerun()


def render_symbol_analysis(discord_mgr: DiscordAlertManager):
    """Render symbol-specific analysis"""
    st.header("🎯 Symbol Analysis")
    
    # Get alerts
    alerts = discord_mgr.get_alerts(limit=100)
    
    if not alerts:
        st.info("No alerts to analyze yet.")
        return
    
    df = pd.DataFrame(alerts)
    symbols = sorted(df['symbol'].unique().tolist())
    
    # Symbol selector
    selected_symbol = st.selectbox("Select Symbol", symbols)
    
    if selected_symbol:
        symbol_alerts = discord_mgr.get_symbol_alerts(selected_symbol, limit=50)
        
        if symbol_alerts:
            symbol_df = pd.DataFrame(symbol_alerts)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Alerts", len(symbol_alerts))
            
            with col2:
                entry_count = len(symbol_df[symbol_df['alert_type'] == 'ENTRY'])
                st.metric("Entries", entry_count)
            
            with col3:
                exit_count = len(symbol_df[symbol_df['alert_type'] == 'EXIT'])
                st.metric("Exits", exit_count)
            
            with col4:
                runner_count = len(symbol_df[symbol_df['alert_type'] == 'RUNNER'])
                st.metric("Runners", runner_count)
            
            # Alert timeline
            st.subheader("📈 Alert Timeline")
            
            # Sort by timestamp
            symbol_df['timestamp_dt'] = pd.to_datetime(symbol_df['timestamp'])
            symbol_df = symbol_df.sort_values('timestamp_dt')
            
            # Display timeline
            for idx, alert in symbol_df.iterrows():
                alert_icon = {
                    'ENTRY': '🟢',
                    'EXIT': '🔴',
                    'RUNNER': '🚀',
                    'ALERT': '🔔',
                    'STOP': '⛔'
                }.get(alert['alert_type'], '📌')
                
                price_str = f"@ ${alert['price']:.2f}" if alert['price'] else ""
                target_str = f"→ ${alert['target']:.2f}" if alert['target'] else ""
                
                st.markdown(
                    f"{alert_icon} **{alert['alert_type']}** {price_str} {target_str} "
                    f"*({alert['timestamp_dt'].strftime('%H:%M:%S')})*"
                )
            
            # Price levels
            st.subheader("💰 Price Levels")
            
            prices = symbol_df[symbol_df['price'].notna()]['price'].tolist()
            targets = symbol_df[symbol_df['target'].notna()]['target'].tolist()
            stops = symbol_df[symbol_df['stop_loss'].notna()]['stop_loss'].tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prices:
                    st.metric("Avg Entry", f"${sum(prices)/len(prices):.2f}")
                    st.metric("Last Entry", f"${prices[-1]:.2f}")
            
            with col2:
                if targets:
                    st.metric("Avg Target", f"${sum(targets)/len(targets):.2f}")
                    st.metric("Last Target", f"${targets[-1]:.2f}")
            
            with col3:
                if stops:
                    st.metric("Avg Stop", f"${sum(stops)/len(stops):.2f}")
                    st.metric("Last Stop", f"${stops[-1]:.2f}")
            
            # Sentiment
            st.subheader("📊 Sentiment")
            
            sentiment = "BULLISH" if entry_count > exit_count else "BEARISH" if exit_count > entry_count else "NEUTRAL"
            sentiment_color = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
            
            st.markdown(f"### {sentiment_color} {sentiment}")
            st.progress((entry_count / (entry_count + exit_count + 1)))


def render_discord_tab():
    """Main Discord tab renderer"""
    
    # Initialize session state
    init_discord_session_state()
    
    st.title("💬 Discord Trading Alerts")
    
    st.markdown("""
    Monitor Discord channels for trading alerts from professional traders.
    Configure your bot, view alerts, and integrate with AI analysis.
    """)
    
    # Check if bot is configured
    if not st.session_state.discord_config.is_configured():
        st.warning("⚠️ Discord bot not configured. Please configure below to get started.")
        with st.expander("📖 Setup Guide", expanded=True):
            st.markdown("""
            ### Quick Setup:
            1. Create a Discord bot at [Discord Developer Portal](https://discord.com/developers/applications)
            2. Enable **MESSAGE CONTENT INTENT** in Bot settings
            3. Copy your bot token
            4. Add bot to your Discord server
            5. Get channel IDs (right-click channel > Copy Channel ID)
            6. Enter bot token and channel IDs below
            
            See **DISCORD_INTEGRATION_GUIDE.md** for detailed instructions.
            """)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Configuration",
        "📊 Alerts Dashboard",
        "🎯 Symbol Analysis",
        "📖 Help"
    ])
    
    with tab1:
        render_discord_config_section()
    
    with tab2:
        if st.session_state.discord_manager and st.session_state.discord_bot_running:
            render_alerts_dashboard(st.session_state.discord_manager)
        else:
            st.info("Start the Discord bot in the Configuration tab to view alerts.")
    
    with tab3:
        if st.session_state.discord_manager and st.session_state.discord_bot_running:
            render_symbol_analysis(st.session_state.discord_manager)
        else:
            st.info("Start the Discord bot to analyze symbol-specific alerts.")
    
    with tab4:
        st.markdown("""
        ## Discord Integration Help
        
        ### What is this?
        Monitor Discord trading alert channels and integrate professional trader signals into your AI analysis.
        
        ### Features
        - ✅ Monitor multiple Discord channels (free and premium)
        - ✅ Automatic parsing of tickers, prices, targets, stop losses
        - ✅ Alert filtering and search
        - ✅ Export to CSV/JSON
        - ✅ Integration with AI trading signals
        
        ### Alert Types
        - **ENTRY** - Buy/Long signals
        - **EXIT** - Sell/Close signals  
        - **RUNNER** - Momentum/breakout alerts
        - **ALERT** - General watch alerts
        - **STOP** - Stop loss hit notifications
        
        ### How AI Integration Works
        When you run AI analysis on a symbol, Discord alerts are automatically included:
        - Recent alerts for that symbol are analyzed
        - AI considers Discord sentiment (bullish/bearish/neutral)
        - Entry/exit signals from Discord are factored into confidence scores
        - Professional trader insights complement technical/news analysis
        
        ### Best Practices
        1. ✅ Start with free channels to test
        2. ✅ Don't blindly follow alerts - use as one data point
        3. ✅ Combine with your technical and fundamental analysis
        4. ✅ Export alerts regularly for your records
        5. ✅ Monitor bot logs for any issues
        
        ### Troubleshooting
        - **Bot won't connect:** Check token and permissions
        - **No alerts showing:** Verify channel IDs and bot has access
        - **Parsing issues:** Review alert message format
        
        For detailed setup instructions, see **DISCORD_INTEGRATION_GUIDE.md**
        """)


if __name__ == "__main__":
    render_discord_tab()

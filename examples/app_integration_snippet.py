"""
Snippet for integrating Position Monitoring into app.py

Copy the relevant sections below into your main Streamlit app.
"""

# ============================================================================
# 1. IMPORTS (Add to top of app.py)
# ============================================================================

from services.position_monitor import PositionMonitor, get_position_monitor
from services.ticker_manager import TickerManager

# ============================================================================
# 2. SESSION STATE INITIALIZATION (Add to setup section)
# ============================================================================

# Initialize ticker manager (for My Tickers)
if 'ticker_manager' not in st.session_state:
    st.session_state.ticker_manager = TickerManager()

# Initialize position monitor (after Tradier client is set up)
if 'position_monitor' not in st.session_state:
    if 'tradier_client' in st.session_state and st.session_state.tradier_client:
        st.session_state.position_monitor = get_position_monitor(
            alert_system=get_alert_system(),
            tradier_client=st.session_state.tradier_client
        )

# ============================================================================
# 3. POSITION MONITORING TAB (Add new tab to your app)
# ============================================================================

def render_position_monitoring_tab():
    """Add this as a new tab in your Streamlit app"""
    
    st.title("📊 Active Position Monitoring")
    
    if 'tradier_client' not in st.session_state or not st.session_state.tradier_client:
        st.warning("⚠️ Tradier client not connected. Go to Tradier Account tab to connect.")
        return
    
    if 'position_monitor' not in st.session_state:
        st.session_state.position_monitor = get_position_monitor(
            alert_system=get_alert_system(),
            tradier_client=st.session_state.tradier_client
        )
    
    position_monitor = st.session_state.position_monitor
    
    # Configuration
    with st.expander("⚙️ Alert Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pnl_threshold = st.number_input(
                "P&L Alert Threshold (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=1.0,
                help="Send alert every X% P&L change"
            )
            position_monitor.pnl_alert_threshold = pnl_threshold
        
        with col2:
            loss_threshold = st.number_input(
                "Significant Loss Threshold (%)",
                min_value=-50.0,
                max_value=-5.0,
                value=-10.0,
                step=1.0,
                help="Alert when position loses more than this %"
            )
            position_monitor.significant_loss_threshold = loss_threshold
        
        with col3:
            gain_threshold = st.number_input(
                "Significant Gain Threshold (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=1.0,
                help="Alert when position gains more than this %"
            )
            position_monitor.significant_gain_threshold = gain_threshold
    
    st.divider()
    
    # Update positions
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh (every 30 seconds)", value=False)
    
    with col2:
        if st.button("🔄 Update Positions Now", type="primary", use_container_width=True):
            with st.spinner("Fetching positions..."):
                success, alerts = position_monitor.update_positions()
                if success:
                    st.success(f"✓ Updated! {len(alerts)} new alerts")
                else:
                    st.error("Failed to update positions")
    
    with col3:
        if st.button("📋 View Alert History", use_container_width=True):
            st.session_state.show_alert_history = True
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    st.divider()
    
    # Display position summary
    summary = position_monitor.get_position_summary()
    
    st.subheader("📈 Portfolio Summary")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Positions", summary['total_positions'])
    
    with metric_col2:
        pnl_color = "normal" if summary['total_pnl'] >= 0 else "inverse"
        st.metric(
            "Total P&L",
            f"${summary['total_pnl']:,.2f}",
            delta=f"{summary['total_pnl']:+,.2f}"
        )
    
    with metric_col3:
        st.metric("Profitable", summary['profitable_positions'], 
                 delta=f"{summary['profitable_positions']}/{summary['total_positions']}")
    
    with metric_col4:
        st.metric("Losing", summary['losing_positions'],
                 delta=f"{summary['losing_positions']}/{summary['total_positions']}")
    
    # Position details
    if summary['positions']:
        st.subheader("📊 Position Details")
        
        for symbol, pos in summary['positions'].items():
            with st.container():
                pnl_emoji = "🟢" if pos['pnl_dollars'] >= 0 else "🔴"
                
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f"### {pnl_emoji} {symbol}")
                
                with col2:
                    st.metric("Quantity", f"{pos['quantity']:.0f}")
                
                with col3:
                    st.metric("Price", f"${pos['current_price']:.2f}")
                
                with col4:
                    st.metric("P&L $", f"${pos['pnl_dollars']:+,.2f}")
                
                with col5:
                    st.metric("P&L %", f"{pos['pnl_percent']:+.2f}%")
                
                # Stop loss and profit target controls
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    stop_price = st.number_input(
                        f"Stop Loss for {symbol}",
                        min_value=0.01,
                        value=pos['current_price'] * 0.95,
                        step=0.01,
                        key=f"stop_{symbol}"
                    )
                    if st.button(f"Set Stop", key=f"set_stop_{symbol}"):
                        position_monitor.set_stop_loss(symbol, stop_price)
                        st.success(f"Stop loss set at ${stop_price:.2f}")
                
                with exp_col2:
                    target_price = st.number_input(
                        f"Profit Target for {symbol}",
                        min_value=0.01,
                        value=pos['current_price'] * 1.10,
                        step=0.01,
                        key=f"target_{symbol}"
                    )
                    if st.button(f"Set Target", key=f"set_target_{symbol}"):
                        position_monitor.set_profit_target(symbol, target_price)
                        st.success(f"Profit target set at ${target_price:.2f}")
                
                st.divider()
    else:
        st.info("No open positions to monitor")
    
    # Alert history
    if st.session_state.get('show_alert_history', False):
        st.subheader("📜 Recent Alerts")
        
        from services.alert_system import get_alert_system
        alert_system = get_alert_system()
        recent_alerts = alert_system.get_recent_alerts(count=20)
        
        if recent_alerts:
            for alert in recent_alerts:
                priority_color = {
                    "CRITICAL": "🔴",
                    "HIGH": "🟠",
                    "MEDIUM": "🟡",
                    "LOW": "⚪"
                }.get(alert.priority.value, "⚪")
                
                st.markdown(f"{priority_color} **{alert.ticker}** - {alert.message}")
                st.caption(f"{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Priority: {alert.priority.value}")
                st.divider()
        else:
            st.info("No recent alerts")

# ============================================================================
# 4. MY TICKERS FILTERING FOR TECHNICAL ALERTS (Update scanner initialization)
# ============================================================================

def initialize_scanner_with_my_tickers():
    """Update your scanner initialization to support My Tickers filtering"""
    
    from services.preset_scanners import PresetScanner
    
    # Option 1: My Tickers only alerts
    scanner_my_tickers_only = PresetScanner(
        my_tickers_only=True,
        ticker_manager=st.session_state.ticker_manager
    )
    
    # Option 2: All tickers (default)
    scanner_all = PresetScanner()
    
    # Add toggle in UI
    my_tickers_filter = st.checkbox(
        "📌 Only alert for My Tickers",
        value=True,  # Default to True - most users want this
        help="If enabled, technical setup alerts will only trigger for stocks you've saved in My Tickers"
    )
    
    scanner = scanner_my_tickers_only if my_tickers_filter else scanner_all
    
    # Show which tickers are being monitored
    if my_tickers_filter:
        my_tickers = st.session_state.ticker_manager.get_all_tickers()
        if my_tickers:
            ticker_symbols = [t['ticker'] for t in my_tickers]
            st.info(f"📌 Alerts limited to: {', '.join(ticker_symbols)}")
        else:
            st.warning("⚠️ No tickers in My Tickers! Add some to receive alerts.")
    
    return scanner

# ============================================================================
# 5. USAGE IN MAIN APP
# ============================================================================

# Add to your tabs
tab1, tab2, tab3, ..., tab_positions = st.tabs([
    "Trading", "Analysis", "Account", ..., "📊 Positions"
])

with tab_positions:
    render_position_monitoring_tab()

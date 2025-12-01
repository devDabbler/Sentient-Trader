"""
Risk Profile Configuration UI for Service Control Panel

Provides:
- Risk tolerance selection (Conservative/Moderate/Aggressive)
- Capital settings configuration
- Position sizing controls
- AI sizing toggle
- Live position size calculator
"""

import streamlit as st
from typing import Dict, Optional
from loguru import logger


def render_risk_profile_config():
    """Render risk profile configuration UI in control panel"""
    
    try:
        from services.risk_profile_config import (
            get_risk_profile_manager, 
            RiskProfile, 
            RISK_PRESETS
        )
    except ImportError as e:
        st.error(f"Could not load risk profile module: {e}")
        return
    
    # Broker Status Section
    render_broker_status()
    st.divider()
    
    st.subheader("💰 Risk Profile & Position Sizing")
    
    # Get manager
    manager = get_risk_profile_manager()
    profile = manager.get_profile()
    
    # Quick preset selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        preset = st.selectbox(
            "Risk Preset",
            options=list(RISK_PRESETS.keys()),
            index=list(RISK_PRESETS.keys()).index(profile.risk_tolerance) 
                  if profile.risk_tolerance in RISK_PRESETS else 1,
            help="Quick presets for risk tolerance levels"
        )
    
    with col2:
        if st.button("Apply Preset", use_container_width=True):
            manager.apply_preset(preset)
            st.success(f"✅ Applied {preset} preset")
            st.rerun()
    
    # Show preset description
    if preset in RISK_PRESETS:
        st.caption(f"ℹ️ {RISK_PRESETS[preset]['description']}")
    
    st.divider()
    
    # Capital Settings
    st.markdown("**💵 Capital Settings**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_capital = st.number_input(
            "Total Capital ($)",
            min_value=100.0,
            max_value=10000000.0,
            value=float(profile.total_capital),
            step=1000.0,
            help="Total portfolio value"
        )
    
    with col2:
        available_capital = st.number_input(
            "Available Capital ($)",
            min_value=0.0,
            max_value=float(total_capital),
            value=float(min(profile.available_capital, total_capital)),
            step=500.0,
            help="Cash available for trading"
        )
    
    with col3:
        reserved_pct = st.number_input(
            "Reserve %",
            min_value=0.0,
            max_value=50.0,
            value=float(profile.reserved_pct),
            step=5.0,
            help="Percentage to keep in reserve"
        )
    
    # Position Sizing
    st.markdown("**📊 Position Sizing Limits**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_position_pct = st.slider(
            "Max Position %",
            min_value=1,
            max_value=50,
            value=int(profile.max_position_pct),
            help="Maximum % of capital per trade"
        )
    
    with col2:
        min_position_pct = st.slider(
            "Min Position %",
            min_value=1,
            max_value=int(max_position_pct),
            value=int(min(profile.min_position_pct, max_position_pct)),
            help="Minimum % of capital per trade"
        )
    
    with col3:
        max_positions = st.number_input(
            "Max Positions",
            min_value=1,
            max_value=50,
            value=int(profile.max_positions),
            step=1,
            help="Maximum concurrent positions"
        )
    
    # Risk Settings
    st.markdown("**⚠️ Risk Management**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_per_trade = st.slider(
            "Risk per Trade %",
            min_value=0.5,
            max_value=10.0,
            value=float(profile.risk_per_trade_pct),
            step=0.5,
            help="Max % of capital to risk per trade"
        )
    
    with col2:
        max_daily_loss = st.slider(
            "Max Daily Loss %",
            min_value=1.0,
            max_value=20.0,
            value=float(profile.max_loss_per_day_pct),
            step=1.0,
            help="Maximum daily drawdown before stopping"
        )
    
    with col3:
        min_confidence = st.slider(
            "Min Confidence %",
            min_value=30,
            max_value=90,
            value=int(profile.min_confidence_to_trade),
            help="Minimum signal confidence to trade"
        )
    
    # AI Settings
    col1, col2 = st.columns(2)
    with col1:
        use_ai_sizing = st.toggle(
            "🤖 AI Position Sizing",
            value=profile.use_ai_sizing,
            help="Use AI to dynamically adjust position sizes"
        )
    
    # Save button
    st.divider()
    if st.button("💾 Save Risk Profile", type="primary", use_container_width=True):
        manager.update_profile(
            risk_tolerance=preset,
            total_capital=total_capital,
            available_capital=available_capital,
            reserved_pct=reserved_pct,
            max_position_pct=float(max_position_pct),
            min_position_pct=float(min_position_pct),
            max_positions=max_positions,
            risk_per_trade_pct=risk_per_trade,
            max_loss_per_day_pct=max_daily_loss,
            min_confidence_to_trade=float(min_confidence),
            use_ai_sizing=use_ai_sizing
        )
        st.success("✅ Risk profile saved!")
    
    # Live Calculator
    st.divider()
    st.markdown("**🧮 Position Size Calculator**")
    
    calc_col1, calc_col2, calc_col3 = st.columns(3)
    
    with calc_col1:
        calc_price = st.number_input(
            "Entry Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=1.0,
            key="calc_price"
        )
    
    with calc_col2:
        calc_stop = st.number_input(
            "Stop Loss ($)",
            min_value=0.01,
            max_value=float(calc_price) * 0.99,
            value=float(calc_price) * 0.95,
            step=0.50,
            key="calc_stop"
        )
    
    with calc_col3:
        calc_confidence = st.slider(
            "Confidence %",
            min_value=30,
            max_value=100,
            value=75,
            key="calc_confidence"
        )
    
    # Calculate and display
    sizing = manager.calculate_position_size(
        price=calc_price,
        stop_loss=calc_stop,
        confidence=calc_confidence
    )
    
    # Display results
    result_col1, result_col2, result_col3, result_col4 = st.columns(4)
    
    with result_col1:
        st.metric(
            "Shares",
            f"{sizing['recommended_shares']:,}",
            help="Recommended number of shares"
        )
    
    with result_col2:
        st.metric(
            "Position Value",
            f"${sizing['recommended_value']:,.2f}",
            help="Total position value"
        )
    
    with result_col3:
        st.metric(
            "Position %",
            f"{sizing['position_pct']:.1f}%",
            help="Percentage of portfolio"
        )
    
    with result_col4:
        risk_color = "🟢" if sizing['risk_pct'] <= 2 else "🟡" if sizing['risk_pct'] <= 3 else "🔴"
        st.metric(
            "Risk",
            f"{risk_color} {sizing['risk_pct']:.1f}%",
            help=f"${sizing['risk_amount']:,.2f} at risk"
        )
    
    # Show calculation details
    with st.expander("📋 Calculation Details"):
        st.write(f"**Sizing Method:** {sizing['sizing_method'].replace('_', ' ').title()}")
        st.write(f"**Confidence Adjustment:** {sizing['confidence_adjustment']:.2f}x")
        st.write(f"**Max Position Value:** ${sizing['max_position_value']:,.2f}")
        st.write(f"**Usable Capital:** ${sizing['usable_capital']:,.2f}")
        st.write(f"**Risk Amount:** ${sizing['risk_amount']:,.2f}")
        
        # Risk/Reward calculation
        if calc_stop < calc_price:
            risk_per_share = calc_price - calc_stop
            # Assume 2:1 target
            target_price = calc_price + (risk_per_share * 2)
            potential_profit = sizing['recommended_shares'] * risk_per_share * 2
            
            st.write(f"**Risk per Share:** ${risk_per_share:.2f}")
            st.write(f"**2:1 Target Price:** ${target_price:.2f}")
            st.write(f"**Potential Profit (2:1):** ${potential_profit:,.2f}")


def render_compact_risk_summary():
    """Render a compact risk profile summary (for sidebar or header)"""
    
    try:
        from services.risk_profile_config import get_risk_profile_manager
    except ImportError:
        return
    
    manager = get_risk_profile_manager()
    profile = manager.get_profile()
    
    # Emoji based on tolerance
    emoji = {"Conservative": "🛡️", "Moderate": "⚖️", "Aggressive": "🚀"}.get(
        profile.risk_tolerance, "📊"
    )
    
    st.markdown(f"""
    **{emoji} {profile.risk_tolerance}** | 
    💰 ${profile.get_usable_capital():,.0f} usable |
    📈 Max {profile.max_position_pct}% |
    ⚠️ Risk {profile.risk_per_trade_pct}%
    """)


def get_position_recommendation_for_trade(
    symbol: str,
    price: float,
    stop_loss: float = None,
    confidence: float = None
) -> Dict:
    """
    Get position sizing recommendation for a specific trade
    
    Args:
        symbol: Stock symbol
        price: Entry price
        stop_loss: Stop loss price
        confidence: Signal confidence
        
    Returns:
        Dict with recommendation
    """
    try:
        from services.risk_profile_config import get_risk_profile_manager
        
        manager = get_risk_profile_manager()
        sizing = manager.calculate_position_size(
            price=price,
            stop_loss=stop_loss,
            confidence=confidence
        )
        
        sizing['symbol'] = symbol
        return sizing
        
    except Exception as e:
        logger.error(f"Error getting position recommendation: {e}")
        return {
            'symbol': symbol,
            'recommended_shares': 0,
            'recommended_value': 0,
            'error': str(e)
        }


def get_broker_status() -> Dict:
    """
    Get broker connection status and account balance
    
    Returns:
        Dict with broker status, balance info, and connection details
    """
    import os
    
    # Default to TRADIER if not set (matches ai_stock_position_manager behavior)
    broker_type = os.getenv('BROKER_TYPE', 'TRADIER').upper()
    
    result = {
        'connected': False,
        'broker_type': broker_type,
        'paper_mode': os.getenv('STOCK_PAPER_MODE', 'true').lower() == 'true',
        'total_equity': 0.0,
        'cash': 0.0,
        'buying_power': 0.0,
        'error': None
    }
    
    try:
        if broker_type == 'TRADIER':
            from src.integrations.tradier_client import TradierClient
            from src.integrations.broker_adapter import TradierAdapter
            
            # Get credentials based on mode
            if result['paper_mode']:
                account_id = os.getenv('TRADIER_PAPER_ACCOUNT_ID') or os.getenv('TRADIER_ACCOUNT_ID')
                access_token = os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or os.getenv('TRADIER_ACCESS_TOKEN')
            else:
                account_id = os.getenv('TRADIER_PROD_ACCOUNT_ID')
                access_token = os.getenv('TRADIER_PROD_ACCESS_TOKEN')
            
            if access_token and account_id:
                client = TradierClient(account_id=account_id, access_token=access_token)
                adapter = TradierAdapter(client)
                
                success, balance = adapter.get_account_balance()
                if success:
                    result['connected'] = True
                    result['total_equity'] = balance.get('total_equity', 0)
                    result['cash'] = balance.get('cash', 0)
                    result['buying_power'] = balance.get('buying_power', 0)
                else:
                    result['error'] = "Failed to get balance"
            else:
                result['error'] = "Credentials not configured"
                
        elif broker_type == 'IBKR':
            from src.integrations.ibkr_client import IBKRClient
            
            if result['paper_mode']:
                port = int(os.getenv('IBKR_PAPER_PORT', '7497'))
                client_id = int(os.getenv('IBKR_PAPER_CLIENT_ID', '1'))
            else:
                port = int(os.getenv('IBKR_LIVE_PORT', '7496'))
                client_id = int(os.getenv('IBKR_LIVE_CLIENT_ID', '2'))
            
            client = IBKRClient(port=port, client_id=client_id)
            if client.connect():
                account_info = client.get_account_info()
                if account_info:
                    result['connected'] = True
                    result['total_equity'] = account_info.net_liquidation
                    result['cash'] = account_info.total_cash_value
                    result['buying_power'] = account_info.buying_power
                else:
                    result['error'] = "Failed to get account info"
                client.disconnect()
            else:
                result['error'] = "Failed to connect to TWS/Gateway"
        else:
            result['error'] = f"Unknown broker type: {broker_type}"
            
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error getting broker status: {e}")
    
    return result


def render_broker_status():
    """Render broker connection status and balance in Streamlit"""
    st.subheader("🏦 Broker Account Status")
    
    status = get_broker_status()
    
    # Connection status
    if status['connected']:
        st.success(f"✅ Connected to **{status['broker_type']}** ({'PAPER' if status['paper_mode'] else 'LIVE'})")
    else:
        st.error(f"❌ Not Connected: {status.get('error', 'Unknown error')}")
        return
    
    # Balance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Equity",
            f"${status['total_equity']:,.2f}",
            help="Total account value including positions"
        )
    
    with col2:
        st.metric(
            "Cash",
            f"${status['cash']:,.2f}",
            help="Available cash in account"
        )
    
    with col3:
        st.metric(
            "Buying Power",
            f"${status['buying_power']:,.2f}",
            help="Available buying power for trades"
        )
    
    # Sync button
    if st.button("🔄 Sync Capital to Risk Profile", use_container_width=True):
        try:
            from services.risk_profile_config import get_risk_profile_manager
            manager = get_risk_profile_manager()
            manager.update_capital(
                total=status['total_equity'],
                available=status['buying_power']
            )
            st.success("✅ Risk profile updated with broker balances!")
            st.rerun()
        except Exception as e:
            st.error(f"Error syncing: {e}")


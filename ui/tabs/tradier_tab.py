"""
Tradier Account Tab
Tradier broker integration and account management

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime, timedelta
import pandas as pd

# Import trading config with fallback
try:
    from src.integrations.trading_config import get_trading_mode_manager, TradingMode, switch_to_paper_mode, switch_to_production_mode
except ImportError:
    logger.debug("Trading config not available, using fallback")
    class TradingMode:
        PAPER = "PAPER"
        PRODUCTION = "PRODUCTION"
    
    class MockTradingModeManager:
        def is_paper_mode(self):
            return True
        def get_current_mode(self):
            return TradingMode.PAPER
    
    def get_trading_mode_manager():
        return MockTradingModeManager()
    
    def switch_to_paper_mode():
        logger.info("Switched to paper mode (fallback)")
    
    def switch_to_production_mode():
        logger.info("Switched to production mode (fallback)")

# Import Tradier client with fallback
try:
    from src.integrations.tradier_client import TradierClient, validate_tradier_connection
except ImportError:
    logger.debug("TradierClient not available")
    TradierClient = None
    def validate_tradier_connection(api_key, account_id, use_sandbox):
        return False, "TradierClient not available"

def render_tab():
    """Main render function called from app.py"""
    st.header("Tradier Account")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    
    # =============================================================================
    # LAZY LOADING: Only initialize Tradier when user interacts with this tab
    # This saves ~1 second on every page load when viewing other tabs
    # =============================================================================
    
    # Track if user has visited this tab (auto-initialize on first view)
    if 'tab10_visited' not in st.session_state:
        st.session_state.tab10_visited = False
    
    # Auto-initialize on first view (no button required)
    if not st.session_state.tab10_visited:
        with st.spinner("⚡ Loading Tradier integration (one-time initialization)..."):
            st.session_state.tab10_visited = True
    
    # Initialize Tradier (only runs once per session)
    if st.session_state.tab10_visited:
        # Initialize Tradier client - cache mode check to prevent unnecessary re-initialization
        from src.integrations.tradier_client import create_tradier_client_from_env
        
        # Cache the last mode we initialized for to prevent unnecessary checks
        if 'tradier_client_last_mode' not in st.session_state:
            st.session_state.tradier_client_last_mode = None
        
        # Check if we need to initialize or refresh the client
        # Only check mode if client doesn't exist or might need refresh
        should_refresh_client = (
            'tradier_client' not in st.session_state or
            st.session_state.tradier_client is None
        )
        
        # If client exists, check if mode has changed (only if we suspect it might have)
        if not should_refresh_client and st.session_state.tradier_client is not None:
            # Only check mode if we suspect it might have changed (e.g., after mode switch)
            # This avoids calling get_trading_mode_manager() on every rerun
            if st.session_state.tradier_client_last_mode is None:
                # First time, cache the mode
                st.session_state.tradier_client_last_mode = st.session_state.tradier_client.trading_mode
            elif 'trading_mode' in st.session_state:
                # Check if session state mode differs from cached client mode
                # Only do this check if trading_mode is in session state (might have changed)
                mode_manager = get_trading_mode_manager()
                current_mode = mode_manager.get_mode()
                if current_mode != st.session_state.tradier_client_last_mode:
                    should_refresh_client = True
        
        if should_refresh_client:
            # Get current mode only when we actually need to refresh
            mode_manager = get_trading_mode_manager()
            current_mode = mode_manager.get_mode()
            
            logger.info("Initializing/refreshing Tradier client from environment")
            logger.info("Current trading mode: %s", current_mode.value)
            try:
                # Use trading mode manager to get client for current mode
                st.session_state.tradier_client = create_tradier_client_from_env(trading_mode=current_mode)
                st.session_state.tradier_client_last_mode = current_mode  # Cache the mode
                logger.info("Tradier client initialized successfully: %s", bool(st.session_state.tradier_client))
                logger.info("Client trading mode: %s", st.session_state.tradier_client.trading_mode.value if st.session_state.tradier_client else "None")
                # Clear cached account data when switching modes
                if 'account_summary' in st.session_state:
                    del st.session_state.account_summary
                    logger.info("Cleared cached account summary due to mode change")
            except Exception as e:
                logger.error(f"Failed to initialize Tradier client: {e}", exc_info=True)
                st.session_state.tradier_client = None
        
        # Trade Journal section
        with st.expander("📓 Trade Journal", expanded=False):
            try:
                from ui.trade_journal_ui import display_trade_journal
                display_trade_journal()
            except Exception as e:
                st.error(f"Failed to load trade journal: {e}")
                logger.error(f"Trade journal error: {e}", exc_info=True)
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Connection Status")
            
            if st.session_state.tradier_client:
                # Test connection
                if st.button("🔍 Test Connection"):
                    with st.spinner("Testing Tradier connection..."):
                        success, message = validate_tradier_connection()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                
                # Connection info
                st.info(f"**Account ID:** {st.session_state.tradier_client.account_id}")
                st.info(f"**API URL:** {st.session_state.tradier_client.api_url}")
                
            else:
                st.error("❌ Tradier client not initialized")
                st.warning("Please check your environment variables:")
                st.code("""
TRADIER_ACCOUNT_ID=your_account_id
TRADIER_ACCESS_TOKEN=your_access_token
TRADIER_API_URL=https://sandbox.tradier.com
                """)
        
        with col2:
            st.subheader("📊 Account Overview")
            
            if st.session_state.tradier_client:
                # Get account summary
                if st.button("🔄 Refresh Account Data"):
                    with st.spinner("Fetching account data..."):
                        success, summary = st.session_state.tradier_client.get_account_summary()
                    
                        if success:
                            st.session_state.account_summary = summary
                            st.success("Account data refreshed!")
                        else:
                            st.error(f"Failed to fetch account data: {summary.get('error', 'Unknown error')}")
            
                # Display account summary if available
                if 'account_summary' in st.session_state:
                    summary = st.session_state.account_summary
                
                    # Balance information
                    balance = summary.get('balance', {})
                    if 'balances' in balance:
                        bal_data = balance['balances']
                    
                        col1, col2, col3, col4 = st.columns(4)
                    
                        with col1:
                            st.metric("Total Cash", f"${float(bal_data.get('total_cash') or 0):,.2f}")
                        with col2:
                            st.metric("Buying Power", f"${float(bal_data.get('buying_power') or 0):,.2f}")
                        with col3:
                            st.metric("Day Trading", f"${float(bal_data.get('day_trading') or 0):,.2f}")
                        with col4:
                            st.metric("Market Value", f"${float(bal_data.get('market_value') or 0):,.2f}")
                
                    # Positions
                    st.subheader("📈 Current Positions")
                    positions = summary.get('positions', [])
                
                    if positions:
                        positions_df = pd.DataFrame(positions)
                    
                        # Display key columns
                        display_cols = ['symbol', 'quantity', 'average_cost', 'market_value', 'gain_loss']
                        if all(col in positions_df.columns for col in display_cols):
                            st.dataframe(
                                positions_df[display_cols],
                                width='stretch',
                                column_config={
                                    "symbol": "Symbol",
                                    "quantity": "Quantity", 
                                    "average_cost": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
                                    "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                                    "gain_loss": st.column_config.NumberColumn("P&L", format="$%.2f")
                                }
                            )
                        else:
                            st.dataframe(positions_df, width='stretch')
                    else:
                        st.info("No positions found")
                
                    # Recent orders
                    st.subheader("📋 Recent Orders")
                    orders = summary.get('recent_orders', [])
                
                    if orders:
                        # Group orders by class (show bracket orders specially)
                        for order in orders:
                            order_class = order.get('class', 'equity')
                            order_id = order.get('id', 'N/A')
                            symbol = order.get('symbol', 'N/A')
                            status = order.get('status', 'N/A')
                        
                            if order_class in ['otoco', 'oco']:
                                # Bracket order - show all legs
                                with st.expander(f"🎯 Bracket Order: {symbol} (ID: {order_id}) - {status}"):
                                    st.write(f"**Order Class:** {order_class.upper()}")
                                    st.write(f"**Status:** {status}")
                                
                                    # Get legs if available
                                    legs = order.get('leg', [])
                                    if not isinstance(legs, list):
                                        legs = [legs] if legs else []
                                
                                    if legs:
                                        st.write("**Order Legs:**")
                                        for i, leg in enumerate(legs, 1):
                                            leg_type = leg.get('type', 'N/A')
                                            leg_side = leg.get('side', 'N/A')
                                            leg_qty = leg.get('quantity', 'N/A')
                                            leg_price = leg.get('price', leg.get('avg_fill_price', ''))
                                            leg_stop = leg.get('stop', '')
                                            leg_status = leg.get('status', 'N/A')
                                        
                                            # Determine leg purpose based on type and position
                                            if leg_type == 'limit' and i == 1:
                                                price_str = f"${leg_price}" if leg_price else "N/A"
                                                st.info(f"**Leg {i} - Entry:** {leg_side.upper()} {leg_qty} @ {price_str} ({leg_status})")
                                            elif leg_type == 'limit' and i == 2:
                                                price_str = f"${leg_price}" if leg_price else "N/A"
                                                st.success(f"**Leg {i} - Take Profit:** {leg_side.upper()} {leg_qty} @ {price_str} ({leg_status})")
                                            elif leg_type in ['stop', 'stop_limit'] or i == 3:
                                                # For stop orders, show stop price
                                                if leg_stop:
                                                    price_display = f"${leg_stop}"
                                                elif leg_price:
                                                    price_display = f"${leg_price}"
                                                else:
                                                    price_display = "N/A"
                                                st.error(f"**Leg {i} - Stop Loss:** {leg_side.upper()} {leg_qty} @ {price_display} ({leg_status})")
                                            else:
                                                # Fallback display
                                                price_info = f"Price: ${leg_price}" if leg_price else ""
                                                stop_info = f", Stop: ${leg_stop}" if leg_stop else ""
                                                st.write(f"**Leg {i}:** {leg_type.upper()} {leg_side.upper()} {leg_qty} - {price_info}{stop_info} ({leg_status})")
                                
                                    # Show full order details
                                    with st.expander("View Full Order JSON"):
                                        st.json(order)
                            else:
                                # Simple order
                                with st.expander(f"📝 {order_class.upper()}: {symbol} (ID: {order_id}) - {status}"):
                                    st.write(f"**Side:** {order.get('side', 'N/A')}")
                                    st.write(f"**Quantity:** {order.get('quantity', 'N/A')}")
                                    st.write(f"**Type:** {order.get('type', 'N/A')}")
                                    st.write(f"**Price:** ${order.get('price', 'N/A')}")
                                    st.write(f"**Status:** {status}")
                                
                                    with st.expander("View Full Order JSON"):
                                        st.json(order)
                    else:
                        st.info("No orders found")
            
                else:
                    st.info("Click 'Refresh Account Data' to load your account information")
        
        # Order management section
        st.subheader("📝 Order Management")
    
        if st.session_state.tradier_client:
            col1, col2 = st.columns(2)
        
            with col1:
                st.write("**Get Order Status**")
                order_id = st.text_input("Order ID", placeholder="Enter order ID")
            
                if st.button("🔍 Get Order Status") and order_id:
                    with st.spinner("Fetching order status..."):
                        success, order_data = st.session_state.tradier_client.get_order_status(order_id)
                    
                        if success:
                            st.success("Order found!")
                            st.json(order_data)
                        else:
                            st.error(f"Failed to get order: {order_data.get('error', 'Unknown error')}")
        
            with col2:
                st.write("**Cancel Order**")
                cancel_order_id = st.text_input("Order ID to Cancel", placeholder="Enter order ID", key="cancel_order")
            
                if st.button("❌ Cancel Order") and cancel_order_id:
                    with st.spinner("Cancelling order..."):
                        success, result = st.session_state.tradier_client.cancel_order(cancel_order_id)
                    
                        if success:
                            st.success("Order cancelled successfully!")
                            st.json(result)
                        else:
                            st.error(f"Failed to cancel order: {result.get('error', 'Unknown error')}")
    
        # Manual order placement section
        st.subheader("🎯 Manual Order Placement")
    
        if st.session_state.tradier_client:
            with st.expander("Place Custom Order"):
                # Order mode selection
                order_mode = st.radio("Order Mode", ["Simple Order", "Bracket Order (OTOCO)"], horizontal=True, key='tab7_order_mode')
            
                if order_mode == "Bracket Order (OTOCO)":
                    st.info("🎯 Bracket orders automatically set take-profit and stop-loss orders after your entry fills")
                
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        symbol = st.text_input("Symbol", placeholder="AAPL", key='tab7_bracket_symbol')
                        side = st.selectbox("Side", ["buy", "sell"], key='tab7_bracket_side')
                        quantity = st.number_input("Quantity", min_value=1, value=10, key='tab7_bracket_qty')
                
                    with col2:
                        entry_price = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01, format="%.2f", key='tab7_bracket_entry')
                        take_profit = st.number_input("Take Profit Price", min_value=0.01, value=105.00, step=0.01, format="%.2f", key='tab7_bracket_profit')
                        stop_loss = st.number_input("Stop Loss Price", min_value=0.01, value=97.00, step=0.01, format="%.2f", key='tab7_bracket_stop')
                
                    with col3:
                        duration = st.selectbox("Duration", ["gtc", "day"], key='tab7_bracket_duration')
                        tag = st.text_input("Tag", value=f"BRACKET_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key='tab7_bracket_tag')
                    
                        # Calculate percentages
                        if side == "buy":
                            profit_pct = ((take_profit - entry_price) / entry_price) * 100
                            loss_pct = ((entry_price - stop_loss) / entry_price) * 100
                        else:
                            profit_pct = ((entry_price - take_profit) / entry_price) * 100
                            loss_pct = ((stop_loss - entry_price) / entry_price) * 100
                    
                        st.metric("Profit Target", f"{profit_pct:.1f}%")
                        st.metric("Max Loss", f"{loss_pct:.1f}%")
                
                    if st.button("🎯 Place Bracket Order", type="primary", key='tab7_bracket_submit'):
                        with st.spinner("Placing bracket order..."):
                            success, result = st.session_state.tradier_client.place_bracket_order(
                                symbol=symbol.upper(),
                                side=side,
                                quantity=quantity,
                                entry_price=entry_price,
                                take_profit_price=take_profit,
                                stop_loss_price=stop_loss,
                                duration=duration,
                                tag=tag
                            )
                        
                            if success:
                                st.success("🎉 Bracket order placed successfully!")
                                st.info(f"✅ Entry: ${entry_price} | 🎯 Target: ${take_profit} | 🛑 Stop: ${stop_loss}")
                                st.json(result)
                            else:
                                st.error(f"Failed to place bracket order: {result.get('error', 'Unknown error')}")
                                st.json(result)
            
                else:
                    # Simple order mode
                    col1, col2 = st.columns(2)
                
                    with col1:
                        order_class = st.selectbox("Order Class", ["equity", "option", "multileg", "combo"], key='tab7_order_class_select')
                        symbol = st.text_input("Symbol", placeholder="AAPL or AAPL240315C150")
                        side = st.selectbox("Side", ["buy", "sell", "buy_to_cover", "sell_short", "sell_to_open", "sell_to_close", "buy_to_open", "buy_to_close"], key='tab7_order_side_select')
                        quantity = st.number_input("Quantity", min_value=1, value=1)
                
                    with col2:
                        order_type = st.selectbox("Order Type", ["market", "limit", "stop", "stop_limit", "credit", "debit"], key='tab7_order_type_select')
                        duration = st.selectbox("Duration", ["day", "gtc", "pre", "post"], key='tab7_order_duration_select')
                        price = st.number_input("Price", min_value=0.0, value=0.0, step=0.01, format="%.2f")
                        tag = st.text_input("Tag", value=f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                
                    if st.button("📤 Place Order", type="primary"):
                        order_data = {
                            "class": order_class,
                            "symbol": symbol.upper(),
                            "side": side,
                            "quantity": str(quantity),
                            "type": order_type,
                            "duration": duration,
                            "tag": tag
                        }
                    
                        if order_type in ["limit", "stop_limit"] and price > 0:
                            order_data["price"] = str(price)
                    
                        with st.spinner("Placing order..."):
                            success, result = st.session_state.tradier_client.place_order(order_data)
                        
                            if success:
                                st.success("Order placed successfully!")
                                st.json(result)
                            else:
                                st.error(f"Failed to place order: {result.get('error', 'Unknown error')}")
    
        # Configuration section
        st.subheader("⚙️ Configuration")
    
        with st.expander("Environment Variables Status"):
            env_vars = {
                "TRADIER_ACCOUNT_ID": os.getenv('TRADIER_ACCOUNT_ID', 'Not set'),
                "TRADIER_ACCESS_TOKEN": os.getenv('TRADIER_ACCESS_TOKEN', 'Not set'),
                "TRADIER_API_URL": os.getenv('TRADIER_API_URL', 'Not set'),
                "OPTION_ALPHA_WEBHOOK_URL": os.getenv('OPTION_ALPHA_WEBHOOK_URL', 'Not set')
            }
        
            for var, value in env_vars.items():
                if 'TOKEN' in var and value != 'Not set':
                    st.code(f"{var}=***{value[-4:] if len(value) > 4 else '***'}")
                else:
                    st.code(f"{var}={value}")


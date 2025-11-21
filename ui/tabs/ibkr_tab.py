"""
IBKR Trading Tab
Interactive Brokers integration

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple

def render_tab():
    """Main render function called from app.py"""
    st.header("IBKR Trading")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    # =============================================================================
    # LAZY LOADING: Only import IBKR modules when user interacts with this tab
    # This saves ~300ms on every page load when viewing other tabs
    # =============================================================================
    
    # Track if user has visited this tab (auto-initialize on first view)
    if 'tab11_visited' not in st.session_state:
        st.session_state.tab11_visited = False
    
    # Auto-initialize on first view (no button required)
    if not st.session_state.tab11_visited:
        with st.spinner("⚡ Loading IBKR integration (one-time initialization)..."):
            st.session_state.tab11_visited = True
    
    # Initialize IBKR (only runs once per session)
    if st.session_state.tab11_visited:
        st.header("📈 IBKR Day Trading / Scalping")
        st.write("Connect to Interactive Brokers for live day trading and scalping. Real-time positions, orders, and execution.")
        
        # Import IBKR client with comprehensive error handling
        ibkr_available = False
        ibkr_error_message = None
        try:
            logger.info("Attempting to import IBKR client modules...")
            from src.integrations.ibkr_client import IBKRClient, create_ibkr_client_from_env, validate_ibkr_connection, IBKRPosition, IBKROrder
            ibkr_available = True
            logger.info("IBKR client modules imported successfully")
        except ImportError as e:
            ibkr_error_message = f"Missing dependency: {e}. Please install: pip install ib_insync"
            logger.error("IBKR ImportError: {}", str(e), exc_info=True)
            st.error(f"⚠️ {ibkr_error_message}")
        except RuntimeError as e:
            ibkr_error_message = f"Event loop error: {e}. This is a known issue with asyncio in Streamlit."
            logger.error("IBKR RuntimeError: {}", str(e), exc_info=True)
            st.error(f"⚠️ {ibkr_error_message}")
            st.info("💡 Try restarting the Streamlit app to resolve event loop issues.")
        except Exception as e:
            ibkr_error_message = f"Unexpected error: {e}"
            logger.error("IBKR unexpected error: {}", str(e), exc_info=True)
            st.error(f"⚠️ {ibkr_error_message}")
            st.code(str(e))
        
        if ibkr_available:
            # Initialize IBKR client in session state
            if 'ibkr_client' not in st.session_state:
                st.session_state.ibkr_client = None
                st.session_state.ibkr_connected = False
        
            # Trade Journal section
            with st.expander("📓 Trade Journal", expanded=False):
                try:
                    from ui.trade_journal_ui import display_trade_journal
                    display_trade_journal()
                except Exception as e:
                    st.error(f"Failed to load trade journal: {e}")
                    logger.error("Trade journal error: {}", str(e), exc_info=True)
            
            st.divider()
            
            # Connection Section
            st.subheader("🔌 Connection Settings")
        
            col1, col2, col3 = st.columns(3)
        
            with col1:
                ibkr_host = st.text_input("Host", value="127.0.0.1", help="IB Gateway/TWS host address")
        
            with col2:
                ibkr_port = st.number_input(
                    "Port", 
                    value=7497, 
                    help="7497 for paper trading, 7496 for live TWS, 4002 for IB Gateway paper, 4001 for IB Gateway live"
                )
        
            with col3:
                ibkr_client_id = st.number_input("Client ID", value=1, min_value=1, max_value=32)
        
            col_conn1, col_conn2 = st.columns(2)
        
            with col_conn1:
                if st.button("🔗 Connect to IBKR", type="primary", width="stretch"):
                    try:
                        with st.status("Connecting to Interactive Brokers...") as status:
                            st.write("Initializing connection...")
                            client = IBKRClient(host=ibkr_host, port=int(ibkr_port), client_id=int(ibkr_client_id))
                        
                            st.write("Connecting to IB Gateway/TWS...")
                            if client.connect(timeout=10):
                                st.session_state.ibkr_client = client
                                st.session_state.ibkr_connected = True
                            
                                st.write("Fetching account information...")
                                account_info = client.get_account_info()
                            
                                if account_info:
                                    status.update(label="✅ Connected to IBKR!", state="complete")
                                    st.success(f"Connected to account: {account_info.account_id}")
                                    st.info(f"💰 Buying Power: ${account_info.buying_power:,.2f} | Net Liquidation: ${account_info.net_liquidation:,.2f}")
                                else:
                                    status.update(label="⚠️ Connected but no account info", state="error")
                            else:
                                st.error("Failed to connect. Make sure IB Gateway or TWS is running with API enabled.")
                                status.update(label="❌ Connection failed", state="error")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
        
            with col_conn2:
                if st.button("🔌 Disconnect", width="stretch"):
                    if st.session_state.ibkr_client:
                        st.session_state.ibkr_client.disconnect()
                        st.session_state.ibkr_client = None
                        st.session_state.ibkr_connected = False
                        st.success("Disconnected from IBKR")
        
            st.divider()
        
            # Show connection status
            if st.session_state.ibkr_connected and st.session_state.ibkr_client:
                if st.session_state.ibkr_client.is_connected():
                    st.success("🟢 Connected to IBKR")
                else:
                    st.warning("🟡 Connection lost - please reconnect")
                    st.session_state.ibkr_connected = False
            else:
                st.info("🔴 Not connected to IBKR")
        
            # Main trading interface (only show if connected)
            if st.session_state.ibkr_connected and st.session_state.ibkr_client:
                client = st.session_state.ibkr_client
            
                # Account Information
                st.subheader("💼 Account Information")
            
                try:
                    account_info = client.get_account_info()
                
                    if account_info:
                        col1, col2, col3, col4 = st.columns(4)
                    
                        with col1:
                            st.metric("Net Liquidation", f"${account_info.net_liquidation:,.2f}")
                    
                        with col2:
                            st.metric("Buying Power", f"${account_info.buying_power:,.2f}")
                    
                        with col3:
                            st.metric("Cash", f"${account_info.total_cash_value:,.2f}")
                    
                        with col4:
                            if account_info.is_pdt:
                                st.metric("Day Trades Left", "Unlimited" if account_info.net_liquidation >= 25000 else str(account_info.day_trades_remaining))
                            else:
                                st.metric("Day Trades Left", str(account_info.day_trades_remaining))
                
                except Exception as e:
                    st.error(f"Error fetching account info: {e}")
            
                st.divider()
            
                # Current Positions
                st.subheader("📊 Current Positions")
            
                if st.button("🔄 Refresh Positions", width="stretch"):
                    st.rerun()
            
                try:
                    positions = client.get_positions()
                
                    if positions:
                        positions_data = []
                        for pos in positions:
                            # Check if position is fractional
                            is_fractional = (pos.position % 1 != 0)
                            qty_display = f"{pos.position:.4f} 📊" if is_fractional else f"{int(pos.position)}"
                            
                            # Calculate ROI percentage
                            roi_pct = ((pos.market_price - pos.avg_cost) / pos.avg_cost * 100) if pos.avg_cost > 0 else 0
                            
                            positions_data.append({
                                'Symbol': pos.symbol,
                                'Quantity': qty_display,
                                'Avg Cost': f"${pos.avg_cost:.2f}",
                                'Market Price': f"${pos.market_price:.2f}",
                                'Market Value': f"${pos.market_value:,.2f}",
                                'Unrealized P&L': f"${pos.unrealized_pnl:,.2f}",
                                'ROI %': f"{roi_pct:+.2f}%",
                                'Realized P&L': f"${pos.realized_pnl:,.2f}"
                            })
                    
                        positions_df = pd.DataFrame(positions_data)
                        st.dataframe(positions_df, width='stretch', hide_index=True)
                    
                        # Quick flatten buttons
                        st.write("**Quick Actions:**")
                        cols = st.columns(min(len(positions), 4))
                        for idx, pos in enumerate(positions[:4]):
                            with cols[idx]:
                                if st.button(f"Close {pos.symbol}", key=f"flatten_{pos.symbol}"):
                                    if client.flatten_position(pos.symbol):
                                        st.success(f"✅ Closing {pos.symbol}")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to close {pos.symbol}")
                    else:
                        st.info("No open positions")
            
                except Exception as e:
                    st.error(f"Error fetching positions: {e}")
            
                st.divider()
            
                # Open Orders
                st.subheader("📝 Open Orders")
            
                try:
                    open_orders = client.get_open_orders()
                
                    if open_orders:
                        orders_data = []
                        for order in open_orders:
                            orders_data.append({
                                'Order ID': order.order_id,
                                'Symbol': order.symbol,
                                'Action': order.action,
                                'Type': order.order_type,
                                'Qty': order.quantity,
                                'Limit': f"${order.limit_price:.2f}" if order.limit_price else "N/A",
                                'Stop': f"${order.stop_price:.2f}" if order.stop_price else "N/A",
                                'Status': order.status,
                                'Filled': order.filled,
                                'Remaining': order.remaining
                            })
                    
                        orders_df = pd.DataFrame(orders_data)
                        st.dataframe(orders_df, width="stretch")
                    
                        # Cancel orders
                        col_cancel1, col_cancel2 = st.columns(2)
                    
                        with col_cancel1:
                            order_id_to_cancel = st.number_input("Order ID to Cancel", min_value=1, step=1)
                            if st.button("❌ Cancel Order"):
                                if client.cancel_order(int(order_id_to_cancel)):
                                    st.success(f"Order {order_id_to_cancel} cancelled")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Failed to cancel order {order_id_to_cancel}")
                    
                        with col_cancel2:
                            st.write("")
                            st.write("")
                            if st.button("❌❌ Cancel ALL Orders", type="secondary"):
                                cancelled = client.cancel_all_orders()
                                st.success(f"Cancelled {cancelled} orders")
                                time.sleep(1)
                                st.rerun()
                    else:
                        st.info("No open orders")
            
                except Exception as e:
                    st.error(f"Error fetching orders: {e}")
            
                st.divider()
                
                # Fractional Shares Configuration (expander)
                with st.expander("📊 Fractional Share Settings (IBKR Only)"):
                    try:
                        from ui.fractional_share_config_ui import display_fractional_share_config
                        display_fractional_share_config()
                    except Exception as e:
                        st.error(f"Error loading fractional share settings: {e}")
                        logger.error("Fractional share UI error: {}", str(e), exc_info=True)
                
                st.divider()
            
                # Place New Order
                st.subheader("🎯 Place Order")
            
                col1, col2 = st.columns(2)
            
                with col1:
                    order_symbol = st.text_input("Symbol", value="", key="order_symbol").upper()
                    order_action = st.selectbox("Action", options=["BUY", "SELL"], key="order_action")
                    order_quantity = st.number_input("Quantity", min_value=1, value=100, step=1, key="order_quantity")
            
                with col2:
                    order_type = st.selectbox(
                        "Order Type", 
                        options=["MARKET", "LIMIT", "STOP"],
                        key="order_type"
                    )
                
                    if order_type == "LIMIT":
                        order_limit_price = st.number_input("Limit Price", min_value=0.01, value=10.0, step=0.01, key="order_limit")
                    elif order_type == "STOP":
                        order_stop_price = st.number_input("Stop Price", min_value=0.01, value=10.0, step=0.01, key="order_stop")
            
                # Place order button
                if st.button("🚀 Place Order", type="primary", width="stretch"):
                    if not order_symbol:
                        st.error("Please enter a symbol")
                    else:
                        try:
                            with st.status(f"Placing {order_type} order...") as status:
                                result = None
                            
                                if order_type == "MARKET":
                                    result = client.place_market_order(order_symbol, order_action, int(order_quantity))
                                elif order_type == "LIMIT":
                                    result = client.place_limit_order(order_symbol, order_action, int(order_quantity), float(order_limit_price))
                                elif order_type == "STOP":
                                    result = client.place_stop_order(order_symbol, order_action, int(order_quantity), float(order_stop_price))
                            
                                if result:
                                    status.update(label="✅ Order placed!", state="complete")
                                    st.success(f"Order placed: {order_action} {order_quantity} {order_symbol}")
                                    st.json({
                                        'Order ID': result.order_id,
                                        'Symbol': result.symbol,
                                        'Action': result.action,
                                        'Type': result.order_type,
                                        'Quantity': result.quantity,
                                        'Status': result.status
                                    })
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    status.update(label="❌ Order failed", state="error")
                                    st.error("Failed to place order")
                    
                        except Exception as e:
                            st.error(f"Error placing order: {e}")
            
                st.divider()
            
                # Market Data
                st.subheader("📊 Real-Time Market Data")
            
                col1, col2 = st.columns([3, 1])
            
                with col1:
                    market_symbol = st.text_input("Symbol for Quote", value="SPY", key="market_symbol").upper()
            
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("📈 Get Quote", width="stretch"):
                        if market_symbol:
                            try:
                                market_data = client.get_market_data(market_symbol)
                            
                                if market_data:
                                    col1, col2, col3, col4 = st.columns(4)
                                
                                    with col1:
                                        st.metric("Last", f"${market_data['last']:.2f}")
                                
                                    with col2:
                                        st.metric("Bid", f"${market_data['bid']:.2f}", delta=f"{market_data['bid_size']}")
                                
                                    with col3:
                                        st.metric("Ask", f"${market_data['ask']:.2f}", delta=f"{market_data['ask_size']}")
                                
                                    with col4:
                                        st.metric("Volume", f"{market_data['volume']:,}")
                                else:
                                    st.error("Failed to fetch market data")
                        
                            except Exception as e:
                                st.error(f"Error fetching market data: {e}")
        
            else:
                st.warning("⚠️ Please connect to IBKR to access trading features")
                st.info("**Setup Instructions:**\n"
                       "1. Download and install IB Gateway or TWS from Interactive Brokers\n"
                       "2. Log in with your IBKR credentials\n"
                       "3. Enable API connections in TWS/Gateway settings\n"
                       "4. Set the port number (7497 for paper, 7496 for live)\n"
                       "5. Click 'Connect to IBKR' above")


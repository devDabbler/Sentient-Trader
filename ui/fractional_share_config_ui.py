"""
Fractional Share Configuration UI
Allows configuration of fractional share trading for IBKR
"""

import streamlit as st
from loguru import logger
from typing import Optional, Dict, List
import pandas as pd


def display_fractional_share_config():
    """
    Display fractional share configuration UI
    Allows users to configure fractional share trading settings
    """
    st.markdown("### 📊 Fractional Share Configuration")
    st.markdown("Configure fractional share trading for expensive stocks (IBKR only)")
    
    try:
        from services.fractional_share_manager import get_fractional_share_manager, FractionalShareConfig
        manager = get_fractional_share_manager()
    except Exception as e:
        st.error(f"Error loading Fractional Share Manager: {e}")
        logger.error("Fractional Share Manager loading error: {}", str(e), exc_info=True)
        return
    
    # Global Settings Section
    st.markdown("#### ⚙️ Global Settings")
    
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        # Enable/disable fractional shares
        enabled = st.checkbox(
            "✅ Enable Fractional Shares",
            value=manager.config.enabled,
            help="Enable fractional share trading for expensive stocks",
            key="fractional_enabled"
        )
        
        # Price threshold slider
        price_threshold = st.slider(
            "Auto-Enable for stocks above:",
            min_value=50.0,
            max_value=500.0,
            value=manager.config.min_price_threshold,
            step=10.0,
            format="$%.0f",
            help="Automatically use fractional shares for stocks above this price",
            key="fractional_price_threshold"
        )
    
    with settings_col2:
        # Dollar amount range
        min_amount = st.number_input(
            "Minimum Dollar Amount",
            min_value=10.0,
            max_value=1000.0,
            value=manager.config.min_dollar_amount,
            step=10.0,
            format="%.2f",
            help="Minimum dollar amount per fractional trade",
            key="fractional_min_amount"
        )
        
        max_amount = st.number_input(
            "Maximum Dollar Amount",
            min_value=50.0,
            max_value=10000.0,
            value=manager.config.max_dollar_amount if manager.config.max_dollar_amount else 1000.0,
            step=50.0,
            format="%.2f",
            help="Maximum dollar amount per fractional trade",
            key="fractional_max_amount"
        )
    
    # Quick amounts section
    st.markdown("#### 💰 Quick Dollar Amounts")
    quick_col1, quick_col2, quick_col3, quick_col4, quick_col5 = st.columns(5)
    
    preferred_amounts = manager.config.preferred_dollar_amounts
    
    with quick_col1:
        st.button(f"${preferred_amounts[0]:.0f}", key="quick_50", width='stretch')
    with quick_col2:
        st.button(f"${preferred_amounts[1]:.0f}", key="quick_100", width='stretch')
    with quick_col3:
        st.button(f"${preferred_amounts[2]:.0f}", key="quick_250", width='stretch')
    with quick_col4:
        st.button(f"${preferred_amounts[3]:.0f}", key="quick_500", width='stretch')
    with quick_col5:
        st.button(f"${preferred_amounts[4]:.0f}", key="quick_1000", width='stretch')
    
    st.markdown("---")
    
    # Save global settings button
    if st.button("💾 Save Global Settings", type="primary", width='stretch'):
        try:
            # Update config
            manager.config.enabled = enabled
            manager.config.min_price_threshold = price_threshold
            manager.config.min_dollar_amount = min_amount
            manager.config.max_dollar_amount = max_amount
            
            # Save state
            manager._save_state()
            
            st.success("✅ Global settings saved successfully!")
            logger.info(f"Fractional share settings saved: enabled={enabled}, threshold=${price_threshold:.2f}")
        except Exception as e:
            st.error(f"Error saving settings: {e}")
            logger.error("Error saving fractional share settings: {}", str(e), exc_info=True)
    
    st.markdown("---")
    
    # Per-Ticker Configuration Section
    st.markdown("#### 📈 Per-Ticker Configuration")
    st.markdown("Set custom dollar amounts for specific tickers")
    
    # Get current custom amounts
    custom_amounts = manager.custom_amounts
    
    # Display existing custom amounts
    if custom_amounts:
        st.markdown("**Current Custom Amounts:**")
        
        # Create DataFrame for display
        custom_data = []
        for symbol, amount in custom_amounts.items():
            custom_data.append({
                'Symbol': symbol,
                'Dollar Amount': f"${amount:.2f}",
                'Shares @ $100': f"{(amount / 100):.2f}",
                'Shares @ $500': f"{(amount / 500):.2f}"
            })
        
        if custom_data:
            df = pd.DataFrame(custom_data)
            st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("No custom amounts configured yet. Add tickers below to set custom amounts.")
    
    st.markdown("---")
    
    # Add new custom amount
    st.markdown("**Add/Update Custom Amount:**")
    
    add_col1, add_col2, add_col3 = st.columns([2, 2, 1])
    
    with add_col1:
        new_symbol = st.text_input(
            "Ticker Symbol",
            placeholder="e.g., NVDA, TSLA, AAPL",
            key="fractional_new_symbol"
        ).upper()
    
    with add_col2:
        new_amount = st.number_input(
            "Dollar Amount",
            min_value=min_amount,
            max_value=max_amount,
            value=250.0,
            step=10.0,
            format="%.2f",
            key="fractional_new_amount"
        )
    
    with add_col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        if st.button("➕ Add/Update", width='stretch'):
            if new_symbol:
                try:
                    manager.set_custom_amount(new_symbol, new_amount)
                    st.success(f"✅ Set {new_symbol} to ${new_amount:.2f}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error setting custom amount: {e}")
                    logger.error("Error setting custom amount for {new_symbol}: {}", str(e), exc_info=True)
            else:
                st.warning("Please enter a ticker symbol")
    
    # Remove custom amount
    if custom_amounts:
        st.markdown("---")
        st.markdown("**Remove Custom Amount:**")
        
        remove_col1, remove_col2 = st.columns([3, 1])
        
        with remove_col1:
            symbol_to_remove = st.selectbox(
                "Select ticker to remove",
                options=list(custom_amounts.keys()),
                key="fractional_remove_symbol"
            )
        
        with remove_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            if st.button("🗑️ Remove", width='stretch'):
                try:
                    manager.remove_custom_amount(symbol_to_remove)
                    st.success(f"✅ Removed {symbol_to_remove}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error removing custom amount: {e}")
                    logger.error("Error removing custom amount for {symbol_to_remove}: {}", str(e), exc_info=True)
    
    st.markdown("---")
    
    # Quick Actions Section
    st.markdown("#### ⚡ Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("🔥 Apply to Expensive Stocks (>$100)", width='stretch'):
            st.info("This will scan your watchlist and apply default amounts to stocks >$100")
            # This would need watchlist integration
    
    with action_col2:
        if st.button("📋 Apply to All Watchlist", width='stretch'):
            st.info("This will apply fractional settings to all watchlist tickers")
            # This would need watchlist integration
    
    with action_col3:
        if st.button("🔄 Reset to Defaults", width='stretch'):
            try:
                manager.custom_amounts = {}
                manager._save_state()
                st.success("✅ Reset to defaults")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting: {e}")
    
    st.markdown("---")
    
    # Information Section
    with st.expander("ℹ️ How Fractional Shares Work"):
        st.markdown("""
        **Fractional shares allow you to buy portions of expensive stocks with smaller capital.**
        
        ### Key Features:
        - **Automatic Detection**: Stocks above your price threshold automatically use fractional shares
        - **Custom Amounts**: Set specific dollar amounts per ticker (e.g., always invest $250 in NVDA)
        - **Precise Sizing**: Use exact dollar amounts instead of rounding to whole shares
        - **Better Diversification**: Spread capital across more positions
        
        ### Example:
        - **Without Fractional**: NVDA @ $500 → Need $500+ to buy 1 share
        - **With Fractional**: NVDA @ $500 → Buy 0.5 shares for $250
        
        ### Requirements:
        - ⚠️ **IBKR Only**: Fractional shares only work with Interactive Brokers
        - ⚠️ **US Stocks**: Only available for US stocks priced above $5
        - ⚠️ **Minimum Order**: IBKR requires $1 minimum per order
        - ✅ **Paper & Live**: Works in both paper trading and live trading modes
        
        ### How Auto-Trader Uses Fractional:
        1. Check if stock price > threshold (default $100)
        2. Check if custom amount exists for this ticker
        3. Calculate fractional quantity from dollar amount
        4. Place order with fractional quantity (e.g., 0.5 shares)
        
        ### Tips:
        - Use custom amounts for stocks you trade frequently
        - Set higher amounts for higher-conviction trades
        - Keep amounts within your risk management rules
        """)
    
    # Preview Section
    with st.expander("👁️ Position Size Preview"):
        st.markdown("**Preview fractional quantities for different stock prices:**")
        
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            preview_symbol = st.text_input(
                "Symbol (optional)",
                placeholder="e.g., NVDA",
                key="fractional_preview_symbol"
            ).upper()
            
            preview_price = st.number_input(
                "Stock Price",
                min_value=1.0,
                max_value=10000.0,
                value=500.0,
                step=10.0,
                key="fractional_preview_price"
            )
        
        with preview_col2:
            preview_amount = st.number_input(
                "Dollar Amount",
                min_value=min_amount,
                max_value=max_amount,
                value=250.0,
                step=10.0,
                key="fractional_preview_amount"
            )
            
            # Calculate preview
            preview_qty = preview_amount / preview_price
            preview_cost = preview_qty * preview_price
            
            st.markdown(f"""
            **Calculated Quantity:**
            - **Shares**: {preview_qty:.4f}
            - **Actual Cost**: ${preview_cost:.2f}
            - **Is Fractional**: {'✅ Yes' if preview_qty % 1 != 0 else '❌ No (whole share)'}
            """)
            
            # Show if custom amount exists
            if preview_symbol and preview_symbol in custom_amounts:
                custom_amt = custom_amounts[preview_symbol]
                custom_qty = custom_amt / preview_price
                st.info(f"**Custom Amount Set**: ${custom_amt:.2f} = {custom_qty:.4f} shares")


def display_fractional_positions():
    """
    Display current fractional positions with ROI metrics
    """
    st.markdown("### 📊 Fractional Positions")
    st.markdown("Current fractional share positions with detailed metrics")
    
    try:
        from services.fractional_share_manager import get_fractional_share_manager
        manager = get_fractional_share_manager()
    except Exception as e:
        st.error(f"Error loading Fractional Share Manager: {e}")
        logger.error("Fractional Share Manager loading error: {}", str(e), exc_info=True)
        return
    
    # This would integrate with broker to get actual positions
    st.info("🚧 Position display will show actual fractional positions from your broker account")
    
    # Example of how positions would be displayed
    with st.expander("📋 Position Display Format (Example)"):
        st.markdown("""
        ```
        📊 NVDA - 0.5 shares
        Entry: $500.00 → Current: $520.00 (+4.0%)
        Cost: $250.00 → Value: $260.00
        Unrealized P&L: +$10.00 (+4.0%) 🟢
        
        📊 AAPL - 2.85 shares
        Entry: $175.44 → Current: $180.00 (+2.6%)
        Cost: $500.00 → Value: $513.00
        Unrealized P&L: +$13.00 (+2.6%) 🟢
        ```
        """)


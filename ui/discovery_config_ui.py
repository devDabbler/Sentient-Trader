"""
Stock Discovery Configuration UI Component

Streamlit UI for configuring stock discovery in the Service Control Panel.
Allows toggling discovery on/off and configuring which discovery modes are active.

Usage in service_control_panel.py:
    from ui.discovery_config_ui import render_discovery_config_panel
    render_discovery_config_panel()
"""

import streamlit as st
from pathlib import Path
from typing import Dict

# Add path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from windows_services.runners.service_discovery_config import (
    load_discovery_config,
    save_discovery_config,
    toggle_discovery,
    toggle_discovery_mode,
    set_mode_universe_size,
    get_active_modes,
    get_mode_descriptions,
)


def render_discovery_config_panel():
    """Render stock discovery configuration panel in Streamlit"""
    
    st.header("🔍 Stock Discovery Universe")
    st.write("""
    Automatically discover trading opportunities outside your watchlist using the same scanner categories from the main app:
    - **Mega Caps** - Options-friendly large caps (AAPL, MSFT, etc.)
    - **High Beta Tech** - Volatile tech stocks (PLTR, SOFI, etc.)
    - **Momentum/Meme** - High momentum and meme stocks
    - **EV/Clean Energy** - Electric vehicle and clean energy stocks
    - **Crypto-Related** - Stocks tied to crypto (MARA, RIOT, COIN)
    - **AI Stocks** - Artificial intelligence related stocks
    - **Biotech** - Biotechnology and pharma stocks
    - **Financial** - Banks and financial services
    - **Energy** - Oil and gas stocks
    - **High IV Options** - High implied volatility for options trading
    - **Penny Stocks** - Low-priced stocks under $5
    """)
    
    # Load current config
    config = load_discovery_config()
    discovery_enabled = config['enabled']
    
    # Master toggle
    st.subheader("Master Control")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_enabled = st.toggle(
            "🚀 Enable Stock Discovery",
            value=discovery_enabled,
            help="Toggle between watchlist-only and discovery scanning"
        )
    
    with col2:
        if new_enabled != discovery_enabled:
            toggle_discovery(new_enabled)
            discovery_enabled = new_enabled
            st.success(f"Discovery {'enabled' if new_enabled else 'disabled'}")
    
    if not discovery_enabled:
        st.info("💡 Discovery is currently disabled. Enable it above to scan stocks outside your watchlist.")
        return
    
    st.divider()
    
    # Discovery modes configuration
    st.subheader("📊 Discovery Modes")
    st.write("Enable/disable specific discovery modes and set their scope:")
    
    modes = config['modes']
    modes_changed = False
    
    # Create columns for better layout
    for mode_name, mode_config in modes.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{mode_name.replace('_', ' ').title()}**")
            st.caption(mode_config['description'])
        
        with col2:
            # Toggle for mode
            old_enabled = mode_config['enabled']
            new_enabled = st.toggle(
                f"Enable {mode_name}",
                value=old_enabled,
                key=f"mode_{mode_name}",
                label_visibility="collapsed"
            )
            
            if new_enabled != old_enabled:
                toggle_discovery_mode(mode_name, new_enabled)
                modes_changed = True
        
        with col3:
            # Universe size slider
            old_size = mode_config['max_universe_size']
            new_size = st.slider(
                f"Size for {mode_name}",
                min_value=10,
                max_value=100,
                value=old_size,
                step=5,
                key=f"size_{mode_name}",
                label_visibility="collapsed"
            )
            
            if new_size != old_size:
                set_mode_universe_size(mode_name, new_size)
                modes_changed = True
        
        st.divider()
    
    if modes_changed:
        st.info("✅ Discovery configuration updated")
    
    # Summary
    st.subheader("📈 Discovery Summary")
    
    active_modes = get_active_modes()
    active_count = sum(1 for v in active_modes.values() if v)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Status", "🚀 Active" if discovery_enabled else "⏸️ Inactive")
    
    with col2:
        st.metric("Active Modes", active_count)
    
    with col3:
        total_universe = sum(
            modes[name]['max_universe_size']
            for name, enabled in active_modes.items()
            if enabled
        )
        st.metric("Max Universe Size", total_universe)
    
    with col4:
        # Estimate of expanded scan
        watchlist_est = 43  # Typical watchlist size
        total_est = watchlist_est + total_universe
        st.metric("Est. Total Scan", total_est)
    
    st.info(
        f"""
        **How it works:**
        - Stock Monitor scans your {watchlist_est} watchlist tickers regularly
        - With discovery enabled, it adds up to {total_universe} additional tickers from enabled modes
        - Each discovered ticker gets the same multi-factor analysis as your watchlist
        - Opportunities found from discovery are tagged with the source mode
        - Discovery runs once per scan cycle, then caches results for efficiency
        """
    )


def render_discovery_status():
    """Render a simple discovery status indicator"""
    config = load_discovery_config()
    discovery_enabled = config['enabled']
    active_modes = get_active_modes()
    active_count = sum(1 for v in active_modes.values() if v)
    
    if discovery_enabled and active_count > 0:
        st.success(f"🔍 Discovery active ({active_count} modes)")
    else:
        st.info("🔍 Discovery disabled - using watchlist only")


if __name__ == "__main__":
    # Test UI
    st.set_page_config(page_title="Discovery Config", layout="wide")
    render_discovery_config_panel()


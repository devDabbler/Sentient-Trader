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
    get_scan_mode,
    set_scan_mode,
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
    current_scan_mode = config.get('scan_mode', 'watchlist_only')
    
    # ============================================================
    # SCAN MODE SELECTOR (Main Control)
    # ============================================================
    st.subheader("🎯 Scan Mode")
    st.write("Choose what the Stock Monitor should scan:")
    
    scan_mode_options = {
        'watchlist_only': '📋 Watchlist Only - Scan only tickers in your watchlist',
        'discovery_only': '🔍 Discovery Only - Scan only discovered stocks (no watchlist)',
        'both': '🚀 Both - Scan watchlist AND discovered stocks'
    }
    
    # Radio buttons for scan mode
    selected_mode = st.radio(
        "Select scan mode:",
        options=list(scan_mode_options.keys()),
        format_func=lambda x: scan_mode_options[x],
        index=list(scan_mode_options.keys()).index(current_scan_mode),
        key="scan_mode_radio",
        horizontal=False
    )
    
    # Apply scan mode change
    if selected_mode != current_scan_mode:
        set_scan_mode(selected_mode)
        st.success(f"✅ Scan mode changed to: {scan_mode_options[selected_mode].split(' - ')[0]}")
        st.rerun()
    
    # Show appropriate info based on mode
    if selected_mode == 'watchlist_only':
        st.info("💡 **Watchlist Only Mode:** The monitor will only scan tickers from your watchlist (My Tickers). Discovery categories below are ignored.")
        return
    
    st.divider()
    
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
    
    # Display scan mode status
    mode_labels = {
        'watchlist_only': '📋 Watchlist Only',
        'discovery_only': '🔍 Discovery Only',
        'both': '🚀 Both'
    }
    
    with col1:
        st.metric("Scan Mode", mode_labels.get(selected_mode, '📋 Watchlist'))
    
    with col2:
        st.metric("Active Discovery Modes", active_count)
    
    with col3:
        total_universe = sum(
            modes[name]['max_universe_size']
            for name, enabled in active_modes.items()
            if enabled
        )
        st.metric("Max Discovery Size", total_universe)
    
    with col4:
        # Estimate of expanded scan
        watchlist_est = 43  # Typical watchlist size
        if selected_mode == 'watchlist_only':
            total_est = watchlist_est
        elif selected_mode == 'discovery_only':
            total_est = total_universe
        else:  # both
            total_est = watchlist_est + total_universe
        st.metric("Est. Total Scan", total_est)
    
    # Mode-specific info
    if selected_mode == 'discovery_only':
        st.info(
            f"""
            **Discovery Only Mode:**
            - Stock Monitor scans ONLY discovered stocks (no watchlist)
            - Up to {total_universe} tickers from {active_count} enabled discovery modes
            - Each discovered ticker gets full multi-factor analysis
            - Opportunities are tagged with the source mode
            - Discovery caches results for 30 minutes for efficiency
            """
        )
    else:  # both
        st.info(
            f"""
            **Watchlist + Discovery Mode:**
            - Stock Monitor scans your ~{watchlist_est} watchlist tickers
            - PLUS up to {total_universe} additional tickers from {active_count} discovery modes
            - Each ticker gets the same multi-factor analysis
            - Opportunities found from discovery are tagged with the source mode
            - Discovery runs once per scan cycle, then caches results for efficiency
            """
        )


def render_discovery_status():
    """Render a simple discovery status indicator"""
    config = load_discovery_config()
    scan_mode = config.get('scan_mode', 'watchlist_only')
    active_modes = get_active_modes()
    active_count = sum(1 for v in active_modes.values() if v)
    
    if scan_mode == 'watchlist_only':
        st.info("📋 Scan Mode: Watchlist only")
    elif scan_mode == 'discovery_only':
        if active_count > 0:
            st.success(f"🔍 Scan Mode: Discovery only ({active_count} modes)")
        else:
            st.warning("🔍 Discovery only mode but no modes enabled!")
    else:  # both
        if active_count > 0:
            st.success(f"🚀 Scan Mode: Watchlist + Discovery ({active_count} modes)")
        else:
            st.info("📋 Scan Mode: Both (but no discovery modes enabled)")


if __name__ == "__main__":
    # Test UI
    st.set_page_config(page_title="Discovery Config", layout="wide")
    render_discovery_config_panel()


"""
Unified Watchlist UI Components

Shared UI patterns for both stock and crypto watchlists to ensure consistency.
Implements user-preferred patterns like multiselect with Select All/Clear All buttons.
"""

import streamlit as st
from typing import List, Dict, Optional, Callable
from loguru import logger


def display_ticker_multiselect(
    tickers: List[str],
    key_prefix: str = "ticker",
    label: str = "Select Tickers",
    default_all: bool = False,
    max_selections: Optional[int] = None,
    help_text: str = "Select tickers to analyze"
) -> List[str]:
    """
    Display a multiselect with Select All / Clear All / Top N buttons.
    
    This is the user-preferred pattern from the Crypto Quick Trade tab.
    
    Args:
        tickers: List of available ticker symbols
        key_prefix: Prefix for session state keys to avoid conflicts
        label: Label for the multiselect
        default_all: If True, select all tickers by default
        max_selections: Optional max number of selections
        help_text: Help text for the multiselect
        
    Returns:
        List of selected ticker symbols
    """
    if not tickers:
        st.info("No tickers available. Add some to your watchlist first.")
        return []
    
    # Initialize session state for selections
    state_key = f"{key_prefix}_selected_tickers"
    if state_key not in st.session_state:
        if default_all:
            st.session_state[state_key] = tickers[:max_selections] if max_selections else tickers.copy()
        else:
            st.session_state[state_key] = []
    
    # Quick selection buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("✅ Select All", key=f"{key_prefix}_select_all", use_container_width=True):
            if max_selections:
                st.session_state[state_key] = tickers[:max_selections]
            else:
                st.session_state[state_key] = tickers.copy()
            st.rerun()
    
    with col2:
        if st.button("❌ Clear All", key=f"{key_prefix}_clear_all", use_container_width=True):
            st.session_state[state_key] = []
            st.rerun()
    
    with col3:
        top_n = min(10, len(tickers))
        if st.button(f"🔝 Top {top_n}", key=f"{key_prefix}_top_n", use_container_width=True):
            st.session_state[state_key] = tickers[:top_n]
            st.rerun()
    
    with col4:
        st.caption(f"📊 {len(st.session_state[state_key])}/{len(tickers)} selected")
    
    # Multiselect widget
    selected = st.multiselect(
        label,
        options=tickers,
        default=st.session_state[state_key],
        key=f"{key_prefix}_multiselect",
        help=help_text,
        max_selections=max_selections
    )
    
    # Sync back to session state
    st.session_state[state_key] = selected
    
    return selected


def display_analysis_mode_selector(
    key_prefix: str = "analysis",
    include_ai: bool = True,
    include_multi_config: bool = True
) -> str:
    """
    Display analysis mode selector with consistent options.
    
    Args:
        key_prefix: Prefix for session state keys
        include_ai: Include AI analysis option
        include_multi_config: Include multi-config option
        
    Returns:
        Selected analysis mode
    """
    modes = ["📊 Single Analysis"]
    
    if include_ai:
        modes.append("🤖 AI-Enhanced Analysis")
    
    modes.append("📋 Bulk Analysis")
    
    if include_multi_config:
        modes.append("🎯 Multi-Config Analysis")
    
    selected_mode = st.radio(
        "Analysis Mode",
        modes,
        horizontal=True,
        key=f"{key_prefix}_mode_selector",
        help="Choose how to analyze your tickers"
    )
    
    return selected_mode


def display_trading_style_selector(
    key_prefix: str = "style",
    default_styles: Optional[List[str]] = None
) -> List[str]:
    """
    Display trading style checkboxes with consistent options.
    
    Args:
        key_prefix: Prefix for session state keys
        default_styles: Default selected styles
        
    Returns:
        List of selected trading styles
    """
    if default_styles is None:
        default_styles = ["SWING", "DAY_TRADE"]
    
    st.write("**Trading Styles**")
    
    col1, col2, col3 = st.columns(3)
    
    selected_styles = []
    
    with col1:
        if st.checkbox("Swing Trading (3:1 R:R)", value="SWING" in default_styles, key=f"{key_prefix}_swing"):
            selected_styles.append("SWING")
    
    with col2:
        if st.checkbox("Day Trading (2:1 R:R)", value="DAY_TRADE" in default_styles, key=f"{key_prefix}_day"):
            selected_styles.append("DAY_TRADE")
    
    with col3:
        if st.checkbox("Scalping (1.5:1 R:R)", value="SCALP" in default_styles, key=f"{key_prefix}_scalp"):
            selected_styles.append("SCALP")
    
    if not selected_styles:
        st.warning("⚠️ Select at least one trading style")
    
    return selected_styles


def display_position_risk_inputs(
    key_prefix: str = "config",
    default_positions: str = "1000,2000,5000",
    default_risks: str = "1.0,2.0,3.0"
) -> tuple:
    """
    Display position size and risk level inputs.
    
    Args:
        key_prefix: Prefix for session state keys
        default_positions: Default position sizes (comma-separated)
        default_risks: Default risk levels (comma-separated)
        
    Returns:
        Tuple of (position_sizes: List[float], risk_levels: List[float])
    """
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Position Sizes (USD)**")
        position_input = st.text_input(
            "Comma-separated values",
            value=default_positions,
            help="e.g., 1000,2000,5000",
            key=f"{key_prefix}_positions"
        )
    
    with col2:
        st.write("**Risk Levels (%)**")
        risk_input = st.text_input(
            "Comma-separated values",
            value=default_risks,
            help="e.g., 1.0,2.0,3.0",
            key=f"{key_prefix}_risks"
        )
    
    # Parse inputs
    try:
        position_sizes = [float(x.strip()) for x in position_input.split(",") if x.strip()]
        risk_levels = [float(x.strip()) for x in risk_input.split(",") if x.strip()]
    except ValueError:
        st.error("⚠️ Invalid format. Use comma-separated numbers (e.g., 1000,2000,5000)")
        return [], []
    
    if not position_sizes or not risk_levels:
        st.error("⚠️ Position sizes and risk levels cannot be empty")
        return [], []
    
    return position_sizes, risk_levels


def display_filter_sort_controls(
    key_prefix: str = "filter",
    action_options: Optional[List[str]] = None,
    sort_options: Optional[List[str]] = None,
    include_confidence_filter: bool = True
) -> Dict:
    """
    Display consistent filter and sort controls.
    
    Args:
        key_prefix: Prefix for session state keys
        action_options: Available action filter options
        sort_options: Available sort options
        include_confidence_filter: Include confidence level filter
        
    Returns:
        Dict with filter settings
    """
    if action_options is None:
        action_options = ["All", "ENTER_NOW", "WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT", "PLACE_LIMIT_ORDER", "DO_NOT_ENTER"]
    
    if sort_options is None:
        sort_options = ["Default", "Confidence (Highest First)", "Analysis Date (Newest First)", "Score (Highest First)"]
    
    filters = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        filters['action'] = st.selectbox(
            "Filter by Action",
            options=action_options,
            key=f"{key_prefix}_action_filter",
            help="Show only items with a specific recommendation"
        )
    
    with col2:
        filters['sort_by'] = st.selectbox(
            "Sort By",
            options=sort_options,
            key=f"{key_prefix}_sort_by",
            help="Sort the list based on selected criteria"
        )
    
    if include_confidence_filter:
        filters['min_confidence'] = st.slider(
            "Min Confidence %",
            min_value=0,
            max_value=100,
            value=0,
            key=f"{key_prefix}_min_confidence"
        )
    
    return filters


def display_analysis_summary_metrics(
    results: List[Dict],
    key_prefix: str = "summary"
):
    """
    Display summary metrics for analysis results.
    
    Args:
        results: List of analysis result dictionaries
        key_prefix: Prefix for keys
    """
    if not results:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(results)
    
    with col1:
        st.metric("Total Analyzed", total)
    
    with col2:
        enter_now = sum(1 for r in results if r.get('Action') == 'ENTER_NOW' or r.get('action') == 'ENTER_NOW')
        pct = (enter_now / total * 100) if total > 0 else 0
        st.metric("ENTER NOW", f"{enter_now} ({pct:.1f}%)")
    
    with col3:
        confidences = [r.get('Confidence', r.get('confidence', 0)) or 0 for r in results]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")
    
    with col4:
        best_conf = max(confidences) if confidences else 0
        st.metric("Best Confidence", f"{best_conf:.1f}%")


def display_action_badge(action: str) -> str:
    """
    Get consistent action badge with emoji.
    
    Args:
        action: Action string (ENTER_NOW, WAIT_FOR_PULLBACK, etc.)
        
    Returns:
        Formatted action string with emoji
    """
    if action == "ENTER_NOW":
        return f"🟢 {action}"
    elif action in ["WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT", "PLACE_LIMIT_ORDER"]:
        return f"🟡 {action}"
    elif action == "DO_NOT_ENTER":
        return f"🔴 {action}"
    elif action == "ERROR":
        return f"⚫ {action}"
    else:
        return f"⚪ {action}"


def display_fast_mode_toggle(key_prefix: str = "fast") -> bool:
    """
    Display fast mode toggle for LLM operations.
    
    Args:
        key_prefix: Prefix for session state key
        
    Returns:
        True if fast mode is enabled
    """
    return st.toggle(
        "⚡ Fast Mode",
        value=True,
        help="Use fast cloud API instead of local Ollama (much faster for bulk operations)",
        key=f"{key_prefix}_mode_toggle"
    )


def display_broker_connection_warning(broker_client, feature_name: str = "this feature"):
    """
    Display consistent broker connection warning.
    
    Args:
        broker_client: The broker client (or None if not connected)
        feature_name: Name of the feature requiring broker
    """
    if not broker_client:
        st.warning(f"⚠️ **Broker Not Connected** - {feature_name} requires a broker connection")
        with st.expander("🔍 How to connect"):
            st.write("**Options:**")
            st.write("1. **Tradier** - Go to Tradier tab and enter API credentials")
            st.write("2. **IBKR** - Go to IBKR tab and connect to TWS/Gateway")
            st.write("")
            st.write("Once connected, return here to use this feature.")
        return False
    return True


def display_llm_connection_warning(llm_analyzer, feature_name: str = "AI analysis"):
    """
    Display consistent LLM connection warning.
    
    Args:
        llm_analyzer: The LLM analyzer (or None if not configured)
        feature_name: Name of the feature requiring LLM
    """
    if not llm_analyzer:
        st.warning(f"⚠️ **LLM Not Configured** - {feature_name} requires an LLM API key")
        with st.expander("🔍 How to configure"):
            st.write("**Options:**")
            st.write("1. **OpenRouter** - Set OPENROUTER_API_KEY in .env")
            st.write("2. **Groq** - Set GROQ_API_KEY in .env (faster)")
            st.write("3. **Local Ollama** - Run Ollama locally with a model")
            st.write("")
            st.write("Restart the app after configuring.")
        return False
    return True

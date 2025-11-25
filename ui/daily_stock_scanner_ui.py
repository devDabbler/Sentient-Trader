"""
Daily Stock/Options Scanner UI

Progressive tiered scanning workflow for stocks/options:
- Tier 1: Quick Filter (lightweight, fast)
- Tier 2: Technical Analysis (RSI, MACD, EMAs)
- Tier 3: Deep Analysis (AI review, options analysis)
- Active Monitors: Add winners to live monitoring
"""

import streamlit as st
from loguru import logger
from datetime import datetime
from typing import List, Dict, Optional


def add_to_monitors(result: Dict, tier: int = 2):
    """Add a stock to active monitors"""
    if 'stock_active_monitors' not in st.session_state:
        st.session_state.stock_active_monitors = []
    
    # Check if already monitoring
    existing = [m['ticker'] for m in st.session_state.stock_active_monitors]
    if result['ticker'] not in existing:
        monitor_entry = {
            'ticker': result['ticker'],
            'score': result.get('score', 0),
            'price': result.get('price', 0),
            'tier': tier,
            'signals': result.get('signals', []),
            'added_at': datetime.now().strftime('%H:%M'),
            'stop_loss': result.get('stop_loss'),
            'take_profit': result.get('take_profit')
        }
        st.session_state.stock_active_monitors.append(monitor_entry)
        logger.info(f"Added {result['ticker']} to monitors (tier {tier})")


def render_daily_stock_scanner():
    """Main entry point for the Daily Stock/Options Scanner."""
    st.subheader("📊 Daily Stock/Options Scanner")
    st.markdown("""
    **Progressive Scanning Workflow** - Start light, go deep on winners:
    - 🏃 **Tier 1: Quick Filter** 100+ stocks (price, volume, momentum only)
    - 📊 **Tier 2: Technical Analysis** on top 10-25 (RSI, MACD, EMAs)
    - 🎯 **Tier 3: Deep dive** on selected (full strategy + AI review)
    - 🤖 **Monitor:** Add winners to live monitoring
    """)
    
    # Initialize scanner
    if 'stock_tiered_scanner' not in st.session_state:
        try:
            from services.stock_tiered_scanner import TieredStockScanner
            st.session_state.stock_tiered_scanner = TieredStockScanner(use_ai=True)
            logger.info("✅ Stock tiered scanner initialized")
        except Exception as e:
            logger.error(f"Error initializing scanner: {e}")
            st.error(f"Failed to initialize scanner: {e}")
            return
    
    scanner = st.session_state.stock_tiered_scanner
    
    # Create tabs for each tier
    tier_tabs = st.tabs([
        "🏃 Tier 1: Quick Filter",
        "📊 Tier 2: Technical Analysis", 
        "🎯 Tier 3: Deep Analysis",
        "🤖 Active Monitors"
    ])
    
    with tier_tabs[0]:
        display_tier1_quick_filter(scanner)
    with tier_tabs[1]:
        display_tier2_medium_analysis(scanner)
    with tier_tabs[2]:
        display_tier3_deep_analysis(scanner)
    with tier_tabs[3]:
        display_active_monitors()


def display_tier1_quick_filter(scanner):
    """Tier 1: Quick filter interface"""
    st.subheader("🏃 Tier 1: Quick Filter")
    st.markdown("Lightweight scan using **only** price, volume, and momentum.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scan_source = st.selectbox(
            "Scan Source",
            ["All Categories (100+ stocks)", "🎯 Mega Caps (Options-friendly)",
             "🚀 High Beta Tech", "🎮 Meme/Momentum Stocks", "⚡ EV/Clean Energy",
             "₿ Crypto-Related Stocks", "🤖 AI Stocks", "💊 Biotech",
             "💰 Penny Stocks (<$5)", "📈 High IV (Options)", "⭐ My Watchlist"],
            key="stock_tier1_scan_source"
        )
    
    with col2:
        max_results = st.slider("Max Results", 10, 50, 25, key="stock_tier1_max_results")
    
    # Custom tickers
    with st.expander("📝 Custom Tickers (Optional)"):
        custom_tickers_input = st.text_input(
            "Comma-separated tickers:", placeholder="AAPL, TSLA, NVDA",
            key="stock_custom_tickers"
        )
    
    if st.button("🚀 Start Quick Scan", key="stock_tier1_scan", type="primary"):
        with st.spinner(f"Scanning {scan_source}..."):
            tickers = _get_tickers_for_source(scanner, scan_source, custom_tickers_input)
            if not tickers:
                st.error("No tickers found")
                return
            
            try:
                results = scanner.tier1_quick_filter(tickers, max_results=max_results)
                st.session_state.stock_tier1_results = results
                st.session_state.stock_tier1_timestamp = datetime.now()
                st.success(f"✅ Found {len(results)} promising stocks!")
            except Exception as e:
                st.error(f"Scan failed: {e}")
                return
    
    # Display results
    results = st.session_state.get('stock_tier1_results', [])
    if results:
        _display_tier1_results(results)


def _get_tickers_for_source(scanner, scan_source: str, custom_input: str) -> List[str]:
    """Get tickers based on scan source selection"""
    if custom_input:
        return [t.strip().upper() for t in custom_input.split(',')]
    
    source_map = {
        "All Categories (100+ stocks)": lambda: scanner.get_all_scan_tickers(),
        "🎯 Mega Caps (Options-friendly)": lambda: scanner.get_tickers_by_category('mega_cap'),
        "🚀 High Beta Tech": lambda: scanner.get_tickers_by_category('high_beta_tech'),
        "🎮 Meme/Momentum Stocks": lambda: scanner.get_tickers_by_category('momentum'),
        "⚡ EV/Clean Energy": lambda: scanner.get_tickers_by_category('ev_energy'),
        "₿ Crypto-Related Stocks": lambda: scanner.get_tickers_by_category('crypto_related'),
        "🤖 AI Stocks": lambda: scanner.get_tickers_by_category('ai_stocks'),
        "💊 Biotech": lambda: scanner.get_tickers_by_category('biotech'),
        "💰 Penny Stocks (<$5)": lambda: scanner.get_tickers_by_category('penny_stocks'),
        "📈 High IV (Options)": lambda: scanner.get_tickers_by_category('high_iv'),
    }
    
    if scan_source == "⭐ My Watchlist":
        if hasattr(st.session_state, 'ticker_manager'):
            return st.session_state.ticker_manager.get_all_tickers()
        return []
    
    return source_map.get(scan_source, lambda: [])()


def _display_tier1_results(results: List[Dict]):
    """Display Tier 1 scan results"""
    timestamp = st.session_state.get('stock_tier1_timestamp', datetime.now())
    st.markdown(f"**Last scan:** {timestamp.strftime('%H:%M:%S')}")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Found", len(results))
    with col2:
        avg = sum(r['score'] for r in results) / len(results) if results else 0
        st.metric("Avg Score", f"{avg:.1f}")
    with col3:
        st.metric("Gainers", sum(1 for r in results if r.get('change_pct', 0) > 0))
    with col4:
        st.metric("High Vol", sum(1 for r in results if r.get('volume_ratio', 0) > 2))
    
    for i, result in enumerate(results[:15], 1):
        emoji = "🟢" if result.get('change_pct', 0) > 0 else "🔴"
        with st.expander(f"#{i} {emoji} **{result['ticker']}** - Score: {result['score']:.0f} | ${result['price']:.2f}", expanded=(i <= 3)):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Price", f"${result['price']:.2f}", f"{result['change_pct']:+.1f}%")
            with c2:
                st.metric("Vol Ratio", f"{result.get('volume_ratio', 0):.1f}x")
            with c3:
                st.metric("Score", f"{result['score']:.0f}")
            
            if st.button(f"⭐ Watch", key=f"w1_{result['ticker']}"):
                if hasattr(st.session_state, 'ticker_manager'):
                    st.session_state.ticker_manager.add_ticker(result['ticker'], "Scanner")
                    st.success(f"Added {result['ticker']}")
    
    if st.button("📤 Send All to Tier 2", key="send_tier2"):
        st.session_state.stock_tier1_results = results
        st.success(f"Ready for Tier 2!")


def display_tier2_medium_analysis(scanner):
    """Tier 2: Technical analysis interface"""
    st.subheader("📊 Tier 2: Technical Analysis")
    
    tier1_results = st.session_state.get('stock_tier1_results', [])
    if not tier1_results:
        st.warning("⚠️ Run Tier 1 Quick Filter first")
        return
    
    st.info(f"📥 {len(tier1_results)} candidates from Tier 1")
    
    col1, col2 = st.columns(2)
    with col1:
        min_score = st.slider("Min Score", 20, 80, 35, key="stock_t2_min")
    with col2:
        top_n = st.slider("Analyze Top N", 1, len(tier1_results), min(20, len(tier1_results)), key="stock_t2_n")
    
    if st.button("📈 Start Technical Analysis", key="stock_t2_btn", type="primary"):
        scanner.tier2_min_score = min_score
        with st.spinner(f"Analyzing {top_n} stocks..."):
            results = scanner.tier2_medium_analysis(tier1_results[:top_n])
            st.session_state.stock_tier2_results = results
            st.session_state.stock_tier2_timestamp = datetime.now()
            st.success(f"✅ {len(results)} stocks passed!")
    
    results = st.session_state.get('stock_tier2_results', [])
    if results:
        _display_tier2_results(results)


def _display_tier2_results(results: List[Dict]):
    """Display Tier 2 results"""
    for i, r in enumerate(results[:10], 1):
        trend = r.get('trend', 'UNKNOWN')
        emoji = "📈" if "UP" in trend else "📉" if "DOWN" in trend else "➡️"
        
        with st.expander(f"#{i} {emoji} **{r['ticker']}** - Score: {r['score']:.0f} | {trend}", expanded=(i <= 3)):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Price", f"${r['price']:.2f}", f"{r['change_pct']:+.1f}%")
            with c2:
                st.metric("RSI", f"{r.get('rsi', 50):.1f}")
            with c3:
                st.metric("Vol", f"{r.get('volume_ratio', 1):.1f}x")
            with c4:
                st.metric("Score", f"{r['score']:.0f}")
            
            signals = r.get('signals', [])
            if signals:
                st.markdown("**Signals:** " + " | ".join(signals[:3]))
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"🤖 Monitor", key=f"m2_{r['ticker']}"):
                    add_to_monitors(r, tier=2)
                    st.success(f"Added {r['ticker']} to monitors!")
            with c2:
                if st.button(f"⭐ Watch", key=f"w2_{r['ticker']}"):
                    if hasattr(st.session_state, 'ticker_manager'):
                        st.session_state.ticker_manager.add_ticker(r['ticker'], "Scanner T2")


def display_tier3_deep_analysis(scanner):
    """Tier 3: Deep analysis interface"""
    st.subheader("🎯 Tier 3: Deep Analysis + AI Review")
    
    tier2_results = st.session_state.get('stock_tier2_results', [])
    if not tier2_results:
        st.warning("⚠️ Run Tier 2 Technical Analysis first")
        return
    
    st.info(f"📥 {len(tier2_results)} candidates from Tier 2")
    
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Analyze Top N", 1, min(10, len(tier2_results)), min(5, len(tier2_results)), key="stock_t3_n")
    with col2:
        include_options = st.checkbox("Include Options Analysis", True, key="stock_t3_opt")
    
    if st.button("🎯 Run Deep Analysis", key="stock_t3_btn", type="primary"):
        with st.spinner(f"Deep analyzing {top_n} stocks..."):
            ai_reviewer = st.session_state.get('ai_scanner')
            results = scanner.tier3_deep_analysis(tier2_results[:top_n], include_options, ai_reviewer)
            st.session_state.stock_tier3_results = results
            st.success(f"✅ {len(results)} stocks analyzed!")
    
    results = st.session_state.get('stock_tier3_results', [])
    if results:
        for i, r in enumerate(results, 1):
            ready = "✅" if r.get('ready_for_monitoring') else "⚠️"
            with st.expander(f"#{i} {ready} **{r['ticker']}** - Score: {r['score']:.0f}", expanded=(i <= 3)):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Price", f"${r['price']:.2f}")
                    st.metric("Support", f"${r.get('support', 0):.2f}")
                with c2:
                    st.metric("Score", f"{r['score']:.0f}")
                    st.metric("Resistance", f"${r.get('resistance', 0):.2f}")
                with c3:
                    st.metric("Stop Loss", f"${r.get('stop_loss', 0):.2f}")
                    st.metric("Take Profit", f"${r.get('take_profit', 0):.2f}")
                
                if r.get('ready_for_monitoring'):
                    if st.button(f"🤖 Add Monitor", key=f"m3_{r['ticker']}", type="primary"):
                        add_to_monitors(r, tier=3)
                        st.success(f"Added {r['ticker']}!")


def display_active_monitors():
    """Display active stock monitors"""
    st.subheader("🤖 Active Stock Monitors")
    
    monitors = st.session_state.get('stock_active_monitors', [])
    
    if not monitors:
        st.info("No active monitors. Add stocks from Tier 2 or Tier 3.")
        col1, col2 = st.columns([3, 1])
        with col1:
            ticker = st.text_input("Quick Add Ticker", key="quick_mon")
        with col2:
            if st.button("Add", key="quick_add"):
                if ticker:
                    add_to_monitors({'ticker': ticker.upper(), 'score': 0, 'price': 0})
                    st.rerun()
        return
    
    st.markdown(f"**{len(monitors)} Active Monitors**")
    
    for i, m in enumerate(monitors):
        emoji = "🎯" if m.get('tier') == 3 else "📊" if m.get('tier') == 2 else "🏃"
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"{emoji} **{m['ticker']}**")
        with col2:
            st.write(f"Score: {m.get('score', 0):.0f}")
        with col3:
            st.write(f"Added: {m.get('added_at', 'N/A')}")
        with col4:
            if st.button("❌", key=f"rm_{i}"):
                monitors.pop(i)
                st.session_state.stock_active_monitors = monitors
                st.rerun()
    
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔗 Send to Stock Monitor Service"):
            _sync_to_monitor_service(monitors)
    with col2:
        if st.button("🗑️ Clear All"):
            st.session_state.stock_active_monitors = []
            st.rerun()


def _sync_to_monitor_service(monitors: List[Dict]):
    """Sync monitors to the stock informational monitor service"""
    try:
        from services.stock_informational_monitor import get_stock_informational_monitor
        
        tickers = [m['ticker'] for m in monitors]
        if not tickers:
            st.warning("No tickers to sync")
            return
        
        # Get or create monitor service
        monitor_service = get_stock_informational_monitor()
        
        # Use the sync method
        monitor_service.sync_from_tiered_scanner(monitors)
        
        st.success(f"✅ Synced {len(tickers)} tickers to monitor service!")
        st.info(f"Monitoring: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        logger.info(f"Synced monitors: {tickers}")
        
    except Exception as e:
        st.error(f"Failed to sync: {e}")
        logger.error(f"Monitor sync error: {e}")

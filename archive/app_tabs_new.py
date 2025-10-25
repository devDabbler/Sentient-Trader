# This file contains the new tab implementations to be inserted into app.py

# ===== TAB 1: DASHBOARD =====
def render_dashboard_tab():
    """Dashboard tab with quick actions and overview"""
    st.header("🏠 Trading Dashboard")
    st.write("Your command center for options and penny stock trading")
    
    # Initialize managers
    if 'ticker_manager' not in st.session_state:
        st.session_state.ticker_manager = TickerManager()
    if 'scanner' not in st.session_state:
        st.session_state.scanner = TopTradesScanner()
    
    tm = st.session_state.ticker_manager
    scanner = st.session_state.scanner
    
    # Quick Stats Row
    st.subheader("📊 Quick Overview")
    
    stats = tm.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💾 Saved Tickers", stats.get('total_tickers', 0))
    with col2:
        st.metric("📋 Watchlists", stats.get('watchlists', 0))
    with col3:
        recent = tm.get_recent_tickers(limit=1)
        st.metric("🕐 Last Viewed", recent[0] if recent else "None")
    with col4:
        popular = tm.get_popular_tickers(limit=1)
        st.metric("🔥 Most Popular", popular[0] if popular else "None")
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("### 🎯 Analyze Stock")
        quick_ticker = st.text_input("Enter Ticker", placeholder="AAPL", key="quick_analyze")
        if st.button("🔍 Analyze Now", use_container_width=True, type="primary"):
            if quick_ticker:
                tm.record_access(quick_ticker.upper())
                tm.add_ticker(quick_ticker.upper(), ticker_type='stock')
                st.success(f"✅ Analyzing {quick_ticker.upper()}... Check Stock Intelligence tab")
                st.session_state.analyze_ticker = quick_ticker.upper()
    
    with action_col2:
        st.markdown("### 🔥 Scan Markets")
        scan_type = st.selectbox("Scan Type", ["Options Trades", "Penny Stocks"], key="quick_scan")
        if st.button("🚀 Run Scan", use_container_width=True, type="primary"):
            with st.spinner(f"Scanning for top {scan_type.lower()}..."):
                if "Options" in scan_type:
                    st.session_state.run_options_scan = True
                else:
                    st.session_state.run_penny_scan = True
                st.success(f"✅ Scan complete! Check {scan_type} tab")
    
    with action_col3:
        st.markdown("### ⭐ Quick Add")
        add_ticker = st.text_input("Ticker to Save", placeholder="TSLA", key="quick_add")
        add_type = st.selectbox("Type", ["stock", "penny_stock", "option"], key="quick_add_type")
        if st.button("💾 Save Ticker", use_container_width=True):
            if add_ticker:
                if tm.add_ticker(add_ticker.upper(), ticker_type=add_type):
                    st.success(f"✅ {add_ticker.upper()} saved!")
                else:
                    st.error("Failed to save ticker")
    
    st.divider()
    
    # Recent Activity
    st.subheader("📈 Recent Activity")
    
    recent_col1, recent_col2 = st.columns(2)
    
    with recent_col1:
        st.markdown("**🕐 Recently Viewed (Last 10)**")
        recent_tickers = tm.get_recent_tickers(limit=10)
        if recent_tickers:
            for ticker in recent_tickers:
                col_tick, col_btn = st.columns([3, 1])
                with col_tick:
                    st.write(f"• `{ticker}`")
                with col_btn:
                    if st.button("🔍", key=f"view_{ticker}_recent"):
                        st.session_state.analyze_ticker = ticker
                        tm.record_access(ticker)
        else:
            st.info("No recent activity")
    
    with recent_col2:
        st.markdown("**🔥 Most Popular (Top 10)**")
        popular_tickers = tm.get_popular_tickers(limit=10)
        if popular_tickers:
            for ticker in popular_tickers:
                ticker_info = tm.get_ticker(ticker)
                access_count = ticker_info.get('access_count', 0) if ticker_info else 0
                col_tick, col_btn = st.columns([3, 1])
                with col_tick:
                    st.write(f"• `{ticker}` ({access_count} views)")
                with col_btn:
                    if st.button("🔍", key=f"view_{ticker}_popular"):
                        st.session_state.analyze_ticker = ticker
                        tm.record_access(ticker)
        else:
            st.info("No popular tickers yet")
    
    st.divider()
    
    # Quick Tips
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        **Dashboard Features:**
        - 🔥 **Top Options/Penny Stocks**: Scan pre-configured lists for best opportunities
        - ⭐ **My Tickers**: Manage your saved tickers and watchlists
        - 🔍 **Stock Intelligence**: Deep dive analysis on any ticker
        - 🎯 **Strategy Advisor**: Get AI-powered strategy recommendations
        
        **Pro Tips:**
        - Save frequently viewed tickers for quick access
        - Create watchlists to organize stocks by strategy or sector
        - Run daily scans to find fresh opportunities
        - Review recent activity to track your research
        """)


# ===== TAB 2: TOP OPTIONS TRADES =====
def render_top_options_tab():
    """Top Options Trades scanner"""
    st.header("🔥 Top Options Trading Opportunities")
    st.write("Scan for the best high-volume, high-momentum options trades")
    
    if 'scanner' not in st.session_state:
        st.session_state.scanner = TopTradesScanner()
    if 'ticker_manager' not in st.session_state:
        st.session_state.ticker_manager = TickerManager()
    
    scanner = st.session_state.scanner
    tm = st.session_state.ticker_manager
    
    # Scan Controls
    col_control1, col_control2, col_control3 = st.columns([2, 1, 1])
    
    with col_control1:
        scan_count = st.slider("Number of trades to find", 5, 50, 20)
    
    with col_control2:
        auto_save = st.checkbox("Auto-save to My Tickers", value=True)
    
    with col_control3:
        st.write("")
        scan_btn = st.button("🚀 Scan Now", type="primary", use_container_width=True)
    
    # Run scan
    if scan_btn or st.session_state.get('run_options_scan', False):
        st.session_state.run_options_scan = False
        
        with st.status("🔍 Scanning markets for top options trades...", expanded=True) as status:
            st.write("📊 Analyzing volume and price action...")
            st.write("📈 Calculating volatility metrics...")
            st.write("🎯 Scoring opportunities...")
            
            trades = scanner.scan_top_options_trades(top_n=scan_count)
            
            if trades:
                status.update(label=f"✅ Found {len(trades)} opportunities!", state="complete")
                st.session_state.top_options_trades = trades
                
                # Auto-save if enabled
                if auto_save:
                    for trade in trades:
                        tm.add_ticker(trade.ticker, ticker_type='option')
            else:
                status.update(label="⚠️ No opportunities found", state="error")
    
    # Display results
    if 'top_options_trades' in st.session_state and st.session_state.top_options_trades:
        trades = st.session_state.top_options_trades
        
        # Quick insights
        insights = scanner.get_quick_insights(trades)
        
        insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
        
        with insight_col1:
            st.metric("📊 Total Opportunities", insights['total'])
        with insight_col2:
            st.metric("🎯 Avg Score", f"{insights['avg_score']:.1f}")
        with insight_col3:
            st.metric("⭐ High Confidence", insights['high_confidence'])
        with insight_col4:
            st.metric("📈 Big Movers", insights['big_movers'])
        
        st.divider()
        
        # Display trades table
        st.subheader("📋 Top Opportunities")
        
        # Convert to dataframe
        trade_data = []
        for trade in trades:
            trade_data.append({
                'Rank': trades.index(trade) + 1,
                'Ticker': trade.ticker,
                'Score': trade.score,
                'Price': f"${trade.price:.2f}",
                'Change %': f"{trade.change_pct:+.2f}%",
                'Volume Ratio': f"{trade.volume_ratio:.2f}x",
                'Confidence': trade.confidence,
                'Risk': trade.risk_level
            })
        
        df = pd.DataFrame(trade_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Detailed view
        st.subheader("🔎 Detailed Analysis")
        
        selected_ticker = st.selectbox(
            "Select ticker for details",
            [t.ticker for t in trades],
            key="options_detail_select"
        )
        
        if selected_ticker:
            selected_trade = next((t for t in trades if t.ticker == selected_ticker), None)
            
            if selected_trade:
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.metric("Score", f"{selected_trade.score:.1f}/100")
                    st.metric("Price", f"${selected_trade.price:.2f}")
                
                with detail_col2:
                    st.metric("Change", f"{selected_trade.change_pct:+.2f}%")
                    st.metric("Volume", f"{selected_trade.volume:,}")
                
                with detail_col3:
                    st.metric("Confidence", selected_trade.confidence)
                    st.metric("Risk Level", selected_trade.risk_level)
                
                st.info(f"**Why this is a good opportunity:** {selected_trade.reason}")
                
                # Actions
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    if st.button(f"💾 Save {selected_ticker} to My Tickers", use_container_width=True):
                        if tm.add_ticker(selected_ticker, ticker_type='option'):
                            st.success(f"✅ {selected_ticker} saved!")
                
                with action_col2:
                    if st.button(f"🔍 Analyze {selected_ticker} in Detail", use_container_width=True):
                        st.session_state.analyze_ticker = selected_ticker
                        tm.record_access(selected_ticker)
                        st.info("Check Stock Intelligence tab for detailed analysis")
    else:
        st.info("👆 Click 'Scan Now' to find top options trading opportunities")


# ===== TAB 3: TOP PENNY STOCKS =====
def render_top_penny_stocks_tab():
    """Top Penny Stocks scanner"""
    st.header("💰 Top Penny Stock Opportunities")
    st.write("Scan for high-potential penny stocks with strong momentum and catalysts")
    
    if 'scanner' not in st.session_state:
        st.session_state.scanner = TopTradesScanner()
    if 'ticker_manager' not in st.session_state:
        st.session_state.ticker_manager = TickerManager()
    if 'watchlist_manager' not in st.session_state:
        st.session_state.watchlist_manager = WatchlistManager()
    
    scanner = st.session_state.scanner
    tm = st.session_state.ticker_manager
    wm = st.session_state.watchlist_manager
    
    # Scan Controls
    col_control1, col_control2, col_control3 = st.columns([2, 1, 1])
    
    with col_control1:
        scan_count = st.slider("Number of stocks to find", 5, 50, 20, key="penny_scan_count")
    
    with col_control2:
        auto_save = st.checkbox("Auto-save to Penny Watchlist", value=True, key="penny_auto_save")
    
    with col_control3:
        st.write("")
        scan_btn = st.button("🚀 Scan Now", type="primary", use_container_width=True, key="penny_scan_btn")
    
    # Run scan
    if scan_btn or st.session_state.get('run_penny_scan', False):
        st.session_state.run_penny_scan = False
        
        with st.status("🔍 Scanning for top penny stocks...", expanded=True) as status:
            st.write("📊 Analyzing momentum scores...")
            st.write("💎 Evaluating valuations...")
            st.write("📰 Checking catalysts...")
            st.write("🎯 Ranking opportunities...")
            
            trades = scanner.scan_top_penny_stocks(top_n=scan_count)
            
            if trades:
                status.update(label=f"✅ Found {len(trades)} opportunities!", state="complete")
                st.session_state.top_penny_trades = trades
                
                # Auto-save if enabled
                if auto_save:
                    for trade in trades:
                        tm.add_ticker(trade.ticker, ticker_type='penny_stock')
                        # Also add to penny watchlist
                        wm.add_stock({
                            'ticker': trade.ticker,
                            'price': trade.price,
                            'pct_change': trade.change_pct,
                            'volume': trade.volume,
                            'composite_score': trade.score,
                            'confidence_level': trade.confidence
                        })
            else:
                status.update(label="⚠️ No opportunities found", state="error")
    
    # Display results
    if 'top_penny_trades' in st.session_state and st.session_state.top_penny_trades:
        trades = st.session_state.top_penny_trades
        
        # Quick insights
        insights = scanner.get_quick_insights(trades)
        
        insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
        
        with insight_col1:
            st.metric("📊 Total Opportunities", insights['total'])
        with insight_col2:
            st.metric("🎯 Avg Score", f"{insights['avg_score']:.1f}")
        with insight_col3:
            st.metric("⭐ High Confidence", insights['high_confidence'])
        with insight_col4:
            st.metric("🚀 Volume Spikes", insights['volume_spikes'])
        
        st.divider()
        
        # Display trades table
        st.subheader("📋 Top Opportunities")
        
        # Convert to dataframe
        trade_data = []
        for trade in trades:
            trade_data.append({
                'Rank': trades.index(trade) + 1,
                'Ticker': trade.ticker,
                'Score': trade.score,
                'Price': f"${trade.price:.2f}",
                'Change %': f"{trade.change_pct:+.2f}%",
                'Volume Ratio': f"{trade.volume_ratio:.2f}x",
                'Confidence': trade.confidence,
                'Risk': trade.risk_level
            })
        
        df = pd.DataFrame(trade_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Detailed view
        st.subheader("🔎 Detailed Analysis")
        
        selected_ticker = st.selectbox(
            "Select ticker for details",
            [t.ticker for t in trades],
            key="penny_detail_select"
        )
        
        if selected_ticker:
            selected_trade = next((t for t in trades if t.ticker == selected_ticker), None)
            
            if selected_trade:
                detail_col1, detail_col2, detail_col3 = st.columns(3)
                
                with detail_col1:
                    st.metric("Composite Score", f"{selected_trade.score:.1f}/100")
                    st.metric("Price", f"${selected_trade.price:.2f}")
                
                with detail_col2:
                    st.metric("Change", f"{selected_trade.change_pct:+.2f}%")
                    st.metric("Volume Ratio", f"{selected_trade.volume_ratio:.2f}x")
                
                with detail_col3:
                    st.metric("Confidence", selected_trade.confidence)
                    st.metric("Risk Level", selected_trade.risk_level)
                
                st.info(f"**Why this is a good opportunity:** {selected_trade.reason}")
                
                # Actions
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button(f"💾 Save to My Tickers", use_container_width=True, key="save_my_tickers"):
                        if tm.add_ticker(selected_ticker, ticker_type='penny_stock'):
                            st.success(f"✅ {selected_ticker} saved!")
                
                with action_col2:
                    if st.button(f"📊 Add to Penny Watchlist", use_container_width=True, key="add_penny_watchlist"):
                        result = {
                            'ticker': selected_ticker,
                            'price': selected_trade.price,
                            'composite_score': selected_trade.score,
                            'confidence_level': selected_trade.confidence
                        }
                        if wm.add_stock(result):
                            st.success(f"✅ Added to watchlist!")
                
                with action_col3:
                    if st.button(f"🔍 Deep Analysis", use_container_width=True, key="deep_analysis"):
                        st.session_state.analyze_ticker = selected_ticker
                        tm.record_access(selected_ticker)
                        st.info("Check Stock Intelligence tab")
    else:
        st.info("👆 Click 'Scan Now' to find top penny stock opportunities")


# ===== TAB 4: MY TICKERS =====
def render_my_tickers_tab():
    """My Tickers management"""
    st.header("⭐ My Saved Tickers")
    st.write("Manage your saved tickers, watchlists, and quick access lists")
    
    if 'ticker_manager' not in st.session_state:
        st.session_state.ticker_manager = TickerManager()
    
    tm = st.session_state.ticker_manager
    
    # Tabs within My Tickers
    subtab1, subtab2, subtab3 = st.tabs(["📋 All Tickers", "📊 Watchlists", "➕ Add New"])
    
    with subtab1:
        # All Tickers View
        st.subheader("📋 All Saved Tickers")
        
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            filter_type = st.selectbox("Filter by type", ["All", "stock", "penny_stock", "option"])
        
        with filter_col2:
            search_query = st.text_input("Search tickers", placeholder="Search by symbol or name")
        
        # Get tickers
        if search_query:
            tickers = tm.search_tickers(search_query)
        elif filter_type == "All":
            tickers = tm.get_all_tickers(limit=200)
        else:
            tickers = tm.get_all_tickers(ticker_type=filter_type, limit=200)
        
        if tickers:
            st.write(f"**Found {len(tickers)} tickers**")
            
            # Display as cards
            for ticker_info in tickers:
                with st.expander(f"🎯 {ticker_info['ticker']} - {ticker_info.get('name', 'N/A')}"):
                    info_col1, info_col2, info_col3 = st.columns(3)
                    
                    with info_col1:
                        st.write(f"**Type:** {ticker_info.get('type', 'N/A')}")
                        st.write(f"**Sector:** {ticker_info.get('sector', 'N/A')}")
                    
                    with info_col2:
                        st.write(f"**Access Count:** {ticker_info.get('access_count', 0)}")
                        st.write(f"**Last Accessed:** {ticker_info.get('last_accessed', 'N/A')[:10]}")
                    
                    with info_col3:
                        if ticker_info.get('tags'):
                            st.write(f"**Tags:** {', '.join(ticker_info['tags'])}")
                    
                    if ticker_info.get('notes'):
                        st.info(f"**Notes:** {ticker_info['notes']}")
                    
                    # Actions
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    with btn_col1:
                        if st.button("🔍 Analyze", key=f"analyze_{ticker_info['ticker']}"):
                            st.session_state.analyze_ticker = ticker_info['ticker']
                            tm.record_access(ticker_info['ticker'])
                    
                    with btn_col2:
                        if st.button("📝 Edit", key=f"edit_{ticker_info['ticker']}"):
                            st.session_state.edit_ticker = ticker_info['ticker']
                    
                    with btn_col3:
                        if st.button("🗑️ Delete", key=f"delete_{ticker_info['ticker']}"):
                            if tm.remove_ticker(ticker_info['ticker']):
                                st.success(f"Deleted {ticker_info['ticker']}")
                                st.rerun()
        else:
            st.info("No tickers found. Add some tickers to get started!")
    
    with subtab2:
        # Watchlists View
        st.subheader("📊 Watchlists")
        
        watchlists = tm.get_watchlists()
        
        if watchlists:
            for wl in watchlists:
                with st.expander(f"📋 {wl['name']} ({len(tm.get_watchlist_tickers(wl['name']))} tickers)"):
                    st.write(f"**Description:** {wl.get('description', 'No description')}")
                    st.write(f"**Created:** {wl['date_created'][:10]}")
                    
                    tickers_in_wl = tm.get_watchlist_tickers(wl['name'])
                    if tickers_in_wl:
                        st.write("**Tickers:**")
                        for ticker in tickers_in_wl:
                            col_t, col_b = st.columns([3, 1])
                            with col_t:
                                st.write(f"• {ticker}")
                            with col_b:
                                if st.button("❌", key=f"remove_{wl['name']}_{ticker}"):
                                    tm.remove_from_watchlist(wl['name'], ticker)
                                    st.rerun()
                    
                    if st.button(f"🗑️ Delete Watchlist", key=f"delete_wl_{wl['name']}"):
                        if tm.delete_watchlist(wl['name']):
                            st.success(f"Deleted {wl['name']}")
                            st.rerun()
        else:
            st.info("No watchlists yet. Create one below!")
        
        st.divider()
        st.write("**Create New Watchlist**")
        
        new_wl_col1, new_wl_col2 = st.columns([2, 1])
        
        with new_wl_col1:
            new_wl_name = st.text_input("Watchlist Name", placeholder="My Tech Stocks")
            new_wl_desc = st.text_input("Description", placeholder="High growth tech companies")
        
        with new_wl_col2:
            st.write("")
            st.write("")
            if st.button("➕ Create Watchlist", use_container_width=True):
                if new_wl_name:
                    if tm.create_watchlist(new_wl_name, new_wl_desc):
                        st.success(f"Created {new_wl_name}!")
                        st.rerun()
                else:
                    st.error("Please enter a name")
    
    with subtab3:
        # Add New Ticker
        st.subheader("➕ Add New Ticker")
        
        add_col1, add_col2 = st.columns(2)
        
        with add_col1:
            new_ticker = st.text_input("Ticker Symbol", placeholder="AAPL").upper()
            new_name = st.text_input("Company Name (optional)", placeholder="Apple Inc.")
            new_sector = st.text_input("Sector (optional)", placeholder="Technology")
        
        with add_col2:
            new_type = st.selectbox("Type", ["stock", "penny_stock", "option", "etf", "crypto"])
            new_tags = st.text_input("Tags (comma separated)", placeholder="tech, growth, large-cap")
            new_notes = st.text_area("Notes", placeholder="Your research notes here...")
        
        if st.button("💾 Save Ticker", type="primary", use_container_width=True):
            if new_ticker:
                tags_list = [t.strip() for t in new_tags.split(',')] if new_tags else None
                
                if tm.add_ticker(
                    new_ticker, 
                    name=new_name or None,
                    sector=new_sector or None,
                    ticker_type=new_type,
                    notes=new_notes or None,
                    tags=tags_list
                ):
                    st.success(f"✅ {new_ticker} saved successfully!")
                    st.balloons()
                else:
                    st.error("Failed to save ticker")
            else:
                st.error("Please enter a ticker symbol")

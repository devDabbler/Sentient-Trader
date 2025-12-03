"""
Watchlist Tab
Manage ticker watchlist, bulk analysis, and strategy-specific monitoring

Extracted from app.py for modularization
Refactored for consistent UI patterns across stock and crypto watchlists.
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
from .common_imports import (
    get_ticker_manager,
    get_llm_analyzer,
    get_cached_stock_data,
    ComprehensiveAnalyzer,
    TradingStyleAnalyzer
)

# Import unified watchlist components
try:
    from ui.components.watchlist_components import (
        display_ticker_multiselect,
        display_analysis_mode_selector,
        display_trading_style_selector,
        display_position_risk_inputs,
        display_filter_sort_controls,
        display_analysis_summary_metrics,
        display_action_badge,
        display_fast_mode_toggle,
        display_broker_connection_warning,
        display_llm_connection_warning
    )
    WATCHLIST_COMPONENTS_AVAILABLE = True
except ImportError:
    logger.debug("Unified watchlist components not available, using inline implementations")
    WATCHLIST_COMPONENTS_AVAILABLE = False

# Import additional modules with fallbacks
try:
    from services.penny_stock_constants import is_penny_stock
except ImportError:
    # Fallback for is_penny_stock
    def is_penny_stock(price, max_price=5.0):
        return price is not None and price > 0 and price <= max_price

try:
    from services.alpha_factors import AlphaFactorCalculator
except ImportError:
    logger.debug("AlphaFactorCalculator not available")
    AlphaFactorCalculator = None

def render_tab():
    """Main render function called from app.py"""
    st.header("⭐ Stock Watchlist")
    st.write("Manage your saved tickers with AI-powered analysis and bulk operations.")
    
    # Use cached ticker manager from session state
    tm = st.session_state.ticker_manager

    # --- Centralized Client and Assistant Initialization ---
    from services.ai_stock_entry_assistant import get_ai_stock_entry_assistant

    # Get broker client (prefer IBKR if available, fallback to Tradier, then BrokerAdapter)
    broker_client = st.session_state.get('ibkr_client') or st.session_state.get('tradier_client') or st.session_state.get('broker_client')
    llm_analyzer = st.session_state.get('llm_analyzer')

    # Show prominent warning if broker not connected
    if not broker_client:
        st.error("🚨 **Broker Not Connected** - Multi-Config Analysis & AI Entry features require a broker connection")
        st.info("👉 Go to **Tradier** or **IBKR** tab to connect a broker, then return here.")
        with st.expander("🔍 Why do I need a broker?"):
            st.write("**AI Entry Analysis & Multi-Config require:**")
            st.write("1. ✅ **Broker Connection** (IBKR or Tradier) - for real-time market data")
            st.write("2. ✅ **LLM API Key** (OpenRouter) - for AI analysis")
            st.write("")
            st.write("**Current Status:**")
            st.write(f"- Broker: {'✅ Connected' if broker_client else '❌ Not connected'}")
            st.write(f"- LLM: {'✅ Configured' if llm_analyzer else '❌ Not configured'}")
        st.divider()

    # Ensure AI Entry Assistant is initialized once and available for all components on this page
    if 'stock_ai_entry_assistant' not in st.session_state:
        if broker_client and llm_analyzer:
            try:
                st.session_state.stock_ai_entry_assistant = get_ai_stock_entry_assistant(
                    broker_client=broker_client,
                    llm_analyzer=llm_analyzer
                )
                st.success("✅ AI Stock Entry Assistant initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize AI Stock Entry Assistant: {e}")
                st.error(f"❌ Failed to initialize AI Entry Assistant: {str(e)}")
    
    # Debug: Show connection status
    if tm.supabase:
        if 'supabase_connection_test' not in st.session_state:
            st.session_state.supabase_connection_test = tm.test_connection()
        
        if st.session_state.supabase_connection_test:
            st.success("✅ Supabase connected")
        else:
            st.error("❌ Supabase connection failed")
            st.stop()
    else:
        st.error("❌ Supabase not connected")
        st.stop()
    
    # Add new ticker
    st.subheader("➕ Add New Ticker")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_ticker = st.text_input("Ticker Symbol").upper()
    with col2:
        new_name = st.text_input("Company Name (optional)")
    with col3:
        new_type = st.selectbox("Type", ["stock", "option", "penny_stock", "crypto"])
    
    new_notes = st.text_area("Notes (optional)")
    
    if st.button("➕ Add Ticker"):
        if new_ticker:
            # Check if ticker already exists in watchlist
            existing = tm.get_ticker(new_ticker)
            if existing:
                st.warning(f"⚠️ {new_ticker} is already in your watchlist!")
            elif tm.add_ticker(new_ticker, name=new_name, ticker_type=new_type, notes=new_notes):
                # Invalidate ticker cache to force refresh
                st.session_state.ticker_cache = {}
                st.session_state.ticker_cache_timestamp = None
                st.success(f"✅ Added {new_ticker}! Refresh the page to see it in your list.")
                # No immediate rerun needed - ticker is added, will show on next natural refresh
            else:
                st.error("❌ Failed to add ticker. Check the logs for details.")
                # Show debug info
                if tm.supabase:
                    st.info("✅ Supabase client is connected")
                else:
                    st.error("❌ Supabase client is not connected - check your secrets")
        else:
            st.warning("⚠️ Ticker symbol is required.")
    
    st.divider()
    
    # Multi-Config Analysis Section - AT THE TOP for easy access
    if broker_client and llm_analyzer:
        with st.expander("🎯 **MULTI-CONFIGURATION BULK ANALYSIS** (Click to Expand)", expanded=False):
            from ui.bulk_ai_entry_analysis_ui import display_multi_config_bulk_analysis
            
            # Get all tickers
            try:
                all_tickers_temp = tm.get_all_tickers()
                if all_tickers_temp:
                    ticker_list_temp = [t['ticker'] for t in all_tickers_temp]
                    entry_assistant = st.session_state.get('stock_ai_entry_assistant')
                    
                    if entry_assistant:
                        st.markdown("#### 🎯 Multi-Configuration Bulk Analysis")
                        st.caption("Test different position sizes, risk levels, and trading styles to find optimal setups")
                        display_multi_config_bulk_analysis(ticker_list_temp, entry_assistant, tm)
                    else:
                        st.warning("AI Entry Assistant not yet initialized. It will initialize when you scroll down.")
                else:
                    st.info("Add tickers to your watchlist below to use multi-config analysis.")
            except Exception as e:
                logger.error(f"Error loading multi-config at top: {e}")
                st.error(f"Error: {str(e)}")
    else:
        st.warning("🎯 **Multi-Config Analysis Unavailable** - Connect a broker (Tradier/IBKR) to enable this feature")
    
    st.divider()
    
    # Trading Style Selector
    st.subheader("⚙️ Analysis Settings")
    col_style, col_refresh_all = st.columns([2, 1])
    
    with col_style:
        analysis_style = st.selectbox(
            "Trading Style for Analysis",
            ["AI", "OPTIONS", "DAY_TRADE", "SWING_TRADE", "SCALP", "WARRIOR_SCALPING", "ORB_FVG", "BUY_AND_HOLD"],
            index=0,
            help="Select the trading style to analyze your tickers with"
        )
        st.session_state.analysis_timeframe = analysis_style
    
    with col_refresh_all:
        if st.button("🔄 Refresh All", help="Refresh analysis for all tickers"):
            # Set flag for refresh - no need to rerun immediately
            st.session_state.refresh_all_tickers = True
            st.info("🔄 Refreshing all tickers... This will update on the next analysis run.")
    
    st.divider()
    
    # AI Entry Analysis Section
    st.subheader("🤖 AI Entry Analysis")
    st.write("Analyze whether now is a good time to enter a stock trade using AI-powered market analysis.")
    
    from ui.stock_ai_entry_ui import display_stock_ai_entry_analysis
    from services.ai_stock_entry_assistant import get_ai_stock_entry_assistant

    # Get broker client (prefer IBKR if available, fallback to Tradier, then BrokerAdapter)
    broker_client = st.session_state.get('ibkr_client') or st.session_state.get('tradier_client') or st.session_state.get('broker_client')

    # Ensure AI Entry Assistant is initialized and available for all components on this page
    if 'stock_ai_entry_assistant' not in st.session_state:
        if broker_client and 'llm_analyzer' in st.session_state:
            st.session_state.stock_ai_entry_assistant = get_ai_stock_entry_assistant(
                broker_client=broker_client,
                llm_analyzer=st.session_state.llm_analyzer
            )
    
    if st.session_state.get('stock_ai_entry_assistant'):
        display_stock_ai_entry_analysis(broker_client, st.session_state.llm_analyzer)
    else:
        st.warning("⚠️ AI Entry Assistant not available. Please connect to a broker and ensure LLM is configured.")
        st.info("💡 Go to the '🏦 Tradier Account' or '📈 IBKR Trading' tab to connect.")
    
    st.divider()
    
    # View saved tickers with pagination
    st.subheader("📋 Your Saved Tickers")

    # --- Filter and Sort Controls ---
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        ai_action_filter = st.selectbox(
            "Filter by AI Action",
            options=["All", "ENTER_NOW", "WAIT_FOR_PULLBACK", "WAIT_FOR_BREAKOUT", "PLACE_LIMIT_ORDER", "DO_NOT_ENTER"],
            key="ai_action_filter",
            help="Show only tickers with a specific AI recommendation."
        )
    with filter_col2:
        sort_by = st.selectbox(
            "Sort By",
            options=["Default", "AI Confidence (Highest First)", "Analysis Date (Newest First)"],
            key="ai_sort_by",
            help="Sort the list of tickers based on AI analysis results."
        )
    
    # Pagination settings
    import math
    items_per_page = 10
    
    # Initialize pagination state
    if 'ticker_page' not in st.session_state:
        st.session_state.ticker_page = 1
    
    # Get total count for pagination with caching
    # Check if we have cached ticker data that's still fresh
    cache_valid = False
    if st.session_state.ticker_cache and st.session_state.ticker_cache_timestamp:
        cache_age = (datetime.now() - st.session_state.ticker_cache_timestamp).total_seconds()
        if cache_age < st.session_state.ticker_cache_ttl:
            cache_valid = True
    
    if cache_valid and 'all_tickers' in st.session_state.ticker_cache:
        all_tickers_full = st.session_state.ticker_cache['all_tickers']
        pass  # logger.debug(f"Using cached ticker data (age: {}s) {cache_age:.1f}")
    else:
        # Fetch fresh data and cache it
        all_tickers_full = tm.get_all_tickers(limit=100)
        st.session_state.ticker_cache['all_tickers'] = all_tickers_full
        st.session_state.ticker_cache_timestamp = datetime.now()
        logger.debug("Fetched fresh ticker data and cached it")

    # Apply filtering
    if ai_action_filter != "All":
        all_tickers_full = [t for t in all_tickers_full if t.get('ai_entry_action') == ai_action_filter]

    # Apply sorting
    if sort_by == "AI Confidence (Highest First)":
        all_tickers_full = sorted(
            all_tickers_full, 
            key=lambda t: t.get('ai_entry_confidence', 0) or 0, 
            reverse=True
        )
    elif sort_by == "Analysis Date (Newest First)":
        all_tickers_full = sorted(
            all_tickers_full, 
            key=lambda t: t.get('ai_entry_timestamp', '1970-01-01T00:00:00+00:00') or '1970-01-01T00:00:00+00:00', 
            reverse=True
        )
    
    total_tickers = len(all_tickers_full)
    total_pages = max(1, math.ceil(total_tickers / items_per_page))
    
    # Ensure current page is within bounds
    if st.session_state.ticker_page > total_pages:
        st.session_state.ticker_page = total_pages
    
    # Display pagination controls at top
    if total_pages > 1:
        col_p1, col_p2, col_p3, col_p4 = st.columns([1, 2, 2, 1])
        with col_p1:
            if st.button("◀ Previous", disabled=st.session_state.ticker_page == 1, key="prev_top"):
                st.session_state.ticker_page -= 1
                # Pagination needs rerun to display new page
                st.rerun()
        with col_p2:
            st.write(f"**Page {st.session_state.ticker_page} of {total_pages}**")
        with col_p3:
            st.write(f"**Showing {min(items_per_page, total_tickers)} of {total_tickers} tickers**")
        with col_p4:
            if st.button("Next ▶", disabled=st.session_state.ticker_page == total_pages, key="next_top"):
                st.session_state.ticker_page += 1
                # Pagination needs rerun to display new page
                st.rerun()
    
    # Get only the tickers for current page
    start_idx = (st.session_state.ticker_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    all_tickers = all_tickers_full[start_idx:end_idx]

    if all_tickers:
        for ticker in all_tickers:
            # Create enhanced ticker card with better visual design
            ticker_symbol = ticker['ticker']
            ticker_name = ticker.get('name', 'Unknown Company')
            ticker_type = ticker.get('type', 'stock')
            ml_score = ticker.get('ml_score')
            notes = ticker.get('notes', '')
            
            # Determine card header styling based on ML score
            if ml_score is not None:
                if ml_score >= 70:
                    score_emoji = "🟢"
                    score_color = "green"
                    confidence_label = "HIGH"
                elif ml_score >= 50:
                    score_emoji = "🟡"
                    score_color = "orange"
                    confidence_label = "MEDIUM"
                else:
                    score_emoji = "🔴"
                    score_color = "red"
                    confidence_label = "LOW"
                expander_title = f"{score_emoji} **{ticker_symbol}** · {ticker_name[:30]}{'...' if len(ticker_name) > 30 else ''} · **{confidence_label}** {ml_score:.0f}/100"
            else:
                expander_title = f"📊 **{ticker_symbol}** · {ticker_name[:30]}{'...' if len(ticker_name) > 30 else ''}"
            
            # Main card container
            with st.container():
                # AI Entry Analysis Badge
                ai_action = ticker.get('ai_entry_action')
                ai_confidence = ticker.get('ai_entry_confidence')
                ai_timestamp_str = ticker.get('ai_entry_timestamp')

                if ai_action and ai_confidence is not None and ai_timestamp_str:
                    try:
                        ai_timestamp = datetime.fromisoformat(ai_timestamp_str.replace('Z', '+00:00'))
                        time_ago = (datetime.now(timezone.utc) - ai_timestamp).total_seconds()
                        if time_ago < 3600:
                            time_label = f"{int(time_ago / 60)}m ago"
                        elif time_ago < 86400:
                            time_label = f"{int(time_ago / 3600)}h ago"
                        else:
                            time_label = f"{int(time_ago / 86400)}d ago"

                        badge_color = "green" if ai_action == "ENTER_NOW" else "yellow" if "WAIT" in ai_action else "red"
                        
                        st.markdown(
                            f'<div style="background-color:{badge_color};color:white;padding:5px 10px;border-radius:5px;margin-bottom:5px;font-size:0.9em;">' 
                            f'🤖 **AI Rec:** {ai_action.replace("_", " ")} ({ai_confidence:.0f}%) - {time_label}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    except (ValueError, TypeError):
                        pass # Ignore parsing errors

                # Card header with action buttons
                col_header, col_actions = st.columns([5, 2])
                
                with col_header:
                    with st.expander(expander_title, expanded=False):
                        # Company info section
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.markdown("**📊 Company Details**")
                            st.write(f"**Symbol:** `{ticker_symbol}`")
                            st.write(f"**Name:** {ticker_name}")
                            st.write(f"**Type:** {ticker_type.replace('_', ' ').title()}")
                            
                            # Sector and tags if available
                            sector = ticker.get('sector')
                            if sector:
                                st.write(f"**Sector:** {sector}")
                            
                            tags = ticker.get('tags')
                            if tags and isinstance(tags, list):
                                st.write(f"**Tags:** {', '.join(tags)}")
                        
                        with info_col2:
                            st.markdown("**📈 Activity & Stats**")
                            
                            # Access count with emoji
                            access_count = ticker.get('access_count', 0)
                            activity_emoji = "🔥" if access_count > 10 else "📊" if access_count > 5 else "📋"
                            st.write(f"**Views:** {activity_emoji} {access_count}")
                            
                            # Date added
                            date_added_str = ticker.get('date_added', 'Unknown')
                            if date_added_str != 'Unknown':
                                try:
                                    dt_utc = datetime.fromisoformat(date_added_str).replace(tzinfo=timezone.utc)
                                    dt_local = dt_utc.astimezone()
                                    friendly_date = dt_local.strftime('%b %d, %Y')
                                    days_ago = (datetime.now(timezone.utc) - dt_utc).days
                                    if days_ago == 0:
                                        time_label = "Today"
                                    elif days_ago == 1:
                                        time_label = "Yesterday"
                                    elif days_ago < 7:
                                        time_label = f"{days_ago} days ago"
                                    else:
                                        time_label = friendly_date
                                    st.write(f"**Added:** {time_label}")
                                except (ValueError, TypeError):
                                    st.write(f"**Added:** {date_added_str}")
                            else:
                                st.write(f"**Added:** Unknown")
                            
                            # Last accessed if available
                            last_accessed = ticker.get('last_accessed')
                            if last_accessed:
                                try:
                                    dt_accessed = datetime.fromisoformat(last_accessed).replace(tzinfo=timezone.utc)
                                    dt_local_accessed = dt_accessed.astimezone()
                                    accessed_ago = (datetime.now(timezone.utc) - dt_accessed).days
                                    if accessed_ago == 0:
                                        access_label = "Today"
                                    elif accessed_ago == 1:
                                        access_label = "Yesterday"
                                    elif accessed_ago < 7:
                                        access_label = f"{accessed_ago} days ago"
                                    else:
                                        access_label = dt_local_accessed.strftime('%b %d, %Y')
                                    st.write(f"**Last View:** {access_label}")
                                except:
                                    pass
                        
                        # Notes section if available
                        if notes and notes.strip():
                            st.markdown("**📝 Notes**")
                            with st.container():
                                st.info(notes)
                        
                        # Comprehensive Analysis section - Enhanced with Dashboard-level data
                        st.divider()
                        st.markdown("**📊 Comprehensive Analysis**")
                        
                        # Check if analysis is stale (older than 1 hour) with caching
                        # Cache key includes ticker and max_age_hours to ensure proper invalidation
                        cache_key = f"{ticker_symbol}_1.0"
                        
                        # Check if we have a cached result that's still valid (cache for 5 minutes to reduce DB queries)
                        needs_update = True  # Default
                        if cache_key in st.session_state.analysis_update_cache:
                            cache_entry = st.session_state.analysis_update_cache[cache_key]
                            cache_timestamp = st.session_state.analysis_update_cache_timestamp.get(cache_key)
                            if cache_timestamp:
                                cache_age = (datetime.now() - cache_timestamp).total_seconds()
                                if cache_age < 300:  # Cache for 5 minutes (increased from 30 seconds)
                                    needs_update = cache_entry
                                    logger.debug(f"Using cached should_update_analysis for {ticker_symbol}: {needs_update}")
                        
                        # If not cached or cache expired, try to use ticker data from cache first
                        if needs_update is True or cache_key not in st.session_state.analysis_update_cache:
                            try:
                                # First, try to get last_analyzed from already-loaded ticker data
                                ticker_data = next((t for t in all_tickers_full if t.get('ticker', '').upper() == ticker_symbol.upper()), None)
                                if ticker_data and ticker_data.get('last_analyzed'):
                                    # Use cached ticker data to avoid database query
                                    try:
                                        last_analyzed = datetime.fromisoformat(ticker_data['last_analyzed']).replace(tzinfo=timezone.utc)
                                        age_hours = (datetime.now(timezone.utc) - last_analyzed).total_seconds() / 3600
                                        needs_update = age_hours >= 1.0
                                        # Cache the result
                                        st.session_state.analysis_update_cache[cache_key] = needs_update
                                        st.session_state.analysis_update_cache_timestamp[cache_key] = datetime.now()
                                    except (ValueError, TypeError) as e:
                                        logger.debug(f"Error parsing last_analyzed from ticker data for {ticker_symbol}: {e}")
                                        # Fall through to database query
                                        needs_update = True
                                else:
                                    # Fallback to database query if not in cached data
                                    if tm and hasattr(tm, 'should_update_analysis'):
                                        needs_update = tm.should_update_analysis(ticker_symbol, max_age_hours=1.0)
                                        # Cache the result
                                        st.session_state.analysis_update_cache[cache_key] = needs_update
                                        st.session_state.analysis_update_cache_timestamp[cache_key] = datetime.now()
                                    else:
                                        # Fallback: check last_analyzed directly from ticker data
                                        needs_update = True
                            except (AttributeError, TypeError) as e:
                                logger.error(f"Error checking analysis staleness for {ticker_symbol}: {e}")
                                needs_update = True  # Default to needing update if check fails
                        
                        last_analyzed_str = ticker.get('last_analyzed')
                        
                        # Display last analysis timestamp
                        if last_analyzed_str:
                            try:
                                last_analyzed_dt = datetime.fromisoformat(last_analyzed_str).replace(tzinfo=timezone.utc)
                                age_hours = (datetime.now(timezone.utc) - last_analyzed_dt).total_seconds() / 3600
                                if age_hours < 1:
                                    st.info(f"📊 Analysis cached ({age_hours*60:.0f} minutes ago) - Click 'Refresh' for latest data")
                                else:
                                    st.warning(f"⚠️ Analysis is {age_hours:.1f} hours old - Click 'Refresh' for latest data")
                            except:
                                pass
                        else:
                            st.info("📊 No recent analysis - Click 'Analyze' to generate")
                        
                        # Show analysis button instead of auto-analyzing
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            analyze_btn = st.button(
                                f"🔍 {'Refresh' if not needs_update else 'Analyze'} {ticker_symbol}",
                                key=f"analyze_btn_{ticker_symbol}",
                                help="Run comprehensive analysis with latest data",
                                type="primary" if needs_update else "secondary"
                            )
                        
                        with col_btn2:
                            show_cached = st.button(
                                "👁️ View Cached Data",
                                key=f"view_cached_{ticker_symbol}",
                                help="View last saved analysis data",
                                disabled=not last_analyzed_str
                            )
                        
                        # Get fresh analysis data only if requested
                        analysis = None
                        try:
                            # Check if user requested fresh analysis
                            should_refresh = (
                                analyze_btn or
                                st.session_state.get(f"refresh_{ticker_symbol}", False) or 
                                st.session_state.get('refresh_all_tickers', False) or
                                st.session_state.get('ml_ticker_to_analyze') == ticker_symbol
                            )
                            
                            if should_refresh:
                                with st.spinner(f"🔄 Analyzing {ticker_symbol} with latest data..."):
                                    analysis = ComprehensiveAnalyzer.analyze_stock(ticker_symbol, st.session_state.get('analysis_timeframe', 'OPTIONS'))
                                    # Fetch historical data for trading style analysis
                                    hist, _ = get_cached_stock_data(ticker_symbol)
                                
                                # Clear refresh flags
                                if f"refresh_{ticker_symbol}" in st.session_state:
                                    del st.session_state[f"refresh_{ticker_symbol}"]
                                
                                st.success(f"✅ Fresh analysis completed for {ticker_symbol}!")
                            elif show_cached or not needs_update:
                                # Show cached data from database without re-analyzing
                                # Display metrics from database directly
                                st.info(f"📊 Displaying cached analysis data for {ticker_symbol}")
                                
                                # Show basic metrics from database
                                db_col1, db_col2, db_col3, db_col4 = st.columns(4)
                                with db_col1:
                                    st.metric("ML Score", f"{ticker.get('ml_score', 'N/A')}/100" if ticker.get('ml_score') else "N/A")
                                with db_col2:
                                    momentum = ticker.get('momentum', 0)
                                    st.metric("Momentum", f"{momentum:+.2f}%" if momentum else "N/A")
                                with db_col3:
                                    rsi = ticker.get('rsi')
                                    st.metric("RSI", f"{rsi:.1f}" if rsi else "N/A")
                                with db_col4:
                                    sentiment = ticker.get('sentiment_score')
                                    st.metric("Sentiment", f"{sentiment:.2f}" if sentiment is not None else "N/A")
                                
                                st.info("💡 Click 'Analyze' above to run fresh comprehensive analysis with latest market data")
                                
                                # Don't show full analysis - just the cached summary
                                analysis = None
                                
                            if analysis:
                                # Update ticker with fresh analysis data
                                tm.update_analysis(ticker_symbol, analysis.__dict__)
                                
                                # Detect characteristics
                                is_penny_stock_check = is_penny_stock(analysis.price)
                                is_otc = analysis.ticker.endswith(('.OTC', '.PK', '.QB'))
                                volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
                                is_runner = volume_vs_avg > 200 and analysis.change_pct > 10
                                
                                # Header metrics (same as dashboard)
                                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                                
                                with metric_col1:
                                    price_display = f"${analysis.price:.4f}" if is_penny_stock else f"${analysis.price:.2f}"
                                    st.metric("Price", price_display, f"{analysis.change_pct:+.2f}%")
                                with metric_col2:
                                    st.metric("Trend", analysis.trend)
                                with metric_col3:
                                    st.metric("Confidence", f"{int(analysis.confidence_score)}%")
                                with metric_col4:
                                    st.metric("IV Rank", f"{analysis.iv_rank}%")
                                with metric_col5:
                                    volume_indicator = "🔥" if volume_vs_avg > 100 else "📊"
                                    st.metric(f"{volume_indicator} Volume", f"{analysis.volume:,}", f"{volume_vs_avg:+.1f}%")
                                
                                # Special alerts for penny stocks and runners
                                if is_runner:
                                    st.warning(f"🚀 **RUNNER DETECTED!** {volume_vs_avg:+.0f}% volume spike with {analysis.change_pct:+.1f}% price move!")
                                
                                if is_penny_stock:
                                    st.info(f"💰 **PENNY STOCK** (${analysis.price:.4f}) - High risk/high reward. Use caution and proper position sizing.")
                                
                                if is_otc:
                                    st.warning("⚠️ **OTC STOCK** - Lower liquidity, wider spreads, higher risk. Limited data may be available.")
                                
                                # Technical Indicators (same as dashboard)
                                st.subheader("📊 Technical Indicators")
                                
                                tech_col1, tech_col2, tech_col3 = st.columns(3)
                                
                                with tech_col1:
                                    st.metric("RSI (14)", f"{analysis.rsi:.1f}")
                                    if analysis.rsi < 30:
                                        st.caption("🟢 Oversold - potential buy")
                                    elif analysis.rsi > 70:
                                        st.caption("🔴 Overbought - potential sell")
                                    else:
                                        st.caption("🟡 Neutral")
                                
                                with tech_col2:
                                    st.metric("MACD Signal", analysis.macd_signal)
                                    if analysis.macd_signal == "BULLISH":
                                        st.caption("🟢 Bullish momentum")
                                    elif analysis.macd_signal == "BEARISH":
                                        st.caption("🔴 Bearish momentum")
                                    else:
                                        st.caption("🟡 Neutral momentum")
                                
                                with tech_col3:
                                    st.metric("Support", f"${analysis.support}")
                                    st.metric("Resistance", f"${analysis.resistance}")
                                
                                # IV Analysis (same as dashboard)
                                st.subheader("📈 Implied Volatility Analysis")
                                
                                iv_col1, iv_col2, iv_col3 = st.columns(3)
                                
                                with iv_col1:
                                    st.metric("IV Rank", f"{analysis.iv_rank}%")
                                    if analysis.iv_rank > 60:
                                        st.caption("🔥 High IV - Great for selling premium")
                                    elif analysis.iv_rank < 40:
                                        st.caption("❄️ Low IV - Good for buying options")
                                    else:
                                        st.caption("➡️ Moderate IV")
                                
                                with iv_col2:
                                    st.metric("IV Percentile", f"{analysis.iv_percentile}%")
                                
                                with iv_col3:
                                    if analysis.iv_rank > 50:
                                        st.info("💡 Consider: Selling puts, covered calls, iron condors")
                                    else:
                                        st.info("💡 Consider: Buying calls/puts, debit spreads")
                                
                                # Catalysts (same as dashboard)
                                st.subheader("📅 Upcoming Catalysts")
                                
                                if analysis.catalysts:
                                    for catalyst in analysis.catalysts:
                                        impact_color = {
                                            'HIGH': '🔴',
                                            'MEDIUM': '🟡',
                                            'LOW': '🟢'
                                        }.get(catalyst['impact'], '⚪')
                                        
                                        with st.expander(f"{impact_color} {catalyst['type']} - {catalyst['date']} ({catalyst.get('days_away', 'N/A')} days away)"):
                                            st.write(f"**Impact Level:** {catalyst['impact']}")
                                            st.write(f"**Details:** {catalyst['description']}")
                                            
                                            if catalyst['type'] == 'Earnings Report' and catalyst.get('days_away', 999) <= 7:
                                                st.warning("⚠️ Earnings within 7 days - expect high volatility!")
                                else:
                                    st.info("No major catalysts identified in the next 60 days")
                                
                                # News & Sentiment (same as dashboard)
                                st.subheader("📰 Recent News & Sentiment")
                                
                                if analysis.recent_news:
                                    st.success(f"✅ Found {len(analysis.recent_news)} recent news articles")
                                else:
                                    st.warning("⚠️ No recent news found - this may indicate low news volume or connectivity issues")
                                
                                sentiment_col1, sentiment_col2 = st.columns([1, 3])
                                
                                with sentiment_col1:
                                    sentiment_label = "POSITIVE" if analysis.sentiment_score > 0.2 else "NEGATIVE" if analysis.sentiment_score < -0.2 else "NEUTRAL"
                                    sentiment_color = "🟢" if analysis.sentiment_score > 0.2 else "🔴" if analysis.sentiment_score < -0.2 else "🟡"
                                    
                                    st.metric("News Sentiment", f"{sentiment_color} {sentiment_label}")
                                    st.metric("Sentiment Score", f"{analysis.sentiment_score:.2f}")
                                    
                                    # Show sentiment signals if available
                                    if hasattr(analysis, 'sentiment_signals') and analysis.sentiment_signals:
                                        with st.expander("📊 Sentiment Analysis Details"):
                                            for signal in analysis.sentiment_signals[:3]:  # Show top 3
                                                st.write(signal)
                                
                                with sentiment_col2:
                                    if analysis.recent_news:
                                        st.write("**Latest News Articles:**")
                                        for idx, article in enumerate(analysis.recent_news[:3]):  # Show top 3
                                            # Create a more informative expander
                                            expander_title = f"📰 {article['title'][:50]}..." if len(article['title']) > 50 else f"📰 {article['title']}"
                                            
                                            with st.expander(expander_title):
                                                col_pub, col_time = st.columns(2)
                                                with col_pub:
                                                    st.write(f"**Publisher:** {article['publisher']}")
                                                with col_time:
                                                    st.write(f"**Published:** {article['published']}")
                                                
                                                # Show summary if available
                                                if article.get('summary'):
                                                    st.write("**Summary:**")
                                                    st.write(article['summary'])
                                                
                                                # Link to full article
                                                if article.get('link'):
                                                    st.write(f"[📖 Read Full Article]({article['link']})")
                                    else:
                                        st.info("📭 No recent news found for this ticker.")
                                
                                # Penny Stock Risk Assessment (if applicable)
                                if is_penny_stock:
                                    st.subheader("⚠️ Penny Stock Risk Assessment")
                                    
                                    risk_col1, risk_col2 = st.columns(2)
                                    
                                    with risk_col1:
                                        st.warning("""
**Penny Stock Risks:**
- 🔴 High volatility (can swing 20-50%+ daily)
- 🔴 Low liquidity (harder to exit positions)
- 🔴 Wide bid-ask spreads (higher trading costs)
- 🔴 Manipulation risk (pump & dump schemes)
- 🔴 Limited financial data/transparency
- 🔴 Higher bankruptcy risk
                                        """)
                                    
                                    with risk_col2:
                                        st.success("""
**Penny Stock Trading Rules:**
- ✅ Never risk more than 1-2% of portfolio
- ✅ Use limit orders (avoid market orders)
- ✅ Set tight stop losses (5-10%)
- ✅ Take profits quickly (don't be greedy)
- ✅ Research company fundamentals
- ✅ Watch for unusual volume spikes
- ✅ Avoid stocks with no news/catalysts
                                        """)
                                    
                                    # Calculate penny stock score with tiered bonuses for extreme moves
                                    penny_score = 0
                                    
                                    # Volume scoring - tiered for magnitude
                                    if volume_vs_avg > 300:
                                        penny_score += 30  # Extreme volume
                                    elif volume_vs_avg > 200:
                                        penny_score += 27
                                    elif volume_vs_avg > 100:
                                        penny_score += 25
                                    elif volume_vs_avg > 50:
                                        penny_score += 15
                                    
                                    # Price change scoring - tiered for magnitude
                                    if abs(analysis.change_pct) > 50:
                                        penny_score += 25  # Extreme move
                                    elif abs(analysis.change_pct) > 20:
                                        penny_score += 22
                                    elif abs(analysis.change_pct) > 10:
                                        penny_score += 20
                                    elif abs(analysis.change_pct) > 5:
                                        penny_score += 15
                                    
                                    # RSI check
                                    if analysis.rsi < 70:
                                        penny_score += 15
                                    
                                    # News and sentiment
                                    if len(analysis.recent_news) > 0:
                                        penny_score += 15
                                    if analysis.sentiment_score > 0:
                                        penny_score += 15
                                    
                                    # Cap at 100
                                    penny_score = min(100, penny_score)
                                    
                                    st.metric("Penny Stock Opportunity Score", f"{penny_score}/100")
                                    
                                    if penny_score > 70:
                                        st.success("🟢 Strong opportunity - but still use caution!")
                                    elif penny_score > 50:
                                        st.info("🟡 Moderate opportunity - proceed carefully")
                                    else:
                                        st.warning("🔴 Weak setup - consider waiting for better entry")
                                
                                # Runner Metrics (if detected)
                                if is_runner or volume_vs_avg > 100:
                                    st.subheader("🚀 Runner / Momentum Metrics")
                                    
                                    runner_col1, runner_col2, runner_col3, runner_col4 = st.columns(4)
                                    
                                    with runner_col1:
                                        st.metric("Volume Spike", f"{volume_vs_avg:+.0f}%")
                                        if volume_vs_avg > 300:
                                            st.caption("🔥 EXTREME volume!")
                                        elif volume_vs_avg > 200:
                                            st.caption("🔥 Very high volume")
                                        else:
                                            st.caption("📈 Elevated volume")
                                    
                                    with runner_col2:
                                        st.metric("Price Change", f"{analysis.change_pct:+.2f}%")
                                        if abs(analysis.change_pct) > 20:
                                            st.caption("🚀 Major move!")
                                        elif abs(analysis.change_pct) > 10:
                                            st.caption("📈 Strong move")
                                    
                                    with runner_col3:
                                        # Calculate momentum score
                                        momentum_score = min(100, (abs(analysis.change_pct) * 2 + volume_vs_avg / 5))
                                        st.metric("Momentum Score", f"{momentum_score:.0f}/100")
                                        if momentum_score > 80:
                                            st.caption("🔥 HOT!")
                                        elif momentum_score > 60:
                                            st.caption("🔥 Strong")
                                    
                                    with runner_col4:
                                        # Risk level for runners
                                        runner_risk = "EXTREME" if is_penny_stock and volume_vs_avg > 300 else "VERY HIGH" if volume_vs_avg > 200 else "HIGH"
                                        st.metric("Runner Risk", runner_risk)
                                        st.caption("⚠️ Use stops!")
                                    
                                    if is_runner:
                                        st.warning("""
**Runner Trading Tips:**
- ✅ Use tight stop losses (3-5%)
- ✅ Take profits quickly (don't be greedy)
- ✅ Watch for volume decline (exit signal)
- ✅ Avoid chasing - wait for pullbacks
- ❌ Don't hold overnight (high gap risk)
                                        """)
                                
                                # Timeframe-Specific Analysis
                                st.subheader("⏰ Trading Style Analysis")
                                
                                # Get trading style from session state or default to OPTIONS
                                trading_style = st.session_state.get('analysis_timeframe', 'OPTIONS')
                                
                                # Calculate timeframe-specific metrics
                                if trading_style == "DAY_TRADE":
                                    # Day trading focus: quick moves, tight stops
                                    timeframe_score = 0
                                    reasons = []
                                    
                                    if volume_vs_avg > 100:
                                        timeframe_score += 30
                                        reasons.append(f"✅ High volume (+{volume_vs_avg:.0f}%) - good for day trading")
                                    else:
                                        reasons.append(f"⚠️ Volume only +{volume_vs_avg:.0f}% - may lack intraday momentum")
                                    
                                    if abs(analysis.change_pct) > 2:
                                        timeframe_score += 25
                                        reasons.append(f"✅ Strong intraday move ({analysis.change_pct:+.1f}%)")
                                    else:
                                        reasons.append("⚠️ Low intraday volatility - limited profit potential")
                                    
                                    if 30 < analysis.rsi < 70:
                                        timeframe_score += 20
                                        reasons.append("✅ RSI in tradeable range (not overbought/oversold)")
                                    
                                    if not is_penny_stock:
                                        timeframe_score += 15
                                        reasons.append("✅ Not a penny stock - better liquidity for day trading")
                                    else:
                                        reasons.append("⚠️ Penny stock - higher risk, use smaller size")
                                    
                                    if analysis.trend != "NEUTRAL":
                                        timeframe_score += 10
                                        reasons.append(f"✅ Clear trend ({analysis.trend}) - easier to trade")
                                    
                                    st.metric("Day Trading Suitability", f"{timeframe_score}/100")
                                    
                                    for reason in reasons:
                                        st.write(reason)
                                    
                                    if timeframe_score > 70:
                                        st.success("🟢 **EXCELLENT** for day trading!")
                                    elif timeframe_score > 50:
                                        st.info("🟡 **GOOD** for day trading with caution")
                                    else:
                                        st.warning("🔴 **POOR** for day trading - consider other timeframes")
                                
                                elif trading_style == "AI":
                                    # AI Analysis
                                    st.write("🤖 **AI-Powered Analysis**")
                                    ai_results = TradingStyleAnalyzer.analyze_ai_style(analysis, hist)
                                    
                                    # Display AI score and prediction
                                    ai_col1, ai_col2, ai_col3 = st.columns(3)
                                    with ai_col1:
                                        st.metric("AI Score", f"{ai_results['score']}/100")
                                    with ai_col2:
                                        st.metric("ML Prediction", ai_results.get('ml_prediction', 'N/A'))
                                    with ai_col3:
                                        st.metric("Risk Level", ai_results.get('risk_level', 'UNKNOWN'))
                                    
                                    # Display signals
                                    if ai_results.get('signals'):
                                        st.write("**📊 AI Signals:**")
                                        for signal in ai_results['signals']:
                                            st.write(signal)
                                    
                                    # Display recommendations
                                    if ai_results.get('recommendations'):
                                        st.write("**💡 AI Recommendations:**")
                                        for rec in ai_results['recommendations']:
                                            st.write(rec)

                                elif trading_style == "SWING_TRADE":
                                    # Swing trading focus: multi-day moves, wider stops
                                    timeframe_score = 0
                                    reasons = []
                                    
                                    if abs(analysis.change_pct) > 5:
                                        timeframe_score += 25
                                        reasons.append(f"✅ Strong momentum ({analysis.change_pct:+.1f}%) - good for swing trades")
                                    
                                    if analysis.trend != "NEUTRAL":
                                        timeframe_score += 30
                                        reasons.append(f"✅ Clear trend ({analysis.trend}) - ideal for swing trading")
                                    
                                    if 40 < analysis.rsi < 60:
                                        timeframe_score += 20
                                        reasons.append("✅ RSI in good swing range")
                                    
                                    if analysis.iv_rank > 30:
                                        timeframe_score += 15
                                        reasons.append("✅ Sufficient volatility for swing moves")
                                    
                                    if len(analysis.recent_news) > 0:
                                        timeframe_score += 10
                                        reasons.append("✅ News catalyst available")
                                    
                                    st.metric("Swing Trading Suitability", f"{timeframe_score}/100")
                                    
                                    for reason in reasons:
                                        st.write(reason)
                                    
                                    if timeframe_score > 70:
                                        st.success("🟢 **EXCELLENT** for swing trading!")
                                    elif timeframe_score > 50:
                                        st.info("🟡 **GOOD** for swing trading")
                                    else:
                                        st.warning("🔴 **POOR** for swing trading - wait for better setup")
                                
                                elif trading_style == "SCALP":
                                    # Scalp Analysis
                                    st.write("⚡ **Scalping Analysis**")
                                    scalp_results = TradingStyleAnalyzer.analyze_scalp_style(analysis, hist)
                                    
                                    # Display scalp score and risk
                                    scalp_col1, scalp_col2 = st.columns(2)
                                    with scalp_col1:
                                        st.metric("Scalping Score", f"{scalp_results['score']}/100")
                                    with scalp_col2:
                                        st.metric("Risk Level", scalp_results.get('risk_level', 'UNKNOWN'))
                                    
                                    # Display signals
                                    if scalp_results.get('signals'):
                                        st.write("**📊 Scalping Signals:**")
                                        for signal in scalp_results['signals']:
                                            st.write(signal)
                                    
                                    # Display targets
                                    if scalp_results.get('targets'):
                                        st.write("**🎯 Scalping Targets:**")
                                        for target in scalp_results['targets']:
                                            st.write(target)
                                    
                                    # Display recommendations
                                    if scalp_results.get('recommendations'):
                                        st.write("**💡 Scalping Strategy:**")
                                        for rec in scalp_results['recommendations']:
                                            st.write(rec)

                                elif trading_style == "WARRIOR_SCALPING":
                                    # Warrior Scalping Analysis
                                    st.write("⚔️ **Warrior Scalping Analysis**")
                                    warrior_results = TradingStyleAnalyzer.analyze_warrior_scalping_style(analysis, hist)
                                    
                                    # Display warrior score and setup type
                                    warrior_col1, warrior_col2, warrior_col3 = st.columns(3)
                                    with warrior_col1:
                                        st.metric("Warrior Score", f"{warrior_results['score']}/100")
                                    with warrior_col2:
                                        st.metric("Setup Type", warrior_results.get('setup_type', 'N/A'))
                                    with warrior_col3:
                                        st.metric("Risk Level", warrior_results.get('risk_level', 'UNKNOWN'))
                                    
                                    # Display signals
                                    if warrior_results.get('signals'):
                                        st.write("**📊 Warrior Signals:**")
                                        for signal in warrior_results['signals']:
                                            st.write(signal)
                                    
                                    # Display targets
                                    if warrior_results.get('targets'):
                                        st.write("**🎯 Warrior Targets:**")
                                        for target in warrior_results['targets']:
                                            st.write(target)
                                    
                                    # Display recommendations
                                    if warrior_results.get('recommendations'):
                                        st.write("**💡 Warrior Strategy:**")
                                        for rec in warrior_results['recommendations']:
                                            st.write(rec)

                                elif trading_style == "BUY_AND_HOLD":
                                    # Buy & Hold Analysis
                                    st.write("💎 **Buy & Hold Analysis**")
                                    hold_results = TradingStyleAnalyzer.analyze_buy_and_hold_style(analysis, hist)
                                    
                                    # Display hold score and risk
                                    hold_col1, hold_col2 = st.columns(2)
                                    with hold_col1:
                                        st.metric("Investment Score", f"{hold_results['score']}/100")
                                    with hold_col2:
                                        st.metric("Risk Level", hold_results.get('risk_level', 'UNKNOWN'))
                                    
                                    # Display valuation metrics
                                    if hold_results.get('valuation'):
                                        st.write("**📊 Valuation Metrics:**")
                                        val_col1, val_col2, val_col3 = st.columns(3)
                                        valuation = hold_results['valuation']
                                        with val_col1:
                                            if '200_day_ma' in valuation:
                                                st.metric("200-Day MA", valuation['200_day_ma'])
                                        with val_col2:
                                            if 'pe_ratio' in valuation:
                                                st.metric("P/E Ratio", valuation['pe_ratio'])
                                        with val_col3:
                                            if 'dividend_yield' in valuation:
                                                st.metric("Dividend Yield", valuation['dividend_yield'])
                                    
                                    # Display signals
                                    if hold_results.get('signals'):
                                        st.write("**📊 Investment Signals:**")
                                        for signal in hold_results['signals']:
                                            st.write(signal)
                                    
                                    # Display long-term targets
                                    if hold_results.get('targets'):
                                        st.write("**🎯 Long-Term Targets:**")
                                        for target in hold_results['targets']:
                                            st.write(target)
                                    
                                    # Display recommendations
                                    if hold_results.get('recommendations'):
                                        st.write("**💡 Investment Strategy:**")
                                        for rec in hold_results['recommendations']:
                                            st.write(rec)

                                elif trading_style == "ORB_FVG":
                                    # ORB+FVG Strategy Analysis
                                    st.write("📊 **Opening Range Breakout + Fair Value Gap Analysis**")
                                    
                                    try:
                                        from analyzers.orb_fvg_strategy import ORBFVGAnalyzer
                                        import yfinance as yf
                                        
                                        # Get intraday data for ORB+FVG analysis
                                        ticker_obj = yf.Ticker(ticker)
                                        intraday_data = ticker_obj.history(period="1d", interval="1m")
                                        
                                        if len(intraday_data) > 0:
                                            orb_analyzer = ORBFVGAnalyzer()
                                            orb_results = orb_analyzer.analyze(ticker, intraday_data, analysis.price)
                                            
                                            # Display key metrics
                                            orb_col1, orb_col2, orb_col3 = st.columns(3)
                                            with orb_col1:
                                                st.metric("Signal", orb_results['signal'])
                                            with orb_col2:
                                                st.metric("Confidence", f"{orb_results['confidence']:.1f}%")
                                            with orb_col3:
                                                st.metric("Risk Level", orb_results['risk_level'])
                                            
                                            # Opening Range info
                                            if orb_results.get('opening_range'):
                                                orb_range = orb_results['opening_range']
                                                st.info(f"📊 **Opening Range:** ${orb_range['orl']:.2f} - ${orb_range['orh']:.2f} ({orb_range['range_pct']:.1f}%)")
                                            
                                            # FVG info
                                            if orb_results.get('fvg_signal') != 'NEUTRAL':
                                                st.success(f"🎯 **Fair Value Gap:** {orb_results['fvg_signal']} (Strength: {orb_results['fvg_strength']})")
                                            
                                            # Trade Setup
                                            if orb_results['signal'] in ['BUY', 'SELL']:
                                                st.write("**📈 Trade Setup:**")
                                                st.write(f"**Entry:** ${orb_results['entry']:.2f} | **Stop:** ${orb_results['stop_loss']:.2f} | **Target:** ${orb_results['target']:.2f}")
                                                st.write(f"**R:R Ratio:** 1:{orb_results['risk_reward_ratio']:.1f}")
                                            
                                            # Key Signals
                                            if orb_results.get('key_signals'):
                                                st.write("**📊 Key Signals:**")
                                                for signal in orb_results['key_signals'][:3]:
                                                    st.write(signal)
                                        else:
                                            st.warning("⚠️ No intraday data available for ORB+FVG analysis")
                                    except Exception as e:
                                        logger.error(f"ORB+FVG analysis error for {ticker}: {e}")
                                        st.error(f"ORB+FVG analysis error: {str(e)}")

                                elif trading_style in ["EMA_HEIKIN_ASHI", "RSI_STOCHASTIC_HAMMER", "FISHER_RSI", "MACD_VOLUME_RSI", "AGGRESSIVE_SCALPING"]:
                                    # Freqtrade Strategy Analysis
                                    st.write("⚡ **Freqtrade Strategy Analysis**")
                                    
                                    strategy_names = {
                                        "EMA_HEIKIN_ASHI": "EMA Crossover + Heikin Ashi",
                                        "RSI_STOCHASTIC_HAMMER": "RSI + Stochastic + Hammer",
                                        "FISHER_RSI": "Fisher RSI Multi-Indicator",
                                        "MACD_VOLUME_RSI": "MACD + Volume + RSI",
                                        "AGGRESSIVE_SCALPING": "Aggressive Scalping"
                                    }
                                    
                                    st.info(f"📊 **Strategy:** {strategy_names.get(trading_style, trading_style)}")
                                    
                                    # Calculate strategy suitability score
                                    strategy_score = 50
                                    signals = []
                                    
                                    # Common indicators for freqtrade strategies
                                    if 30 < analysis.rsi < 70:
                                        strategy_score += 15
                                        signals.append("✅ RSI in optimal range")
                                    elif analysis.rsi < 30:
                                        strategy_score += 10
                                        signals.append("✅ RSI oversold - potential buy signal")
                                    elif analysis.rsi > 70:
                                        strategy_score += 10
                                        signals.append("✅ RSI overbought - potential sell signal")
                                    
                                    if analysis.macd_signal == "BULLISH":
                                        strategy_score += 15
                                        signals.append("✅ MACD bullish crossover")
                                    elif analysis.macd_signal == "BEARISH":
                                        strategy_score += 10
                                        signals.append("⚠️ MACD bearish crossover")
                                    
                                    if volume_vs_avg > 1.5:
                                        strategy_score += 15
                                        signals.append(f"✅ Volume surge: {volume_vs_avg:.1f}x average")
                                    
                                    if analysis.trend != "NEUTRAL":
                                        strategy_score += 10
                                        signals.append(f"✅ Clear trend: {analysis.trend}")
                                    
                                    # Strategy-specific scoring
                                    if trading_style == "AGGRESSIVE_SCALPING":
                                        if abs(analysis.change_pct) > 2:
                                            strategy_score += 10
                                            signals.append(f"✅ High volatility: {analysis.change_pct:+.2f}%")
                                    elif trading_style == "EMA_HEIKIN_ASHI":
                                        if analysis.trend == "BULLISH":
                                            strategy_score += 10
                                            signals.append("✅ Heikin Ashi bullish setup")
                                    
                                    strategy_score = min(100, strategy_score)
                                    
                                    # Display results
                                    strat_col1, strat_col2 = st.columns(2)
                                    with strat_col1:
                                        st.metric("Strategy Score", f"{strategy_score}/100")
                                    with strat_col2:
                                        if strategy_score >= 75:
                                            st.metric("Signal", "🟢 STRONG BUY")
                                        elif strategy_score >= 60:
                                            st.metric("Signal", "🟡 BUY")
                                        elif strategy_score >= 40:
                                            st.metric("Signal", "⚪ HOLD")
                                        else:
                                            st.metric("Signal", "🔴 SELL")
                                    
                                    if signals:
                                        st.write("**📊 Strategy Signals:**")
                                        for signal in signals:
                                            st.write(signal)
                                    
                                    # Targets
                                    target1 = analysis.price * 1.02
                                    target2 = analysis.price * 1.05
                                    target3 = analysis.price * 1.10
                                    st.write("**🎯 Profit Targets:**")
                                    st.write(f"T1: ${target1:.2f} (+2%)")
                                    st.write(f"T2: ${target2:.2f} (+5%)")
                                    st.write(f"T3: ${target3:.2f} (+10%)")
                                    
                                    # Stop loss
                                    stop_loss = analysis.price * 0.98
                                    st.write(f"**🛑 Stop Loss:** ${stop_loss:.2f} (-2%)")
                                
                                else:  # OPTIONS trading
                                    # Options trading focus: IV, time decay, volatility
                                    timeframe_score = 0
                                    reasons = []
                                    
                                    if analysis.iv_rank > 60:
                                        timeframe_score += 30
                                        reasons.append(f"✅ High IV Rank ({analysis.iv_rank}%) - great for selling premium")
                                    elif analysis.iv_rank < 40:
                                        timeframe_score += 20
                                        reasons.append(f"✅ Low IV Rank ({analysis.iv_rank}%) - good for buying options")
                                    else:
                                        reasons.append(f"⚠️ Moderate IV Rank ({analysis.iv_rank}%) - mixed signals")
                                    
                                    if analysis.trend != "NEUTRAL":
                                        timeframe_score += 25
                                        reasons.append(f"✅ Clear trend ({analysis.trend}) - easier to pick direction")
                                    
                                    if 30 < analysis.rsi < 70:
                                        timeframe_score += 20
                                        reasons.append("✅ RSI in tradeable range")
                                    
                                    if volume_vs_avg > 50:
                                        timeframe_score += 15
                                        reasons.append(f"✅ Good volume activity (+{volume_vs_avg:.0f}%)")
                                    
                                    if len(analysis.catalysts) > 0:
                                        timeframe_score += 10
                                        reasons.append("✅ Upcoming catalysts for volatility")
                                    
                                    st.metric("Options Trading Suitability", f"{timeframe_score}/100")
                                    
                                    for reason in reasons:
                                        st.write(reason)
                                    
                                    if timeframe_score > 70:
                                        st.success("🟢 **EXCELLENT** for options trading!")
                                    elif timeframe_score > 50:
                                        st.info("🟡 **GOOD** for options trading")
                                    else:
                                        st.warning("🔴 **POOR** for options trading - wait for better setup")
                                    
                                    # Options-specific recommendations
                                    if analysis.iv_rank > 60:
                                        st.info("💡 **High IV Strategy:** Consider selling puts, covered calls, or iron condors")
                                    elif analysis.iv_rank < 40:
                                        st.info("💡 **Low IV Strategy:** Consider buying calls/puts or debit spreads")
                                
                                # Last analysis timestamp
                                st.caption(f"🕒 Analysis updated: Just now")
                                
                            elif should_refresh:
                                # Only show error if user actually requested analysis but it failed
                                st.error(f"❌ Could not analyze {ticker_symbol}. Check ticker symbol or try again.")
                            # else: neither button clicked, just waiting for user action - no error needed
                                
                        except Exception as e:
                            st.error(f"❌ Error analyzing {ticker_symbol}: {str(e)}")
                            # Fallback to stored analysis if available
                            if ml_score is not None:
                                st.info("📊 Showing cached analysis data:")
                                
                                # Confidence score with color coding
                                if ml_score >= 70:
                                    st.success(f"✅ **HIGH CONFIDENCE** - Score: {ml_score:.0f}/100")
                                elif ml_score >= 50:
                                    st.info(f"📊 **MEDIUM CONFIDENCE** - Score: {ml_score:.0f}/100")
                                else:
                                    st.warning(f"⚠️ **LOW CONFIDENCE** - Score: {ml_score:.0f}/100")
                                
                                # Analysis metrics in grid
                                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                
                                with metric_col1:
                                    trend = ticker.get('trend', 'N/A')
                                    trend_emoji = "📈" if trend == "BULLISH" else "📉" if trend == "BEARISH" else "➡️"
                                    st.metric("Trend", f"{trend_emoji} {trend}")
                                
                                with metric_col2:
                                    sentiment = ticker.get('sentiment_score')
                                    if sentiment is not None:
                                        sentiment_emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😟"
                                        st.metric("Sentiment", f"{sentiment_emoji} {sentiment:.2f}")
                                    else:
                                        st.metric("Sentiment", "N/A")
                                
                                with metric_col3:
                                    rsi = ticker.get('rsi')
                                    if rsi is not None:
                                        rsi_emoji = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                                        st.metric("RSI", f"{rsi_emoji} {rsi:.1f}")
                                    else:
                                        st.metric("RSI", "N/A")
                                
                                with metric_col4:
                                    momentum = ticker.get('momentum')
                                    if momentum is not None:
                                        momentum_emoji = "🚀" if momentum > 5 else "📈" if momentum > 0 else "📉"
                                        st.metric("Momentum", f"{momentum_emoji} {momentum:.1f}%")
                                    else:
                                        st.metric("Momentum", "N/A")
                                
                                # Recommendation if available
                                recommendation = ticker.get('recommendation')
                                if recommendation and recommendation != 'N/A':
                                    rec_emoji = "💰" if "BUY" in recommendation.upper() else "⏱️" if "HOLD" in recommendation.upper() else "🚨"
                                    st.markdown(f"**💡 Recommendation:** {rec_emoji} {recommendation}")
                                
                                # Last analysis timestamp
                                last_analyzed_str = ticker.get('last_analyzed')
                                if last_analyzed_str:
                                    try:
                                        dt_analyzed = datetime.fromisoformat(last_analyzed_str).replace(tzinfo=timezone.utc)
                                        analyzed_ago = (datetime.now(timezone.utc) - dt_analyzed).total_seconds() / 3600
                                        if analyzed_ago < 1:
                                            time_str = "Just now"
                                        elif analyzed_ago < 24:
                                            time_str = f"{analyzed_ago:.0f} hours ago"
                                        else:
                                            time_str = f"{analyzed_ago/24:.0f} days ago"
                                        st.caption(f"🕒 Analysis updated: {time_str}")
                                    except:
                                        pass
                
                with col_actions:
                    st.write("")  # Add some spacing
                    
                    # Quick analyze button with timeframe selection
                    if st.button("🔍 Analyze", key=f"analyze_{ticker_symbol}", help="Run fresh comprehensive analysis", width="stretch"):
                        st.session_state.ml_ticker_to_analyze = ticker_symbol
                        st.session_state.analysis_timeframe = "OPTIONS"  # Default to options
                        # Analysis trigger needs rerun to start analysis process
                        st.rerun()
                    
                    # Refresh analysis button
                    if st.button("🔄 Refresh", key=f"refresh_{ticker_symbol}", help="Refresh analysis data", width="stretch"):
                        st.session_state[f"refresh_{ticker_symbol}"] = True
                        # No rerun needed - flag will be processed on next render
                        st.success(f"⏳ Queued refresh for {ticker_symbol}")
                    
                    # Quick trade button
                    if st.button("⚡ Trade", key=f"trade_{ticker_symbol}", help="Open quick trade interface", width="stretch"):
                        st.session_state.selected_ticker = ticker_symbol
                        st.session_state.show_quick_trade = True
                        st.info(f"💡 Switch to '🚀 Quick Trade' tab to trade {ticker_symbol}")
                    
                    # Edit notes button
                    if st.button("✏️ Edit", key=f"edit_{ticker_symbol}", help="Edit ticker details", width="stretch"):
                        st.session_state[f"editing_{ticker_symbol}"] = True
                        # Edit mode toggle needs rerun to show edit form
                        st.rerun()
                    
                    # Remove button
                    if st.button("🗑️ Remove", key=f"remove_{ticker_symbol}", help="Remove from saved tickers", width="stretch"):
                        if tm.remove_ticker(ticker_symbol):
                            # Invalidate ticker cache to force refresh
                            st.session_state.ticker_cache = {}
                            st.session_state.ticker_cache_timestamp = None
                            st.success(f"🗑️ Removed {ticker_symbol}!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to remove {ticker_symbol}.")
                
                # Edit mode popup
                if st.session_state.get(f"editing_{ticker_symbol}", False):
                    with st.expander(f"✏️ Edit {ticker_symbol}", expanded=True):
                        edit_col1, edit_col2 = st.columns(2)
                        with edit_col1:
                            new_name = st.text_input("Company Name", value=ticker_name, key=f"edit_name_{ticker_symbol}")
                            new_notes = st.text_area("Notes", value=notes, key=f"edit_notes_{ticker_symbol}")
                        with edit_col2:
                            new_sector = st.text_input("Sector", value=ticker.get('sector', ''), key=f"edit_sector_{ticker_symbol}")
                            new_type = st.selectbox("Type", ["stock", "option", "penny_stock", "crypto"], 
                                                   index=["stock", "option", "penny_stock", "crypto"].index(ticker_type) if ticker_type in ["stock", "option", "penny_stock", "crypto"] else 0,
                                                   key=f"edit_type_{ticker_symbol}")
                        
                        button_col1, button_col2 = st.columns(2)
                        with button_col1:
                            if st.button("� Save Changes", key=f"save_{ticker_symbol}"):
                                if tm.add_ticker(ticker_symbol, name=new_name, sector=new_sector, ticker_type=new_type, notes=new_notes):
                                    st.success(f"✅ Updated {ticker_symbol}!")
                                    st.session_state[f"editing_{ticker_symbol}"] = False
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to update ticker.")
                        with button_col2:
                            if st.button("❌ Cancel", key=f"cancel_{ticker_symbol}"):
                                st.session_state[f"editing_{ticker_symbol}"] = False
                                st.rerun()
                
                st.divider()  # Separator between cards
        
        # Show ML analysis if requested
        if 'ml_ticker_to_analyze' in st.session_state:
            ticker_to_analyze = st.session_state.ml_ticker_to_analyze
            st.divider()
            st.subheader(f"🧠 ML-Enhanced Analysis: {ticker_to_analyze}")
            
            with st.spinner(f"Analyzing {ticker_to_analyze} with 50+ alpha factors..."):
                if AlphaFactorCalculator is None:
                    st.warning("AlphaFactorCalculator module not available")
                    if st.button("❌ Close"):
                        del st.session_state.ml_ticker_to_analyze
                        st.rerun()
                else:
                    try:
                        alpha_calc = AlphaFactorCalculator()
                        alpha_factors = alpha_calc.calculate_factors(ticker_to_analyze)
                        
                        if alpha_factors:
                            col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
                            
                            with col_ml1:
                                momentum = alpha_factors.get('return_20d', 0) * 100
                                st.metric("20-Day Return", f"{momentum:+.1f}%")
                            
                            with col_ml2:
                                vol_ratio = alpha_factors.get('volume_5d_ratio', 1)
                                st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
                            
                            with col_ml3:
                                rsi = alpha_factors.get('rsi_14', 50)
                                st.metric("RSI", f"{rsi:.1f}")
                            
                            with col_ml4:
                                volatility = alpha_factors.get('volatility_20d', 0) * 100
                                st.metric("Volatility", f"{volatility:.1f}%")
                            
                            # Calculate ML score
                            ml_score = 50  # baseline
                            if momentum > 5:
                                ml_score += 15
                            elif momentum < -5:
                                ml_score -= 15
                            if vol_ratio > 1.5:
                                ml_score += 10
                            if 30 < rsi < 70:
                                ml_score += 10
                            
                            ml_score = max(0, min(100, ml_score))
                            
                            if ml_score >= 70:
                                st.success(f"✅ **HIGH CONFIDENCE** - ML Score: {ml_score:.1f}/100")
                                st.write("Strong signals across multiple factors. Good opportunity.")
                            elif ml_score >= 50:
                                st.info(f"📊 **MEDIUM CONFIDENCE** - ML Score: {ml_score:.1f}/100")
                                st.write("Mixed signals. Monitor for better entry.")
                            else:
                                st.warning(f"⚠️ **LOW CONFIDENCE** - ML Score: {ml_score:.1f}/100")
                                st.write("Weak signals. Consider waiting or passing.")
                            
                            if st.button("❌ Close Analysis"):
                                del st.session_state.ml_ticker_to_analyze
                                st.rerun()
                        else:
                            st.error(f"Could not calculate alpha factors for {ticker_to_analyze}")
                            if st.button("❌ Close"):
                                del st.session_state.ml_ticker_to_analyze
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        if st.button("❌ Close"):
                            del st.session_state.ml_ticker_to_analyze
                            st.rerun()
        
        # Pagination controls at bottom
        if total_pages > 1:
            st.divider()
            col_p1_btm, col_p2_btm, col_p3_btm, col_p4_btm = st.columns([1, 2, 2, 1])
            with col_p1_btm:
                if st.button("◀ Previous", disabled=st.session_state.ticker_page == 1, key="prev_bottom"):
                    st.session_state.ticker_page -= 1
                    st.rerun()
            with col_p2_btm:
                st.write(f"**Page {st.session_state.ticker_page} of {total_pages}**")
            with col_p3_btm:
                st.write(f"**Showing {min(items_per_page, total_tickers)} of {total_tickers} tickers**")
            with col_p4_btm:
                if st.button("Next ▶", disabled=st.session_state.ticker_page == total_pages, key="next_bottom"):
                    st.session_state.ticker_page += 1
                    st.rerun()
    else:
        st.info("No saved tickers yet. Add some above!")
    
    # Clear refresh_all_tickers flag if it was set
    if st.session_state.get('refresh_all_tickers', False):
        st.session_state.refresh_all_tickers = False
    
    # Note: Multi-config analysis is now at the TOP of this page (line 130)
    # No need for duplicate section at bottom


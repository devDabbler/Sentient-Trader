"""
Crypto Watchlist UI Components
Streamlit UI components for displaying and managing crypto watchlists
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
from services.crypto_watchlist_manager import CryptoWatchlistManager



def display_crypto_watchlist_header():
    """Display header for crypto watchlist section"""
    st.markdown("### 📊 My Crypto Watchlist")
    st.write("Track your favorite cryptocurrencies with real-time analysis and signals")


def display_add_crypto_form(manager: CryptoWatchlistManager):
    """Display form to add crypto to watchlist"""
    with st.expander("➕ Add Crypto to Watchlist", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_symbol = st.text_input(
                "Crypto Pair Symbol",
                placeholder="e.g., BTC/USD, ETH/USD, SOL/USD",
                help="Enter crypto pair symbol (format: BASE/QUOTE)"
            ).upper()
        
        with col2:
            if st.button("Add to Watchlist", type="primary", use_container_width=True):
                if new_symbol:
                    if '/' not in new_symbol:
                        st.error("❌ Invalid format. Use format like: BTC/USD")
                    else:
                        with st.spinner(f"Adding {new_symbol}..."):
                            success = manager.add_crypto(new_symbol)
                            if success:
                                st.success(f"✅ Added {new_symbol} to watchlist!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to add {new_symbol}")
                else:
                    st.warning("Please enter a symbol")


def display_watchlist_filters():
    """Display filter controls for watchlist"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sort_by = st.selectbox(
            "Sort By",
            ['Score (High to Low)', 'Score (Low to High)', 
             'Change % (High to Low)', 'Change % (Low to High)',
             'Volume (High to Low)', 'Recently Added'],
            key="crypto_wl_sort"
        )
    
    with col2:
        strategy_filter = st.multiselect(
            "Strategy",
            ['ALL', 'SCALP', 'MOMENTUM', 'SWING', 'BUZZING', 'HOTTEST', 'BREAKOUT'],
            default=['ALL'],
            key="crypto_wl_strategy"
        )
    
    with col3:
        risk_filter = st.multiselect(
            "Risk Level",
            ['LOW', 'MEDIUM', 'HIGH', 'EXTREME'],
            default=['LOW', 'MEDIUM', 'HIGH', 'EXTREME'],
            key="crypto_wl_risk"
        )
    
    with col4:
        confidence_filter = st.multiselect(
            "Confidence",
            ['HIGH', 'MEDIUM', 'LOW'],
            default=['HIGH', 'MEDIUM', 'LOW'],
            key="crypto_wl_confidence"
        )
    
    return {
        'sort_by': sort_by,
        'strategy': strategy_filter,
        'risk': risk_filter,
        'confidence': confidence_filter
    }


def apply_watchlist_filters(watchlist: List[Dict], filters: Dict) -> List[Dict]:
    """Apply filters to watchlist data"""
    filtered = watchlist.copy()
    
    # Strategy filter
    if 'ALL' not in filters['strategy']:
        filtered = [w for w in filtered if w.get('strategy') in filters['strategy']]
    
    # Risk filter
    filtered = [w for w in filtered if w.get('risk_level') in filters['risk']]
    
    # Confidence filter
    filtered = [w for w in filtered if w.get('confidence_level') in filters['confidence']]
    
    # Sorting
    sort_by = filters['sort_by']
    if 'Score (High to Low)' in sort_by:
        filtered = sorted(filtered, key=lambda x: x.get('composite_score', 0), reverse=True)
    elif 'Score (Low to High)' in sort_by:
        filtered = sorted(filtered, key=lambda x: x.get('composite_score', 0))
    elif 'Change % (High to Low)' in sort_by:
        filtered = sorted(filtered, key=lambda x: x.get('change_pct_24h', 0), reverse=True)
    elif 'Change % (Low to High)' in sort_by:
        filtered = sorted(filtered, key=lambda x: x.get('change_pct_24h', 0))
    elif 'Volume (High to Low)' in sort_by:
        filtered = sorted(filtered, key=lambda x: x.get('volume_24h', 0), reverse=True)
    elif 'Recently Added' in sort_by:
        filtered = sorted(filtered, key=lambda x: x.get('date_added', ''), reverse=True)
    
    return filtered


def display_watchlist_summary(watchlist: List[Dict]):
    """Display summary statistics for watchlist"""
    if not watchlist:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Cryptos", len(watchlist))
    
    with col2:
        avg_score = sum(w.get('composite_score', 0) for w in watchlist) / len(watchlist)
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    with col3:
        high_conf = sum(1 for w in watchlist if w.get('confidence_level') == 'HIGH')
        st.metric("High Confidence", f"{high_conf}/{len(watchlist)}")
    
    with col4:
        gainers = sum(1 for w in watchlist if w.get('change_pct_24h', 0) > 0)
        st.metric("24h Gainers", f"{gainers}/{len(watchlist)}")
    
    with col5:
        low_risk = sum(1 for w in watchlist if w.get('risk_level') == 'LOW')
        st.metric("Low Risk", f"{low_risk}/{len(watchlist)}")


def display_crypto_card(crypto: Dict, index: int, manager: CryptoWatchlistManager):
    """Display a single crypto watchlist card"""
    symbol = crypto.get('symbol', 'N/A')
    score = crypto.get('composite_score', 0)
    confidence = crypto.get('confidence_level', 'UNKNOWN')
    risk = crypto.get('risk_level', 'UNKNOWN')
    strategy = crypto.get('strategy', 'N/A')
    
    # Build card title
    change_pct = crypto.get('change_pct_24h', 0)
    direction = "🟢" if change_pct > 0 else "🔴"
    
    card_title = f"#{index} | {symbol} | Score: {score:.1f} | {confidence} Conf | {risk} Risk | {direction} {change_pct:.2f}%"
    
    with st.expander(card_title, expanded=(index <= 3)):
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            price = crypto.get('current_price')
            if price:
                st.metric("Current Price", f"${price:,.2f}")
            else:
                st.metric("Current Price", "N/A")
            st.metric("Strategy", strategy.upper())
        
        with col2:
            st.metric("24h Change", f"{direction} {change_pct:.2f}%")
            vol = crypto.get('volatility_24h')
            if vol:
                st.metric("Volatility", f"{vol:.2f}%")
            else:
                st.metric("Volatility", "N/A")
        
        with col3:
            volume = crypto.get('volume_24h')
            if volume:
                st.metric("Volume 24h", f"${volume:,.0f}")
            else:
                st.metric("Volume 24h", "N/A")
            
            vol_ratio = crypto.get('volume_ratio')
            if vol_ratio:
                st.metric("Vol Ratio", f"{vol_ratio:.2f}x")
            else:
                st.metric("Vol Ratio", "N/A")
        
        with col4:
            st.metric("Confidence", confidence)
            st.metric("Risk Level", risk)
        
        # Technical indicators
        with st.expander("📊 Technical Indicators", expanded=False):
            tcol1, tcol2 = st.columns(2)
            
            with tcol1:
                st.markdown("**Momentum**")
                rsi = crypto.get('rsi')
                if rsi:
                    st.text(f"RSI: {rsi:.2f}")
                
                momentum_score = crypto.get('momentum_score')
                if momentum_score:
                    st.text(f"Momentum Score: {momentum_score:.1f}")
                
                st.markdown("**Moving Averages**")
                ema8 = crypto.get('ema_8')
                ema20 = crypto.get('ema_20')
                if ema8:
                    st.text(f"EMA8: ${ema8:,.2f}")
                if ema20:
                    st.text(f"EMA20: ${ema20:,.2f}")
            
            with tcol2:
                st.markdown("**MACD**")
                macd_line = crypto.get('macd_line')
                macd_signal = crypto.get('macd_signal')
                macd_hist = crypto.get('macd_histogram')
                
                if macd_line is not None:
                    st.text(f"MACD Line: {macd_line:.4f}")
                if macd_signal is not None:
                    st.text(f"Signal: {macd_signal:.4f}")
                if macd_hist is not None:
                    hist_direction = "🟢" if macd_hist > 0 else "🔴"
                    st.text(f"Histogram: {hist_direction} {macd_hist:.4f}")
                
                st.markdown("**Bollinger Bands**")
                bb_upper = crypto.get('bb_upper')
                bb_middle = crypto.get('bb_middle')
                bb_lower = crypto.get('bb_lower')
                
                if bb_upper:
                    st.text(f"Upper: ${bb_upper:,.2f}")
                if bb_middle:
                    st.text(f"Middle: ${bb_middle:,.2f}")
                if bb_lower:
                    st.text(f"Lower: ${bb_lower:,.2f}")
        
        # Analysis & reasoning
        reasoning = crypto.get('reasoning')
        if reasoning:
            st.markdown("**📝 Analysis:**")
            st.info(reasoning)
        
        notes = crypto.get('notes')
        if notes:
            st.markdown("**💭 Notes:**")
            st.text_area("Notes", value=notes, key=f"notes_view_{symbol}", disabled=True)
        
        # Timestamps
        date_added = crypto.get('date_added')
        last_updated = crypto.get('last_updated')
        
        if date_added or last_updated:
            st.divider()
            tcol1, tcol2 = st.columns(2)
            
            with tcol1:
                if date_added:
                    st.text(f"Added: {date_added}")
            
            with tcol2:
                if last_updated:
                    st.text(f"Updated: {last_updated}")
        
        st.divider()
        
        # Action buttons
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        
        with bcol1:
            if st.button("📈 Generate Signal", key=f"gen_signal_{symbol}_{index}"):
                st.session_state.crypto_signal_symbol = symbol
                st.info(f"Generating signals for {symbol}...")
        
        with bcol2:
            if st.button("🔄 Refresh Data", key=f"refresh_{symbol}_{index}"):
                with st.spinner(f"Refreshing {symbol}..."):
                    # Trigger refresh by setting flag
                    st.session_state.crypto_refresh_symbol = symbol
                    st.rerun()
        
        with bcol3:
            if st.button("🏷️ Manage Tags", key=f"tags_{symbol}_{index}"):
                st.session_state.crypto_manage_tags_symbol = symbol
                st.info(f"Tag management for {symbol} - Coming soon!")
        
        with bcol4:
            if st.button("🗑️ Remove", key=f"remove_{symbol}_{index}", type="secondary"):
                if st.session_state.get(f"confirm_remove_{symbol}"):
                    with st.spinner(f"Removing {symbol}..."):
                        success = manager.remove_crypto(symbol)
                        if success:
                            st.success(f"✅ Removed {symbol}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to remove {symbol}")
                else:
                    st.session_state[f"confirm_remove_{symbol}"] = True
                    st.warning("⚠️ Click again to confirm removal")


def display_empty_watchlist():
    """Display message when watchlist is empty"""
    st.info("""
    🔍 **Your crypto watchlist is empty!**
    
    Get started by:
    1. Use the scanner to find opportunities
    2. Click "Save to Watchlist" on promising cryptos
    3. Or manually add symbols using the form above
    
    Your watchlist will persist across sessions and sync via Supabase!
    """)


def display_crypto_watchlist_actions(manager: CryptoWatchlistManager):
    """Display bulk actions for watchlist"""
    with st.expander("⚙️ Bulk Actions", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh All Data", use_container_width=True):
                with st.spinner("Refreshing all cryptos..."):
                    # Trigger full refresh
                    st.session_state.crypto_refresh_all = True
                    st.rerun()
        
        with col2:
            if st.button("📊 Export to CSV", use_container_width=True):
                try:
                    watchlist = manager.get_all_cryptos()
                    if watchlist:
                        df = pd.DataFrame(watchlist)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "crypto_watchlist.csv",
                            "text/csv",
                            key="download_crypto_csv"
                        )
                    else:
                        st.warning("Watchlist is empty")
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        with col3:
            if st.button("🗑️ Clear Watchlist", use_container_width=True, type="secondary"):
                if st.session_state.get('confirm_clear_crypto_watchlist'):
                    # Perform clear
                    st.warning("⚠️ Clear functionality - requires confirmation in database")
                    st.session_state.confirm_clear_crypto_watchlist = False
                else:
                    st.session_state.confirm_clear_crypto_watchlist = True
                    st.warning("⚠️ Click again to confirm clearing entire watchlist")


def render_crypto_watchlist_tab(manager: CryptoWatchlistManager):
    """Main function to render complete crypto watchlist tab"""
    display_crypto_watchlist_header()
    
    # Add crypto form
    display_add_crypto_form(manager)
    
    st.divider()
    
    # Get watchlist
    try:
        watchlist = manager.get_all_cryptos()
        
        if watchlist:
            # Display summary
            display_watchlist_summary(watchlist)
            
            st.divider()
            
            # Filters
            filters = display_watchlist_filters()
            
            # Apply filters
            filtered_watchlist = apply_watchlist_filters(watchlist, filters)
            
            st.divider()
            
            # Bulk actions
            display_crypto_watchlist_actions(manager)
            
            st.divider()
            
            # Display count
            st.markdown(f"**Showing {len(filtered_watchlist)} of {len(watchlist)} cryptos**")
            
            # Display each crypto
            for i, crypto in enumerate(filtered_watchlist, 1):
                display_crypto_card(crypto, i, manager)
        
        else:
            display_empty_watchlist()
    
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        logger.error(f"Crypto watchlist error: {e}", exc_info=True)

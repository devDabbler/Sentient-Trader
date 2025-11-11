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


def display_crypto_card(crypto: Dict, index: int, manager: CryptoWatchlistManager, kraken_client=None):
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
        
        # News & Sentiment Analysis
        sentiment_data = crypto.get('sentiment_data') or crypto.get('news_sentiment')
        if sentiment_data:
            with st.expander("📰 News & Sentiment Analysis", expanded=False):
                sent_col1, sent_col2 = st.columns(2)
                
                with sent_col1:
                    if isinstance(sentiment_data, dict):
                        news_count = sentiment_data.get('news_count', 0)
                        overall_sentiment = sentiment_data.get('overall_sentiment', 'NEUTRAL')
                        sentiment_score = sentiment_data.get('sentiment_score', 0.0)
                        bullish = sentiment_data.get('bullish_articles', 0)
                        bearish = sentiment_data.get('bearish_articles', 0)
                        
                        st.metric("News Articles", news_count)
                        sentiment_emoji = "🟢" if overall_sentiment == 'BULLISH' else "🔴" if overall_sentiment == 'BEARISH' else "⚪"
                        st.metric("Overall Sentiment", f"{sentiment_emoji} {overall_sentiment}")
                        st.metric("Sentiment Score", f"{sentiment_score:+.2f}")
                
                with sent_col2:
                    if isinstance(sentiment_data, dict):
                        st.markdown("**Article Breakdown:**")
                        st.text(f"🟢 Bullish: {bullish}")
                        st.text(f"🔴 Bearish: {bearish}")
                        st.text(f"⚪ Neutral: {sentiment_data.get('neutral_articles', 0)}")
                        
                        # Social sentiment if available
                        social = sentiment_data.get('social_sentiment', {})
                        if social:
                            st.markdown("**Social Mentions:**")
                            reddit = social.get('reddit_mentions', 0)
                            twitter = social.get('twitter_mentions', 0)
                            if reddit > 0:
                                st.text(f"Reddit: {reddit}")
                            if twitter > 0:
                                st.text(f"Twitter: {twitter}")
                        
                        # Major catalysts
                        catalysts = sentiment_data.get('major_catalysts', [])
                        if catalysts:
                            st.markdown("**Major Catalysts:**")
                            for catalyst in catalysts[:3]:  # Show top 3
                                st.text(f"• {catalyst}")
        
        # Whale Wallet Activity
        whale_data = crypto.get('whale_data') or crypto.get('whale_insights')
        whale_alert = crypto.get('whale_alert')
        if whale_data or whale_alert:
            with st.expander("🐋 Whale Wallet Activity", expanded=False):
                whale_col1, whale_col2 = st.columns(2)
                
                with whale_col1:
                    if isinstance(whale_data, dict):
                        whale_score = whale_data.get('whale_activity_score', 0)
                        st.metric("Whale Activity Score", f"{whale_score:.0f}/100")
                        
                        exchange_flow = whale_data.get('exchange_flow', {})
                        net_flow = exchange_flow.get('net_flow', 0)
                        if net_flow != 0:
                            direction = "🟢 INFLOW" if net_flow > 0 else "🔴 OUTFLOW"
                            st.metric("Exchange Flow", direction)
                            st.text(f"Amount: ${abs(net_flow):,.0f}")
                
                with whale_col2:
                    if whale_alert:
                        if isinstance(whale_alert, dict):
                            alert_desc = whale_alert.get('description', '')
                            confidence = whale_alert.get('confidence', 0.0)
                            direction = whale_alert.get('direction', 'UNKNOWN')
                        else:
                            alert_desc = str(whale_alert)
                            confidence = 0.0
                            direction = 'UNKNOWN'
                        
                        st.markdown("**🐋 Whale Alert:**")
                        st.text(alert_desc)
                        if confidence > 0:
                            st.metric("Confidence", f"{confidence:.0%}")
                        if direction != 'UNKNOWN':
                            flow_emoji = "🟢" if direction == 'INFLOW' else "🔴"
                            st.text(f"Direction: {flow_emoji} {direction}")
        
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
                # Generate signals inline without switching tabs
                with st.spinner(f"Generating signals for {symbol}..."):
                    try:
                        from ui.crypto_signal_ui import get_all_crypto_strategies, analyze_symbol_with_strategies
                        from loguru import logger
                        
                        # Check if Kraken client is available
                        if not kraken_client:
                            st.error("Kraken client not available. Please ensure you're connected to Kraken.")
                        else:
                            # Get all strategies
                            all_strategies = get_all_crypto_strategies()
                            
                            # Analyze with default strategies (first 3)
                            selected_strategies = list(all_strategies.keys())[:3]
                            timeframe = '15m'
                            
                            signals = analyze_symbol_with_strategies(
                                symbol=symbol,
                                strategies=all_strategies,
                                selected_strategy_names=selected_strategies,
                                kraken_client=kraken_client,
                                timeframe=timeframe
                            )
                            
                            # Display results inline
                            if signals:
                                st.success(f"✅ Generated {len(signals)} signal(s)")
                                
                                # Sort by confidence
                                signals.sort(key=lambda x: x.confidence, reverse=True)
                                
                                # Display in compact format
                                for sig in signals:
                                    sig_color = "🟢" if sig.signal_type == 'BUY' else "🔴" if sig.signal_type == 'SELL' else "⚪"
                                    st.markdown(f"""
                                    **{sig_color} {sig.strategy}** - {sig.signal_type} ({sig.confidence:.1f}% confidence)
                                    - Entry: ${sig.entry_price:.6f} | Stop: ${sig.stop_loss:.6f} | Target: ${sig.take_profit:.6f}
                                    - Risk/Reward: {sig.risk_reward_ratio:.2f}x | Risk: {sig.risk_level}
                                    - {sig.reasoning[:150]}...
                                    """)
                            else:
                                st.info("No signals generated. Market conditions may not meet strategy criteria.")
                    except Exception as e:
                        st.error(f"Error generating signals: {e}")
                        logger.error(f"Inline signal generation error: {e}", exc_info=True)
        
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


def display_crypto_watchlist_actions(manager: CryptoWatchlistManager, watchlist: List[Dict] = None):
    """Display bulk actions for watchlist"""
    with st.expander("⚙️ Bulk Actions", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        # Check if AI, whale tracking, and news/sentiment are available
        feature_hints = []
        try:
            import os
            if os.getenv('OPENROUTER_API_KEY'):
                feature_hints.append("AI/ML")
            # Etherscan API key works for both Ethereum and BSC (merged services)
            if os.getenv('ETHERSCAN_API_KEY') or os.getenv('SOLSCAN_API_KEY'):
                feature_hints.append("Whale")
            # News/sentiment uses CoinGecko API (free tier, no key required)
            feature_hints.append("News")
        except:
            pass
        
        hint_text = f" ({', '.join(feature_hints)} enabled)" if feature_hints else ""
        
        with col1:
            if st.button(f"🔄 Refresh All Data{hint_text}", use_container_width=True):
                with st.spinner("Refreshing all cryptos..."):
                    # Trigger full refresh
                    st.session_state.crypto_refresh_all = True
                    st.rerun()
        
        with col2:
            # Refresh selected cryptos
            selected_count = len(st.session_state.get('crypto_selected_symbols', []))
            if watchlist:
                selected_text = f" ({selected_count} selected)" if selected_count > 0 else ""
                if st.button(f"🔄 Refresh Selected{selected_text}", use_container_width=True, 
                           disabled=selected_count == 0):
                    if selected_count > 0:
                        st.session_state.crypto_refresh_selected = True
                        st.rerun()
                    else:
                        st.warning("Please select at least one crypto to refresh")
            else:
                st.button("🔄 Refresh Selected", use_container_width=True, disabled=True)
        
        with col3:
            if st.button("📊 Export to CSV", use_container_width=True):
                try:
                    if watchlist is None:
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
        
        with col4:
            if st.button("🗑️ Clear Watchlist", use_container_width=True, type="secondary"):
                if st.session_state.get('confirm_clear_crypto_watchlist'):
                    # Perform clear
                    st.warning("⚠️ Clear functionality - requires confirmation in database")
                    st.session_state.confirm_clear_crypto_watchlist = False
                else:
                    st.session_state.confirm_clear_crypto_watchlist = True
                    st.warning("⚠️ Click again to confirm clearing entire watchlist")


def bulk_refresh_cryptos(
    manager: CryptoWatchlistManager,
    kraken_client,
    crypto_config,
    symbols: List[str],
    use_ai: bool = True,
    use_whale_tracking: bool = True,
    use_news_sentiment: bool = True,
    use_filtered: bool = False
) -> Dict[str, bool]:
    """
    Bulk refresh/analyze cryptos in the watchlist with comprehensive hybrid analysis
    
    Analysis includes:
    - Technical indicators (RSI, EMA, volume, volatility)
    - AI/ML analysis (LLM-powered assessment)
    - Whale tracking (large wallet movements)
    - News & sentiment analysis (news articles + social media)
    
    All signals are validated through hybrid analysis for maximum confidence.
    
    Args:
        manager: CryptoWatchlistManager instance
        kraken_client: KrakenClient instance
        crypto_config: Crypto trading configuration
        symbols: List of symbols to refresh
        use_ai: If True, use AI/ML analysis (requires OPENROUTER_API_KEY)
        use_whale_tracking: If True, track whale wallets (requires blockchain API keys)
        use_news_sentiment: If True, analyze news and sentiment (uses CoinGecko API)
        use_filtered: If True, only refresh symbols that match current filters
        
    Returns:
        Dict mapping symbol to success status
    """
    from services.crypto_scanner import CryptoOpportunityScanner
    from utils.crypto_pair_utils import normalize_crypto_pair, extract_base_asset
    
    results = {}
    
    # Use AI scanner if requested and available, otherwise use base scanner
    if use_ai:
        try:
            from services.ai_crypto_scanner import AICryptoScanner
            scanner = AICryptoScanner(kraken_client, crypto_config)
            base_scanner = scanner.base_scanner
            logger.info("🤖 Using AI-Enhanced Crypto Scanner for analysis")
        except Exception as e:
            logger.warning(f"⚠️ AI scanner not available, using base scanner: {e}")
            scanner = None
            base_scanner = CryptoOpportunityScanner(kraken_client, crypto_config)
            use_ai = False
    else:
        scanner = None
        base_scanner = CryptoOpportunityScanner(kraken_client, crypto_config)
    
    # Initialize whale tracker if requested
    whale_tracker = None
    if use_whale_tracking:
        try:
            from services.crypto_whale_tracker import CryptoWhaleTracker
            whale_tracker = CryptoWhaleTracker()
            logger.info("🐋 Whale tracking enabled")
        except Exception as e:
            logger.warning(f"⚠️ Whale tracker not available: {e}")
            whale_tracker = None
            use_whale_tracking = False
    
    # Initialize news/sentiment analyzer if requested
    news_analyzer = None
    if use_news_sentiment:
        try:
            from services.crypto_news_analyzer import CryptoNewsAnalyzer
            news_analyzer = CryptoNewsAnalyzer()
            logger.info("📰 News & sentiment analysis enabled")
        except Exception as e:
            logger.warning(f"⚠️ News analyzer not available: {e}")
            news_analyzer = None
            use_news_sentiment = False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        total = len(symbols)
        for idx, symbol in enumerate(symbols):
            progress = (idx + 1) / total
            progress_bar.progress(progress)
            
            # Update status with analysis types
            analysis_types = []
            if use_ai and scanner:
                analysis_types.append("AI")
            if use_whale_tracking and whale_tracker:
                analysis_types.append("Whale")
            if use_news_sentiment and news_analyzer:
                analysis_types.append("News")
            status_suffix = f" with {', '.join(analysis_types)}..." if analysis_types else "..."
            status_text.text(f"Analyzing {symbol} ({idx + 1}/{total}){status_suffix}")
            
            try:
                # Try different strategies to find best match
                strategies = ['momentum', 'scalp', 'swing']
                best_opportunity = None
                
                for strategy in strategies:
                    opportunity = base_scanner._analyze_crypto_pair(symbol, strategy)
                    if opportunity:
                        if best_opportunity is None or opportunity.score > best_opportunity.score:
                            best_opportunity = opportunity
                
                if best_opportunity:
                    # Get news/sentiment analysis if available (hybrid analysis)
                    news_sentiment = None
                    if use_news_sentiment and news_analyzer:
                        try:
                            base_asset = extract_base_asset(symbol)
                            
                            # Use asyncio to run async functions in sync context
                            import asyncio
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            news_sentiment = loop.run_until_complete(
                                news_analyzer.analyze_comprehensive_sentiment(
                                    base_asset,
                                    include_social=True,
                                    hours=24
                                )
                            )
                            
                            logger.info(f"📰 News & sentiment analysis complete for {symbol}")
                        except Exception as e:
                            logger.warning(f"⚠️ News/sentiment analysis failed for {symbol}, continuing without: {e}")
                            news_sentiment = None
                    
                    # Get whale insights if available (hybrid analysis)
                    whale_insights = None
                    whale_alert = None
                    if use_whale_tracking and whale_tracker:
                        try:
                            base_asset = extract_base_asset(symbol)
                            # Determine chain based on symbol (simplified - could be enhanced)
                            chain = 'ethereum'  # Default, could be determined by symbol
                            if base_asset in ['SOL', 'USDC-SPL']:
                                chain = 'solana'
                            elif base_asset in ['BNB', 'BUSD']:
                                chain = 'bsc'
                            
                            # Use asyncio to run async functions in sync context
                            import asyncio
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            whale_insights = loop.run_until_complete(
                                whale_tracker.get_whale_insights(
                                    base_asset,
                                    chain=chain,
                                    hours=24
                                )
                            )
                            
                            # Generate whale alert with hybrid validation
                            if whale_insights:
                                # Get technical data for validation
                                normalized_symbol = normalize_crypto_pair(symbol)
                                ticker = kraken_client.get_ticker_data(normalized_symbol)
                                technical_data = {
                                    'change_pct_24h': best_opportunity.change_pct_24h,
                                    'volume_ratio': best_opportunity.volume_ratio,
                                    'rsi': None  # Will be calculated below
                                }
                                
                                volume_data = {
                                    'volume_24h': best_opportunity.volume_24h,
                                    'volume_ratio': best_opportunity.volume_ratio
                                }
                                
                                # Include news/sentiment data for whale validation
                                news_data = None
                                sentiment_data = None
                                if news_sentiment:
                                    news_data = {
                                        'overall_sentiment': news_sentiment.overall_sentiment,
                                        'overall_sentiment_score': news_sentiment.sentiment_score,
                                        'news_count': news_sentiment.news_count
                                    }
                                    # Also include social sentiment if available
                                    if news_sentiment.social_sentiment:
                                        sentiment_data = news_sentiment.social_sentiment
                                
                                # Generate alert with hybrid validation
                                whale_alert = loop.run_until_complete(
                                    whale_tracker.generate_whale_alert(
                                        base_asset,
                                        [],  # Transactions would be passed here in full implementation
                                        volume_data=volume_data,
                                        technical_data=technical_data,
                                        sentiment_data=sentiment_data,
                                        news_data=news_data
                                    )
                                )
                                
                                logger.info(f"🐋 Whale analysis complete for {symbol}")
                        except Exception as e:
                            logger.warning(f"⚠️ Whale tracking failed for {symbol}, continuing without: {e}")
                            whale_insights = None
                    
                    # Add AI/ML analysis if available
                    ai_opportunity = None
                    if use_ai and scanner and hasattr(scanner, '_add_ai_confidence'):
                        try:
                            # Enhance AI analysis with whale data if available
                            if whale_insights and hasattr(best_opportunity, 'whale_activity_score'):
                                best_opportunity.whale_activity_score = whale_insights.get('whale_activity_score', 0)
                                if whale_alert:
                                    best_opportunity.whale_alert = whale_alert.description
                                    best_opportunity.exchange_flow_direction = whale_alert.direction
                                    best_opportunity.whale_confidence = 'HIGH' if whale_alert.confidence > 0.7 else 'MEDIUM'
                            
                            ai_opportunity = scanner._add_ai_confidence(best_opportunity)
                            logger.info(f"🤖 AI analysis complete for {symbol}")
                        except Exception as e:
                            logger.warning(f"⚠️ AI analysis failed for {symbol}, using base analysis: {e}")
                            ai_opportunity = None
                    
                    # Get additional technical indicators if available
                    try:
                        normalized_symbol = normalize_crypto_pair(symbol)
                        ticker = kraken_client.get_ticker_data(normalized_symbol)
                        ohlc_5m = kraken_client.get_ohlc_data(normalized_symbol, interval=5)
                        ohlc_1h = kraken_client.get_ohlc_data(normalized_symbol, interval=60)
                        
                        # Calculate technical indicators
                        rsi = None
                        ema_8 = None
                        ema_20 = None
                        
                        if ohlc_5m and len(ohlc_5m) >= 14:
                            closes_5m = [candle['close'] for candle in ohlc_5m[-14:]]
                            rsi = base_scanner._calculate_rsi(closes_5m, period=14)
                        
                        if ohlc_5m and len(ohlc_5m) >= 20:
                            closes_5m = [candle['close'] for candle in ohlc_5m[-20:]]
                            ema_8 = base_scanner._calculate_ema(closes_5m, period=8)
                        
                        if ohlc_1h and len(ohlc_1h) >= 40:
                            closes_1h = [candle['close'] for candle in ohlc_1h[-40:]]
                            ema_20 = base_scanner._calculate_ema(closes_1h, period=20)
                    except Exception as e:
                        logger.warning(f"Could not get additional indicators for {symbol}: {e}")
                    
                    # Build update data - use AI data if available
                    if ai_opportunity:
                        # Use AI-enhanced opportunity data
                        update_data = {
                            'current_price': ai_opportunity.current_price,
                            'change_pct_24h': ai_opportunity.change_pct_24h,
                            'volume_24h': ai_opportunity.volume_24h,
                            'volume_ratio': ai_opportunity.volume_ratio,
                            'volatility_24h': ai_opportunity.volatility_24h,
                            'composite_score': ai_opportunity.score,
                            'confidence_level': ai_opportunity.confidence,
                            'risk_level': ai_opportunity.risk_level,
                            'strategy': ai_opportunity.strategy,
                            'reasoning': ai_opportunity.reason
                        }
                        
                        # Add AI analysis fields to reasoning if available
                        ai_reasoning_parts = []
                        if hasattr(ai_opportunity, 'ai_reasoning') and ai_opportunity.ai_reasoning:
                            ai_reasoning_parts.append(f"AI Analysis: {ai_opportunity.ai_reasoning}")
                        if hasattr(ai_opportunity, 'ai_risks') and ai_opportunity.ai_risks:
                            ai_reasoning_parts.append(f"Risks: {ai_opportunity.ai_risks}")
                        if hasattr(ai_opportunity, 'social_narrative') and ai_opportunity.social_narrative:
                            ai_reasoning_parts.append(f"Narrative: {ai_opportunity.social_narrative}")
                        if hasattr(ai_opportunity, 'market_cycle_phase') and ai_opportunity.market_cycle_phase:
                            ai_reasoning_parts.append(f"Cycle: {ai_opportunity.market_cycle_phase}")
                        
                        # Add news/sentiment insights to reasoning if available
                        news_reasoning_parts = []
                        if news_sentiment:
                            if news_sentiment.news_count > 0:
                                news_reasoning_parts.append(f"📰 News: {news_sentiment.news_count} articles ({news_sentiment.bullish_articles} bullish, {news_sentiment.bearish_articles} bearish)")
                            
                            if news_sentiment.overall_sentiment != 'NEUTRAL':
                                sentiment_emoji = "🟢" if news_sentiment.overall_sentiment == 'BULLISH' else "🔴"
                                news_reasoning_parts.append(f"Sentiment: {sentiment_emoji} {news_sentiment.overall_sentiment} (Score: {news_sentiment.sentiment_score:+.2f})")
                            
                            if news_sentiment.major_catalysts:
                                catalyst = news_sentiment.major_catalysts[0]  # Top catalyst
                                news_reasoning_parts.append(f"Catalyst: {catalyst}")
                            
                            # Social sentiment if available
                            if news_sentiment.social_sentiment:
                                social = news_sentiment.social_sentiment
                                reddit_count = social.get('reddit_mentions', 0)
                                twitter_count = social.get('twitter_mentions', 0)
                                if reddit_count > 0 or twitter_count > 0:
                                    news_reasoning_parts.append(f"Social: {reddit_count} Reddit, {twitter_count} Twitter")
                        
                        # Add whale insights to reasoning if available
                        whale_reasoning_parts = []
                        if whale_insights:
                            whale_score = whale_insights.get('whale_activity_score', 0)
                            if whale_score > 50:
                                whale_reasoning_parts.append(f"🐋 Whale Activity Score: {whale_score:.0f}/100")
                            
                            exchange_flow = whale_insights.get('exchange_flow', {})
                            if exchange_flow.get('net_flow', 0) != 0:
                                direction = 'INFLOW' if exchange_flow['net_flow'] > 0 else 'OUTFLOW'
                                whale_reasoning_parts.append(f"Exchange Flow: {direction} ${abs(exchange_flow['net_flow']):,.0f}")
                            
                            if whale_alert:
                                whale_reasoning_parts.append(f"Whale Alert: {whale_alert.description} (Confidence: {whale_alert.confidence:.0%})")
                        
                        # Combine all reasoning
                        if ai_reasoning_parts:
                            update_data['reasoning'] = f"{update_data['reasoning']}\n\n🤖 {' | '.join(ai_reasoning_parts)}"
                        
                        if news_reasoning_parts:
                            update_data['reasoning'] = f"{update_data['reasoning']}\n\n📰 {' | '.join(news_reasoning_parts)}"
                        
                        if whale_reasoning_parts:
                            update_data['reasoning'] = f"{update_data['reasoning']}\n\n🐋 {' | '.join(whale_reasoning_parts)}"
                        
                        # Save sentiment and whale data separately (for UI display)
                        if news_sentiment:
                            sentiment_dict = {
                                'news_count': news_sentiment.news_count,
                                'overall_sentiment': news_sentiment.overall_sentiment,
                                'sentiment_score': news_sentiment.sentiment_score,
                                'bullish_articles': news_sentiment.bullish_articles,
                                'bearish_articles': news_sentiment.bearish_articles,
                                'neutral_articles': news_sentiment.neutral_articles,
                                'major_catalysts': news_sentiment.major_catalysts[:5] if news_sentiment.major_catalysts else [],
                            }
                            if hasattr(news_sentiment, 'social_sentiment') and news_sentiment.social_sentiment:
                                sentiment_dict['social_sentiment'] = news_sentiment.social_sentiment
                            update_data['sentiment_data'] = sentiment_dict
                            update_data['news_sentiment'] = sentiment_dict
                        
                        if whale_insights or whale_alert:
                            whale_dict = {}
                            if whale_insights:
                                whale_dict = {
                                    'whale_activity_score': whale_insights.get('whale_activity_score', 0),
                                    'exchange_flow': whale_insights.get('exchange_flow', {}),
                                    'large_transactions': whale_insights.get('large_transactions', []),
                                }
                            if whale_alert:
                                whale_dict['whale_alert'] = {
                                    'description': whale_alert.description if hasattr(whale_alert, 'description') else str(whale_alert),
                                    'confidence': whale_alert.confidence if hasattr(whale_alert, 'confidence') else 0.0,
                                    'direction': whale_alert.direction if hasattr(whale_alert, 'direction') else 'UNKNOWN'
                                }
                            update_data['whale_data'] = whale_dict
                            update_data['whale_insights'] = whale_dict
                            if whale_alert:
                                update_data['whale_alert'] = whale_dict.get('whale_alert')
                        
                        # HYBRID CONFIDENCE SCORING - Combine all signals intelligently
                        confidence_boost = 0
                        confidence_signals = []
                        
                        # News/sentiment contribution (25% weight)
                        if news_sentiment:
                            if news_sentiment.overall_sentiment == 'BULLISH' and news_sentiment.sentiment_score > 0.3:
                                confidence_boost += 0.25
                                confidence_signals.append("Strong bullish news/sentiment")
                            elif news_sentiment.overall_sentiment == 'BEARISH' and news_sentiment.sentiment_score < -0.3:
                                confidence_boost -= 0.15  # Reduce confidence on bearish news
                                confidence_signals.append("Bearish news/sentiment")
                            
                            if news_sentiment.news_count >= 5:  # Significant news volume
                                confidence_boost += 0.1
                                confidence_signals.append("High news volume")
                        
                        # Whale contribution (20% weight)
                        if whale_alert and whale_alert.confidence > 0.6:
                            confidence_boost += 0.2
                            confidence_signals.append("Validated whale signal")
                        elif whale_insights and whale_insights.get('whale_activity_score', 0) > 70:
                            confidence_boost += 0.15
                            confidence_signals.append("High whale activity")
                        
                        # AI contribution (already handled above, but can boost further)
                        if hasattr(ai_opportunity, 'ai_confidence') and ai_opportunity.ai_confidence == 'HIGH':
                            confidence_boost += 0.15
                            confidence_signals.append("High AI confidence")
                        
                        # Technical confirmation (15% weight)
                        if best_opportunity.volume_ratio > 2.0 and best_opportunity.change_pct_24h > 5.0:
                            confidence_boost += 0.15
                            confidence_signals.append("Strong technical momentum")
                        
                        # Apply confidence boost
                        current_confidence = update_data['confidence_level']
                        if confidence_boost >= 0.4:  # Strong alignment across signals
                            if current_confidence == 'MEDIUM':
                                update_data['confidence_level'] = 'HIGH'
                            elif current_confidence == 'LOW':
                                update_data['confidence_level'] = 'MEDIUM'
                        elif confidence_boost >= 0.25:  # Moderate alignment
                            if current_confidence == 'LOW':
                                update_data['confidence_level'] = 'MEDIUM'
                        
                        # Add confidence reasoning if multiple signals align
                        if len(confidence_signals) >= 3:
                            update_data['reasoning'] = f"{update_data['reasoning']}\n\n✅ **Hybrid Validation**: {len(confidence_signals)} signals aligned - {', '.join(confidence_signals[:3])}"
                        
                        # Update confidence level with AI confidence if available
                        if hasattr(ai_opportunity, 'ai_confidence') and ai_opportunity.ai_confidence:
                            # Combine quantitative and AI confidence
                            if ai_opportunity.ai_confidence == 'HIGH':
                                update_data['confidence_level'] = 'HIGH'
                            elif ai_opportunity.ai_confidence == 'MEDIUM' and update_data['confidence_level'] != 'HIGH':
                                update_data['confidence_level'] = 'MEDIUM'
                            elif ai_opportunity.ai_confidence == 'LOW':
                                update_data['confidence_level'] = 'LOW'
                    else:
                        # Use base opportunity data
                        update_data = {
                            'current_price': best_opportunity.current_price,
                            'change_pct_24h': best_opportunity.change_pct_24h,
                            'volume_24h': best_opportunity.volume_24h,
                            'volume_ratio': best_opportunity.volume_ratio,
                            'volatility_24h': best_opportunity.volatility_24h,
                            'composite_score': best_opportunity.score,
                            'confidence_level': best_opportunity.confidence,
                            'risk_level': best_opportunity.risk_level,
                            'strategy': best_opportunity.strategy,
                            'reasoning': best_opportunity.reason
                        }
                        
                        # Add news/sentiment insights to base reasoning if available
                        news_reasoning_parts = []
                        if news_sentiment:
                            if news_sentiment.news_count > 0:
                                news_reasoning_parts.append(f"📰 News: {news_sentiment.news_count} articles ({news_sentiment.bullish_articles} bullish, {news_sentiment.bearish_articles} bearish)")
                            
                            if news_sentiment.overall_sentiment != 'NEUTRAL':
                                sentiment_emoji = "🟢" if news_sentiment.overall_sentiment == 'BULLISH' else "🔴"
                                news_reasoning_parts.append(f"Sentiment: {sentiment_emoji} {news_sentiment.overall_sentiment} (Score: {news_sentiment.sentiment_score:+.2f})")
                            
                            if news_sentiment.major_catalysts:
                                catalyst = news_sentiment.major_catalysts[0]
                                news_reasoning_parts.append(f"Catalyst: {catalyst}")
                            
                            if news_sentiment.social_sentiment:
                                social = news_sentiment.social_sentiment
                                reddit_count = social.get('reddit_mentions', 0)
                                twitter_count = social.get('twitter_mentions', 0)
                                if reddit_count > 0 or twitter_count > 0:
                                    news_reasoning_parts.append(f"Social: {reddit_count} Reddit, {twitter_count} Twitter")
                        
                        # Add whale insights to base reasoning if available
                        whale_reasoning_parts = []
                        if whale_insights:
                            whale_score = whale_insights.get('whale_activity_score', 0)
                            if whale_score > 50:
                                whale_reasoning_parts.append(f"🐋 Whale Activity Score: {whale_score:.0f}/100")
                            
                            exchange_flow = whale_insights.get('exchange_flow', {})
                            if exchange_flow.get('net_flow', 0) != 0:
                                direction = 'INFLOW' if exchange_flow['net_flow'] > 0 else 'OUTFLOW'
                                whale_reasoning_parts.append(f"Exchange Flow: {direction} ${abs(exchange_flow['net_flow']):,.0f}")
                            
                            if whale_alert:
                                whale_reasoning_parts.append(f"Whale Alert: {whale_alert.description}")
                        
                        # Combine all reasoning
                        if news_reasoning_parts:
                            update_data['reasoning'] = f"{update_data['reasoning']}\n\n📰 {' | '.join(news_reasoning_parts)}"
                        
                        if whale_reasoning_parts:
                            update_data['reasoning'] = f"{update_data['reasoning']}\n\n🐋 {' | '.join(whale_reasoning_parts)}"
                        
                        # HYBRID CONFIDENCE SCORING - Combine all signals intelligently
                        confidence_boost = 0
                        confidence_signals = []
                        
                        # News/sentiment contribution (25% weight)
                        if news_sentiment:
                            if news_sentiment.overall_sentiment == 'BULLISH' and news_sentiment.sentiment_score > 0.3:
                                confidence_boost += 0.25
                                confidence_signals.append("Strong bullish news/sentiment")
                            elif news_sentiment.overall_sentiment == 'BEARISH' and news_sentiment.sentiment_score < -0.3:
                                confidence_boost -= 0.15
                                confidence_signals.append("Bearish news/sentiment")
                            
                            if news_sentiment.news_count >= 5:
                                confidence_boost += 0.1
                                confidence_signals.append("High news volume")
                        
                        # Whale contribution (20% weight)
                        if whale_alert and whale_alert.confidence > 0.6:
                            confidence_boost += 0.2
                            confidence_signals.append("Validated whale signal")
                        elif whale_insights and whale_insights.get('whale_activity_score', 0) > 70:
                            confidence_boost += 0.15
                            confidence_signals.append("High whale activity")
                        
                        # Technical confirmation (15% weight)
                        if best_opportunity.volume_ratio > 2.0 and best_opportunity.change_pct_24h > 5.0:
                            confidence_boost += 0.15
                            confidence_signals.append("Strong technical momentum")
                        
                        # Apply confidence boost
                        if confidence_boost >= 0.4:
                            if update_data['confidence_level'] == 'MEDIUM':
                                update_data['confidence_level'] = 'HIGH'
                            elif update_data['confidence_level'] == 'LOW':
                                update_data['confidence_level'] = 'MEDIUM'
                        elif confidence_boost >= 0.25:
                            if update_data['confidence_level'] == 'LOW':
                                update_data['confidence_level'] = 'MEDIUM'
                        
                        # Add confidence reasoning if multiple signals align
                        if len(confidence_signals) >= 3:
                            update_data['reasoning'] = f"{update_data['reasoning']}\n\n✅ **Hybrid Validation**: {len(confidence_signals)} signals aligned - {', '.join(confidence_signals[:3])}"
                    
                    # Add technical indicators if available
                    if rsi is not None:
                        update_data['rsi'] = rsi
                    if ema_8 is not None:
                        update_data['ema_8'] = ema_8
                    if ema_20 is not None:
                        update_data['ema_20'] = ema_20
                    
                    # Save sentiment and whale data separately (for UI display)
                    if news_sentiment:
                        # Convert sentiment object to dict for storage
                        sentiment_dict = {
                            'news_count': news_sentiment.news_count,
                            'overall_sentiment': news_sentiment.overall_sentiment,
                            'sentiment_score': news_sentiment.sentiment_score,
                            'bullish_articles': news_sentiment.bullish_articles,
                            'bearish_articles': news_sentiment.bearish_articles,
                            'neutral_articles': news_sentiment.neutral_articles,
                            'major_catalysts': news_sentiment.major_catalysts[:5] if news_sentiment.major_catalysts else [],
                        }
                        # Include social sentiment if available
                        if hasattr(news_sentiment, 'social_sentiment') and news_sentiment.social_sentiment:
                            sentiment_dict['social_sentiment'] = news_sentiment.social_sentiment
                        update_data['sentiment_data'] = sentiment_dict
                        update_data['news_sentiment'] = sentiment_dict  # Also save with this key for compatibility
                    
                    if whale_insights or whale_alert:
                        # Convert whale data to dict for storage
                        whale_dict = {}
                        if whale_insights:
                            whale_dict = {
                                'whale_activity_score': whale_insights.get('whale_activity_score', 0),
                                'exchange_flow': whale_insights.get('exchange_flow', {}),
                                'large_transactions': whale_insights.get('large_transactions', []),
                            }
                        if whale_alert:
                            whale_dict['whale_alert'] = {
                                'description': whale_alert.description if hasattr(whale_alert, 'description') else str(whale_alert),
                                'confidence': whale_alert.confidence if hasattr(whale_alert, 'confidence') else 0.0,
                                'direction': whale_alert.direction if hasattr(whale_alert, 'direction') else 'UNKNOWN'
                            }
                        update_data['whale_data'] = whale_dict
                        update_data['whale_insights'] = whale_dict  # Also save with this key for compatibility
                        if whale_alert:
                            update_data['whale_alert'] = whale_dict.get('whale_alert')
                    
                    # Update watchlist
                    success = manager.update_crypto_price(symbol, update_data)
                    results[symbol] = success
                    
                    if success:
                        ai_status = " (with AI)" if ai_opportunity else ""
                        logger.info(f"✅ Refreshed {symbol}{ai_status}")
                    else:
                        logger.warning(f"⚠️ Failed to update {symbol} in database")
                        results[symbol] = False
                else:
                    logger.warning(f"⚠️ Could not analyze {symbol}")
                    results[symbol] = False
                    
            except Exception as e:
                logger.error(f"Error refreshing {symbol}: {e}", exc_info=True)
                results[symbol] = False
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        return results
        
    except Exception as e:
        logger.error(f"Bulk refresh error: {e}", exc_info=True)
        status_text.empty()
        return results


def render_crypto_watchlist_tab(
    manager: CryptoWatchlistManager,
    kraken_client=None,
    crypto_config=None
):
    """
    Main function to render complete crypto watchlist tab
    
    Args:
        manager: CryptoWatchlistManager instance
        kraken_client: Optional KrakenClient instance (required for refresh)
        crypto_config: Optional crypto trading config (required for refresh)
    """
    display_crypto_watchlist_header()
    
    # Show refresh status messages if refresh was just completed
    if st.session_state.get('crypto_refresh_status'):
        status = st.session_state.crypto_refresh_status
        if status.get('successful', 0) > 0:
            st.success(f"✅ Successfully refreshed {status['successful']} cryptos")
        if status.get('failed', 0) > 0:
            st.warning(f"⚠️ Failed to refresh {status['failed']} cryptos")
        if status.get('error'):
            st.error(f"Error during refresh: {status['error']}")
        # Clear status after showing
        st.session_state.crypto_refresh_status = None
    
    # Handle bulk refresh
    if st.session_state.get('crypto_refresh_all', False):
        if not kraken_client or not crypto_config:
            st.error("⚠️ Refresh requires Kraken connection. Please ensure you're connected to Kraken.")
            st.session_state.crypto_refresh_all = False
        else:
            try:
                watchlist = manager.get_all_cryptos()
                if watchlist:
                    symbols = [crypto.get('symbol') for crypto in watchlist if crypto.get('symbol')]
                    
                    if symbols:
                        results = bulk_refresh_cryptos(
                            manager,
                            kraken_client,
                            crypto_config,
                            symbols
                        )
                        
                        successful = sum(1 for v in results.values() if v)
                        failed = len(results) - successful
                        
                        # Store status for next render
                        st.session_state.crypto_refresh_status = {
                            'successful': successful,
                            'failed': failed,
                            'error': None
                        }
                    else:
                        st.session_state.crypto_refresh_status = {
                            'successful': 0,
                            'failed': 0,
                            'error': "No symbols found in watchlist"
                        }
                else:
                    st.session_state.crypto_refresh_status = {
                        'successful': 0,
                        'failed': 0,
                        'error': "Watchlist is empty"
                    }
            except Exception as e:
                logger.error(f"Bulk refresh error: {e}", exc_info=True)
                st.session_state.crypto_refresh_status = {
                    'successful': 0,
                    'failed': 0,
                    'error': str(e)
                }
        
        # Clear the flag and rerun
        st.session_state.crypto_refresh_all = False
        st.rerun()
    
    # Handle selected cryptos refresh
    if st.session_state.get('crypto_refresh_selected'):
        selected_symbols = st.session_state.get('crypto_selected_symbols', [])
        if not kraken_client or not crypto_config:
            st.error(f"⚠️ Refresh requires Kraken connection. Cannot refresh selected cryptos.")
            st.session_state.crypto_refresh_selected = False
        elif not selected_symbols:
            st.warning("⚠️ No cryptos selected for refresh")
            st.session_state.crypto_refresh_selected = False
        else:
            try:
                with st.spinner(f"Refreshing {len(selected_symbols)} selected cryptos..."):
                    results = bulk_refresh_cryptos(
                        manager,
                        kraken_client,
                        crypto_config,
                        selected_symbols,
                        use_ai=True,
                        use_whale_tracking=True,
                        use_news_sentiment=True
                    )
                    
                    successful = sum(1 for v in results.values() if v)
                    failed = len(results) - successful
                    
                    if successful > 0:
                        st.success(f"✅ Successfully refreshed {successful} of {len(selected_symbols)} cryptos")
                    if failed > 0:
                        failed_symbols = [s for s, v in results.items() if not v]
                        st.warning(f"⚠️ Failed to refresh {failed} cryptos: {', '.join(failed_symbols)}")
                    
                    # Store status for display
                    st.session_state.crypto_refresh_status = {
                        'successful': successful,
                        'failed': failed,
                        'total': len(selected_symbols)
                    }
            except Exception as e:
                st.error(f"Error refreshing selected cryptos: {e}")
                logger.error(f"Selected refresh error: {e}", exc_info=True)
                st.session_state.crypto_refresh_status = {
                    'successful': 0,
                    'failed': len(selected_symbols),
                    'error': str(e)
                }
        
        # Clear the flag and rerun
        st.session_state.crypto_refresh_selected = False
        st.rerun()
    
    # Handle individual refresh
    if st.session_state.get('crypto_refresh_symbol'):
        symbol = st.session_state.crypto_refresh_symbol
        if not kraken_client or not crypto_config:
            st.error(f"⚠️ Refresh requires Kraken connection. Cannot refresh {symbol}.")
            st.session_state.crypto_refresh_symbol = None
        else:
            try:
                results = bulk_refresh_cryptos(
                    manager,
                    kraken_client,
                    crypto_config,
                    [symbol]
                )
                
                if results.get(symbol, False):
                    st.success(f"✅ Successfully refreshed {symbol}")
                else:
                    st.warning(f"⚠️ Failed to refresh {symbol}")
            except Exception as e:
                st.error(f"Error refreshing {symbol}: {e}")
                logger.error(f"Individual refresh error: {e}", exc_info=True)
        
        # Clear the flag and rerun
        st.session_state.crypto_refresh_symbol = None
        st.rerun()
    
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
            display_crypto_watchlist_actions(manager, watchlist)
            
            st.divider()
            
            # Selection UI for custom refresh
            st.markdown("### 📋 Select Cryptos for Custom Refresh")
            st.caption("Select specific cryptos to refresh individually. Useful for testing or focusing on specific coins.")
            
            # Initialize selected symbols in session state if not exists
            if 'crypto_selected_symbols' not in st.session_state:
                st.session_state.crypto_selected_symbols = []
            
            # Create selection checkboxes
            col_select1, col_select2, col_select3 = st.columns(3)
            all_symbols = [crypto.get('symbol') for crypto in filtered_watchlist if crypto.get('symbol')]
            
            # Select all / Deselect all buttons
            select_col1, select_col2, select_col3 = st.columns([1, 1, 4])
            with select_col1:
                if st.button("✅ Select All", use_container_width=True, key="select_all_crypto"):
                    st.session_state.crypto_selected_symbols = all_symbols.copy()
                    st.rerun()
            
            with select_col2:
                if st.button("❌ Deselect All", use_container_width=True, key="deselect_all_crypto"):
                    st.session_state.crypto_selected_symbols = []
                    st.rerun()
            
            # Display checkboxes in columns
            # Update session state based on checkbox values
            current_selected = st.session_state.crypto_selected_symbols.copy()
            
            for i, crypto in enumerate(filtered_watchlist):
                symbol = crypto.get('symbol')
                if not symbol:
                    continue
                
                col_idx = i % 3
                col = [col_select1, col_select2, col_select3][col_idx]
                
                with col:
                    is_selected = symbol in current_selected
                    checkbox_key = f"crypto_select_{symbol}"
                    
                    # Use checkbox - Streamlit will manage the state via the key
                    checkbox_value = st.checkbox(
                        symbol,
                        value=is_selected,
                        key=checkbox_key,
                        label_visibility="visible"
                    )
                    
                    # Update session state based on checkbox state
                    if checkbox_value and symbol not in st.session_state.crypto_selected_symbols:
                        st.session_state.crypto_selected_symbols.append(symbol)
                    elif not checkbox_value and symbol in st.session_state.crypto_selected_symbols:
                        st.session_state.crypto_selected_symbols.remove(symbol)
            
            # Show selected count
            selected_count = len(st.session_state.crypto_selected_symbols)
            if selected_count > 0:
                st.info(f"✅ {selected_count} cryptos selected: {', '.join(st.session_state.crypto_selected_symbols)}")
            
            st.divider()
            
            # Display count
            st.markdown(f"**Showing {len(filtered_watchlist)} of {len(watchlist)} cryptos**")
            
            # Display each crypto
            for i, crypto in enumerate(filtered_watchlist, 1):
                display_crypto_card(crypto, i, manager, kraken_client)
        
        else:
            display_empty_watchlist()
    
    except Exception as e:
        st.error(f"Error loading watchlist: {e}")
        logger.error(f"Crypto watchlist error: {e}", exc_info=True)

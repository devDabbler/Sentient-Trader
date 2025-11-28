"""
Daily Scanner UI - Progressive crypto scanning workflow

Provides a tiered daily scanning workflow:
1. Quick Filter (100+ coins)
2. Medium Analysis (Top 20)
3. Deep Analysis (Selected 5)
4. Add to Monitoring
"""

import streamlit as st
from loguru import logger
from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import pandas as pd

def display_daily_scanner(kraken_client, crypto_config, ai_trade_reviewer=None):
    """
    Display the daily scanner interface with progressive workflow
    
    Workflow:
    1. Tier 1: Quick scan of all coins (lightweight)
    2. Tier 2: Technical analysis of promising coins
    3. Tier 3: Deep strategy + AI review
    4. Add best to monitoring
    """
    # ========== CHECK FOR MULTI-CONFIG BUTTON CLICKS (BEFORE RENDERING) ==========
    # This catches button clicks from previous renders and transfers setup to Quick Trade
    if 'multi_config_results' in st.session_state and st.session_state.multi_config_results is not None:
        results_df = st.session_state.multi_config_results
        
        # Check if any "Use This Setup" button was clicked (best-per-pair OR filtered results)
        selected_config_idx = None
        for idx in results_df.index:
            # Check both button types: use_config_{idx} and use_filtered_{idx}
            if st.session_state.get(f'use_config_{idx}_clicked', False):
                selected_config_idx = idx
                st.session_state[f'use_config_{idx}_clicked'] = False  # Reset flag
                logger.info(f"🔘 DAILY SCANNER - Detected button click for config {idx} (best-per-pair)")
                break
            elif st.session_state.get(f'use_filtered_{idx}_clicked', False):
                selected_config_idx = idx
                st.session_state[f'use_filtered_{idx}_clicked'] = False  # Reset flag
                logger.info(f"🔘 DAILY SCANNER - Detected button click for filtered {idx} (filtered results)")
                break
        
        # If a config was selected, transfer to Quick Trade
        if selected_config_idx is not None:
            row = results_df.loc[selected_config_idx]
            pair = row.get('pair', 'UNKNOWN')
            trade_type = row.get('trade_type', 'UNKNOWN')
            
            logger.info(f"🔘 DAILY SCANNER - Transferring setup for {pair} - {trade_type}")
            
            # Store complete setup with REAL market data
            st.session_state.crypto_scanner_opportunity = {
                'symbol': row.get('pair', 'UNKNOWN'),
                'strategy': row.get('strategy', 'Unknown'),
                'confidence': row.get('ai_approved', False),
                'risk_level': 'Medium' if (row.get('leverage', 0) or 0) <= 2 else 'High',
                'score': row.get('ai_score', 0),
                'current_price': row.get('current_price', 0),
                'change_24h': row.get('change_24h', 0),
                'volume_ratio': (row.get('volume_24h', 0) or 0) / 1000000 if (row.get('volume_24h', 0) or 0) > 0 else 1.0,
                'volatility': row.get('volatility', 0),
                'reason': f"{row.get('trade_type', 'UNKNOWN')} recommended",
                'ai_reasoning': row.get('ai_recommendation', ''),
                'ai_confidence': 'High' if row.get('ai_confidence', 0) >= 75 else 'Medium' if row.get('ai_confidence', 0) >= 50 else 'Low',
                'ai_rating': row.get('ai_confidence', 0) / 10,
                'ai_risks': row.get('ai_risks', [])
            }
            
            st.session_state.crypto_quick_pair = row.get('pair', 'UNKNOWN')
            st.session_state.crypto_quick_trade_pair = row.get('pair', 'UNKNOWN')
            st.session_state.crypto_quick_direction = row.get('direction', 'BUY')
            st.session_state.crypto_trading_mode = row.get('trading_mode', 'Spot Trading')
            st.session_state.crypto_quick_leverage = row.get('leverage', 1)
            st.session_state.crypto_quick_position_size = row.get('position_size', 100)
            st.session_state.crypto_quick_stop_pct = row.get('stop_pct', 2.0)
            st.session_state.crypto_quick_target_pct = row.get('target_pct', 5.0)
            
            logger.info(f"📝 DAILY SCANNER - Session state set: pair={pair}, direction={row.get('direction', 'BUY')}, leverage={row.get('leverage', 1)}, position=${row.get('position_size', 100)}")
            
            # Switch to Quick Trade main tab AND Execute Trade subtab
            st.session_state.active_crypto_tab = "⚡ Quick Trade"
            st.session_state.quick_trade_subtab = "⚡ Execute Trade"
            
            st.success(f"✅ Trade setup loaded for {pair} ({trade_type})! Switching to Execute Trade tab...")
            st.balloons()
            st.rerun()
    
    st.header("🔍 Daily Crypto Scanner")
    st.markdown("""
    **Progressive Scanning Workflow** - Start light, go deep on winners
    - 🏃 **Tier 1**: Quick filter 100+ coins (price, volume, momentum only)
    - 📊 **Tier 2**: Technical analysis on top 20 (RSI, MACD, EMAs)
    - 🎯 **Tier 3**: Deep dive on selected (full strategy + AI review)
    - 🤖 **Monitor**: Add winners to live monitoring
    """)
    
    # Store ai_trade_reviewer in session state for monitor buttons
    if ai_trade_reviewer:
        st.session_state.ai_trade_reviewer = ai_trade_reviewer
    
    # Initialize scanner
    if 'tiered_scanner' not in st.session_state:
        try:
            from services.crypto_tiered_scanner import TieredCryptoScanner
            st.session_state.tiered_scanner = TieredCryptoScanner(kraken_client, crypto_config)
            logger.info("✅ Tiered scanner initialized")
        except Exception as e:
            logger.error("Error initializing scanner: {}", str(e), exc_info=True)
            st.error(f"Failed to initialize scanner: {e}")
            return
    
    scanner = st.session_state.tiered_scanner
    
    # Create tabs for each tier
    tier_tabs = st.tabs([
        "🏃 Tier 1: Quick Filter",
        "📊 Tier 2: Technical Analysis",
        "🎯 Tier 3: Deep Analysis",
        "🤖 Active Monitors"
    ])
    
    # ========== TIER 1: QUICK FILTER ==========
    with tier_tabs[0]:
        display_tier1_quick_filter(scanner, kraken_client, crypto_config)
    
    # ========== TIER 2: TECHNICAL ANALYSIS ==========
    with tier_tabs[1]:
        display_tier2_medium_analysis(scanner)
    
    # ========== TIER 3: DEEP ANALYSIS ==========
    with tier_tabs[2]:
        display_tier3_deep_analysis(scanner, ai_trade_reviewer)
    
    # ========== ACTIVE MONITORS ==========
    with tier_tabs[3]:
        display_active_monitors(ai_trade_reviewer, kraken_client)


def display_tier1_quick_filter(scanner, kraken_client, crypto_config):
    """Tier 1: Quick filter interface"""
    st.subheader("🏃 Tier 1: Quick Filter")
    st.markdown("""
    Lightweight scan using **only** price, volume, and momentum.
    Choose between analyzing your **watchlist** or **discovering new coins**.
    """)
    
    # Mode selection - Watchlist vs Discovery
    scan_mode = st.radio(
        "Scan Mode",
        ["📋 Watchlist Analysis", "🔍 New Discovery"],
        horizontal=True,
        key="tier1_scan_mode",
        help="Watchlist = analyze coins you track | Discovery = find NEW coins on Kraken"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if scan_mode == "📋 Watchlist Analysis":
            # Watchlist-focused options
            scan_source = st.selectbox(
                "Scan Source",
                [
                    "⭐ My Watchlist",
                    "📈 Top Gainers (watchlist)",
                    "📊 High Volume (watchlist)",
                    "🗣️ Social Buzz (watchlist)",
                ],
                key="tier1_scan_source_watchlist",
                help="Analyze coins from your saved watchlist"
            )
            
            # Get watchlist count for info
            try:
                if 'crypto_watchlist_manager' in st.session_state:
                    wl_count = len(st.session_state.crypto_watchlist_manager.get_watchlist_symbols())
                    st.caption(f"📋 Your watchlist has {wl_count} coins")
            except:
                pass
        else:
            # Discovery-focused options  
            scan_source = st.selectbox(
                "Discovery Source",
                [
                    "🌐 All Kraken USD Pairs",
                    "🔥 CoinGecko Trending",
                    "💰 Penny Cryptos (<$1)",
                    "🔬 Sub-Penny (<$0.01)",
                    "🚀 Potential Runners",
                    "🆕 New to Kraken (exclude watchlist)",
                ],
                key="tier1_scan_source_discovery",
                help="Discover NEW coins not in your watchlist"
            )
            
            # Show discovery info
            st.caption("🔍 Discovery mode finds coins outside your watchlist")
        
        # Show sentiment indicator for sentiment-enabled sources
        if "sentiment" in scan_source.lower() or "Trending" in scan_source or "Buzz" in scan_source:
            st.info("🧠 Includes Reddit & social sentiment + 🐦 X/Twitter")
    
    with col2:
        # Filter settings
        max_results = st.slider(
            "Max Results",
            min_value=10,
            max_value=50,
            value=20,
            key="tier1_max_results"
        )
        
        # Show estimated scan time
        if scan_mode == "🔍 New Discovery" and "All Kraken" in scan_source:
            st.caption("⏱️ Discovery scans may take 30-60 seconds")
    
    # Scan button
    if st.button("🚀 Start Quick Scan", key="tier1_scan", type="primary"):
        logger.info("=" * 100)
        logger.info("🔘 SCAN BUTTON CLICKED")
        logger.info("   Selected scan source: '{}'", str(scan_source))
        logger.info(f"   Scan source length: {len(scan_source)}")
        logger.info(f"   Scan source bytes: {scan_source.encode('utf-8')}")
        logger.info("=" * 100)
        
        # Import discovery services
        from services.crypto_sentiment_analyzer import CryptoSentimentAnalyzer
        from services.sub_penny_discovery import SubPennyDiscovery
        from services.social_sentiment_analyzer import SocialSentimentAnalyzer
        
        with st.spinner(f"Scanning {scan_source}..."):
            # Get pairs to scan based on source
            pairs = []
            
            logger.info(f"🔍 Determining which scan source to use...")
            
            # Defensive check: ensure scanner is the correct type
            from services.crypto_tiered_scanner import TieredCryptoScanner
            if not isinstance(scanner, TieredCryptoScanner):
                logger.error("Scanner is not TieredCryptoScanner, got {}: {}", type(scanner), scanner)
                st.error(f"Internal error: Scanner is {type(scanner).__name__}, expected TieredCryptoScanner")
                # Try to reinitialize
                try:
                    st.session_state.tiered_scanner = TieredCryptoScanner(kraken_client, crypto_config)
                    scanner = st.session_state.tiered_scanner
                    st.info("Scanner reinitialized. Please try again.")
                    return
                except Exception as reinit_error:
                    st.error(f"Failed to reinitialize scanner: {reinit_error}")
                    return
            
            # ============= WATCHLIST ANALYSIS MODE =============
            if scan_mode == "📋 Watchlist Analysis":
                logger.info("📋 WATCHLIST ANALYSIS MODE")
                
                # Load user's watchlist
                try:
                    if 'crypto_watchlist_manager' not in st.session_state:
                        from services.crypto_watchlist_manager import CryptoWatchlistManager
                        st.session_state.crypto_watchlist_manager = CryptoWatchlistManager()
                    
                    crypto_wl_manager = st.session_state.crypto_watchlist_manager
                    watchlist_pairs = crypto_wl_manager.get_watchlist_symbols()
                    
                    if not watchlist_pairs:
                        st.warning("⚠️ Your watchlist is empty! Add coins from the Watchlist tab first.")
                        st.info("💡 Or switch to Discovery mode to find new coins.")
                        return
                    
                    pairs = watchlist_pairs
                    logger.info(f"   Loaded {len(pairs)} pairs from user watchlist")
                    
                    # Apply filter based on sub-option
                    if "Top Gainers" in scan_source:
                        st.info(f"📈 Scanning {len(pairs)} watchlist coins, sorted by 24h gains")
                    elif "High Volume" in scan_source:
                        st.info(f"📊 Scanning {len(pairs)} watchlist coins, sorted by volume")
                    elif "Social Buzz" in scan_source:
                        st.info(f"🗣️ Scanning {len(pairs)} watchlist coins with social sentiment")
                    else:
                        st.info(f"⭐ Scanning {len(pairs)} coins from your watchlist")
                        
                except Exception as e:
                    logger.error(f"Error loading watchlist: {e}", exc_info=True)
                    st.error(f"Failed to load watchlist: {e}")
                    return
            
            # ============= DISCOVERY MODE =============
            else:
                logger.info("🔍 DISCOVERY MODE")
                
                if "All Kraken USD" in scan_source:
                    logger.info("✅ Matched: All Kraken USD Pairs")
                    pairs = scanner.get_all_kraken_usd_pairs()
                    st.info(f"🌐 Scanning ALL {len(pairs)} USD pairs on Kraken")
                
                elif "New to Kraken" in scan_source or "exclude watchlist" in scan_source:
                    logger.info("✅ Matched: New to Kraken (exclude watchlist)")
                    pairs = scanner.get_discovery_pairs(exclude_watchlist=True)
                    st.info(f"🆕 Scanning {len(pairs)} coins NOT in your watchlist")
                
                elif "CoinGecko Trending" in scan_source or "Trending" in scan_source:
                    logger.info("✅ Matched: CoinGecko Trending")
                    try:
                        sentiment_analyzer = CryptoSentimentAnalyzer()
                        status_placeholder = st.empty()
                        status_placeholder.info("🔥 Fetching CoinGecko trending + sentiment...")
                        
                        trending_with_sentiment = asyncio.run(
                            sentiment_analyzer.get_trending_with_sentiment(top_n=20)
                        )
                        
                        # Get ALL Kraken pairs to match against (not just hardcoded 34)
                        all_kraken_pairs = scanner.get_all_kraken_usd_pairs()
                        trending_pairs = []
                        
                        for trend in trending_with_sentiment:
                            symbol = trend.get('symbol', '').upper()
                            # Try to find matching Kraken pair
                            for pair in all_kraken_pairs:
                                if symbol == pair.split('/')[0]:
                                    trending_pairs.append(pair)
                                    break
                        
                        if trending_pairs:
                            pairs = trending_pairs
                            status_placeholder.success(f"🔥 Found {len(pairs)} trending coins available on Kraken")
                        else:
                            pairs = scanner.get_all_kraken_usd_pairs()[:50]
                            status_placeholder.warning("No trending coins on Kraken, scanning top 50 pairs")
                    except Exception as e:
                        logger.error(f"CoinGecko trending error: {e}", exc_info=True)
                        pairs = scanner.get_all_kraken_usd_pairs()[:50]
                        st.warning("Trending unavailable, scanning top 50 Kraken pairs")
                
                elif "Penny Cryptos" in scan_source:
                    logger.info("✅ Matched: Penny Cryptos (Discovery)")
                    # Get ALL Kraken pairs under $1
                    try:
                        all_kraken = scanner.get_all_kraken_usd_pairs()
                        st.info(f"💰 Scanning {len(all_kraken)} Kraken pairs for coins under $1...")
                        pairs = all_kraken  # Will be filtered by price in tier1_quick_filter
                    except Exception as e:
                        logger.error(f"Penny discovery error: {e}")
                        pairs = scanner.get_all_kraken_usd_pairs()[:50]
                        st.warning("Using top 50 Kraken pairs")
                
                elif "Sub-Penny" in scan_source:
                    logger.info("✅ Matched: Sub-Penny (Discovery)")
                    # Get ALL Kraken pairs under $0.01
                    try:
                        all_kraken = scanner.get_all_kraken_usd_pairs()
                        st.info(f"🔬 Scanning {len(all_kraken)} Kraken pairs for coins under $0.01...")
                        pairs = all_kraken  # Will be filtered by price in tier1_quick_filter
                    except Exception as e:
                        logger.error(f"Sub-penny discovery error: {e}")
                        pairs = scanner.get_all_kraken_usd_pairs()[:50]
                        st.warning("Using top 50 Kraken pairs")
                
                elif "Potential Runners" in scan_source:
                    logger.info("✅ Matched: Potential Runners (Discovery)")
                    # Get ALL Kraken pairs and find runners
                    try:
                        all_kraken = scanner.get_all_kraken_usd_pairs()
                        st.info(f"🚀 Scanning {len(all_kraken)} Kraken pairs for potential runners...")
                        pairs = all_kraken
                    except Exception as e:
                        logger.error(f"Runner discovery error: {e}")
                        pairs = scanner.get_all_kraken_usd_pairs()[:50]
                        st.warning("Using top 50 Kraken pairs")
                
                else:
                    # Fallback for discovery mode
                    logger.warning(f"⚠️ Unknown discovery source: {scan_source}, using all Kraken pairs")
                    pairs = scanner.get_all_kraken_usd_pairs()[:50]
                    st.info(f"🔍 Scanning top 50 Kraken USD pairs")
            
            # Defensive check: ensure pairs is a list
            if not isinstance(pairs, list):
                logger.error("pairs is not a list, got {}: {}", type(pairs), pairs)
                st.error(f"Internal error: Expected list of pairs, got {type(pairs).__name__}")
                return
            
            if not pairs:
                logger.error("=" * 100)
                logger.error(f"❌ NO PAIRS FOUND TO SCAN!")
                logger.error("   pairs = {}", pairs)
                logger.error("   scan_source was = '{}'", str(scan_source))
                logger.error("=" * 100)
                st.warning("No pairs found to scan")
                st.info("💡 Try selecting a different scan source or check your watchlist")
                return
            
            # Run Tier 1 scan
            try:
                logger.debug(f"Running tier1_quick_filter with {len(pairs)} pairs")
                results = asyncio.run(
                    scanner.tier1_quick_filter(pairs, max_results=max_results)
                )
                
                logger.debug(f"tier1_quick_filter returned {type(results)}")
                
                # Store results - ensure it's a list
                if isinstance(results, list):
                    # Merge X sentiment data if available (from CoinGecko Trending scan)
                    trending_sentiment = st.session_state.get('trending_sentiment_data', {})
                    if trending_sentiment:
                        for result in results:
                            pair = result.get('pair', '')
                            if pair in trending_sentiment:
                                sentiment_data = trending_sentiment[pair]
                                result['x_tweet_count'] = sentiment_data.get('x_tweet_count', 0)
                                result['x_sentiment_score'] = sentiment_data.get('x_sentiment_score', 0)
                                result['x_bullish_count'] = sentiment_data.get('x_bullish_count', 0)
                                result['x_bearish_count'] = sentiment_data.get('x_bearish_count', 0)
                                result['trending_score'] = sentiment_data.get('trending_score', 0)
                        logger.info(f"Merged X sentiment data for {len([r for r in results if r.get('x_tweet_count', 0) > 0])} coins")
                    
                    st.session_state.tier1_results = results
                    st.session_state.tier1_timestamp = datetime.now()
                    logger.debug(f"Stored {len(results)} results in session state")
                else:
                    logger.error("Expected list from tier1_quick_filter, got {}", type(results))
                    st.error("Internal error: Invalid scan results format")
                    return
                
                st.success(f"✅ Found {len(results)} promising coins!")
                
            except Exception as e:
                st.error(f"Scan failed: {e}")
                logger.error("Tier 1 scan error: {}", str(e), exc_info=True)
                return
    
    # Display results
    results = st.session_state.get('tier1_results')
    if results and isinstance(results, list):
        timestamp = st.session_state.get('tier1_timestamp', datetime.now())
        
        st.markdown(f"**Last scan:** {timestamp.strftime('%H:%M:%S')}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Coins Found", len(results))
        with col2:
            avg_score = sum(r['score'] for r in results) / len(results)
            st.metric("Avg Score", f"{avg_score:.1f}")
        with col3:
            avg_change = sum(r['change_24h'] for r in results) / len(results)
            st.metric("Avg 24h Change", f"{avg_change:+.2f}%")
        with col4:
            high_potential = sum(1 for r in results if r['score'] >= 60)
            st.metric("High Potential", high_potential)
        
        # Results table
        st.markdown("### 📊 Filtered Coins")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(results)
        
        # Check if X sentiment data is available
        has_x_sentiment = any('x_tweet_count' in r or 'x_sentiment_score' in r for r in results)
        
        if has_x_sentiment:
            # Include X sentiment columns
            columns_to_show = ['pair', 'score', 'price', 'change_24h', 'volume_24h']
            
            # Add X sentiment columns if they exist
            if 'x_tweet_count' in df.columns:
                columns_to_show.append('x_tweet_count')
            if 'x_sentiment_score' in df.columns:
                columns_to_show.append('x_sentiment_score')
            
            df = df[[c for c in columns_to_show if c in df.columns]]
            
            # Rename columns
            column_names = {'pair': 'Pair', 'score': 'Score', 'price': 'Price', 
                          'change_24h': '24h %', 'volume_24h': 'Volume',
                          'x_tweet_count': '🐦 Tweets', 'x_sentiment_score': '🐦 X Score'}
            df.columns = [column_names.get(c, c) for c in df.columns]
        else:
            df = df[['pair', 'score', 'price', 'change_24h', 'volume_24h']]
            df.columns = ['Pair', 'Score', 'Price', '24h %', 'Volume']
        
        # Format columns
        df['Price'] = df['Price'].apply(lambda x: f"${x:.8f}" if x < 0.01 else f"${x:.4f}")
        df['24h %'] = df['24h %'].apply(lambda x: f"{x:+.2f}%")
        df['Volume'] = df['Volume'].apply(lambda x: f"${x:,.0f}")
        
        # Format X sentiment columns if present
        if '🐦 Tweets' in df.columns:
            df['🐦 Tweets'] = df['🐦 Tweets'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "—")
        if '🐦 X Score' in df.columns:
            df['🐦 X Score'] = df['🐦 X Score'].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x > 0 else "—")
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Quick actions
        st.markdown("### ⚡ Next Steps")
        st.info("👆 **Click the '📊 Tier 2: Technical Analysis' tab above** to analyze these coins further with RSI, MACD, and EMAs")
        
        # Selective save section
        st.markdown("### 💾 Save to Watchlist")
        st.markdown("**Select coins to save:**")
        
        # Show checkboxes FIRST (before button)
        select_all = st.checkbox("✅ Select All", key="tier1_select_all")
        
        # Create checkboxes in columns for better layout
        num_results = len(results)
        num_to_show = min(20, num_results)  # Show up to 20
        
        cols_per_row = 3
        for i in range(0, num_to_show, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < num_to_show:
                    result = results[idx]
                    pair_key = result['pair'].replace('/', '_')
                    with col:
                        st.checkbox(
                            f"{result['pair']} ({result['score']:.1f})",
                            value=select_all,
                            key=f"tier1_select_{pair_key}_{idx}"  # Add idx to make unique
                        )
        
        if num_results > num_to_show:
            st.info(f"ℹ️ Showing first {num_to_show} of {num_results} results")
        
        # Save button AFTER checkboxes
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save Selected", key="tier1_save_selected", type="primary"):
                # Initialize watchlist manager
                if 'crypto_watchlist_manager' not in st.session_state:
                    from services.crypto_watchlist_manager import CryptoWatchlistManager
                    st.session_state.crypto_watchlist_manager = CryptoWatchlistManager()
                
                wl_manager = st.session_state.crypto_watchlist_manager
                saved_count = 0
                skipped_count = 0
                
                with st.spinner("Saving coins..."):
                    for idx, result in enumerate(results):
                        pair_key = result['pair'].replace('/', '_')
                        # Check if selected (check indexed key for displayed items, or select_all)
                        is_selected = False
                        if select_all:
                            is_selected = True
                        elif idx < num_to_show:
                            is_selected = st.session_state.get(f"tier1_select_{pair_key}_{idx}", False)
                        
                        if is_selected:
                            # Check for duplicates
                            if wl_manager.is_in_watchlist(result['pair']):
                                logger.info(f"Skipping duplicate: {result['pair']}")
                                skipped_count += 1
                                continue
                            
                            # Prepare opportunity data
                            opportunity_data = {
                                'current_price': result.get('price', 0),
                                'change_pct_24h': result.get('change_24h', 0),
                                'volume_24h': result.get('volume_24h', 0),
                                'score': result.get('score', 0),
                                'strategy': 'tier1_scan',
                                'reason': f"Tier 1 scan: Score {result.get('score', 0):.1f}"
                            }
                            
                            # Save to watchlist
                            if wl_manager.add_crypto(result['pair'], opportunity_data):
                                saved_count += 1
                                logger.info(f"Saved {result['pair']} to watchlist")
                
                if saved_count > 0:
                    st.success(f"✅ Saved {saved_count} coins to watchlist!")
                if skipped_count > 0:
                    st.info(f"ℹ️ Skipped {skipped_count} coins (already in watchlist)")
                if saved_count == 0 and skipped_count == 0:
                    st.warning("⚠️ No coins selected. Check boxes to save.")


def display_tier2_medium_analysis(scanner):
    """Tier 2: Medium analysis interface"""
    from datetime import datetime
    st.subheader("📊 Tier 2: Technical Analysis")
    st.markdown("""
    Add technical indicators to filter: RSI, MACD, EMAs, volume analysis.
    Analyzes top performers from Tier 1.
    """)
    
    # Check if Tier 1 results exist
    tier1_results = st.session_state.get('tier1_results')
    if not tier1_results or not isinstance(tier1_results, list):
        st.warning("⚠️ Run Tier 1 Quick Filter first to get candidates")
        st.info("💡 Go to the '🏃 Tier 1: Quick Filter' tab above and click 'Start Scan'")
        return
    
    st.info(f"📥 {len(tier1_results)} candidates from Tier 1 (avg score: {sum(r['score'] for r in tier1_results)/len(tier1_results):.1f})")
    
    # Filter settings
    col1, col2 = st.columns(2)
    
    with col1:
        min_tier2_score = st.slider(
            "Minimum Score",
            min_value=20,
            max_value=80,
            value=35,
            help="Lower threshold = more results. 35 is recommended for daily scanning.",
            key="tier2_min_score"
        )
    
    with col2:
        max_candidates = len(tier1_results)
        # Guard slider creation when max == min (e.g., only 1 candidate)
        default_top = min(20, max_candidates)
        if max_candidates <= 1:
            st.write("Analyze Top N:", max_candidates)
            analyze_top_n = max_candidates
        else:
            analyze_top_n = st.slider(
                "Analyze Top N",
                min_value=1,
                max_value=max_candidates,
                value=default_top,
                key="tier2_top_n"
            )
    
    # Analyze button
    if st.button("📈 Start Technical Analysis", key="tier2_analyze", type="primary"):
        with st.spinner(f"Analyzing {analyze_top_n} coins..."):
            # Take top N from Tier 1
            candidates = tier1_results[:analyze_top_n]
            
            try:
                # Update scanner min_score from UI
                scanner.tier2_min_score = min_tier2_score
                
                results = asyncio.run(
                    scanner.tier2_medium_analysis(candidates)
                )
                
                # Results already filtered by scanner.tier2_min_score
                # Store results
                st.session_state.tier2_results = results
                st.session_state.tier2_timestamp = datetime.now()
                
                if len(results) > 0:
                    st.success(f"✅ {len(results)} coins passed technical analysis!")
                else:
                    st.warning(f"⚠️ No coins passed with minimum score {min_tier2_score}. Try lowering the threshold.")
                    st.info(f"💡 TIP: Lower the 'Minimum Score' slider to 25-30 to see more results")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error("Tier 2 analysis error: {}", str(e), exc_info=True)
                return
    
    # Display results
    results = st.session_state.get('tier2_results')
    if results and isinstance(results, list):
        timestamp = st.session_state.get('tier2_timestamp', datetime.now())
        
        st.markdown(f"**Results from:** {timestamp.strftime('%I:%M %p')}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Coins Passed", len(results))
        with col2:
            avg_score = sum(r['score'] for r in results) / len(results)
            st.metric("Avg Score", f"{avg_score:.1f}")
        with col3:
            bullish_signals = sum(len([s for s in r.get('signals', []) if '📈' in s or '🟢' in s]) for r in results)
            st.metric("Bullish Signals", bullish_signals)
        with col4:
            ready_count = sum(1 for r in results if r['score'] >= 70)
            st.metric("Ready for Deep", ready_count)
        
        # Display each coin
        st.markdown("### 📊 Analysis Results")
        
        for i, result in enumerate(results):
            with st.expander(f"{'🏆' if i < 3 else '📊'} {result['pair']} - Score: {result['score']:.1f}", expanded=i < 3):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**💰 Price Info**")
                    price = result.get('price', 0) or 0
                    price_str = f"${price:.8f}" if price < 0.01 else f"${price:.4f}"
                    st.write(f"Price: {price_str}")
                    st.write(f"24h: {result['change_24h']:+.2f}%")
                    st.write(f"Volume: ${result['volume_24h']:,.0f}")
                
                with col2:
                    st.markdown("**📈 Technical**")
                    st.write(f"RSI: {result.get('rsi', 0):.1f}")
                    st.write(f"MACD: {result.get('macd', 0):.4f}")
                    st.write(f"Vol Ratio: {result.get('volume_ratio', 1):.2f}x")
                
                with col3:
                    st.markdown("**🎯 Signals**")
                    signals = result.get('signals', [])
                    if signals:
                        for signal in signals[:5]:
                            st.write(signal)
                    else:
                        st.write("No strong signals")
                
                # Actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Use coin pair in key for stability across reruns
                    pair_key = result['pair'].replace('/', '_')
                    if st.button(f"🎯 Select for Deep Analysis", key=f"deep_{pair_key}_{i}"):
                        st.session_state.selected_for_tier3 = [result]
                        st.session_state.tier3_auto_switch = True  # Signal to show info in Tier 3
                        logger.info(f"Selected {result['pair']} for deep analysis")
                        st.success(f"✅ {result['pair']} selected! Now switch to '🎯 Tier 3: Deep Analysis' tab above ⬆️")
                
                with col2:
                    if st.button(f"💾 Save {result['pair']}", key=f"save_{pair_key}_{i}"):
                        # Initialize watchlist manager
                        if 'crypto_watchlist_manager' not in st.session_state:
                            from services.crypto_watchlist_manager import CryptoWatchlistManager
                            st.session_state.crypto_watchlist_manager = CryptoWatchlistManager()
                        
                        wl_manager = st.session_state.crypto_watchlist_manager
                        
                        # Check for duplicates
                        if wl_manager.is_in_watchlist(result['pair']):
                            st.warning(f"⚠️ {result['pair']} is already in your watchlist")
                        else:
                            # Prepare opportunity data
                            opportunity_data = {
                                'current_price': result.get('price', 0),
                                'change_pct_24h': result.get('change_24h', 0),
                                'volume_24h': result.get('volume_24h', 0),
                                'volume_ratio': result.get('volume_ratio', 1),
                                'rsi': result.get('rsi', 50),
                                'score': result.get('score', 0),
                                'strategy': 'tier2_technical',
                                'reason': f"Tier 2 analysis: Score {result.get('score', 0):.1f}"
                            }
                            
                            # Save to watchlist
                            if wl_manager.add_crypto(result['pair'], opportunity_data):
                                st.success(f"✅ Saved {result['pair']} to watchlist!")
                                logger.info(f"Saved {result['pair']} to watchlist from Tier 2")
                            else:
                                st.error(f"❌ Failed to save {result['pair']}")
                
                with col3:
                    if st.button(f"🤖 Monitor", key=f"monitor_t2_{pair_key}_{i}"):
                        if 'ai_trade_reviewer' in st.session_state and st.session_state.ai_trade_reviewer:
                            reviewer = st.session_state.ai_trade_reviewer
                            trade_id = f"{result['pair'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            reviewer.start_trade_monitoring(
                                trade_id=trade_id,
                                pair=result['pair'],
                                side='BUY',
                                entry_price=result.get('price', 0),
                                current_price=result.get('price', 0),
                                volume=1.0,
                                stop_loss=result.get('price', 0) * 0.98,  # 2% stop
                                take_profit=result.get('price', 0) * 1.05,  # 5% target
                                strategy='Tier2_Technical'
                            )
                            
                            st.session_state.active_trade_monitors = reviewer.active_monitors
                            st.success(f"✅ {result['pair']} monitoring!")
                        else:
                            st.error("❌ Monitor unavailable")
        
        # Bulk actions
        st.markdown("### ⚡ Bulk Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎯 Select Top 5 for Deep Analysis", type="primary", key="tier2_select_top5"):
                st.session_state.selected_for_tier3 = results[:5]
                st.session_state.tier3_auto_switch = True
                logger.info(f"Selected top 5 coins for deep analysis: {[r['pair'] for r in results[:5]]}")
                st.success(f"✅ Top 5 coins selected! Now switch to '🎯 Tier 3: Deep Analysis' tab above ⬆️")
        
        with col2:
            if st.button("💾 Save All to Watchlist", key="tier2_save_all"):
                # Initialize watchlist manager
                if 'crypto_watchlist_manager' not in st.session_state:
                    from services.crypto_watchlist_manager import CryptoWatchlistManager
                    st.session_state.crypto_watchlist_manager = CryptoWatchlistManager()
                
                wl_manager = st.session_state.crypto_watchlist_manager
                saved_count = 0
                skipped_count = 0
                
                with st.spinner("Saving all coins..."):
                    for result in results:
                        # Check for duplicates
                        if wl_manager.is_in_watchlist(result['pair']):
                            logger.info(f"Skipping duplicate: {result['pair']}")
                            skipped_count += 1
                            continue
                        
                        # Prepare opportunity data
                        opportunity_data = {
                            'current_price': result.get('price', 0),
                            'change_pct_24h': result.get('change_24h', 0),
                            'volume_24h': result.get('volume_24h', 0),
                            'volume_ratio': result.get('volume_ratio', 1),
                            'rsi': result.get('rsi', 50),
                            'score': result.get('score', 0),
                            'strategy': 'tier2_technical',
                            'reason': f"Tier 2 analysis: Score {result.get('score', 0):.1f}"
                        }
                        
                        # Save to watchlist
                        if wl_manager.add_crypto(result['pair'], opportunity_data):
                            saved_count += 1
                            logger.info(f"Saved {result['pair']} to watchlist")
                
                if saved_count > 0:
                    st.success(f"✅ Saved {saved_count} coins to watchlist!")
                if skipped_count > 0:
                    st.info(f"ℹ️ Skipped {skipped_count} coins (already in watchlist)")
                if saved_count == 0 and skipped_count == 0:
                    st.warning("⚠️ No coins to save")
        
        with col3:
            if st.button("📊 Export to CSV"):
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download CSV",
                    csv,
                    f"tier2_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )


def display_tier3_deep_analysis(scanner, ai_trade_reviewer):
    """Tier 3: Deep analysis interface with multiple modes"""
    st.subheader("🎯 Tier 3: Deep Analysis")
    st.markdown("""
    **Comprehensive Analysis Suite**
    - 🎯 Quick: Best strategy only
    - 📊 Standard: Multiple strategies comparison
    - 🔬 Multi-Config: Test all directions + leverage levels
    - 🏆 Ultimate: Everything combined (all strategies + all configs)
    """)
    
    # Check if Tier 2 results exist or user selected coins
    tier2_results = st.session_state.get('tier2_results', [])
    selected_coins = st.session_state.get('selected_for_tier3', [])
    
    if not tier2_results and not selected_coins:
        st.warning("⚠️ Run Tier 2 Technical Analysis first, or select coins manually")
        st.info("💡 Go to the '📊 Tier 2: Technical Analysis' tab and run an analysis")
        return
    
    # Show selection notification if auto-switched
    if st.session_state.get('tier3_auto_switch', False):
        st.success("✅ Coins loaded from Tier 2 selection! Ready for deep analysis below ⬇️")
        st.session_state.tier3_auto_switch = False  # Clear flag after showing
    
    # Source selection
    if selected_coins:
        st.success(f"📥 {len(selected_coins)} coins ready for deep analysis")
        # Show which coins
        coin_list = ", ".join([c['pair'] for c in selected_coins])
        st.info(f"**Selected:** {coin_list}")
        candidates = selected_coins
    else:
        st.info(f"📥 {len(tier2_results)} candidates from Tier 2 (avg score: {sum(r['score'] for r in tier2_results)/len(tier2_results):.1f})")
        candidates = tier2_results
    
    # Analysis mode selection
    st.markdown("### 🎯 Analysis Mode")
    analysis_mode = st.radio(
        "Select Analysis Type",
        [
            "🎯 Quick (Best Strategy Only)",
            "📊 Standard (Multiple Strategies)",
            "🔬 Multi-Config (All Directions + Leverage)",
            "🏆 Ultimate (Everything Combined)"
        ],
        key="tier3_mode",
        horizontal=True
    )
    
    # Position Size Configuration (shown for all modes)
    st.markdown("### 💰 Position Sizing")
    col_preset, col_custom = st.columns([1, 2])
    
    with col_preset:
        preset = st.selectbox(
            "Quick Preset",
            ["💎 Small ($100)", "📊 Medium ($500)", "💰 Large ($1000)", "🏦 Custom"],
            key="tier3_preset",
            help="Quick position size presets"
        )
    
    with col_custom:
        if "Custom" in preset:
            position_size = st.number_input(
                "Custom Position Size per Coin ($USD)",
                min_value=10.0,  # Allow as low as $10 for testing
                max_value=100000.0,
                value=250.0,
                step=10.0,
                key="tier3_position_size",
                help="Amount to invest per coin (supports fractional purchases)"
            )
        elif "Small" in preset:
            position_size = 100.0
            st.info("💎 Testing mode: $100 per coin")
        elif "Medium" in preset:
            position_size = 500.0
            st.info("📊 Balanced: $500 per coin")
        else:  # Large
            position_size = 1000.0
            st.info("💰 Aggressive: $1000 per coin")
    
    # Configuration based on mode
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Only show slider if there's more than 1 candidate
        if len(candidates) > 1:
            max_analyze = st.slider(
                "Max Coins to Analyze",
                min_value=1,
                max_value=min(10, len(candidates)),
                value=min(5, len(candidates)),
                key="tier3_max",
                help="Number of top coins to analyze"
            )
        else:
            max_analyze = 1
            st.info("1 coin to analyze")
        
        # Show total exposure calculation
        total_exposure = position_size * max_analyze
        st.metric("Total Exposure", f"${total_exposure:,.2f}")
    
    # Mode-specific configuration
    if "Quick" in analysis_mode:
        with col2:
            strategy = st.selectbox(
                "Strategy",
                [
                    "ema_crossover",  # Default - most reliable
                    "rsi_stoch_hammer",  # Was rsi_stochastic
                    "fisher_rsi_multi",  # Was fisher_rsi
                    "macd_volume",
                    "aggressive_scalp",
                    "orb_fvg"
                ],
                key="tier3_strategy",
                help="Select one strategy - all strategies are AI-enhanced"
            )
            st.caption("🧠 ML + AI + Sentiment included")
        with col3:
            st.info("Single strategy analysis")
    
    elif "Standard" in analysis_mode:
        with col2:
            strategies_to_test = st.multiselect(
                "Strategies to Test",
                [
                    "ema_crossover",
                    "rsi_stoch_hammer",
                    "fisher_rsi_multi",
                    "macd_volume",
                    "aggressive_scalp",
                    "orb_fvg"
                ],
                default=["ema_crossover", "rsi_stoch_hammer", "fisher_rsi_multi"],
                key="tier3_strategies",
                help="Compare multiple ML-enhanced strategies"
            )
            st.caption("🧠 Each strategy gets ML + AI scoring")
        with col3:
            if strategies_to_test:
                st.info(f"Testing {len(strategies_to_test)} strategies")
            else:
                st.warning("Select at least 1 strategy")
    
    elif "Multi-Config" in analysis_mode:
        with col2:
            directions = st.multiselect(
                "Directions",
                ["BUY", "SELL"],
                default=["BUY"],
                key="tier3_directions",
                help="Test long and/or short positions"
            )
        with col3:
            leverage_levels = st.multiselect(
                "Leverage Levels",
                ["1x (Spot)", "2x", "3x", "5x"],
                default=["1x (Spot)", "2x"],
                key="tier3_leverage",
                help="Test different leverage levels"
            )
    
    else:  # Ultimate
        with col2:
            st.info("🏆 All strategies will be tested")
        with col3:
            st.info("🔬 All configurations will be tested")
    
    # Analyze button
    if st.button("🚀 Start Deep Analysis", key="tier3_analyze", type="primary"):
        # Take top N candidates
        to_analyze = candidates[:max_analyze]
        
        # Get pairs list
        pairs_to_analyze = [c['pair'] for c in to_analyze]
        
        try:
            if "Quick" in analysis_mode:
                # Quick mode - single strategy
                with st.spinner(f"Analyzing {len(to_analyze)} coins with {strategy}..."):
                    results = asyncio.run(
                        scanner.tier3_deep_analysis(
                            to_analyze,
                            strategy=strategy,
                            ai_reviewer=ai_trade_reviewer
                        )
                    )
                    
                    st.session_state.tier3_results = results
                    st.session_state.tier3_analysis_mode = "quick"  # Changed to avoid widget key conflict
                    st.session_state.tier3_timestamp = datetime.now()
                    
                    ready_count = sum(1 for r in results if r.get('ready_for_monitoring', False))
                    st.success(f"✅ {strategy} analysis complete! {ready_count}/{len(results)} ready for monitoring")
            
            elif "Standard" in analysis_mode:
                # Standard mode - multiple strategies
                if not strategies_to_test:
                    st.error("Please select at least one strategy")
                    return
                
                with st.spinner(f"Testing {len(strategies_to_test)} strategies on {len(to_analyze)} coins..."):
                    all_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, strat in enumerate(strategies_to_test):
                        status_text.text(f"Testing {strat}... ({idx+1}/{len(strategies_to_test)})")
                        
                        results = asyncio.run(
                            scanner.tier3_deep_analysis(
                                to_analyze,
                                strategy=strat,
                                ai_reviewer=ai_trade_reviewer
                            )
                        )
                        
                        # Add strategy name to results
                        for r in results:
                            r['tested_strategy'] = strat
                        
                        all_results.extend(results)
                        progress_bar.progress((idx + 1) / len(strategies_to_test))
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Group by pair and find best strategy for each
                    best_results = {}
                    for r in all_results:
                        pair = r['pair']
                        if pair not in best_results or r['score'] > best_results[pair]['score']:
                            best_results[pair] = r
                    
                    final_results = list(best_results.values())
                    final_results.sort(key=lambda x: x['score'], reverse=True)
                    
                    st.session_state.tier3_results = final_results
                    st.session_state.tier3_all_strategy_results = all_results
                    st.session_state.tier3_analysis_mode = "standard"  # Changed to avoid widget key conflict
                    st.session_state.tier3_timestamp = datetime.now()
                    
                    ready_count = sum(1 for r in final_results if r.get('ready_for_monitoring', False))
                    st.success(f"✅ Tested {len(strategies_to_test)} strategies! {ready_count}/{len(final_results)} ready for monitoring")
            
            elif "Multi-Config" in analysis_mode:
                # Multi-config mode - different directions and leverage
                if not directions:
                    st.error("Please select at least one direction (BUY/SELL)")
                    return
                if not leverage_levels:
                    st.error("Please select at least one leverage level")
                    return
                
                # Parse leverage levels
                leverage_vals = []
                for lev in leverage_levels:
                    if "1x" in lev:
                        leverage_vals.append(1.0)
                    elif "2x" in lev:
                        leverage_vals.append(2.0)
                    elif "3x" in lev:
                        leverage_vals.append(3.0)
                    elif "5x" in lev:
                        leverage_vals.append(5.0)
                
                total_configs = len(pairs_to_analyze) * len(directions) * len(leverage_vals)
                
                with st.spinner(f"Testing {total_configs} configurations ({len(directions)} directions × {len(leverage_vals)} leverage)..."):
                    # Import multi-config function
                    from ui.crypto_quick_trade_ui import analyze_multi_config_bulk
                    from clients.kraken_client import KrakenClient
                    
                    # Get kraken client from session state or scanner
                    kraken_client = scanner.kraken_client
                    
                    # Get position size from presets/custom input
                    pos_size = position_size
                    
                    # Run multi-config analysis
                    test_configs = {
                        'directions': directions,
                        'leverage_levels': leverage_vals,
                        'risk_pct': 2.0,
                        'take_profit_pct': 5.0
                    }
                    
                    analyze_multi_config_bulk(
                        kraken_client=kraken_client,
                        pairs=pairs_to_analyze,
                        position_size=pos_size,  # User-configured position size
                        test_configs=test_configs
                    )
                    
                    # Results are stored in session state by analyze_multi_config_bulk
                    if 'multi_config_results' in st.session_state:
                        st.session_state.tier3_results = st.session_state.multi_config_results
                        st.session_state.tier3_analysis_mode = "multi_config"  # Changed to avoid widget key conflict
                        st.session_state.tier3_timestamp = datetime.now()
                        st.success(f"✅ Multi-config analysis complete! {total_configs} configurations tested")
                    else:
                        st.error("Multi-config analysis failed to produce results")
                        return
            
            else:  # Ultimate mode
                # Ultimate mode - everything combined
                with st.spinner("🏆 Running ULTIMATE analysis (all strategies + all configs)..."):
                    ultimate_results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Step 1: All strategies
                    status_text.text("Phase 1/2: Testing all strategies...")
                    all_strategies = ["momentum", "ema_crossover", "rsi_stochastic", "fisher_rsi", "macd_volume", "aggressive_scalp"]
                    strategy_results = []
                    
                    for idx, strat in enumerate(all_strategies):
                        results = asyncio.run(
                            scanner.tier3_deep_analysis(
                                to_analyze,
                                strategy=strat,
                                ai_reviewer=ai_trade_reviewer
                            )
                        )
                        
                        for r in results:
                            r['tested_strategy'] = strat
                            r['analysis_type'] = 'strategy'
                        
                        strategy_results.extend(results)
                        progress_bar.progress(0.5 * (idx + 1) / len(all_strategies))
                    
                    # Step 2: Multi-config
                    status_text.text("Phase 2/2: Testing all configurations...")
                    from ui.crypto_quick_trade_ui import analyze_multi_config_bulk
                    
                    kraken_client = scanner.kraken_client
                    
                    # Use position size from UI
                    pos_size = position_size
                    
                    test_configs = {
                        'directions': ['BUY'],  # Focus on BUY for Ultimate mode
                        'leverage_levels': [1.0, 2.0, 3.0],  # Skip 5x for safety
                        'risk_pct': 2.0,
                        'take_profit_pct': 5.0
                    }
                    
                    analyze_multi_config_bulk(
                        kraken_client=kraken_client,
                        pairs=pairs_to_analyze,
                        position_size=pos_size,
                        test_configs=test_configs
                    )
                    
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Combine results
                    if 'multi_config_results' in st.session_state:
                        config_results_df = st.session_state.multi_config_results
                        # Convert DataFrame to list of dicts if needed
                        if isinstance(config_results_df, pd.DataFrame):
                            config_results = config_results_df.to_dict('records')
                        else:
                            config_results = config_results_df
                        
                        # Add analysis type to each config result
                        for r in config_results:
                            r['analysis_type'] = 'config'
                        
                        ultimate_results = strategy_results + config_results
                    else:
                        ultimate_results = strategy_results
                    
                    st.session_state.tier3_results = ultimate_results
                    st.session_state.tier3_analysis_mode = "ultimate"  # Changed to avoid widget key conflict
                    st.session_state.tier3_timestamp = datetime.now()
                    
                    st.success(f"🏆 ULTIMATE analysis complete! {len(strategy_results)} strategy tests + {len(config_results) if 'config_results' in locals() else 0} config tests = {len(ultimate_results)} total results")
        
        except Exception as e:
            st.error(f"Deep analysis failed: {e}")
            logger.error("Tier 3 analysis error: {}", str(e), exc_info=True)
            return
    
    # Display results
    tier3_results = st.session_state.get('tier3_results')
    if tier3_results is not None:
        # Get mode from session state early to ensure it's always available
        mode = st.session_state.get('tier3_analysis_mode', 'quick')
        # Check if results is empty (works for both list and DataFrame)
        is_empty = False
        if isinstance(tier3_results, pd.DataFrame):
            is_empty = tier3_results.empty
            results = tier3_results.to_dict('records')  # Convert to list of dicts for consistent handling
        elif isinstance(tier3_results, list):
            is_empty = len(tier3_results) == 0
            results = tier3_results
        else:
            is_empty = True
            results = []
        
        if not is_empty:
            timestamp = st.session_state.get('tier3_timestamp', datetime.now())
            
            st.markdown("---")
            st.markdown(f"**Results from:** {timestamp.strftime('%I:%M %p')} | **Mode:** {mode.upper()}")
            
            # Mode-specific summary metrics
            if mode == "multi_config" or mode == "ultimate":
                # Multi-config or Ultimate mode - different metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    unique_pairs = len(set(r.get('pair', '') for r in results))
                    st.metric("📊 Pairs Analyzed", unique_pairs)
                with col2:
                    total_configs = len(results)
                    st.metric("🔬 Total Configurations", total_configs)
                with col3:
                    buy_signals = sum(1 for r in results if r.get('side', r.get('strategy_signal')) == 'BUY')
                    st.metric("🟢 BUY Signals", buy_signals)
                with col4:
                    if results and any('ai_score' in r or 'strategy_confidence' in r for r in results):
                        confidences = [r.get('ai_score', r.get('strategy_confidence', 0)) for r in results if r.get('ai_score', r.get('strategy_confidence', 0)) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        st.metric("🎯 Avg Score", f"{avg_confidence:.1f}%")
                    else:
                        st.metric("🎯 Avg Score", "N/A")
        else:
            # Quick or Standard mode - traditional metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Analyzed", len(results))
            with col2:
                ready_count = sum(1 for r in results if r.get('ready_for_monitoring', False))
                st.metric("Ready to Trade", ready_count)
            with col3:
                buy_signals = sum(1 for r in results if r.get('strategy_signal') == 'BUY')
                st.metric("BUY Signals", buy_signals)
            with col4:
                if results:
                    raw_scores = [r.get('strategy_confidence') for r in results]
                    numeric_scores = [float(score) for score in raw_scores if isinstance(score, (int, float))]
                    avg_confidence = (sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Display based on mode
        if mode == "standard":
            st.markdown("### 📊 Strategy Comparison Results")
            st.info("💡 Showing best strategy for each pair. Expand to see all tested strategies.")
        elif mode == "multi_config":
            st.markdown("### 🔬 Multi-Configuration Results")
            st.info("💡 Showing all tested configurations (directions × leverage). Best results at top.")
        elif mode == "ultimate":
            st.markdown("### 🏆 ULTIMATE Analysis Results")
            st.info("💡 Combined results from all strategies AND all configurations. Sorted by score.")
        else:
            st.markdown("### 🎯 Deep Analysis Results")
        
        # Display results based on mode
        if mode == "multi_config" or (mode == "ultimate" and any('analysis_type' in r and r['analysis_type'] == 'config' for r in results)):
            # Multi-config display
            display_multi_config_results(results, mode)
        elif mode == "standard":
            # Standard mode - show best strategy per pair + all strategies option
            display_standard_mode_results(results)
        else:
            # Quick mode or traditional display
            display_quick_mode_results(results)


def display_quick_mode_results(results):
    """Display results for Quick mode (single strategy)"""
    from datetime import datetime
    for i, result in enumerate(results):
        is_ready = result.get('ready_for_monitoring', False)
        icon = "🏆" if is_ready and i < 3 else "✅" if is_ready else "📊"
        
        with st.expander(
            f"{icon} {result['pair']} - {result.get('strategy_signal', 'HOLD')} "
            f"(Score: {result['score']:.1f})",
            expanded=i < 2
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💰 Trade Setup**")
                st.write(f"Signal: **{result.get('strategy_signal', 'N/A')}**")
                price = result.get('entry_price', result.get('price', 0)) or 0
                price_str = f"${price:.8f}" if price < 0.01 else f"${price:.4f}"
                st.write(f"Entry: {price_str}")
                if result.get('stop_loss'):
                    st.write(f"Stop: ${result['stop_loss']:.8f}")
                if result.get('take_profit'):
                    st.write(f"Target: ${result['take_profit']:.8f}")
                st.write(f"Risk: {result.get('risk_level', 'N/A')}")
            
            with col2:
                st.markdown("**📊 Analysis**")
                st.write(f"Strategy: {result.get('strategy', 'N/A')}")
                confidence = result.get('strategy_confidence') or 0
                st.write(f"Confidence: {float(confidence):.1f}%")
                st.write(f"Score: {result['score']:.1f}/100")
                
                signals = result.get('signals', [])
                if signals:
                    st.write("Signals:")
                    for sig in signals[:3]:
                        st.write(f"• {sig}")
            
            with col3:
                st.markdown("**🤖 AI Review**")
                ai_rec = result.get('ai_recommendation', 'Not available')
                st.write(f"{ai_rec}")
                if result.get('ai_confidence'):
                    st.write(f"Confidence: {result['ai_confidence']:.1f}%")
                
                ai_risks = result.get('ai_risks', [])
                if ai_risks and isinstance(ai_risks, list):
                    st.write("⚠️ Risks:")
                    for risk in ai_risks[:2]:
                        st.write(f"• {risk}")
                elif ai_risks and isinstance(ai_risks, str):
                    st.write(f"⚠️ Risk: {ai_risks}")
            
            # Action buttons
            acol1, acol2 = st.columns(2)
            
            with acol1:
                # Use on_click callback for reliable session state setting in expanders
                def set_quick_setup(result_data, idx):
                    logger.info(f"🔘 QUICK SETUP CALLBACK! Setting up {result_data['pair']}")
                    # Transfer setup to Quick Trade
                    st.session_state.crypto_scanner_opportunity = {
                        'symbol': result_data['pair'],
                        'strategy': result_data.get('strategy', 'Unknown'),
                        'confidence': result_data.get('strategy_confidence', 0) > 70,
                        'risk_level': result_data.get('risk_level', 'Medium'),
                        'score': result_data.get('score', 0),
                        'current_price': result_data.get('entry_price', result_data.get('price', 0)),
                        'change_24h': result_data.get('change_24h', 0),
                        'volume_ratio': result_data.get('volume_ratio', 1.0),
                        'volatility': result_data.get('volatility', 0),
                        'reason': f"{result_data.get('strategy_signal', 'HOLD')} recommended",
                        'ai_reasoning': result_data.get('ai_recommendation', ''),
                        'ai_confidence': 'High' if result_data.get('ai_confidence', 0) >= 75 else 'Medium',
                        'ai_rating': result_data.get('ai_confidence', 0) / 10,
                        'ai_risks': result_data.get('ai_risks', [])
                    }
                    st.session_state.crypto_quick_pair = result_data['pair']
                    st.session_state.crypto_quick_trade_pair = result_data['pair']
                    st.session_state.crypto_quick_direction = 'BUY'
                    st.session_state.crypto_trading_mode = 'Spot Trading'
                    st.session_state.crypto_quick_leverage = 1
                    st.session_state.crypto_quick_position_size = 200
                    st.session_state.crypto_quick_stop_pct = 3.0
                    st.session_state.crypto_quick_target_pct = 10.0  # Higher target to beat Kraken fees
                    
                    # Navigate to Quick Trade
                    st.session_state.active_crypto_tab = "⚡ Quick Trade"
                    st.session_state.quick_trade_subtab = "⚡ Execute Trade"
                    st.session_state.show_quick_setup_success = {'pair': result_data['pair']}
                
                st.button(
                    f"✅ Use This Setup", 
                    key=f"use_quick_{i}", 
                    type="primary" if is_ready else "secondary",
                    on_click=set_quick_setup,
                    args=(result, i)
                )
            
            with acol2:
                if st.button(f"🤖 Add to Monitor", key=f"monitor_quick_{i}", type="secondary"):
                    # Add to monitoring
                    if 'ai_trade_reviewer' in st.session_state and st.session_state.ai_trade_reviewer:
                        reviewer = st.session_state.ai_trade_reviewer
                        trade_id = f"{result['pair'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        reviewer.start_trade_monitoring(
                            trade_id=trade_id,
                            pair=result['pair'],
                            side='BUY',
                            entry_price=result.get('entry_price', result.get('price', 0)),
                            current_price=result.get('price', result.get('entry_price', 0)),
                            volume=1.0,  # Default volume
                            stop_loss=result.get('stop_loss', 0),
                            take_profit=result.get('take_profit', 0),
                            strategy=result.get('strategy', 'Unknown')
                        )
                        
                        # Update session state
                        st.session_state.active_trade_monitors = reviewer.active_monitors
                        st.success(f"✅ {result['pair']} added to monitoring!")
                        st.info("👉 Go to 'Active Monitors' tab to track it")
                    else:
                        st.error("❌ Trade reviewer not available")


def display_standard_mode_results(results):
    """Display results for Standard mode (multiple strategies comparison)"""
    from datetime import datetime
    # Get all strategy results from session state
    all_strategy_results = st.session_state.get('tier3_all_strategy_results', [])
    
    for i, result in enumerate(results):
        pair = result['pair']
        is_ready = result.get('ready_for_monitoring', False)
        icon = "🏆" if is_ready and i < 3 else "✅" if is_ready else "📊"
        
        # Get all results for this pair
        pair_results = [r for r in all_strategy_results if r['pair'] == pair]
        num_strategies = len(pair_results)
        
        with st.expander(
            f"{icon} {pair} - Best: {result.get('tested_strategy', 'N/A')} "
            f"(Score: {result['score']:.1f}, {num_strategies} strategies tested)",
            expanded=i < 2
        ):
            # Show best strategy details
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🏆 Best Strategy**")
                st.write(f"Strategy: {result.get('tested_strategy', 'N/A')}")
                st.write(f"Signal: **{result.get('strategy_signal', 'N/A')}**")
                st.write(f"Score: {result['score']:.1f}/100")
                confidence = result.get('strategy_confidence') or 0
                st.write(f"Confidence: {float(confidence):.1f}%")
            
            with col2:
                st.markdown("**💰 Trade Setup**")
                price = result.get('entry_price', result.get('price', 0)) or 0
                st.write(f"Entry: ${price:.8f}" if price < 0.01 else f"Entry: ${price:.4f}")
                if result.get('stop_loss'):
                    st.write(f"Stop: ${result['stop_loss']:.8f}")
                if result.get('take_profit'):
                    st.write(f"Target: ${result['take_profit']:.8f}")
            
            # Show all strategy results in a table
            if pair_results:
                st.markdown("**📊 All Strategies Comparison**")
                comparison_df = pd.DataFrame([
                    {
                        'Strategy': r.get('tested_strategy', 'N/A'),
                        'Signal': r.get('strategy_signal', 'N/A'),
                        'Score': r['score'],
                        'Confidence': float(r.get('strategy_confidence') or 0)
                    }
                    for r in sorted(pair_results, key=lambda x: x['score'], reverse=True)
                ])
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
            
            # Action buttons
            acol1, acol2 = st.columns(2)
            
            with acol1:
                if st.button(f"� Use Best Strategy", key=f"use_std_{i}", type="primary" if is_ready else "secondary"):
                    # Transfer setup to Quick Trade
                    st.session_state.crypto_scanner_opportunity = {
                        'symbol': result['pair'],
                        'strategy': result.get('tested_strategy', 'Unknown'),
                        'confidence': result.get('strategy_confidence', 0) > 70,
                        'risk_level': result.get('risk_level', 'Medium'),
                        'score': result.get('score', 0),
                        'current_price': result.get('entry_price', result.get('price', 0)),
                        'change_24h': result.get('change_24h', 0),
                        'volume_ratio': result.get('volume_ratio', 1.0),
                        'volatility': result.get('volatility', 0),
                        'reason': f"{result.get('strategy_signal', 'HOLD')} - {result.get('tested_strategy')}",
                        'ai_reasoning': result.get('ai_recommendation', ''),
                        'ai_confidence': 'High' if result.get('ai_confidence', 0) >= 75 else 'Medium',
                        'ai_rating': result.get('ai_confidence', 0) / 10,
                        'ai_risks': result.get('ai_risks', [])
                    }
                    st.session_state.crypto_quick_pair = result['pair']
                    st.session_state.crypto_quick_trade_pair = result['pair']
                    st.session_state.crypto_quick_direction = 'BUY'
                    st.session_state.crypto_trading_mode = 'Spot Trading'
                    st.session_state.crypto_quick_leverage = 1
                    st.session_state.crypto_quick_position_size = 200
                    st.session_state.crypto_quick_stop_pct = 3.0
                    st.session_state.crypto_quick_target_pct = 10.0  # Higher target to beat Kraken fees
                    
                    # Navigate to Quick Trade
                    st.session_state.active_crypto_tab = "⚡ Quick Trade"
                    st.session_state.quick_trade_subtab = "⚡ Execute Trade"
                    
                    st.success(f"✅ {result.get('tested_strategy')} strategy ready! Switching to Execute Trade...")
                    st.balloons()
                    st.rerun()
            
            with acol2:
                if st.button(f"🤖 Add to Monitor", key=f"monitor_std_{i}", type="secondary"):
                    # Add to monitoring
                    if 'ai_trade_reviewer' in st.session_state and st.session_state.ai_trade_reviewer:
                        reviewer = st.session_state.ai_trade_reviewer
                        trade_id = f"{result['pair'].replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        reviewer.start_trade_monitoring(
                            trade_id=trade_id,
                            pair=result['pair'],
                            side='BUY',
                            entry_price=result.get('entry_price', result.get('price', 0)),
                            current_price=result.get('price', result.get('entry_price', 0)),
                            volume=1.0,  # Default volume
                            stop_loss=result.get('stop_loss', 0),
                            take_profit=result.get('take_profit', 0),
                            strategy=result.get('tested_strategy', 'Unknown')
                        )
                        
                        # Update session state
                        st.session_state.active_trade_monitors = reviewer.active_monitors
                        st.success(f"✅ {result['pair']} added to monitoring!")
                        st.info("👉 Go to 'Active Monitors' tab to track it")
                    else:
                        st.error("❌ Trade reviewer not available")


def display_multi_config_results(results, mode):
    """Display results for Multi-Config or Ultimate mode"""
    # Group by pair
    pairs = {}
    for r in results:
        pair = r.get('pair', 'Unknown')
        if pair not in pairs:
            pairs[pair] = []
        pairs[pair].append(r)
    
    # Sort pairs by best score
    sorted_pairs = sorted(pairs.items(), key=lambda x: max(r.get('ai_score', r.get('score', 0)) for r in x[1]), reverse=True)
    
    for pair_idx, (pair, configs) in enumerate(sorted_pairs):
        # Sort configs by score
        configs.sort(key=lambda x: x.get('ai_score', x.get('score', 0)), reverse=True)
        best_config = configs[0]
        
        # Determine icon
        best_score = best_config.get('ai_score', best_config.get('score', 0))
        icon = "🏆" if best_score >= 70 and pair_idx < 3 else "✅" if best_score >= 60 else "📊"
        
        # Get trade type
        if 'trade_type' in best_config:
            trade_type = best_config['trade_type']
        else:
            direction = best_config.get('side', best_config.get('strategy_signal', 'BUY'))
            trade_type = direction
        
        with st.expander(
            f"{icon} {pair} - Best: {trade_type} "
            f"(Score: {best_score:.1f}, {len(configs)} configs tested)",
            expanded=pair_idx < 2
        ):
            # Best config details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🏆 Best Configuration**")
                st.write(f"Type: {best_config.get('trade_type', 'N/A')}")
                st.write(f"Score: {best_score:.1f}")
                if best_config.get('ai_approved') is not None:
                    approved_icon = "✅" if best_config['ai_approved'] else "❌"
                    st.write(f"AI Approved: {approved_icon}")
            
            with col2:
                st.markdown("**💰 Prices**")
                current_price = best_config.get('current_price', 0) or 0
                st.write(f"Entry: ${current_price:.8f}" if current_price < 0.01 else f"Entry: ${current_price:.4f}")
                if best_config.get('stop_loss_price'):
                    st.write(f"Stop: ${best_config['stop_loss_price']:.8f}")
                if best_config.get('take_profit_price'):
                    st.write(f"Target: ${best_config['take_profit_price']:.8f}")
            
            with col3:
                st.markdown("**📊 Risk/Reward**")
                if best_config.get('risk_reward_ratio'):
                    st.write(f"R:R Ratio: {best_config['risk_reward_ratio']:.2f}:1")
                if best_config.get('leverage'):
                    st.write(f"Leverage: {best_config['leverage']}x")
                if best_config.get('effective_position'):
                    st.write(f"Position: ${best_config['effective_position']:.2f}")
            
            # All configurations table
            st.markdown(f"**🔬 All {len(configs)} Configurations**")
            config_df = pd.DataFrame([
                {
                    'Type': c.get('trade_type', 'N/A'),
                    'Score': c.get('ai_score', c.get('score', 0)),
                    'AI': "✅" if c.get('ai_approved') else "❌" if c.get('ai_approved') is not None else "?",
                    'Leverage': f"{c.get('leverage', 1)}x",
                    'Stop': f"${c.get('stop_loss_price', 0):.4f}",
                    'Target': f"${c.get('take_profit_price', 0):.4f}"
                }
                for c in configs
            ])
            st.dataframe(config_df, hide_index=True, use_container_width=True)
            
            if st.button(f"🚀 Use Best Config", key=f"use_multi_{pair_idx}", type="primary"):
                # Transfer setup to Quick Trade
                st.session_state.crypto_scanner_opportunity = {
                    'symbol': best_config.get('pair', 'UNKNOWN'),
                    'strategy': best_config.get('strategy', 'Unknown'),
                    'confidence': best_config.get('ai_approved', False),
                    'risk_level': 'Medium' if (best_config.get('leverage', 0) or 0) <= 2 else 'High',
                    'score': best_config.get('score', 0),
                    'current_price': best_config.get('current_price', 0),
                    'change_24h': best_config.get('change_24h', 0),
                    'volume_ratio': best_config.get('volume_ratio', 1.0),
                    'volatility': best_config.get('volatility', 0),
                    'reason': f"{best_config.get('trade_type')} recommended",
                    'ai_reasoning': best_config.get('ai_recommendation', ''),
                    'ai_confidence': 'High' if best_config.get('ai_confidence', 0) >= 75 else 'Medium',
                    'ai_rating': best_config.get('ai_confidence', 0) / 10,
                    'ai_risks': best_config.get('ai_risks', [])
                }
                st.session_state.crypto_quick_pair = best_config.get('pair', 'UNKNOWN')
                st.session_state.crypto_quick_trade_pair = best_config.get('pair', 'UNKNOWN')
                st.session_state.crypto_quick_direction = best_config.get('direction', 'BUY')
                st.session_state.crypto_trading_mode = best_config.get('trading_mode', 'Spot Trading')
                st.session_state.crypto_quick_leverage = best_config.get('leverage', 1)
                st.session_state.crypto_quick_position_size = best_config.get('position_size', 100)
                st.session_state.crypto_quick_stop_pct = best_config.get('stop_pct', 2.0)
                st.session_state.crypto_quick_target_pct = best_config.get('target_pct', 5.0)
                
                # Navigate to Quick Trade
                st.session_state.active_crypto_tab = "⚡ Quick Trade"
                st.session_state.quick_trade_subtab = "⚡ Execute Trade"
                
                st.success(f"✅ {best_config.get('trade_type')} configuration ready! Switching to Execute Trade...")
                st.balloons()
                st.rerun()


def display_active_monitors(ai_trade_reviewer, kraken_client):
    """Display active trade monitors"""
    st.subheader("🤖 Active Trade Monitors")
    st.markdown("""
    View and manage actively monitored trades from Deep Analysis.
    """)
    
    if not ai_trade_reviewer:
        st.warning("⚠️ AI Trade Reviewer not available")
        return
    
    # Sync monitors from session state
    if 'active_trade_monitors' in st.session_state:
        ai_trade_reviewer.active_monitors = st.session_state.active_trade_monitors
    
    active_monitors = ai_trade_reviewer.active_monitors
    
    if not active_monitors:
        st.info("📊 No active trades being monitored")
        st.markdown("""
        **How to add monitors:**
        1. Run Tier 1 → Tier 2 → Tier 3 analysis
        2. Click "🤖 Add to Monitoring" on any coin
        3. Monitor will appear here with live P&L tracking
        """)
        return
    
    st.success(f"📊 Monitoring {len(active_monitors)} active trade(s)")
    
    # Display each monitor
    for monitor_id, monitor_data in active_monitors.items():
        pair = monitor_data.get('pair', 'Unknown')
        entry_price = monitor_data.get('entry_price', 0)
        stop_loss = monitor_data.get('stop_loss', 0)
        take_profit = monitor_data.get('take_profit', 0)
        
        # Get current price
        try:
            ticker = kraken_client.get_ticker_info(pair.replace('/USD', ''))
            current_price = float(ticker.get('c', [0, 0])[0]) if ticker else entry_price
        except:
            current_price = entry_price
        
        # Calculate P&L
        pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        pnl_color = "green" if pnl_pct > 0 else "red"
        
        with st.expander(f"{'🟢' if pnl_pct > 0 else '🔴'} {pair} - P&L: {pnl_pct:+.2f}%"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**💰 Prices**")
                st.write(f"Entry: ${entry_price:.8f}")
                st.write(f"Current: ${current_price:.8f}")
                st.write(f"Stop: ${stop_loss:.8f}")
                st.write(f"Target: ${take_profit:.8f}")
            
            with col2:
                st.markdown("**📊 Performance**")
                st.write(f"P&L: :{pnl_color}[{pnl_pct:+.2f}%]")
                
                # Distance to targets
                stop_dist = ((stop_loss - current_price) / current_price) * 100
                target_dist = ((take_profit - current_price) / current_price) * 100
                st.write(f"To Stop: {stop_dist:.2f}%")
                st.write(f"To Target: {target_dist:.2f}%")
            
            with col3:
                st.markdown("**🎯 Actions**")
                if st.button(f"🤖 AI Recommendation", key=f"ai_rec_{monitor_id}"):
                    st.info("Getting AI recommendation...")
                
                if st.button(f"🚪 Close Position", key=f"close_{monitor_id}"):
                    # Close monitoring
                    ai_trade_reviewer.stop_monitoring(monitor_id)
                    if 'active_trade_monitors' in st.session_state:
                        st.session_state.active_trade_monitors = ai_trade_reviewer.active_monitors
                    st.success(f"✅ Stopped monitoring {pair}")
                    st.rerun()

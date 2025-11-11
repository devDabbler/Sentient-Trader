"""
CoinMarketCap Features UI
Displays trending, sentiment, and new cryptocurrencies from CoinMarketCap

References:
- Trending: https://coinmarketcap.com/trending-cryptocurrencies/
- Sentiment: https://coinmarketcap.com/sentiment/
- New Coins: https://coinmarketcap.com/new/
"""

import streamlit as st
import pandas as pd
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from loguru import logger
from services.coinmarketcap_features import (
    CoinMarketCapFeatures,
    TrendingCrypto,
    CryptoSentiment,
    NewCrypto
)


def display_trending_cryptos(trending: List[TrendingCrypto]):
    """Display trending cryptocurrencies"""
    if not trending:
        st.info("No trending cryptos available. Make sure CoinMarketCap API key is set.")
        return
    
    st.markdown(f"### 🔥 Trending Cryptocurrencies ({len(trending)} coins)")
    st.caption("Cryptocurrencies with the highest 24h price changes and volume from CoinMarketCap")
    
    # Create dataframe
    data = []
    for crypto in trending:
        change_emoji = "🟢" if crypto.change_24h > 0 else "🔴"
        data.append({
            'Rank': crypto.rank,
            'Symbol': crypto.symbol,
            'Name': crypto.name,
            'Price': f"${crypto.price:,.6f}" if crypto.price < 1 else f"${crypto.price:,.2f}",
            '1h Change': f"{change_emoji} {crypto.change_1h:+.2f}%",
            '24h Change': f"{change_emoji} {crypto.change_24h:+.2f}%",
            'Market Cap': f"${crypto.market_cap:,.0f}" if crypto.market_cap else "N/A",
            'Volume 24h': f"${crypto.volume_24h:,.0f}" if crypto.volume_24h else "N/A"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Display detailed cards
    st.markdown("### 📊 Detailed View")
    for i, crypto in enumerate(trending[:10], 1):  # Show top 10
        change_emoji = "🟢" if crypto.change_24h > 0 else "🔴"
        with st.expander(f"#{i} {crypto.symbol} - {crypto.name} | {change_emoji} {crypto.change_24h:+.2f}%", expanded=(i <= 3)):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price", f"${crypto.price:,.6f}" if crypto.price < 1 else f"${crypto.price:,.2f}")
                st.metric("Market Cap", f"${crypto.market_cap:,.0f}" if crypto.market_cap else "N/A")
            
            with col2:
                st.metric("1h Change", f"{change_emoji} {crypto.change_1h:+.2f}%")
                st.metric("24h Change", f"{change_emoji} {crypto.change_24h:+.2f}%")
            
            with col3:
                st.metric("Volume 24h", f"${crypto.volume_24h:,.0f}" if crypto.volume_24h else "N/A")
                st.metric("Rank", f"#{crypto.rank}" if crypto.rank else "N/A")
            
            with col4:
                if st.button(f"➕ Add to Watchlist", key=f"add_trending_{crypto.symbol}_{i}"):
                    st.session_state.crypto_add_to_watchlist = crypto.symbol
                    st.success(f"✅ {crypto.symbol} will be added to watchlist")
            
            # Add link to CoinMarketCap
            st.markdown(f"[View on CoinMarketCap](https://coinmarketcap.com/currencies/{crypto.name.lower().replace(' ', '-')}/)")


def display_sentiment_analysis(sentiment: Optional[CryptoSentiment], symbol: str):
    """Display sentiment analysis for a cryptocurrency"""
    if not sentiment:
        st.info(f"Sentiment data not available for {symbol}. This may require a paid CoinMarketCap API plan.")
        return
    
    st.markdown(f"### 📊 Sentiment Analysis: {symbol}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Sentiment score gauge
        score_color = "🟢" if sentiment.sentiment_score > 20 else "🔴" if sentiment.sentiment_score < -20 else "⚪"
        st.metric("Sentiment Score", f"{score_color} {sentiment.sentiment_score:.1f}/100")
        
        # Bullish percentage
        st.metric("Bullish", f"{sentiment.bullish_percent:.1f}%")
    
    with col2:
        st.metric("Bearish", f"{sentiment.bearish_percent:.1f}%")
        st.metric("Neutral", f"{sentiment.neutral_percent:.1f}%")
    
    with col3:
        st.metric("Social Volume", f"{sentiment.social_volume:,}" if sentiment.social_volume else "N/A")
        
        # Visual sentiment bar
        st.markdown("**Sentiment Breakdown:**")
        st.progress(sentiment.bullish_percent / 100, text=f"Bullish: {sentiment.bullish_percent:.1f}%")
        st.progress(sentiment.bearish_percent / 100, text=f"Bearish: {sentiment.bearish_percent:.1f}%")
        st.progress(sentiment.neutral_percent / 100, text=f"Neutral: {sentiment.neutral_percent:.1f}%")


def display_new_cryptos(new_coins: List[NewCrypto]):
    """Display newly added cryptocurrencies"""
    if not new_coins:
        st.info("No new cryptos available. Make sure CoinMarketCap API key is set.")
        return
    
    st.markdown(f"### 🆕 New Cryptocurrencies ({len(new_coins)} coins)")
    st.caption("Recently added cryptocurrencies from CoinMarketCap (last 30 days)")
    
    # Create dataframe
    data = []
    for crypto in new_coins:
        change_emoji = "🟢" if crypto.change_24h > 0 else "🔴"
        data.append({
            'Symbol': crypto.symbol,
            'Name': crypto.name,
            'Price': f"${crypto.price:,.8f}" if crypto.price < 0.01 else f"${crypto.price:,.6f}" if crypto.price < 1 else f"${crypto.price:,.2f}",
            '1h Change': f"{change_emoji} {crypto.change_1h:+.2f}%",
            '24h Change': f"{change_emoji} {crypto.change_24h:+.2f}%",
            'Market Cap': f"${crypto.market_cap:,.0f}" if crypto.market_cap else "N/A",
            'Volume 24h': f"${crypto.volume_24h:,.0f}" if crypto.volume_24h else "N/A",
            'Blockchain': crypto.blockchain,
            'Added': crypto.added_date
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Display detailed cards
    st.markdown("### 📊 Detailed View")
    for i, crypto in enumerate(new_coins[:15], 1):  # Show top 15
        change_emoji = "🟢" if crypto.change_24h > 0 else "🔴"
        with st.expander(f"#{i} {crypto.symbol} - {crypto.name} | {change_emoji} {crypto.change_24h:+.2f}% | {crypto.blockchain}", expanded=(i <= 5)):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Price", f"${crypto.price:,.8f}" if crypto.price < 0.01 else f"${crypto.price:,.6f}" if crypto.price < 1 else f"${crypto.price:,.2f}")
                st.metric("Market Cap", f"${crypto.market_cap:,.0f}" if crypto.market_cap else "N/A")
            
            with col2:
                st.metric("1h Change", f"{change_emoji} {crypto.change_1h:+.2f}%")
                st.metric("24h Change", f"{change_emoji} {crypto.change_24h:+.2f}%")
            
            with col3:
                st.metric("Volume 24h", f"${crypto.volume_24h:,.0f}" if crypto.volume_24h else "N/A")
                st.metric("Blockchain", crypto.blockchain)
            
            with col4:
                st.metric("Added Date", crypto.added_date)
                if st.button(f"➕ Add to Watchlist", key=f"add_new_{crypto.symbol}_{i}"):
                    st.session_state.crypto_add_to_watchlist = crypto.symbol
                    st.success(f"✅ {crypto.symbol} will be added to watchlist")
            
            # Add link to CoinMarketCap
            st.markdown(f"[View on CoinMarketCap](https://coinmarketcap.com/currencies/{crypto.name.lower().replace(' ', '-')}/)")


def render_coinmarketcap_features_tab(watchlist_manager=None):
    """
    Render CoinMarketCap Features tab
    
    Features:
    - Trending cryptocurrencies
    - Sentiment analysis
    - New coin listings
    
    References:
    - https://coinmarketcap.com/trending-cryptocurrencies/
    - https://coinmarketcap.com/sentiment/
    - https://coinmarketcap.com/new/
    """
    st.markdown("### 🔥 CoinMarketCap Features")
    st.caption("Access trending cryptocurrencies, sentiment analysis, and new coin listings from CoinMarketCap")
    
    # Check for API key
    import os
    api_key = os.getenv('COINMARKETCAP_API_KEY')
    if not api_key:
        st.warning("⚠️ **CoinMarketCap API key not found**")
        st.info("""
        To use CoinMarketCap features, you need to:
        1. Sign up for a free CoinMarketCap API key at https://coinmarketcap.com/api/
        2. Add `COINMARKETCAP_API_KEY=your_key_here` to your `.env` file
        3. Free tier: 333 API calls/day
        
        **Note**: Some features (like direct sentiment API) may require a paid plan.
        """)
        return
    
    # Sub-tabs for different features
    feature_tabs = st.tabs(["🔥 Trending", "📊 Sentiment", "🆕 New Coins"])
    
    # Trending Cryptos Tab
    with feature_tabs[0]:
        st.markdown("#### 🔥 Trending Cryptocurrencies")
        st.caption("Cryptocurrencies with the highest visibility and price changes in the last 24 hours")
        st.markdown("*Source: [CoinMarketCap Trending](https://coinmarketcap.com/trending-cryptocurrencies/)*")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            limit = st.slider("Number of trending cryptos to show", 10, 50, 25, key="trending_limit")
        
        # Initialize session state (before widgets)
        if 'trending_cryptos' not in st.session_state:
            st.session_state.trending_cryptos = None
        
        with col2:
            refresh_clicked = st.button("🔄 Refresh Trending", key="refresh_trending_btn")
        
        # Fetch trending cryptos if needed (button clicked or no data)
        should_fetch = refresh_clicked or st.session_state.trending_cryptos is None
        if should_fetch:
            with st.spinner("Fetching trending cryptos from CoinMarketCap..."):
                try:
                    cmc = CoinMarketCapFeatures()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    trending = loop.run_until_complete(cmc.get_trending_cryptos(limit))
                    loop.close()
                    
                    st.session_state.trending_cryptos = trending
                except Exception as e:
                    st.error(f"Error fetching trending cryptos: {e}")
                    logger.error(f"Error fetching trending cryptos: {e}", exc_info=True)
                    st.session_state.trending_cryptos = []
        
        if st.session_state.get('trending_cryptos'):
            display_trending_cryptos(st.session_state.trending_cryptos)
    
    # Sentiment Analysis Tab
    with feature_tabs[1]:
        st.markdown("#### 📊 Cryptocurrency Sentiment Analysis")
        st.caption("Analyze sentiment for any cryptocurrency from CoinMarketCap")
        st.markdown("*Source: [CoinMarketCap Sentiment](https://coinmarketcap.com/sentiment/)*")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            sentiment_symbol = st.text_input(
                "Enter Cryptocurrency Symbol",
                value="BTC",
                placeholder="e.g., BTC, ETH, SOL",
                key="sentiment_symbol_input"
            ).upper()
        
        with col2:
            analyze_clicked = st.button("📊 Analyze Sentiment", key="analyze_sentiment_btn")
        
        # Handle sentiment analysis
        sentiment_cache_key = f'sentiment_{sentiment_symbol}' if sentiment_symbol else None
        
        if analyze_clicked and sentiment_symbol:
            with st.spinner(f"Analyzing sentiment for {sentiment_symbol}..."):
                try:
                    cmc = CoinMarketCapFeatures()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    sentiment = loop.run_until_complete(cmc.get_crypto_sentiment(sentiment_symbol))
                    loop.close()
                    
                    if sentiment:
                        st.session_state[sentiment_cache_key] = sentiment
                        st.success(f"✅ Sentiment analysis complete for {sentiment_symbol}")
                    else:
                        st.warning(f"Could not fetch sentiment data for {sentiment_symbol}. This may require a paid CoinMarketCap API plan.")
                except Exception as e:
                    st.error(f"Error analyzing sentiment: {e}")
                    logger.error(f"Error analyzing sentiment: {e}", exc_info=True)
        
        # Show sentiment analysis if available in cache
        if sentiment_cache_key and sentiment_cache_key in st.session_state:
            st.divider()
            display_sentiment_analysis(st.session_state[sentiment_cache_key], sentiment_symbol)
    
    # New Coins Tab
    with feature_tabs[2]:
        st.markdown("#### 🆕 New Cryptocurrency Listings")
        st.caption("Discover newly added cryptocurrencies to CoinMarketCap")
        st.markdown("*Source: [CoinMarketCap New Coins](https://coinmarketcap.com/new/)*")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_limit = st.slider("Number of new cryptos to show", 10, 50, 25, key="new_coins_limit")
        
        # Initialize session state (before widgets)
        if 'new_cryptos' not in st.session_state:
            st.session_state.new_cryptos = None
        
        with col2:
            refresh_clicked = st.button("🔄 Refresh New Coins", key="refresh_new_coins_btn")
        
        # Fetch new cryptos if needed (button clicked or no data)
        should_fetch = refresh_clicked or st.session_state.new_cryptos is None
        if should_fetch:
            with st.spinner("Fetching new cryptos from CoinMarketCap..."):
                try:
                    cmc = CoinMarketCapFeatures()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    new_coins = loop.run_until_complete(cmc.get_new_cryptos(new_limit))
                    loop.close()
                    
                    st.session_state.new_cryptos = new_coins
                except Exception as e:
                    st.error(f"Error fetching new cryptos: {e}")
                    logger.error(f"Error fetching new cryptos: {e}", exc_info=True)
                    st.session_state.new_cryptos = []
        
        if st.session_state.get('new_cryptos'):
            display_new_cryptos(st.session_state.new_cryptos)
    
    # Handle adding to watchlist
    if st.session_state.get('crypto_add_to_watchlist') and watchlist_manager:
        symbol = st.session_state.crypto_add_to_watchlist
        if watchlist_manager.add_crypto(symbol):
            st.success(f"✅ Added {symbol} to watchlist!")
        else:
            st.warning(f"⚠️ {symbol} may already be in watchlist or failed to add")
        st.session_state.crypto_add_to_watchlist = None


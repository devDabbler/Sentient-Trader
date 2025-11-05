"""
Crypto Sentiment Analyzer - Social sentiment for cryptocurrency markets

Combines:
1. CoinGecko trending API (emerging coins gaining attention)
2. Reddit sentiment (r/cryptocurrency, r/CryptoCurrency, r/defi)
3. Twitter sentiment (crypto hashtags via Nitter)
4. On-chain metrics (optional)

Optimized for finding monster runners by focusing on trending/emerging coins.
"""

import os
import asyncio
import httpx
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CryptoSocialMention:
    """Social media mention for crypto"""
    source: str  # 'coingecko', 'reddit', 'twitter', 'stocktwits'
    symbol: str  # e.g., 'BTC', 'ETH', 'SHIB'
    text: str
    sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    sentiment_score: float  # -1.0 to 1.0
    author: str
    timestamp: str
    url: str
    engagement: int  # upvotes, likes, etc
    relevance_score: float  # 0-1


@dataclass
class CryptoTrendingData:
    """CoinGecko trending coin data"""
    symbol: str
    name: str
    market_cap_rank: Optional[int]
    price_usd: float
    price_change_24h: float
    volume_24h: float
    market_cap: float
    trending_score: int  # 1-7 (1 = most trending)
    sentiment: str  # BULLISH, NEUTRAL, BEARISH
    sentiment_score: float


class CryptoSentimentAnalyzer:
    """
    Analyzes sentiment for cryptocurrencies focusing on trending/emerging coins.
    Uses CoinGecko trending API + social sentiment from Reddit/Twitter.
    """
    
    def __init__(self):
        """Initialize the crypto sentiment analyzer"""
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.last_coingecko_call = 0
        self.coingecko_cache = {}
        self.cache_ttl = 3600  # 1 hour cache for trending data
        
        # Sentiment keywords (crypto-specific)
        self.bullish_keywords = {
            'moon', 'rocket', '🚀', 'bullish', 'buy', 'long', 'hodl',
            'pump', 'surge', 'breakout', 'bull', 'gains', 'lambo',
            'diamond hands', 'to the moon', 'mooning', 'rip', 'up',
            'green', 'winner', 'strong', 'beat', 'crushing', 'killing',
            'bullrun', 'rally', 'bull trap', 'squeeze', 'yolo'
        }
        
        self.bearish_keywords = {
            'crash', 'dump', 'bearish', 'sell', 'short', 'rekt',
            'down', 'red', 'loss', 'dead', 'collapse', 'plunge',
            'bear', 'weak', 'loser', 'miss', 'disappointing',
            'bag', 'bagholding', 'rug', 'scam', 'ponzi', 'dead coin'
        }
        
        logger.info("🔧 Crypto Sentiment Analyzer initialized")
        logger.info("   • CoinGecko trending API (10 calls/min limit)")
        logger.info("   • Reddit sentiment (RSS feeds + API)")
        logger.info("   • Twitter sentiment (Nitter mirrors)")
        logger.info("   • 1-hour cache for trending data")
    
    async def get_trending_cryptos(self, top_n: int = 10) -> List[CryptoTrendingData]:
        """
        Get trending cryptocurrencies from CoinGecko
        
        Args:
            top_n: Number of trending coins to return
            
        Returns:
            List of CryptoTrendingData objects
        """
        try:
            # Check cache first
            cache_key = f"trending_{top_n}"
            if cache_key in self.coingecko_cache:
                cached_data, cached_time = self.coingecko_cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    logger.info(f"📊 Using cached trending data (age: {int(time.time() - cached_time)}s)")
                    return cached_data
            
            # Rate limiting: max 10 calls/min = 1 call every 6 seconds
            elapsed = time.time() - self.last_coingecko_call
            if elapsed < 6:
                await asyncio.sleep(6 - elapsed)
            
            logger.info("🔍 Fetching trending cryptos from CoinGecko...")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.coingecko_api}/search/trending",
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                
                self.last_coingecko_call = time.time()
                
                if response.status_code != 200:
                    logger.error(f"CoinGecko API error: {response.status_code}")
                    return []
                
                data = response.json()
                coins = data.get('coins', [])
                
                trending_data = []
                for i, coin_data in enumerate(coins[:top_n]):
                    try:
                        coin_info = coin_data.get('item', {})
                        symbol = coin_info.get('symbol', '').upper()
                        name = coin_info.get('name', '')
                        price = coin_info.get('price_btc', 0)  # In BTC, convert if needed
                        market_cap_rank = coin_info.get('market_cap_rank', None)
                        
                        # Get USD price if available
                        price_usd = coin_info.get('data', {}).get('price', 0)
                        if isinstance(price_usd, str):
                            try:
                                price_usd = float(price_usd.replace('$', '').replace(',', ''))
                            except:
                                price_usd = 0
                        
                        # Analyze sentiment from name/description
                        sentiment, sentiment_score = self._analyze_text_sentiment(name)
                        
                        trending = CryptoTrendingData(
                            symbol=symbol,
                            name=name,
                            market_cap_rank=market_cap_rank,
                            price_usd=price_usd,
                            price_change_24h=0.0,  # Not in trending endpoint
                            volume_24h=0.0,  # Not in trending endpoint
                            market_cap=0.0,  # Not in trending endpoint
                            trending_score=i + 1,  # 1 = most trending
                            sentiment=sentiment,
                            sentiment_score=sentiment_score
                        )
                        trending_data.append(trending)
                        
                    except Exception as e:
                        logger.debug(f"Error parsing coin {i}: {e}")
                        continue
                
                # Cache the results
                self.coingecko_cache[cache_key] = (trending_data, time.time())
                
                logger.info(f"✅ Found {len(trending_data)} trending cryptos")
                return trending_data
                
        except Exception as e:
            logger.error(f"Error fetching trending cryptos: {e}")
            return []
    
    async def get_trending_with_sentiment(self, top_n: int = 10) -> List[Dict]:
        """
        Get trending cryptos with social sentiment analysis
        
        Args:
            top_n: Number of trending coins to analyze
            
        Returns:
            List of dicts with trending data + sentiment
        """
        trending_coins = await self.get_trending_cryptos(top_n)
        
        results = []
        for coin in trending_coins:
            try:
                # Get social sentiment for this coin
                social_sentiment = await self.analyze_crypto_sentiment(coin.symbol)
                
                result = {
                    'symbol': coin.symbol,
                    'name': coin.name,
                    'price_usd': coin.price_usd,
                    'market_cap_rank': coin.market_cap_rank,
                    'trending_score': coin.trending_score,
                    'trending_sentiment': coin.sentiment,
                    'trending_sentiment_score': coin.sentiment_score,
                    'social_sentiment': social_sentiment,
                    'combined_score': self._calculate_combined_score(
                        coin.sentiment_score,
                        social_sentiment.get('overall_sentiment_score', 0)
                    ),
                    'runner_potential': self._assess_runner_potential(
                        coin.trending_score,
                        social_sentiment,
                        coin.sentiment_score
                    )
                }
                results.append(result)
                
            except Exception as e:
                logger.debug(f"Error analyzing sentiment for {coin.symbol}: {e}")
                continue
        
        return results
    
    async def analyze_crypto_sentiment(self, symbol: str) -> Dict:
        """
        Analyze social sentiment for a specific crypto
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'SHIB')
            
        Returns:
            Dict with sentiment analysis
        """
        try:
            # Get sentiment from multiple sources
            reddit_sentiment = await self._get_reddit_sentiment(symbol)
            twitter_sentiment = await self._get_twitter_sentiment(symbol)
            
            # Combine sentiments
            all_mentions = reddit_sentiment['mentions'] + twitter_sentiment['mentions']
            
            if not all_mentions:
                return {
                    'reddit_mentions': 0,
                    'twitter_mentions': 0,
                    'overall_sentiment': 'NEUTRAL',
                    'overall_sentiment_score': 0.0,
                    'bullish_count': 0,
                    'bearish_count': 0,
                    'neutral_count': 0
                }
            
            # Calculate aggregate sentiment
            bullish_count = sum(1 for m in all_mentions if m['sentiment'] == 'BULLISH')
            bearish_count = sum(1 for m in all_mentions if m['sentiment'] == 'BEARISH')
            neutral_count = sum(1 for m in all_mentions if m['sentiment'] == 'NEUTRAL')
            
            # Overall sentiment
            if bullish_count > bearish_count * 1.5:
                overall_sentiment = 'BULLISH'
            elif bearish_count > bullish_count * 1.5:
                overall_sentiment = 'BEARISH'
            else:
                overall_sentiment = 'NEUTRAL'
            
            # Sentiment score (-1 to 1)
            total = bullish_count + bearish_count + neutral_count
            overall_score = (bullish_count - bearish_count) / total if total > 0 else 0
            
            return {
                'reddit_mentions': len(reddit_sentiment['mentions']),
                'twitter_mentions': len(twitter_sentiment['mentions']),
                'overall_sentiment': overall_sentiment,
                'overall_sentiment_score': overall_score,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'neutral_count': neutral_count,
                'mentions': all_mentions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            return {
                'reddit_mentions': 0,
                'twitter_mentions': 0,
                'overall_sentiment': 'NEUTRAL',
                'overall_sentiment_score': 0.0,
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0
            }
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict:
        """Get Reddit sentiment for crypto symbol"""
        try:
            # Import from existing social sentiment analyzer
            from services.social_sentiment_analyzer import SocialSentimentAnalyzer
            
            analyzer = SocialSentimentAnalyzer()
            
            # Use RSS feeds for fast, rate-limit-free access
            mentions = await analyzer._scrape_reddit_via_rss(symbol)
            
            return {
                'mentions': [
                    {
                        'source': m.source,
                        'text': m.text,
                        'sentiment': m.sentiment,
                        'author': m.author,
                        'timestamp': m.timestamp
                    }
                    for m in mentions
                ]
            }
            
        except Exception as e:
            logger.debug(f"Reddit sentiment error for {symbol}: {e}")
            return {'mentions': []}
    
    async def _get_twitter_sentiment(self, symbol: str) -> Dict:
        """Get Twitter sentiment for crypto symbol"""
        try:
            # Import from existing social sentiment analyzer
            from services.social_sentiment_analyzer import SocialSentimentAnalyzer
            
            analyzer = SocialSentimentAnalyzer()
            
            # Use Nitter mirrors for fast, rate-limit-free access
            mentions = await analyzer._scrape_twitter_via_nitter(symbol)
            
            return {
                'mentions': [
                    {
                        'source': m.source,
                        'text': m.text,
                        'sentiment': m.sentiment,
                        'author': m.author,
                        'timestamp': m.timestamp
                    }
                    for m in mentions
                ]
            }
            
        except Exception as e:
            logger.debug(f"Twitter sentiment error for {symbol}: {e}")
            return {'mentions': []}
    
    def _analyze_text_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of text
        
        Returns:
            (sentiment, score) where sentiment is BULLISH/BEARISH/NEUTRAL
            and score is -1.0 to 1.0
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for keyword in self.bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in self.bearish_keywords if keyword in text_lower)
        
        if bullish_count > bearish_count:
            sentiment = 'BULLISH'
            score = min(bullish_count / 10, 1.0)
        elif bearish_count > bullish_count:
            sentiment = 'BEARISH'
            score = -min(bearish_count / 10, 1.0)
        else:
            sentiment = 'NEUTRAL'
            score = 0.0
        
        return sentiment, score
    
    def _calculate_combined_score(self, trending_score: float, social_score: float) -> float:
        """
        Calculate combined score from trending + social sentiment
        
        Args:
            trending_score: CoinGecko trending sentiment (-1 to 1)
            social_score: Social sentiment score (-1 to 1)
            
        Returns:
            Combined score (-1 to 1)
        """
        return (trending_score + social_score) / 2
    
    def _assess_runner_potential(self, trending_rank: int, social_sentiment: Dict, trending_score: float) -> Dict:
        """
        Assess monster runner potential based on trending + sentiment
        
        Args:
            trending_rank: Position in trending list (1-7)
            social_sentiment: Social sentiment analysis
            trending_score: Trending sentiment score
            
        Returns:
            Dict with runner potential assessment
        """
        runner_score = 0.0
        signals = []
        
        # Trending rank (1-3 = highest potential)
        if trending_rank <= 3:
            runner_score += 30
            signals.append(f"🔥 Top {trending_rank} trending")
        elif trending_rank <= 5:
            runner_score += 20
            signals.append(f"📈 Top {trending_rank} trending")
        else:
            runner_score += 10
            signals.append(f"📊 Trending #{trending_rank}")
        
        # Social sentiment
        bullish = social_sentiment.get('bullish_count', 0)
        bearish = social_sentiment.get('bearish_count', 0)
        
        if bullish > bearish * 2:
            runner_score += 25
            signals.append(f"🚀 Strong bullish sentiment ({bullish} mentions)")
        elif bullish > bearish:
            runner_score += 15
            signals.append(f"📈 Bullish sentiment ({bullish} mentions)")
        
        # Trending sentiment score
        if trending_score > 0.5:
            runner_score += 20
            signals.append("💬 Positive trending narrative")
        elif trending_score > 0:
            runner_score += 10
            signals.append("📝 Neutral trending narrative")
        
        # Determine confidence
        if runner_score >= 60:
            confidence = 'HIGH'
        elif runner_score >= 40:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'runner_score': runner_score,
            'confidence': confidence,
            'signals': signals,
            'recommendation': 'MONITOR' if confidence == 'HIGH' else 'WATCH'
        }
    
    def get_analyzer_stats(self) -> Dict:
        """Get analyzer statistics"""
        return {
            'data_sources': ['CoinGecko Trending', 'Reddit RSS', 'Twitter (Nitter)', 'StockTwits'],
            'rate_limit': '10 calls/min (CoinGecko)',
            'cache_ttl': f'{self.cache_ttl}s',
            'sentiment_keywords': {
                'bullish': len(self.bullish_keywords),
                'bearish': len(self.bearish_keywords)
            },
            'focus': 'Trending/emerging coins for monster runner detection'
        }

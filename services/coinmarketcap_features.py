"""
CoinMarketCap Features Integration
Integrates trending, sentiment, and new cryptocurrencies from CoinMarketCap

References:
- Trending: https://coinmarketcap.com/trending-cryptocurrencies/
- Sentiment: https://coinmarketcap.com/sentiment/
- New Coins: https://coinmarketcap.com/new/
"""

import os
import asyncio
import httpx
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TrendingCrypto:
    """Trending cryptocurrency data from CoinMarketCap"""
    symbol: str
    name: str
    price: float
    change_1h: float
    change_24h: float
    market_cap: float
    volume_24h: float
    rank: int
    cmc_id: int


@dataclass
class CryptoSentiment:
    """Cryptocurrency sentiment data from CoinMarketCap"""
    symbol: str
    name: str
    sentiment_score: float  # -100 to 100
    bullish_percent: float  # 0-100
    bearish_percent: float  # 0-100
    neutral_percent: float  # 0-100
    social_volume: int
    cmc_id: int


@dataclass
class NewCrypto:
    """New cryptocurrency listing from CoinMarketCap"""
    symbol: str
    name: str
    price: float
    change_1h: float
    change_24h: float
    market_cap: float
    volume_24h: float
    blockchain: str
    added_date: str
    cmc_id: int


class CoinMarketCapFeatures:
    """Integrates CoinMarketCap trending, sentiment, and new coins features"""
    
    def __init__(self):
        """Initialize CoinMarketCap features"""
        self.api_base = "https://pro-api.coinmarketcap.com/v1"
        self.api_key = os.getenv('COINMARKETCAP_API_KEY')
        self.last_call_time = 0
        self.rate_limit = 260.0  # Free tier: 333 calls/day = 260 seconds between calls
        
        if not self.api_key:
            logger.warning("⚠️ CoinMarketCap API key not found. Features will be limited.")
        else:
            logger.info("✅ CoinMarketCap Features initialized")
            logger.info("   • Trending cryptocurrencies")
            logger.info("   • Sentiment analysis")
            logger.info("   • New coin listings")
    
    async def _rate_limit_check(self):
        """Check and enforce rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        
        if time_since_last < self.rate_limit:
            wait_time = self.rate_limit - time_since_last
            logger.debug(f"⏱️ CoinMarketCap rate limit: waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
    
    async def get_trending_cryptos(self, limit: int = 25) -> List[TrendingCrypto]:
        """
        Get trending cryptocurrencies from CoinMarketCap
        
        Note: CoinMarketCap API doesn't have a direct "trending" endpoint,
        but we can use the listings endpoint sorted by volume/percent_change_24h
        which represents what's trending.
        
        Args:
            limit: Number of trending cryptos to return
            
        Returns:
            List of trending cryptocurrencies
        """
        if not self.api_key:
            logger.warning("⚠️ CoinMarketCap API key required for trending cryptos")
            return []
        
        await self._rate_limit_check()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get listings sorted by percent change (trending = high change)
                response = await client.get(
                    f"{self.api_base}/cryptocurrency/listings/latest",
                    params={
                        'start': 1,
                        'limit': limit * 2,  # Get more to filter
                        'convert': 'USD',
                        'sort': 'percent_change_24h',  # Sort by 24h change (trending indicator)
                        'sort_dir': 'desc'  # Highest change first
                    },
                    headers={
                        'X-CMC_PRO_API_KEY': self.api_key,
                        'Accept': 'application/json'
                    }
                )
                
                self.last_call_time = time.time()
                
                if response.status_code == 429:
                    logger.warning("⚠️ CoinMarketCap rate limited (429)")
                    return []
                
                if response.status_code != 200:
                    logger.error(f"❌ CoinMarketCap API error: {response.status_code}")
                    return []
                
                data = response.json()
                coins = data.get('data', [])
                
                # Filter for trending (high volume + significant change)
                trending = []
                for coin in coins[:limit]:
                    try:
                        quote = coin.get('quote', {}).get('USD', {})
                        
                        # Only include if significant change and volume
                        change_24h = quote.get('percent_change_24h', 0)
                        volume_24h = quote.get('volume_24h', 0)
                        
                        if abs(change_24h) > 5.0 and volume_24h > 100000:  # 5%+ change, $100k+ volume
                            trending.append(TrendingCrypto(
                                symbol=coin.get('symbol', ''),
                                name=coin.get('name', ''),
                                price=quote.get('price', 0),
                                change_1h=quote.get('percent_change_1h', 0),
                                change_24h=change_24h,
                                market_cap=quote.get('market_cap', 0),
                                volume_24h=volume_24h,
                                rank=coin.get('cmc_rank', 0),
                                cmc_id=coin.get('id', 0)
                            ))
                    except Exception as e:
                        logger.debug(f"Error processing trending crypto: {e}")
                        continue
                
                logger.info(f"✅ Found {len(trending)} trending cryptos from CoinMarketCap")
                return trending
                
        except Exception as e:
            logger.error(f"❌ Error fetching trending cryptos: {e}")
            return []
    
    async def get_crypto_sentiment(self, symbol: str) -> Optional[CryptoSentiment]:
        """
        Get sentiment analysis for a specific cryptocurrency
        
        Note: CoinMarketCap API doesn't have a direct sentiment endpoint,
        but we can use their community data which includes social metrics.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            
        Returns:
            CryptoSentiment object or None
        """
        if not self.api_key:
            logger.warning("⚠️ CoinMarketCap API key required for sentiment analysis")
            return None
        
        await self._rate_limit_check()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # First, get the coin ID from symbol
                response = await client.get(
                    f"{self.api_base}/cryptocurrency/map",
                    params={
                        'symbol': symbol.upper()
                    },
                    headers={
                        'X-CMC_PRO_API_KEY': self.api_key,
                        'Accept': 'application/json'
                    }
                )
                
                if response.status_code != 200:
                    logger.debug(f"Could not find coin ID for {symbol}")
                    return None
                
                data = response.json()
                coins = data.get('data', [])
                if not coins:
                    return None
                
                coin_id = coins[0].get('id')
                
                # Get market data with community metrics
                await self._rate_limit_check()
                response = await client.get(
                    f"{self.api_base}/cryptocurrency/quotes/latest",
                    params={
                        'id': coin_id,
                        'convert': 'USD'
                    },
                    headers={
                        'X-CMC_PRO_API_KEY': self.api_key,
                        'Accept': 'application/json'
                    }
                )
                
                self.last_call_time = time.time()
                
                if response.status_code != 200:
                    return None
                
                data = response.json()
                coin_data = data.get('data', {}).get(str(coin_id), {})
                quote = coin_data.get('quote', {}).get('USD', {})
                
                # Calculate sentiment from price change and volume
                # (CoinMarketCap doesn't provide direct sentiment scores in free tier)
                change_24h = quote.get('percent_change_24h', 0)
                
                # Simple sentiment calculation based on price action
                if change_24h > 5:
                    sentiment_score = min(100, 50 + (change_24h * 2))
                    bullish = min(100, 60 + (change_24h * 2))
                    bearish = max(0, 20 - (change_24h))
                    neutral = 100 - bullish - bearish
                elif change_24h < -5:
                    sentiment_score = max(-100, -50 + (change_24h * 2))
                    bearish = min(100, 60 - (change_24h * 2))
                    bullish = max(0, 20 + (change_24h))
                    neutral = 100 - bullish - bearish
                else:
                    sentiment_score = change_24h * 5  # Scale small changes
                    bullish = 40
                    bearish = 30
                    neutral = 30
                
                return CryptoSentiment(
                    symbol=symbol.upper(),
                    name=coin_data.get('name', ''),
                    sentiment_score=sentiment_score,
                    bullish_percent=bullish,
                    bearish_percent=bearish,
                    neutral_percent=neutral,
                    social_volume=int(quote.get('volume_24h', 0) / 1000000),  # Approximate
                    cmc_id=coin_id
                )
                
        except Exception as e:
            logger.debug(f"Error fetching sentiment for {symbol}: {e}")
            return None
    
    async def get_new_cryptos(self, limit: int = 25, days: int = 30) -> List[NewCrypto]:
        """
        Get newly added cryptocurrencies from CoinMarketCap
        
        Args:
            limit: Number of new cryptos to return
            days: Number of days to look back (max 30 for free tier)
            
        Returns:
            List of new cryptocurrencies
        """
        if not self.api_key:
            logger.warning("⚠️ CoinMarketCap API key required for new cryptos")
            return []
        
        await self._rate_limit_check()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Get all listings and filter by recently added
                # Note: CoinMarketCap API doesn't have a direct "new" endpoint,
                # so we'll use listings sorted by date added (if available)
                # or filter by low market cap (new coins typically start small)
                
                response = await client.get(
                    f"{self.api_base}/cryptocurrency/listings/latest",
                    params={
                        'start': 1,
                        'limit': 500,  # Get more to find new ones
                        'convert': 'USD',
                        'sort': 'market_cap',  # New coins often have low market cap
                        'sort_dir': 'asc'  # Lowest first (likely newer)
                    },
                    headers={
                        'X-CMC_PRO_API_KEY': self.api_key,
                        'Accept': 'application/json'
                    }
                )
                
                self.last_call_time = time.time()
                
                if response.status_code != 200:
                    logger.error(f"❌ CoinMarketCap API error: {response.status_code}")
                    return []
                
                data = response.json()
                coins = data.get('data', [])
                
                new_coins = []
                for coin in coins[:limit * 3]:  # Check more to find truly new ones
                    try:
                        quote = coin.get('quote', {}).get('USD', {})
                        market_cap = quote.get('market_cap', 0)
                        
                        # Filter for new coins (low market cap, recent activity)
                        # This is a heuristic since API doesn't provide date_added
                        if market_cap < 10000000 and quote.get('volume_24h', 0) > 10000:
                            # Get platform/blockchain info
                            platform = coin.get('platform', {})
                            blockchain = platform.get('name', 'Unknown') if platform else 'Unknown'
                            
                            new_coins.append(NewCrypto(
                                symbol=coin.get('symbol', ''),
                                name=coin.get('name', ''),
                                price=quote.get('price', 0),
                                change_1h=quote.get('percent_change_1h', 0),
                                change_24h=quote.get('percent_change_24h', 0),
                                market_cap=market_cap,
                                volume_24h=quote.get('volume_24h', 0),
                                blockchain=blockchain,
                                added_date=datetime.now().strftime('%Y-%m-%d'),  # Approximate
                                cmc_id=coin.get('id', 0)
                            ))
                            
                            if len(new_coins) >= limit:
                                break
                                
                    except Exception as e:
                        logger.debug(f"Error processing new crypto: {e}")
                        continue
                
                logger.info(f"✅ Found {len(new_coins)} new cryptos from CoinMarketCap")
                return new_coins
                
        except Exception as e:
            logger.error(f"❌ Error fetching new cryptos: {e}")
            return []


# Convenience functions
async def get_trending_cryptos(limit: int = 25) -> List[TrendingCrypto]:
    """Get trending cryptos from CoinMarketCap"""
    cmc = CoinMarketCapFeatures()
    return await cmc.get_trending_cryptos(limit)


async def get_crypto_sentiment(symbol: str) -> Optional[CryptoSentiment]:
    """Get sentiment for a crypto from CoinMarketCap"""
    cmc = CoinMarketCapFeatures()
    return await cmc.get_crypto_sentiment(symbol)


async def get_new_cryptos(limit: int = 25) -> List[NewCrypto]:
    """Get new cryptos from CoinMarketCap"""
    cmc = CoinMarketCapFeatures()
    return await cmc.get_new_cryptos(limit)


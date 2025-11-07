"""
Multi-Source Crypto Data Aggregator

Aggregates crypto market data from multiple sources:
- CoinGecko (comprehensive, free tier)
- CoinMarketCap (if API key available)

Does NOT use Binance (as per user requirement).

Fetches ALL available coins without limiting results.
"""

import os
import asyncio
import httpx
import time
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AggregatedCryptoData:
    """Aggregated crypto data from multiple sources"""
    symbol: str
    name: str
    price_usd: float
    market_cap: float
    volume_24h: float
    change_24h: float
    change_7d: float
    market_cap_rank: Optional[int]
    sources: List[str]  # Which sources provided this data
    coingecko_id: Optional[str] = None
    coinmarketcap_id: Optional[str] = None


class CryptoDataAggregator:
    """
    Aggregates crypto market data from multiple sources.
    Fetches ALL available coins without artificial limits.
    """
    
    def __init__(self):
        """Initialize the aggregator"""
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.coinmarketcap_api = "https://pro-api.coinmarketcap.com/v1"
        self.coinmarketcap_api_key = os.getenv('COINMARKETCAP_API_KEY')
        
        self.last_coingecko_call = 0
        self.last_coinmarketcap_call = 0
        
        # Rate limits
        self.coingecko_rate_limit = 7.0  # 10 calls/min = 1 call every 6s, use 7s to be safe
        # CoinMarketCap free tier: 333 calls/day = ~1 call every 4.3 minutes = 258 seconds
        # Use 260 seconds (4.33 minutes) to be safe and avoid rate limits
        self.coinmarketcap_rate_limit = 260.0  # Free tier: 333 calls/day
        
        logger.info("🔗 Crypto Data Aggregator initialized")
        logger.info(f"   • CoinGecko: Enabled (free tier)")
        logger.info(f"   • CoinMarketCap: {'Enabled' if self.coinmarketcap_api_key else 'Disabled (no API key)'}")
    
    async def fetch_all_coins(
        self,
        max_price: Optional[float] = None,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_volume_24h: Optional[float] = None,
        max_coins: Optional[int] = None,
        max_pages: Optional[int] = None
    ) -> List[AggregatedCryptoData]:
        """
        Fetch coins from all available sources with smart limits.
        
        Args:
            max_price: Optional maximum price filter
            min_market_cap: Optional minimum market cap filter
            max_market_cap: Optional maximum market cap filter
            min_volume_24h: Optional minimum 24h volume filter
            max_coins: Optional maximum number of coins to fetch (stops early when reached)
            max_pages: Optional maximum pages to fetch (default: 20 for sub-penny, 200 for general)
            
        Returns:
            List of aggregated crypto data
        """
        logger.info("🔍 Fetching coins from all available sources...")
        if max_coins:
            logger.info(f"   • Max coins: {max_coins}")
        if max_pages:
            logger.info(f"   • Max pages: {max_pages}")
        
        all_coins = {}
        
        # Fetch from CoinGecko (always available)
        try:
            coingecko_coins = await self._fetch_coingecko_all(
                max_price, min_market_cap, max_market_cap, min_volume_24h,
                max_coins=max_coins, max_pages=max_pages
            )
            logger.info(f"✅ CoinGecko: Found {len(coingecko_coins)} coins")
            
            # Merge into aggregated data
            for coin in coingecko_coins:
                symbol = coin['symbol'].upper()
                if symbol not in all_coins:
                    all_coins[symbol] = {
                        'symbol': symbol,
                        'name': coin.get('name', ''),
                        'price_usd': coin.get('current_price', 0),
                        'market_cap': coin.get('market_cap', 0) or 0,
                        'volume_24h': coin.get('total_volume', 0) or 0,
                        'change_24h': coin.get('price_change_percentage_24h', 0) or 0,
                        'change_7d': coin.get('price_change_percentage_7d_in_currency', 0) or 0,
                        'market_cap_rank': coin.get('market_cap_rank'),
                        'sources': ['coingecko'],
                        'coingecko_id': coin.get('id', ''),
                        'coinmarketcap_id': None
                    }
                else:
                    # Update with CoinGecko data if better
                    existing = all_coins[symbol]
                    if coin.get('market_cap', 0) > existing.get('market_cap', 0):
                        existing.update({
                            'price_usd': coin.get('current_price', existing['price_usd']),
                            'market_cap': coin.get('market_cap', existing['market_cap']),
                            'volume_24h': coin.get('total_volume', existing['volume_24h']),
                            'change_24h': coin.get('price_change_percentage_24h', existing['change_24h']),
                            'change_7d': coin.get('price_change_percentage_7d_in_currency', existing['change_7d']),
                            'market_cap_rank': coin.get('market_cap_rank', existing['market_cap_rank']),
                        })
                    existing['sources'].append('coingecko')
        except Exception as e:
            logger.error(f"❌ CoinGecko fetch failed: {e}")
        
        # Fetch from CoinMarketCap if API key available (only if we need more coins)
        if self.coinmarketcap_api_key and (not max_coins or len(all_coins) < max_coins):
            try:
                cmc_coins = await self._fetch_coinmarketcap_all(
                    max_price, min_market_cap, max_market_cap, min_volume_24h,
                    max_coins=max_coins
                )
                logger.info(f"✅ CoinMarketCap: Found {len(cmc_coins)} coins")
                
                # Merge into aggregated data
                for coin in cmc_coins:
                    # CoinMarketCap uses 'symbol' field
                    symbol = coin.get('symbol', '').upper()
                    if not symbol:
                        continue
                    if symbol not in all_coins:
                        all_coins[symbol] = {
                            'symbol': symbol,
                            'name': coin.get('name', ''),
                            'price_usd': coin.get('quote', {}).get('USD', {}).get('price', 0),
                            'market_cap': coin.get('quote', {}).get('USD', {}).get('market_cap', 0) or 0,
                            'volume_24h': coin.get('quote', {}).get('USD', {}).get('volume_24h', 0) or 0,
                            'change_24h': coin.get('quote', {}).get('USD', {}).get('percent_change_24h', 0) or 0,
                            'change_7d': coin.get('quote', {}).get('USD', {}).get('percent_change_7d', 0) or 0,
                            'market_cap_rank': coin.get('cmc_rank'),
                            'sources': ['coinmarketcap'],
                            'coingecko_id': None,
                            'coinmarketcap_id': str(coin.get('id', ''))
                        }
                    else:
                        # Update with CoinMarketCap data
                        existing = all_coins[symbol]
                        existing['sources'].append('coinmarketcap')
                        # Use CoinMarketCap data if more recent or better
                        cmc_price = coin.get('quote', {}).get('USD', {}).get('price', 0)
                        if cmc_price > 0:
                            existing['price_usd'] = cmc_price
                        cmc_mcap = coin.get('quote', {}).get('USD', {}).get('market_cap', 0)
                        if cmc_mcap > existing.get('market_cap', 0):
                            existing['market_cap'] = cmc_mcap
                            existing['market_cap_rank'] = coin.get('cmc_rank', existing['market_cap_rank'])
            except Exception as e:
                logger.error(f"❌ CoinMarketCap fetch failed: {e}")
        
        # Convert to AggregatedCryptoData objects
        aggregated = []
        for symbol, data in all_coins.items():
            aggregated.append(AggregatedCryptoData(
                symbol=data['symbol'],
                name=data['name'],
                price_usd=data['price_usd'],
                market_cap=data['market_cap'],
                volume_24h=data['volume_24h'],
                change_24h=data['change_24h'],
                change_7d=data['change_7d'],
                market_cap_rank=data['market_cap_rank'],
                sources=data['sources'],
                coingecko_id=data.get('coingecko_id'),
                coinmarketcap_id=data.get('coinmarketcap_id')
            ))
        
        logger.info(f"✅ Aggregated {len(aggregated)} unique coins from all sources")
        return aggregated
    
    async def _fetch_coingecko_all(
        self,
        max_price: Optional[float] = None,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_volume_24h: Optional[float] = None,
        max_coins: Optional[int] = None,
        max_pages: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch coins from CoinGecko with smart limits.
        Stops early when max_coins is reached or max_pages is hit.
        """
        all_coins = []
        page = 1
        per_page = 250  # Max per page
        consecutive_empty = 0
        # Default max_pages: 20 for sub-penny (faster), 200 for general
        if max_pages is None:
            max_pages = 20 if max_price and max_price <= 0.01 else 200
        
        while True:
            # Rate limiting
            elapsed = time.time() - self.last_coingecko_call
            if elapsed < self.coingecko_rate_limit:
                await asyncio.sleep(self.coingecko_rate_limit - elapsed)
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{self.coingecko_api}/coins/markets",
                        params={
                            'vs_currency': 'usd',
                            'order': 'market_cap_desc',
                            'per_page': per_page,
                            'page': page,
                            'sparkline': False,
                            'locale': 'en'
                        },
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    
                    self.last_coingecko_call = time.time()
                    
                    if response.status_code == 429:
                        logger.warning(f"Rate limited on page {page}, waiting 30s...")
                        await asyncio.sleep(30)
                        continue
                    
                    if response.status_code != 200:
                        logger.error(f"CoinGecko API error: {response.status_code}")
                        break
                    
                    data = response.json()
                    
                    if not data:
                        consecutive_empty += 1
                        if consecutive_empty >= 3:
                            logger.debug(f"No more data after {consecutive_empty} empty pages")
                            break
                        page += 1
                        continue
                    
                    consecutive_empty = 0
                    
                    # Apply filters
                    for coin in data:
                        price = coin.get('current_price', 0)
                        market_cap = coin.get('market_cap', 0) or 0
                        volume = coin.get('total_volume', 0) or 0
                        
                        if max_price and price > max_price:
                            continue
                        if min_market_cap and market_cap < min_market_cap:
                            continue
                        if max_market_cap and market_cap > max_market_cap:
                            continue
                        if min_volume_24h and volume < min_volume_24h:
                            continue
                        
                        all_coins.append(coin)
                    
                    logger.debug(f"Page {page}: Found {len(data)} coins (total: {len(all_coins)})")
                    
                    # Stop early if we have enough coins
                    if max_coins and len(all_coins) >= max_coins:
                        logger.info(f"✅ Reached max_coins limit ({max_coins}), collected {len(all_coins)} coins")
                        break
                    
                    # Safety limit: don't fetch more than max_pages
                    if page >= max_pages:
                        logger.info(f"Reached page limit ({max_pages} pages), collected {len(all_coins)} coins")
                        break
                    
                    page += 1
                    
            except Exception as e:
                logger.error(f"Error fetching CoinGecko page {page}: {e}")
                break
        
        return all_coins
    
    async def _fetch_coinmarketcap_all(
        self,
        max_price: Optional[float] = None,
        min_market_cap: Optional[float] = None,
        max_market_cap: Optional[float] = None,
        min_volume_24h: Optional[float] = None,
        max_coins: Optional[int] = None
    ) -> List[Dict]:
        """
        Fetch coins from CoinMarketCap with smart limits.
        Stops early when max_coins is reached.
        """
        if not self.coinmarketcap_api_key:
            return []
        
        all_coins = []
        start = 1
        limit = 5000  # Max per request
        
        while True:
            # Rate limiting (free tier: 333 calls/day = ~1 call every 4.3 minutes)
            elapsed = time.time() - self.last_coinmarketcap_call
            if elapsed < self.coinmarketcap_rate_limit:
                wait_time = self.coinmarketcap_rate_limit - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s before next CoinMarketCap call...")
                await asyncio.sleep(wait_time)
            
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(
                        f"{self.coinmarketcap_api}/cryptocurrency/listings/latest",
                        params={
                            'start': start,
                            'limit': limit,
                            'convert': 'USD'
                        },
                        headers={
                            'X-CMC_PRO_API_KEY': self.coinmarketcap_api_key,
                            'Accept': 'application/json'
                        }
                    )
                    
                    self.last_coinmarketcap_call = time.time()
                    
                    if response.status_code == 429:
                        logger.warning("CoinMarketCap rate limited (429), waiting 5 minutes...")
                        await asyncio.sleep(300)
                        continue
                    
                    if response.status_code == 401:
                        logger.error("CoinMarketCap API authentication failed - check your API key")
                        break
                    
                    if response.status_code != 200:
                        error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                        error_msg = error_data.get('status', {}).get('error_message', f'HTTP {response.status_code}')
                        logger.error(f"CoinMarketCap API error ({response.status_code}): {error_msg}")
                        break
                    
                    data = response.json()
                    coins = data.get('data', [])
                    
                    if not coins:
                        break
                    
                    # Apply filters
                    for coin in coins:
                        quote = coin.get('quote', {}).get('USD', {})
                        price = quote.get('price', 0)
                        market_cap = quote.get('market_cap', 0) or 0
                        volume = quote.get('volume_24h', 0) or 0
                        
                        if max_price and price > max_price:
                            continue
                        if min_market_cap and market_cap < min_market_cap:
                            continue
                        if max_market_cap and market_cap > max_market_cap:
                            continue
                        if min_volume_24h and volume < min_volume_24h:
                            continue
                        
                        all_coins.append(coin)
                    
                    logger.debug(f"CoinMarketCap: Fetched {len(coins)} coins (total: {len(all_coins)})")
                    
                    # Stop early if we have enough coins
                    if max_coins and len(all_coins) >= max_coins:
                        logger.info(f"✅ Reached max_coins limit ({max_coins}), collected {len(all_coins)} coins")
                        break
                    
                    # Check if there are more results
                    if len(coins) < limit:
                        break
                    
                    start += limit
                    
            except Exception as e:
                logger.error(f"Error fetching CoinMarketCap: {e}")
                break
        
        return all_coins


"""
Sub-Penny Crypto Discovery - Find ultra-low coins (0.00000+) with monster runner potential

Uses multi-source aggregator (CoinGecko, CoinMarketCap) to discover all coins under $0.01,
combines with sentiment analysis and technical scoring.

No state restrictions - works globally!
"""

import asyncio
import httpx
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv
from services.crypto_data_aggregator import CryptoDataAggregator, AggregatedCryptoData

load_dotenv()


@dataclass
class SubPennyCoin:
    """Ultra-low priced coin discovery"""
    symbol: str
    name: str
    price_usd: float
    price_decimals: int  # Number of decimal places
    market_cap: float
    market_cap_rank: Optional[int]
    volume_24h: float
    change_24h: float
    change_7d: float
    market_cap_change_24h: float
    circulating_supply: float
    total_supply: float
    ath: float  # All-time high
    atl: float  # All-time low
    ath_change_percentage: float
    runner_potential_score: float
    discovery_reason: str  # Why this coin was flagged


class SubPennyDiscovery:
    """
    Discovers ultra-low priced cryptocurrencies with monster runner potential
    using CoinGecko's comprehensive markets data.
    """
    
    def __init__(self):
        """Initialize the sub-penny discovery engine"""
        self.aggregator = CryptoDataAggregator()
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        logger.info("🔬 Sub-Penny Discovery Engine initialized")
        logger.info("   • Data sources: Multi-source aggregator (CoinGecko, CoinMarketCap)")
        logger.info("   • Coverage: Up to 3,000 coins under $0.01 (optimized for valid Kraken pairs)")
        logger.info("   • No state restrictions - works globally")
    
    async def discover_sub_penny_runners(
        self,
        max_price: float = 0.01,
        min_market_cap: float = 0,
        max_market_cap: float = 1_000_000,
        top_n: int = 20,
        sort_by: str = "runner_potential"  # or 'volume', 'market_cap', 'change_24h'
    ) -> List[SubPennyCoin]:
        """
        Discover sub-penny coins with monster runner potential
        
        Args:
            max_price: Maximum price filter (default: $0.01)
            min_market_cap: Minimum market cap filter
            max_market_cap: Maximum market cap filter
            top_n: Number of results to return
            sort_by: Sort criteria
            
        Returns:
            List of SubPennyCoin objects
        """
        try:
            logger.info(f"🔬 Discovering sub-penny runners (max ${max_price})...")
            
            # Fetch coins from multi-source aggregator with smart limits
            # For sub-penny discovery, we need MANY MORE coins because most won't be valid Kraken pairs
            # Most sub-penny coins are NOT on Kraken, so we need to fetch 5000-10000 to get 15-20 valid pairs
            # This prevents the search from going on forever while ensuring enough valid results
            max_coins_to_fetch = 10000  # Increased significantly to get more valid Kraken pairs
            max_pages_to_fetch = 40  # Stop after 40 pages (10,000 coins max from CoinGecko)
            
            aggregated_data = await self.aggregator.fetch_all_coins(
                max_price=max_price,
                min_market_cap=min_market_cap,
                max_market_cap=max_market_cap,
                max_coins=max_coins_to_fetch,
                max_pages=max_pages_to_fetch
            )
            
            if not aggregated_data:
                logger.warning("No coins found under specified price")
                return []
            
            logger.info(f"📊 Analyzing {len(aggregated_data)} coins under ${max_price}...")
            logger.info(f"   • Market cap filter: ${min_market_cap:,.0f} - ${max_market_cap:,.0f}")
            
            # Convert aggregated data to SubPennyCoin objects
            all_coins = []
            for agg in aggregated_data:
                try:
                    coin = SubPennyCoin(
                        symbol=agg.symbol,
                        name=agg.name,
                        price_usd=agg.price_usd,
                        price_decimals=self._count_decimals(agg.price_usd),
                        market_cap=agg.market_cap,
                        market_cap_rank=agg.market_cap_rank,
                        volume_24h=agg.volume_24h,
                        change_24h=agg.change_24h,
                        change_7d=agg.change_7d,
                        market_cap_change_24h=0.0,  # Not available from aggregator
                        circulating_supply=0.0,  # Not available from aggregator
                        total_supply=0.0,  # Not available from aggregator
                        ath=0.0,  # Not available from aggregator
                        atl=0.0,  # Not available from aggregator
                        ath_change_percentage=0.0,  # Not available from aggregator
                        runner_potential_score=0.0,
                        discovery_reason=""
                    )
                    all_coins.append(coin)
                except Exception as e:
                    logger.debug(f"Error converting {agg.symbol}: {e}")
                    continue
            
            # Score each coin for runner potential
            runners = []
            filtered_by_market_cap = 0
            for coin in all_coins:
                try:
                    # Filter by market cap
                    if coin.market_cap < min_market_cap or coin.market_cap > max_market_cap:
                        filtered_by_market_cap += 1
                        continue
                    
                    # Score runner potential
                    runner_score, reason = self._score_runner_potential(coin)
                    coin.runner_potential_score = runner_score
                    coin.discovery_reason = reason
                    
                    runners.append(coin)
                    
                except Exception as e:
                    logger.debug(f"Error scoring {coin.symbol}: {e}")
                    continue
            
            if filtered_by_market_cap > 0:
                logger.info(f"   • Filtered out {filtered_by_market_cap} coins by market cap filter")
            
            # Sort by criteria
            if sort_by == "runner_potential":
                runners.sort(key=lambda x: x.runner_potential_score, reverse=True)
            elif sort_by == "volume":
                runners.sort(key=lambda x: x.volume_24h, reverse=True)
            elif sort_by == "market_cap":
                runners.sort(key=lambda x: x.market_cap, reverse=True)
            elif sort_by == "change_24h":
                runners.sort(key=lambda x: abs(x.change_24h), reverse=True)
            
            logger.info(f"✅ Found {len(runners)} potential runners")
            return runners[:top_n]
            
        except Exception as e:
            logger.error(f"Error discovering sub-penny runners: {e}")
            return []
    
    async def _fetch_all_coins_legacy(self, max_price: float) -> List[SubPennyCoin]:
        """
        Fetch all coins under max_price from CoinGecko
        
        Uses pagination to get all coins efficiently
        """
        try:
            coins = []
            page = 1
            per_page = 250  # Max per page
            low_yield_pages = 0  # Track consecutive pages with few sub-penny coins
            
            while True:
                logger.debug(f"Fetching page {page}...")
                
                # Rate limiting: 10 calls/min = 1 call every 6 seconds (use 7s to be safe)
                elapsed = time.time() - self.last_api_call
                if elapsed < 7:
                    await asyncio.sleep(7 - elapsed)
                
                # Retry logic for rate limits - only retry once (initial attempt + 1 retry)
                max_retries = 2
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        async with httpx.AsyncClient(timeout=15.0) as client:
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
                            
                            self.last_api_call = time.time()
                            
                            # Handle rate limiting - wait once and if still rate limited, give up
                            if response.status_code == 429:
                                retry_count += 1
                                if retry_count >= max_retries:
                                    logger.warning(f"Rate limited (429) after {retry_count} attempts, giving up on this page")
                                    return coins  # Return what we have so far
                                wait_time = 30  # Single wait of 30s
                                logger.warning(f"Rate limited (429), waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                                await asyncio.sleep(wait_time)
                                continue
                            
                            if response.status_code != 200:
                                logger.error(f"CoinGecko API error: {response.status_code}")
                                return coins  # Return what we have so far
                            
                            # Success - break retry loop
                            break
                            
                    except Exception as e:
                        logger.error(f"Request error: {e}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error("Max retries reached, returning coins collected so far")
                            return coins
                        await asyncio.sleep(10)
                
                # If we exhausted retries, return what we have
                if retry_count >= max_retries:
                    logger.warning("Giving up after max retries, returning collected coins")
                    return coins
                
                # Parse response data
                data = response.json()
                
                if not data:
                    logger.debug("No more coins to fetch")
                    break
                
                # Process coins on this page
                # Note: CoinGecko returns coins sorted by market cap, NOT price
                # So we need to check each coin individually, not stop at first expensive one
                page_sub_penny_count = 0
                for coin_data in data:
                    try:
                        price = coin_data.get('current_price', 0)
                        
                        # Skip coins with no price or above threshold
                        if not price or price <= 0 or price > max_price:
                            continue
                        
                        # Found a sub-penny coin!
                        page_sub_penny_count += 1
                        
                        coin = SubPennyCoin(
                            symbol=coin_data.get('symbol', '').upper(),
                            name=coin_data.get('name', ''),
                            price_usd=price,
                            price_decimals=self._count_decimals(price),
                            market_cap=coin_data.get('market_cap', 0) or 0,
                            market_cap_rank=coin_data.get('market_cap_rank'),
                            volume_24h=coin_data.get('total_volume', 0) or 0,
                            change_24h=coin_data.get('price_change_percentage_24h', 0) or 0,
                            change_7d=coin_data.get('price_change_percentage_7d_in_currency', 0) or 0,
                            market_cap_change_24h=coin_data.get('market_cap_change_percentage_24h', 0) or 0,
                            circulating_supply=coin_data.get('circulating_supply', 0) or 0,
                            total_supply=coin_data.get('total_supply', 0) or 0,
                            ath=coin_data.get('ath', 0) or 0,
                            atl=coin_data.get('atl', 0) or 0,
                            ath_change_percentage=coin_data.get('ath_change_percentage', 0) or 0,
                            runner_potential_score=0.0,
                            discovery_reason=""
                        )
                        coins.append(coin)
                    
                    except Exception as e:
                        logger.debug(f"Error parsing coin: {e}")
                        continue
                
                logger.debug(f"Page {page}: Found {page_sub_penny_count} sub-penny coins (total: {len(coins)})")
                
                # If the page has no sub-penny coins, just continue to the next one
                if page_sub_penny_count == 0 and page > 1:
                    logger.debug(f"Page {page} has no sub-penny coins, continuing...")
                
                if len(coins) >= 5000:
                    logger.info(f"Collected {len(coins)} coins - sufficient for analysis")
                    break
                
                page += 1
                
                # Safety limit: don't fetch more than 50 pages
                if page > 50:
                    logger.warning("Reached page limit (50 pages)")
                    break
            
            logger.info(f"📊 Fetched {len(coins)} coins under ${max_price}")
            return coins
            
        except Exception as e:
            logger.error(f"Error fetching coins: {e}")
            return []
    
    def _score_runner_potential(self, coin: SubPennyCoin) -> Tuple[float, str]:
        """
        Score a coin's monster runner potential
        
        Returns:
            (score, reason) tuple
        """
        score = 0.0
        reasons = []
        
        # 1. Price recovery potential (ATH vs current)
        if coin.ath > 0 and coin.price_usd > 0:
            recovery_potential = (coin.ath - coin.price_usd) / coin.price_usd * 100
            if recovery_potential > 1000:
                score += 30
                reasons.append(f"🚀 Extreme recovery potential ({recovery_potential:.0f}%)")
            elif recovery_potential > 500:
                score += 25
                reasons.append(f"🔥 High recovery potential ({recovery_potential:.0f}%)")
            elif recovery_potential > 100:
                score += 15
                reasons.append(f"📈 Recovery potential ({recovery_potential:.0f}%)")
        
        # 2. Recent momentum (24h change)
        if abs(coin.change_24h) > 20:
            score += 25
            reasons.append(f"🚀 EXTREME 24h move ({coin.change_24h:+.1f}%)")
        elif abs(coin.change_24h) > 10:
            score += 20
            reasons.append(f"🔥 Strong 24h move ({coin.change_24h:+.1f}%)")
        elif abs(coin.change_24h) > 5:
            score += 10
            reasons.append(f"📈 Positive momentum ({coin.change_24h:+.1f}%)")
        
        # 3. Volume activity
        if coin.volume_24h > 0 and coin.market_cap > 0:
            volume_ratio = coin.volume_24h / coin.market_cap if coin.market_cap > 0 else 0
            if volume_ratio > 2.0:
                score += 20
                reasons.append(f"💥 High volume activity ({volume_ratio:.1f}x market cap)")
            elif volume_ratio > 1.0:
                score += 15
                reasons.append(f"📊 Good volume ({volume_ratio:.1f}x market cap)")
        
        # 4. Low market cap (high upside potential)
        if coin.market_cap > 0:
            if coin.market_cap < 100_000:
                score += 25
                reasons.append(f"💎 Micro-cap (${coin.market_cap:,.0f})")
            elif coin.market_cap < 1_000_000:
                score += 20
                reasons.append(f"💎 Low market cap (${coin.market_cap:,.0f})")
            elif coin.market_cap < 10_000_000:
                score += 10
                reasons.append(f"📈 Small market cap (${coin.market_cap:,.0f})")
        
        # 5. Price decimals (ultra-low = more room to run)
        if coin.price_decimals >= 8:
            score += 15
            reasons.append(f"🔬 Ultra-low price ({coin.price_decimals} decimals)")
        elif coin.price_decimals >= 6:
            score += 10
            reasons.append(f"🔬 Sub-penny ({coin.price_decimals} decimals)")
        
        # 6. Supply dynamics
        if coin.circulating_supply > 0 and coin.total_supply > 0:
            supply_ratio = coin.circulating_supply / coin.total_supply
            if supply_ratio < 0.1:  # <10% circulating
                score += 15
                reasons.append(f"📊 Low circulating supply ({supply_ratio*100:.1f}%)")
            elif supply_ratio < 0.5:  # <50% circulating
                score += 10
                reasons.append(f"📊 Moderate supply ({supply_ratio*100:.1f}%)")
        
        # 7. Market cap change (growing interest)
        if coin.market_cap_change_24h > 10:
            score += 10
            reasons.append(f"📈 Growing market cap ({coin.market_cap_change_24h:+.1f}%)")
        
        reason = " | ".join(reasons) if reasons else "Emerging opportunity"
        
        return score, reason
    
    def _count_decimals(self, price: float) -> int:
        """Count decimal places in price"""
        if price == 0:
            return 0
        
        price_str = f"{price:.15f}".rstrip('0')
        if '.' in price_str:
            return len(price_str.split('.')[1])
        return 0
    
    def get_discovery_stats(self) -> Dict:
        """Get discovery engine statistics"""
        return {
            'data_source': 'CoinGecko Markets API',
            'coverage': 'All cryptocurrencies globally',
            'state_restrictions': 'None - works everywhere',
            'rate_limit': '10-50 calls/min',
            'price_range': '$0.00000001 - $0.01',
            'scoring_factors': [
                'Price recovery potential (ATH vs current)',
                'Recent momentum (24h change)',
                'Volume activity',
                'Market cap (lower = higher upside)',
                'Price decimals (ultra-low)',
                'Supply dynamics',
                'Market cap growth'
            ],
            'update_frequency': 'Real-time (CoinGecko API)',
            'cache_ttl': '1 hour'
        }

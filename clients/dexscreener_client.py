"""
DexScreener API Client

Official API documentation: https://docs.dexscreener.com/api/reference

Features:
- New pairs discovery across all chains
- Token search and filtering
- Liquidity, volume, and price tracking
- Real-time pair updates
"""

import os
import asyncio
import httpx
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv
from models.dex_models import DexPair, TokenLaunch, Chain, LaunchStage, ContractSafety

load_dotenv()


class DexScreenerClient:
    """Client for DexScreener API"""
    
    BASE_URL = "https://api.dexscreener.com/latest"
    BASE_URL_V1 = "https://api.dexscreener.com"  # For newer v1 endpoints
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize DexScreener client
        
        Args:
            api_key: Optional API key for higher rate limits (if available)
        """
        self.api_key = api_key or os.getenv("DEXSCREENER_API_KEY")
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests (free tier)
        
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """
        Make API request with rate limiting
        
        Args:
            endpoint: API endpoint (e.g., "/dex/tokens/0x...")
            params: Query parameters
            
        Returns:
            (success, response_data)
        """
        try:
            # Rate limiting
            elapsed = datetime.now().timestamp() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            
            url = f"{self.BASE_URL}{endpoint}"
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)
                self.last_request_time = datetime.now().timestamp()
                
                if response.status_code == 200:
                    return True, response.json()
                elif response.status_code == 429:
                    logger.warning("DexScreener rate limit hit, waiting 60s")
                    await asyncio.sleep(60)
                    return False, {"error": "Rate limit exceeded"}
                elif response.status_code == 404:
                    logger.debug(f"Token not found on DexScreener (404)")
                    return False, {"error": "Token not found (404)"}
                else:
                    error_text = response.text[:200] if response.text else "No error message"
                    logger.warning(f"DexScreener API error: {response.status_code} - {error_text}")
                    return False, {"error": f"HTTP {response.status_code}: {error_text}"}
                    
        except httpx.TimeoutException as e:
            error_msg = f"Request timeout: {str(e)}"
            logger.warning(f"DexScreener API timeout: {error_msg}")
            return False, {"error": error_msg}
        except httpx.NetworkError as e:
            error_msg = f"Network error: {str(e)}"
            logger.warning(f"DexScreener network error: {error_msg}")
            return False, {"error": error_msg}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error calling DexScreener API: {error_msg}", exc_info=True)
            return False, {"error": error_msg}
    
    async def get_token_pairs(self, token_address: str, chain: Optional[str] = None) -> Tuple[bool, List[DexPair]]:
        """
        Get all DEX pairs for a token address
        
        Args:
            token_address: Contract address
            chain: Optional chain filter (ethereum, bsc, solana, etc.)
            
        Returns:
            (success, list of DexPair objects)
        """
        endpoint = f"/dex/tokens/{token_address}"
        success, data = await self._make_request(endpoint)
        
        # Check if request failed or data is invalid
        if not success:
            error_detail = data.get("error", "Unknown error") if isinstance(data, dict) else str(data) if data else "Request failed"
            
            # Provide more specific error messages
            if "404" in str(error_detail) or "not found" in str(error_detail).lower():
                logger.debug(f"Token not found on DexScreener: {token_address}")
            elif "timeout" in str(error_detail).lower():
                logger.warning(f"DexScreener request timeout for token {token_address}")
            elif "network" in str(error_detail).lower():
                logger.warning(f"DexScreener network error for token {token_address}")
            else:
                logger.warning(f"DexScreener API request failed for token {token_address}: {error_detail}")
            
            return False, []
        
        # Ensure data is a dict and has pairs
        if not data or not isinstance(data, dict):
            logger.debug(f"Invalid data returned from DexScreener for token {token_address}")
            return False, []
        
        if "pairs" not in data:
            logger.debug(f"No 'pairs' key in DexScreener response for token {token_address}")
            return False, []
        
        pairs = []
        pairs_data = data.get("pairs", [])
        if not pairs_data or not isinstance(pairs_data, list):
            logger.debug(f"Invalid pairs data for token {token_address}")
            return False, []
        
        for pair_data in pairs_data:
            try:
                # Filter by chain if specified
                if chain and pair_data.get("chainId", "").lower() != chain.lower():
                    continue
                
                pair = self._parse_pair(pair_data)
                if pair:
                    pairs.append(pair)
            except Exception as e:
                logger.debug(f"Error parsing pair: {e}")
                continue
        
        return True, pairs
    
    async def search_pairs(
        self,
        query: str,
        chain: Optional[str] = None
    ) -> Tuple[bool, List[DexPair]]:
        """
        Search for pairs by token symbol or name
        
        Args:
            query: Search query (symbol or name)
            chain: Optional chain filter
            
        Returns:
            (success, list of DexPair objects)
        """
        endpoint = f"/dex/search"
        params = {"q": query}
        success, data = await self._make_request(endpoint, params)
        
        # Check if request failed or data is invalid
        if not success:
            logger.debug(f"DexScreener search request failed for query: {query}")
            return False, []
        
        # Ensure data is a dict and has pairs
        if not data or not isinstance(data, dict):
            logger.debug(f"Invalid data returned from DexScreener search for query: {query}")
            return False, []
        
        if "pairs" not in data:
            logger.debug(f"No 'pairs' key in DexScreener search response for query: {query}")
            return False, []
        
        pairs = []
        pairs_data = data.get("pairs", [])
        if not pairs_data or not isinstance(pairs_data, list):
            logger.debug(f"Invalid pairs data from DexScreener search for query: {query}")
            return False, []
        
        for pair_data in pairs_data:
            try:
                if chain and pair_data.get("chainId", "").lower() != chain.lower():
                    continue
                
                pair = self._parse_pair(pair_data)
                if pair:
                    pairs.append(pair)
            except Exception as e:
                logger.debug(f"Error parsing pair: {e}")
                continue
        
        return True, pairs
    
    async def get_new_pairs(
        self,
        chains: Optional[List[str]] = None,
        min_liquidity: float = 1000.0,  # Lowered from 5000 - catch earlier
        max_liquidity: Optional[float] = None,
        max_age_hours: float = 24.0,
        limit: int = 50,
        discovery_mode: str = "aggressive"  # "conservative", "balanced", "aggressive"
    ) -> Tuple[bool, List[DexPair]]:
        """
        Get new token pairs across supported chains using MULTIPLE sources
        
        Args:
            chains: List of chain names to filter (e.g., ['ethereum', 'bsc'])
            min_liquidity: Minimum liquidity USD
            max_liquidity: Maximum liquidity USD
            max_age_hours: Maximum pair age in hours
            limit: Maximum pairs to return
            discovery_mode: How aggressive to be in finding tokens
            
        Returns:
            (success, list of DexPair objects)
        """
        logger.info(f"Searching for new pairs via DexScreener (mode={discovery_mode})...")
        
        all_pairs = []
        seen_addresses = set()
        
        # Adjust filters based on mode
        if discovery_mode == "aggressive":
            effective_min_liq = min(min_liquidity, 100)  # Very low - catch super early launches
            effective_min_vol = 0  # No volume filter - new tokens may have 0 volume initially
            max_search_queries = 25  # More queries
            effective_max_age = max_age_hours * 2  # Allow older tokens too (48h default)
        elif discovery_mode == "balanced":
            effective_min_liq = min_liquidity
            effective_min_vol = 200
            max_search_queries = 15
            effective_max_age = max_age_hours
        else:  # conservative
            effective_min_liq = max(min_liquidity, 5000)
            effective_min_vol = 500
            max_search_queries = 8
            effective_max_age = max_age_hours
        
        # Scam filter patterns
        scam_names = {"coin", "token", "test", "scam", "rug", "fake"}
        
        # Stats for logging
        stats = {
            "source0_found": 0, "source1_found": 0, "source2_found": 0, "source3_found": 0,
            "skipped_seen": 0, "skipped_chain": 0, "skipped_scam": 0,
            "skipped_liquidity": 0, "skipped_volume": 0, "skipped_address": 0,
            "skipped_age": 0
        }
        
        try:
            # SOURCE 0 (NEW!): Token Profiles & Boosts - catches actively promoted tokens
            profile_boost_pairs = []
            try:
                # Get latest token profiles (tokens that paid for promotion)
                success, profiles = await self.get_latest_token_profiles()
                if success and profiles:
                    logger.info(f"📋 Found {len(profiles)} token profiles")
                    print(f"[DEX] Source 0a: {len(profiles)} token profiles", flush=True)
                    profile_pairs = await self.get_token_profile_pairs(profiles, chains)
                    profile_boost_pairs.extend(profile_pairs)
                
                # Get latest boosted tokens
                success, boosts = await self.get_latest_boosted_tokens()
                if success and boosts:
                    logger.info(f"🚀 Found {len(boosts)} boosted tokens")
                    print(f"[DEX] Source 0b: {len(boosts)} boosted tokens", flush=True)
                    boost_pairs = await self.get_token_profile_pairs(boosts, chains)
                    profile_boost_pairs.extend(boost_pairs)
                
                # Get top boosted tokens (highest conviction)
                success, top_boosts = await self.get_top_boosted_tokens()
                if success and top_boosts:
                    logger.info(f"🔥 Found {len(top_boosts)} top boosted tokens")
                    print(f"[DEX] Source 0c: {len(top_boosts)} top boosted tokens", flush=True)
                    top_pairs = await self.get_token_profile_pairs(top_boosts, chains)
                    profile_boost_pairs.extend(top_pairs)
                
                stats["source0_found"] = len(profile_boost_pairs)
                logger.info(f"Source 0: Found {len(profile_boost_pairs)} from profiles/boosts (NEW!)")
            except Exception as e:
                logger.warning(f"Source 0 (profiles/boosts) failed: {e}")
            
            # SOURCE 1: Get latest pairs (with timeout to prevent hangs)
            try:
                latest_pairs = await asyncio.wait_for(self._get_latest_pairs(chains), timeout=30.0)
                stats["source1_found"] = len(latest_pairs)
                logger.info(f"Source 1: Found {len(latest_pairs)} from latest pairs")
            except asyncio.TimeoutError:
                logger.warning("Source 1 timed out, continuing...")
                latest_pairs = []
            except Exception as e:
                logger.warning(f"Source 1 failed: {e}")
                latest_pairs = []
            
            # SOURCE 2: Get trending/boosted tokens (with timeout)
            try:
                trending_pairs = await asyncio.wait_for(self._get_trending_pairs(chains), timeout=30.0)
                stats["source2_found"] = len(trending_pairs)
                logger.info(f"Source 2: Found {len(trending_pairs)} from trending")
            except asyncio.TimeoutError:
                logger.warning("Source 2 timed out, continuing...")
                trending_pairs = []
            except Exception as e:
                logger.warning(f"Source 2 failed: {e}")
                trending_pairs = []
            
            # SOURCE 3: Expanded search queries for better coverage
            logger.info("Source 3: Searching by keywords...")
            search_queries = [
                # High priority - common meme patterns
                "pepe", "doge", "shiba", "inu", "cat", "meme",
                # Chain-specific searches for fresh tokens
                "solana", "sol", "pump", "fun",
                # Trending themes
                "ai", "gpt", "elon", "trump", "based",
                # Classic patterns
                "moon", "rocket", "gem", "new", "launch",
                # Animal memes
                "frog", "dog", "bear", "bull",
                # Pop culture
                "wojak", "chad", "virgin", "giga",
                # Finance themes
                "safe", "baby", "mini", "micro",
                # Numbers/symbols often in meme coins
                "1000x", "100x", "x",
            ]
            
            # Combine all sources (profile/boost pairs first - they're actively promoted!)
            all_source_pairs = profile_boost_pairs + latest_pairs + trending_pairs
            
            # Add search results
            search_count = 0
            for query in search_queries[:max_search_queries]:
                if len(all_source_pairs) >= limit * 4:  # Get 4x limit for better filtering
                    break
                
                try:
                    success, results = await self.search_pairs(query)
                    if success and results:
                        all_source_pairs.extend(results)
                        search_count += len(results)
                    await asyncio.sleep(0.3)  # Faster delay for more queries
                except Exception as e:
                    logger.debug(f"Error searching '{query}': {e}")
                    continue
            
            stats["source3_found"] = search_count
            logger.info(f"Source 3: Found {search_count} from {max_search_queries} keyword searches")
            
            # Process and filter all pairs
            for pair in all_source_pairs:
                if len(all_pairs) >= limit:
                    break
                
                # Skip if already seen
                if pair.base_token_address.lower() in seen_addresses:
                    stats["skipped_seen"] += 1
                    continue
                
                # Filter by chain
                if chains and pair.chain.value not in chains:
                    stats["skipped_chain"] += 1
                    continue
                
                # SCAM FILTER 1: Generic names
                symbol_lower = pair.base_token_symbol.lower()
                if symbol_lower in scam_names:
                    stats["skipped_scam"] += 1
                    continue
                
                # SCAM FILTER 2: Very short symbols (< 2 chars)
                if len(pair.base_token_symbol) < 2:
                    stats["skipped_scam"] += 1
                    continue
                
                # Filter by liquidity (use effective values based on mode)
                if pair.liquidity_usd < effective_min_liq:
                    stats["skipped_liquidity"] += 1
                    continue
                if max_liquidity and pair.liquidity_usd > max_liquidity:
                    stats["skipped_liquidity"] += 1
                    continue
                
                # QUALITY FILTER: Require meaningful volume (adjusted by mode)
                if pair.volume_24h < effective_min_vol:
                    stats["skipped_volume"] += 1
                    continue
                
                # VALIDATION: Check address format before adding
                if not self._is_valid_address_format(pair.base_token_address, pair.chain):
                    stats["skipped_address"] += 1
                    continue
                
                # Filter by age (if available) - use effective_max_age based on mode
                if pair.pair_age_hours > 0 and pair.pair_age_hours > effective_max_age:
                    stats["skipped_age"] += 1
                    continue
                
                # Add to results
                all_pairs.append(pair)
                seen_addresses.add(pair.base_token_address.lower())
                
                if len(all_pairs) >= limit:
                    break
            
            # Sort by age (newest first) if age data available
            all_pairs.sort(key=lambda p: p.pair_age_hours if p.pair_age_hours > 0 else 999)
            
            # Log detailed stats
            total_found = stats["source1_found"] + stats["source2_found"] + stats["source3_found"]
            logger.info(f"Discovery stats: total_raw={total_found}, final={len(all_pairs)}")
            logger.info(f"  Filtered: seen={stats['skipped_seen']}, chain={stats['skipped_chain']}, "
                       f"scam={stats['skipped_scam']}, liq={stats['skipped_liquidity']}, "
                       f"vol={stats['skipped_volume']}, addr={stats['skipped_address']}, age={stats['skipped_age']}")
            
            # Print to console for visibility
            print(f"[DEXSCREENER] Discovery: {total_found} raw → {len(all_pairs)} passed (mode={discovery_mode})", flush=True)
            print(f"[DEXSCREENER] Filter breakdown: dupes={stats['skipped_seen']}, liq={stats['skipped_liquidity']}, vol={stats['skipped_volume']}, age={stats['skipped_age']}", flush=True)
            
            return True, all_pairs[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching new pairs: {e}", exc_info=True)
            return False, []
    
    def _parse_pair(self, pair_data: Dict) -> Optional[DexPair]:
        """Parse API response into DexPair object"""
        try:
            # Parse chain
            chain_id = pair_data.get("chainId", "").lower()
            chain_map = {
                "ethereum": Chain.ETH,
                "bsc": Chain.BSC,
                "solana": Chain.SOLANA,
                "base": Chain.BASE,
                "arbitrum": Chain.ARBITRUM,
                "polygon": Chain.POLYGON,
                "avalanche": Chain.AVALANCHE
            }
            chain = chain_map.get(chain_id, Chain.ETH)
            
            # Parse timestamps
            created_at = None
            pair_age_hours = 0.0
            if "pairCreatedAt" in pair_data:
                try:
                    created_at = datetime.fromtimestamp(pair_data["pairCreatedAt"] / 1000)
                    pair_age_hours = (datetime.now() - created_at).total_seconds() / 3600
                except:
                    pass
            
            # Parse price changes
            price_change = pair_data.get("priceChange", {})
            
            # Parse transaction counts
            txns = pair_data.get("txns", {})
            txns_5m = txns.get("m5", {})
            txns_1h = txns.get("h1", {})
            txns_24h = txns.get("h24", {})
            
            pair = DexPair(
                pair_address=pair_data.get("pairAddress", ""),
                dex_name=pair_data.get("dexId", "unknown"),
                chain=chain,
                base_token_symbol=pair_data.get("baseToken", {}).get("symbol", ""),
                base_token_address=pair_data.get("baseToken", {}).get("address", ""),
                quote_token_symbol=pair_data.get("quoteToken", {}).get("symbol", "USDT"),
                liquidity_usd=float(pair_data.get("liquidity", {}).get("usd", 0)),
                volume_24h=float(pair_data.get("volume", {}).get("h24", 0)),
                price_usd=float(pair_data.get("priceUsd", 0)),
                price_change_5m=float(price_change.get("m5", 0)),
                price_change_1h=float(price_change.get("h1", 0)),
                price_change_6h=float(price_change.get("h6", 0)),
                price_change_24h=float(price_change.get("h24", 0)),
                txn_count_5m=txns_5m.get("buys", 0) + txns_5m.get("sells", 0),
                txn_count_1h=txns_1h.get("buys", 0) + txns_1h.get("sells", 0),
                txn_count_24h=txns_24h.get("buys", 0) + txns_24h.get("sells", 0),
                buys_5m=txns_5m.get("buys", 0),
                sells_5m=txns_5m.get("sells", 0),
                created_at=created_at,
                pair_age_hours=pair_age_hours,
                url=pair_data.get("url", "")
            )
            
            return pair
            
        except Exception as e:
            logger.error(f"Error parsing pair data: {e}", exc_info=True)
            return None
    
    def _is_valid_address_format(self, address: str, chain: Chain) -> bool:
        """Validate address format for chain"""
        if not address or len(address) < 10:
            return False
        
        # EVM chains: Must start with 0x and be 42 chars
        if chain in [Chain.ETH, Chain.BSC, Chain.BASE, Chain.ARBITRUM, Chain.POLYGON]:
            return address.startswith("0x") and len(address) == 42
        
        # Solana: Base58, 32-44 chars, no 0x or special chars
        if chain == Chain.SOLANA:
            # Check for invalid patterns like "ONE-f9954f"
            if "-" in address or address.startswith("0x"):
                return False
            return 32 <= len(address) <= 44
        
        return True  # Other chains: accept for now
    
    async def _get_latest_pairs(self, chains: Optional[List[str]] = None) -> List[DexPair]:
        """Get latest pairs using multiple search strategies"""
        pairs = []
        try:
            target_chains = chains if chains else ["solana", "ethereum", "bsc"]
            
            # Strategy 1: Chain-specific searches with different keywords
            chain_keywords = {
                "solana": ["pump.fun", "raydium", "jupiter", "meteora"],
                "ethereum": ["uniswap", "launch", "presale"],
                "bsc": ["pancake", "launch", "gem"]
            }
            
            for chain in target_chains[:3]:
                keywords = chain_keywords.get(chain, ["launch", "new"])
                for keyword in keywords[:2]:  # 2 keywords per chain
                    try:
                        success, results = await asyncio.wait_for(
                            self.search_pairs(f"{keyword}"),
                            timeout=8.0
                        )
                        if success and results:
                            # Filter to target chain
                            chain_pairs = [p for p in results if p.chain.value == chain]
                            pairs.extend(chain_pairs[:15])
                        await asyncio.sleep(0.3)
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        continue
            
            # Strategy 2: Search for recently trending symbols (often new launches)
            trending_searches = ["$", "sol", "eth", "bnb"]  # $ often appears in new meme coins
            for search in trending_searches[:2]:
                try:
                    success, results = await asyncio.wait_for(
                        self.search_pairs(search),
                        timeout=8.0
                    )
                    if success and results:
                        # Take youngest pairs (sorted by age)
                        sorted_by_age = sorted(results, key=lambda p: p.pair_age_hours if p.pair_age_hours > 0 else 999)
                        pairs.extend(sorted_by_age[:20])
                    await asyncio.sleep(0.3)
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Error getting latest pairs: {e}")
        
        logger.debug(f"Latest pairs source found: {len(pairs)} total")
        return pairs
    
    async def _get_trending_pairs(self, chains: Optional[List[str]] = None) -> List[DexPair]:
        """Get trending/boosted tokens with momentum"""
        pairs = []
        try:
            # Search for tokens with high momentum keywords (expanded list)
            momentum_keywords = [
                "100x", "1000x", "pump", "moon", "rocket",
                "gem", "alpha", "call", "bullish", "send"
            ]
            
            for keyword in momentum_keywords[:5]:  # 5 keywords
                try:
                    success, results = await self.search_pairs(keyword)
                    if success and results:
                        # Filter for tokens with actual momentum (lower threshold for discovery)
                        momentum_pairs = [p for p in results if p.volume_24h > 1000]  # Lowered from 5000
                        pairs.extend(momentum_pairs[:10])
                    await asyncio.sleep(0.3)
                except:
                    continue
            
            # Also search for common meme formats
            meme_patterns = ["inu", "pepe", "wojak", "chad", "giga"]
            for pattern in meme_patterns[:3]:
                try:
                    success, results = await self.search_pairs(pattern)
                    if success and results:
                        # Sort by age to get freshest ones
                        fresh_pairs = sorted(results, key=lambda p: p.pair_age_hours if p.pair_age_hours > 0 else 999)
                        pairs.extend(fresh_pairs[:10])
                    await asyncio.sleep(0.3)
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Error getting trending pairs: {e}")
        
        logger.debug(f"Trending pairs source found: {len(pairs)} total")
        return pairs
    
    async def get_trending_pairs(self, limit: int = 50) -> Tuple[bool, List[DexPair]]:
        """
        Get trending pairs (high volume/velocity)
        
        Note: This requires monitoring the DexScreener website or using paid API
        For free tier, we'll filter by volume/price change
        """
        logger.warning("Trending pairs requires premium features or web scraping")
        return True, []
    
    # ========================================================================
    # NEW V1 ENDPOINTS - Token Profiles & Boosts (FREE!)
    # ========================================================================
    
    async def _make_request_v1(self, endpoint: str, params: Optional[Dict] = None) -> Tuple[bool, any]:
        """
        Make API request to v1 endpoints (token-profiles, token-boosts)
        
        Args:
            endpoint: API endpoint (e.g., "/token-profiles/latest/v1")
            params: Query parameters
            
        Returns:
            (success, response_data) - can be list or dict
        """
        try:
            # Rate limiting
            elapsed = datetime.now().timestamp() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            
            url = f"{self.BASE_URL_V1}{endpoint}"
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)
                self.last_request_time = datetime.now().timestamp()
                
                if response.status_code == 200:
                    return True, response.json()
                elif response.status_code == 429:
                    logger.warning("DexScreener rate limit hit on v1 endpoint")
                    await asyncio.sleep(60)
                    return False, {"error": "Rate limit exceeded"}
                else:
                    logger.warning(f"DexScreener v1 API error: {response.status_code}")
                    return False, {"error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error calling DexScreener v1 API: {e}")
            return False, {"error": str(e)}
    
    async def get_latest_token_profiles(self) -> Tuple[bool, List[Dict]]:
        """
        Get latest token profiles (tokens that paid for a profile)
        
        These are tokens actively being promoted - often indicates
        active development and marketing push.
        
        Rate limit: 60 requests per minute
        
        Returns:
            (success, list of token profile dicts)
        """
        success, data = await self._make_request_v1("/token-profiles/latest/v1")
        
        if not success:
            return False, []
        
        # Response can be a single object or array
        if isinstance(data, dict):
            return True, [data] if data.get("tokenAddress") else []
        elif isinstance(data, list):
            return True, data
        
        return True, []
    
    async def get_latest_boosted_tokens(self) -> Tuple[bool, List[Dict]]:
        """
        Get latest boosted tokens (tokens receiving community boosts)
        
        Boosted tokens have community members paying to promote them,
        indicating active interest and potential pumps.
        
        Rate limit: 60 requests per minute
        
        Returns:
            (success, list of boosted token dicts)
        """
        success, data = await self._make_request_v1("/token-boosts/latest/v1")
        
        if not success:
            return False, []
        
        # Response can be a single object or array
        if isinstance(data, dict):
            return True, [data] if data.get("tokenAddress") else []
        elif isinstance(data, list):
            return True, data
        
        return True, []
    
    async def get_top_boosted_tokens(self) -> Tuple[bool, List[Dict]]:
        """
        Get tokens with most active boosts (highest community conviction)
        
        These are the tokens with the MOST boost activity - strongest
        community conviction signals.
        
        Rate limit: 60 requests per minute
        
        Returns:
            (success, list of top boosted token dicts)
        """
        success, data = await self._make_request_v1("/token-boosts/top/v1")
        
        if not success:
            return False, []
        
        # Response can be a single object or array
        if isinstance(data, dict):
            return True, [data] if data.get("tokenAddress") else []
        elif isinstance(data, list):
            return True, data
        
        return True, []
    
    async def get_token_profile_pairs(
        self,
        profiles: List[Dict],
        target_chains: Optional[List[str]] = None,
        max_profiles: int = 15  # Limit to prevent timeout
    ) -> List[DexPair]:
        """
        Convert token profiles/boosts to DexPairs by looking up their trading pairs
        
        Args:
            profiles: List of token profile/boost dicts from v1 API
            target_chains: Optional list of chains to filter (e.g., ["solana"])
            max_profiles: Maximum profiles to process (prevents timeout)
            
        Returns:
            List of DexPair objects
        """
        pairs = []
        processed = 0
        
        for profile in profiles:
            # Limit to prevent timeout
            if processed >= max_profiles:
                logger.debug(f"Reached max_profiles limit ({max_profiles})")
                break
                
            try:
                chain_id = profile.get("chainId", "")
                token_address = profile.get("tokenAddress", "")
                
                # Skip if not in target chains
                if target_chains and chain_id not in target_chains:
                    continue
                
                if not token_address:
                    continue
                
                processed += 1
                
                # Get the pairs for this token
                success, token_pairs = await self.get_token_pairs(token_address, chain=chain_id)
                
                if success and token_pairs:
                    # Take the first/main pair
                    pairs.extend(token_pairs[:1])
                
                await asyncio.sleep(0.2)  # Reduced rate limiting
                
            except Exception as e:
                logger.debug(f"Error getting pairs for profile: {e}")
                continue
        
        return pairs
    
    async def get_all_new_launches(
        self,
        chains: Optional[List[str]] = None,
        min_liquidity: float = 1000.0,
        max_liquidity: float = 1000000.0,
        include_profiles: bool = True,
        include_boosts: bool = True,
        limit: int = 50
    ) -> Tuple[bool, List[DexPair]]:
        """
        COMPREHENSIVE new launch discovery using ALL available free endpoints.
        
        This combines:
        1. Latest token profiles (actively promoted)
        2. Latest boosted tokens (community pushing)
        3. Top boosted tokens (strongest conviction)
        4. Keyword searches (existing method)
        
        Args:
            chains: List of chains to scan (default: solana)
            min_liquidity: Minimum liquidity in USD
            max_liquidity: Maximum liquidity in USD
            include_profiles: Include token profiles
            include_boosts: Include boosted tokens
            limit: Maximum pairs to return
            
        Returns:
            (success, list of DexPair objects)
        """
        target_chains = chains or ["solana"]
        all_pairs: List[DexPair] = []
        seen_addresses = set()
        
        logger.info(f"🔍 Comprehensive launch scan starting...")
        print(f"[DEX] Starting comprehensive launch scan on {target_chains}", flush=True)
        
        # SOURCE 1: Latest Token Profiles
        if include_profiles:
            try:
                success, profiles = await self.get_latest_token_profiles()
                if success and profiles:
                    logger.info(f"📋 Found {len(profiles)} token profiles")
                    print(f"[DEX] Source 1: {len(profiles)} token profiles", flush=True)
                    
                    profile_pairs = await self.get_token_profile_pairs(profiles, target_chains)
                    for pair in profile_pairs:
                        if pair.base_token_address.lower() not in seen_addresses:
                            if min_liquidity <= pair.liquidity_usd <= max_liquidity:
                                all_pairs.append(pair)
                                seen_addresses.add(pair.base_token_address.lower())
            except Exception as e:
                logger.warning(f"Token profiles source failed: {e}")
        
        # SOURCE 2: Latest Boosted Tokens
        if include_boosts:
            try:
                success, boosts = await self.get_latest_boosted_tokens()
                if success and boosts:
                    logger.info(f"🚀 Found {len(boosts)} boosted tokens")
                    print(f"[DEX] Source 2: {len(boosts)} boosted tokens", flush=True)
                    
                    boost_pairs = await self.get_token_profile_pairs(boosts, target_chains)
                    for pair in boost_pairs:
                        if pair.base_token_address.lower() not in seen_addresses:
                            if min_liquidity <= pair.liquidity_usd <= max_liquidity:
                                all_pairs.append(pair)
                                seen_addresses.add(pair.base_token_address.lower())
            except Exception as e:
                logger.warning(f"Boosted tokens source failed: {e}")
        
        # SOURCE 3: Top Boosted Tokens
        if include_boosts:
            try:
                success, top_boosts = await self.get_top_boosted_tokens()
                if success and top_boosts:
                    logger.info(f"🔥 Found {len(top_boosts)} top boosted tokens")
                    print(f"[DEX] Source 3: {len(top_boosts)} top boosted tokens", flush=True)
                    
                    top_pairs = await self.get_token_profile_pairs(top_boosts, target_chains)
                    for pair in top_pairs:
                        if pair.base_token_address.lower() not in seen_addresses:
                            if min_liquidity <= pair.liquidity_usd <= max_liquidity:
                                all_pairs.append(pair)
                                seen_addresses.add(pair.base_token_address.lower())
            except Exception as e:
                logger.warning(f"Top boosted source failed: {e}")
        
        logger.info(f"✅ Comprehensive scan complete: {len(all_pairs)} unique pairs")
        print(f"[DEX] Comprehensive scan found {len(all_pairs)} unique pairs", flush=True)
        
        return True, all_pairs[:limit]
    
    async def validate_connection(self) -> Tuple[bool, str]:
        """Test API connection"""
        try:
            # Test with a known token (e.g., WETH)
            success, data = await self._make_request("/dex/tokens/0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
            
            if success:
                return True, "DexScreener API connected successfully"
            else:
                return False, f"DexScreener API error: {data.get('error', 'Unknown')}"
                
        except Exception as e:
            return False, f"DexScreener connection failed: {str(e)}"


# For on-chain monitoring as alternative to API
class OnChainMonitor:
    """
    Monitor blockchain events for new pair creation
    
    This is the BEST way to catch launches early since DexScreener
    free tier doesn't have real-time new pair alerts.
    
    You'd use Web3.py to listen to DEX factory events:
    - Uniswap: PairCreated events
    - PancakeSwap: PairCreated events
    - Raydium: InitializeInstruction
    """
    
    def __init__(self, rpc_url: str, chain: Chain):
        self.rpc_url = rpc_url
        self.chain = chain
        # TODO: Implement Web3 connection
        logger.info(f"OnChainMonitor initialized for {chain.value}")
    
    async def listen_for_new_pairs(self):
        """
        Listen for PairCreated events from DEX factories
        
        This would use websockets to get real-time events:
        - Uniswap V2/V3 Factory
        - PancakeSwap Factory
        - Raydium program
        """
        logger.warning("On-chain monitoring not yet implemented - requires Web3.py integration")
        # TODO: Implement event listening
        pass

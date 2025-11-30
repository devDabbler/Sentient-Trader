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
        import sys
        print(f"DEBUG [_make_request]: endpoint={endpoint}, params={params}", file=sys.stdout, flush=True)
        try:
            # Rate limiting
            elapsed = datetime.now().timestamp() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                wait_time = self.rate_limit_delay - elapsed
                print(f"DEBUG [_make_request]: Rate limiting, waiting {wait_time:.2f}s", file=sys.stdout, flush=True)
                await asyncio.sleep(wait_time)
            
            url = f"{self.BASE_URL}{endpoint}"
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            
            print(f"DEBUG [_make_request]: Making GET request to {url}", file=sys.stdout, flush=True)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params, headers=headers)
                print(f"DEBUG [_make_request]: Got response status={response.status_code}", file=sys.stdout, flush=True)
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
        min_liquidity: float = 5000.0,
        max_liquidity: Optional[float] = None,
        max_age_hours: float = 24.0,
        limit: int = 50
    ) -> Tuple[bool, List[DexPair]]:
        """
        Get new token pairs across supported chains using MULTIPLE sources
        
        Args:
            chains: List of chain names to filter (e.g., ['ethereum', 'bsc'])
            min_liquidity: Minimum liquidity USD
            max_liquidity: Maximum liquidity USD
            max_age_hours: Maximum pair age in hours
            limit: Maximum pairs to return
            
        Returns:
            (success, list of DexPair objects)
        """
        import sys
        print("DEBUG [get_new_pairs]: Method entered!", file=sys.stdout, flush=True)
        logger.info("Searching for new pairs via DexScreener (FREE API - MULTIPLE SOURCES)...")
        print("DEBUG [get_new_pairs]: After first logger.info", file=sys.stdout, flush=True)
        
        all_pairs = []
        seen_addresses = set()
        
        # Scam filter patterns
        scam_names = {"coin", "token", "test", "scam", "rug", "fake"}
        
        try:
            # SOURCE 1: Get latest pairs (with timeout to prevent hangs)
            print("DEBUG [get_new_pairs]: Source 1 - Fetching latest pairs...", file=sys.stdout, flush=True)
            logger.info("Source 1: Fetching latest pairs...")
            try:
                latest_pairs = await asyncio.wait_for(self._get_latest_pairs(chains), timeout=30.0)
                print(f"DEBUG [get_new_pairs]: Source 1 returned {len(latest_pairs)} pairs", file=sys.stdout, flush=True)
                logger.info(f"Found {len(latest_pairs)} from latest pairs")
            except asyncio.TimeoutError:
                print("DEBUG [get_new_pairs]: Source 1 TIMEOUT after 30s", file=sys.stdout, flush=True)
                logger.warning("Source 1 timed out after 30s, continuing...")
                latest_pairs = []
            except Exception as e:
                print(f"DEBUG [get_new_pairs]: Source 1 ERROR: {e}", file=sys.stdout, flush=True)
                logger.warning(f"Source 1 failed: {e}")
                latest_pairs = []
            
            # SOURCE 2: Get trending/boosted tokens (with timeout)
            print("DEBUG [get_new_pairs]: Source 2 - Fetching trending tokens...", file=sys.stdout, flush=True)
            logger.info("Source 2: Fetching trending tokens...")
            try:
                trending_pairs = await asyncio.wait_for(self._get_trending_pairs(chains), timeout=30.0)
                print(f"DEBUG [get_new_pairs]: Source 2 returned {len(trending_pairs)} pairs", file=sys.stdout, flush=True)
                logger.info(f"Found {len(trending_pairs)} from trending")
            except asyncio.TimeoutError:
                print("DEBUG [get_new_pairs]: Source 2 TIMEOUT after 30s", file=sys.stdout, flush=True)
                logger.warning("Source 2 timed out after 30s, continuing...")
                trending_pairs = []
            except Exception as e:
                print(f"DEBUG [get_new_pairs]: Source 2 ERROR: {e}", file=sys.stdout, flush=True)
                logger.warning(f"Source 2 failed: {e}")
                trending_pairs = []
            
            # SOURCE 3: Search queries (existing)
            logger.info("Source 3: Searching by keywords...")
            search_queries = [
                "new",      # Tokens with "new" in name
                "launch",   # Launch-related tokens  
                "moon",     # Common meme token keyword
                "inu",      # Popular token suffix
                "pepe",     # Trending meme theme
                "doge",     # Dog meme tokens
                "shiba",    # Shiba variants
                "elon",     # Elon-themed tokens
                "safe",     # SafeMoon-style tokens
                "baby",     # Baby variants
                "mini",     # Mini variants
                "gem",      # Hidden gem tokens
                "ai",       # AI tokens (trending)
                "x",        # X/Twitter-themed tokens
                "cat",      # Cat meme tokens
                "wojak",    # Wojak meme tokens
                "based",    # Based tokens
            ]
            
            # Combine all sources
            all_source_pairs = latest_pairs + trending_pairs
            
            # Add search results
            for query in search_queries[:10]:  # Limit to 10 queries to save time
                if len(all_source_pairs) >= limit * 3:  # Get 3x limit for filtering
                    break
                
                try:
                    success, results = await self.search_pairs(query)
                    if success and results:
                        all_source_pairs.extend(results)
                    await asyncio.sleep(0.5)  # Faster delay
                except Exception as e:
                    logger.debug(f"Error searching '{query}': {e}")
                    continue
            
            # Process and filter all pairs
            for pair in all_source_pairs:
                if len(all_pairs) >= limit:
                    break
                
                # Skip if already seen
                if pair.base_token_address.lower() in seen_addresses:
                    continue
                
                # Filter by chain
                if chains and pair.chain.value not in chains:
                    continue
                
                # SCAM FILTER 1: Generic names
                symbol_lower = pair.base_token_symbol.lower()
                if symbol_lower in scam_names:
                    logger.debug(f"Skipping generic name: {pair.base_token_symbol}")
                    continue
                
                # SCAM FILTER 2: Very short symbols (< 2 chars)
                if len(pair.base_token_symbol) < 2:
                    logger.debug(f"Skipping very short symbol: {pair.base_token_symbol}")
                    continue
                
                # Filter by liquidity
                if pair.liquidity_usd < min_liquidity:
                    continue
                if max_liquidity and pair.liquidity_usd > max_liquidity:
                    continue
                
                # QUALITY FILTER: Require meaningful volume (not dead)
                if pair.volume_24h < 500:  # At least $500 volume for active tokens
                    logger.debug(f"Skipping low volume: {pair.base_token_symbol} (${pair.volume_24h:.0f})")
                    continue
                
                # VALIDATION: Check address format before adding
                if not self._is_valid_address_format(pair.base_token_address, pair.chain):
                    logger.debug(f"Skipping invalid address: {pair.base_token_symbol} ({pair.base_token_address[:20]}...)")
                    continue
                
                # Filter by age (if available)
                if pair.pair_age_hours > 0 and pair.pair_age_hours > max_age_hours:
                    continue
                
                # Add to results
                all_pairs.append(pair)
                seen_addresses.add(pair.base_token_address.lower())
                
                if len(all_pairs) >= limit:
                    break
            
            # Sort by age (newest first) if age data available
            all_pairs.sort(key=lambda p: p.pair_age_hours if p.pair_age_hours > 0 else 999)
            
            logger.info(f"Found {len(all_pairs)} new pairs matching criteria")
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
        """Get latest pairs from profile pages (NEW SOURCE!)"""
        import sys
        print("DEBUG [_get_latest_pairs]: Method entered!", file=sys.stdout, flush=True)
        pairs = []
        try:
            # Try to get pairs from popular chains' latest listings
            # DexScreener shows latest pairs on chain-specific pages
            target_chains = chains if chains else ["solana", "ethereum", "bsc"]
            print(f"DEBUG [_get_latest_pairs]: target_chains = {target_chains}", file=sys.stdout, flush=True)
            
            for chain in target_chains[:3]:  # Limit to 3 chains
                try:
                    print(f"DEBUG [_get_latest_pairs]: Searching for '{chain} launch'...", file=sys.stdout, flush=True)
                    # Search for very recent tokens with generic but recent keywords
                    # Add timeout to individual search
                    success, results = await asyncio.wait_for(
                        self.search_pairs(f"{chain} launch"),
                        timeout=10.0
                    )
                    print(f"DEBUG [_get_latest_pairs]: Search returned success={success}, {len(results) if results else 0} results", file=sys.stdout, flush=True)
                    if success and results:
                        pairs.extend(results[:20])  # Take top 20 from each
                    await asyncio.sleep(0.5)
                except asyncio.TimeoutError:
                    print(f"DEBUG [_get_latest_pairs]: Search for '{chain} launch' TIMEOUT", file=sys.stdout, flush=True)
                    continue
                except Exception as e:
                    print(f"DEBUG [_get_latest_pairs]: Search for '{chain} launch' ERROR: {e}", file=sys.stdout, flush=True)
                    continue
        except Exception as e:
            print(f"DEBUG [_get_latest_pairs]: Overall error: {e}", file=sys.stdout, flush=True)
            logger.debug(f"Error getting latest pairs: {e}")
        
        print(f"DEBUG [_get_latest_pairs]: Returning {len(pairs)} pairs", file=sys.stdout, flush=True)
        return pairs
    
    async def _get_trending_pairs(self, chains: Optional[List[str]] = None) -> List[DexPair]:
        """Get trending/boosted tokens (NEW SOURCE!)"""
        pairs = []
        try:
            # Search for tokens with high momentum keywords
            momentum_keywords = ["100x", "1000x", "pump", "rocket", "exploding", "viral"]
            
            for keyword in momentum_keywords[:3]:  # Limit to 3 keywords
                try:
                    success, results = await self.search_pairs(keyword)
                    if success and results:
                        # Filter for tokens with actual momentum (high volume)
                        momentum_pairs = [p for p in results if p.volume_24h > 5000]
                        pairs.extend(momentum_pairs[:10])  # Top 10 per keyword
                    await asyncio.sleep(0.5)
                except:
                    continue
        except Exception as e:
            logger.debug(f"Error getting trending pairs: {e}")
        
        return pairs
    
    async def get_trending_pairs(self, limit: int = 50) -> Tuple[bool, List[DexPair]]:
        """
        Get trending pairs (high volume/velocity)
        
        Note: This requires monitoring the DexScreener website or using paid API
        For free tier, we'll filter by volume/price change
        """
        logger.warning("Trending pairs requires premium features or web scraping")
        return True, []
    
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

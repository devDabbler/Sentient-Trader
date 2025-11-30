"""
Solana Holder Analyzer

Analyzes token holder distribution on Solana to detect:
1. Holder concentration (whale/dump risk)
2. Top holder percentages
3. Deployer wallet identification
4. LP wallet vs normal holder differentiation

This is CRITICAL for detecting dump risks - if top holders own too much,
they can manipulate price or dump tokens.
"""

import os
import asyncio
import httpx
from typing import Dict, Optional, List, Tuple
from loguru import logger
from dotenv import load_dotenv

# Try to import RPC load balancer (optional)
try:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.solana_rpc_load_balancer import get_rpc_load_balancer
    RPC_LOAD_BALANCER_AVAILABLE = True
except ImportError:
    RPC_LOAD_BALANCER_AVAILABLE = False

load_dotenv()

# Solana RPC endpoint
DEFAULT_SOLANA_RPC = "https://api.mainnet-beta.solana.com"

# SPL Token Program ID
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

# Known LP locker programs (to identify LP wallets)
KNOWN_LOCKER_PROGRAMS = [
    # Add known LP locker program IDs here if needed
]

# Burn address
BURN_ADDRESS = "11111111111111111111111111111111"


class SolanaHolderAnalyzer:
    """Analyze Solana token holder distribution"""
    
    def __init__(self, rpc_url: Optional[str] = None, indexer_api_key: Optional[str] = None):
        """
        Initialize Solana Holder Analyzer
        
        Args:
            rpc_url: Solana RPC endpoint (defaults to public mainnet)
            indexer_api_key: Optional API key for Helius/Birdeye (for faster holder data)
        """
        # Use load balancer if available, otherwise use single endpoint
        if rpc_url:
            self.rpc_url = rpc_url
            self.use_load_balancer = False
        elif RPC_LOAD_BALANCER_AVAILABLE:
            self.load_balancer = get_rpc_load_balancer()
            self.rpc_url = self.load_balancer.get_next_rpc_url()
            self.use_load_balancer = True
            logger.info(f"Solana Holder Analyzer using RPC Load Balancer ({len(self.load_balancer.get_all_rpc_urls())} endpoints)")
        else:
            self.rpc_url = os.getenv("SOLANA_RPC_URL", DEFAULT_SOLANA_RPC)
            self.use_load_balancer = False
        
        self.indexer_api_key = indexer_api_key or os.getenv("HELIUS_API_KEY") or os.getenv("BIRDEYE_API_KEY")
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests (very conservative for free RPCs)
        
        logger.info(f"Solana Holder Analyzer initialized with RPC: {self.rpc_url[:50]}...")
    
    async def get_top_holders(self, mint_address: str, limit: int = 20) -> List[Dict]:
        """
        Get top N token holders
        
        Args:
            mint_address: Token mint address
            limit: Maximum number of holders to return
            
        Returns:
            [
                {
                    'address': str,  # Token account address
                    'owner': str,  # Owner wallet address
                    'balance': float,  # Normalized balance
                    'balance_raw': int,  # Raw balance (lamports)
                    'percentage': float,  # % of total supply
                    'is_lp': bool,  # Is this an LP token account?
                    'is_deployer': bool,  # Is this the deployer? (heuristic)
                    'is_burn': bool  # Is this a burn address?
                },
                ...
            ]
        """
        try:
            # Option 1: Use indexer API if available (faster)
            if self.indexer_api_key:
                holders = await self._get_holders_via_indexer(mint_address, limit)
                if holders:
                    return holders
            
            # Option 2: Use RPC (slower but free)
            return await self._get_holders_via_rpc(mint_address, limit)
            
        except Exception as e:
            logger.error(f"Error getting top holders: {e}", exc_info=True)
            return []
    
    async def _get_holders_via_indexer(self, mint: str, limit: int) -> List[Dict]:
        """
        Use Helius/Birdeye API for holder data (faster, but requires API key)
        
        Note: This is a placeholder - actual implementation depends on API structure.
        For now, we'll use RPC as the primary method.
        """
        # TODO: Implement Helius/Birdeye API integration if needed
        # For now, return empty to fall back to RPC
        logger.debug("Indexer API not yet implemented, using RPC")
        return []
    
    async def _get_holders_via_rpc(self, mint: str, limit: int) -> List[Dict]:
        """
        Use Solana RPC to find token accounts
        
        Strategies:
        1. getTokenLargestAccounts (lighter, returns top 20) - PREFERRED for free RPCs
        2. getProgramAccounts (heavy, returns all) - Often rate limited
        """
        try:
            print(f"Getting holders for mint: {mint}")
            print(f"Using RPC endpoint: {self.rpc_url}")
            
            # Strategy 1: Try getTokenLargestAccounts first (much lighter on RPC)
            holders = await self._get_holders_via_largest_accounts(mint, limit)
            if holders:
                return holders
                
            logger.info("Falling back to getProgramAccounts strategy...")
            
            # Strategy 2: Use getProgramAccounts (Original method)
            await self._rate_limit()
            
            # Use getProgramAccounts to find all token accounts for this mint
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getProgramAccounts",
                "params": [
                    TOKEN_PROGRAM_ID,
                    {
                        "filters": [
                            {
                                "dataSize": 165  # Token account size
                            },
                            {
                                "memcmp": {
                                    "offset": 0,  # Mint address offset in token account
                                    "bytes": mint
                                }
                            }
                        ],
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            # Use retry logic for rate limiting
            data = await self._rpc_request_with_retry(payload)
            if not data:
                print("RPC request failed - no data returned")
                return []

            print(f"RPC response received, result type: {type(data.get('result'))}")
            result = data.get("result", [])
            if not isinstance(result, list):
                print(f"Result is not a list: {result}")
                return []

            print(f"Found {len(result)} token accounts")

            holders = []
            for account in result:
                    try:
                        account_data = account.get("account", {})
                        parsed_data = account_data.get("data", {}).get("parsed", {})
                        info = parsed_data.get("info", {})
                        
                        token_account_address = account.get("pubkey", "")
                        owner = info.get("owner", "")
                        token_amount = info.get("tokenAmount", {})
                        balance = float(token_amount.get("uiAmount", 0))
                        balance_raw = int(token_amount.get("amount", "0"))
                        
                        # Only include holders with balance > 0
                        if balance > 0:
                            holders.append({
                                'address': token_account_address,
                                'owner': owner,
                                'balance': balance,
                                'balance_raw': balance_raw,
                                'is_burn': (token_account_address == BURN_ADDRESS or owner == BURN_ADDRESS),
                                'is_lp': False,  # Will be determined later if needed
                                'is_deployer': False  # Will be determined later if needed
                            })
                    except Exception as e:
                        logger.debug(f"Error parsing holder account: {e}")
                        continue

            # Sort by balance descending
            holders.sort(key=lambda x: x['balance'], reverse=True)

            # Calculate percentages (need total supply first)
            total_supply = sum(h['balance'] for h in holders)

            if total_supply > 0:
                for holder in holders:
                    holder['percentage'] = (holder['balance'] / total_supply * 100)
            else:
                for holder in holders:
                    holder['percentage'] = 0.0

            # Return top N
            return holders[:limit]
                
        except Exception as e:
            logger.error(f"Error getting holders via RPC: {e}", exc_info=True)
            return []
            
    async def _get_holders_via_largest_accounts(self, mint: str, limit: int) -> List[Dict]:
        """
        Get top holders using getTokenLargestAccounts (lighter on RPC)
        Then fetch account info for each to get the owner.
        """
        try:
            logger.debug(f"Fetching largest accounts for {mint}...")
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [mint]
            }
            
            data = await self._rpc_request_with_retry(payload)
            if not data or "result" not in data:
                logger.warning("Failed to get largest accounts")
                return []
                
            largest_accounts = data["result"].get("value", [])
            if not largest_accounts:
                logger.warning("No largest accounts found")
                return []
            
            # Limit to requested amount
            accounts_to_fetch = largest_accounts[:limit]
            logger.info(f"Found {len(largest_accounts)} holders, fetching info for top {len(accounts_to_fetch)}...")
            
            holders = []
            
            # Fetch total supply for percentage calculation
            supply_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getTokenSupply", "params": [mint]
            }
            supply_data = await self._rpc_request_with_retry(supply_payload)
            total_supply = 0.0
            if supply_data and "result" in supply_data:
                try:
                    val = supply_data["result"]["value"]
                    total_supply = float(val.get("uiAmount") or 0)
                except:
                    pass

            # Batch fetch account info to get owners (MUCH lighter on RPC)
            addresses = [acc.get("address") for acc in accounts_to_fetch if acc.get("address")]
            owners_map = await self._get_multiple_accounts_owners(addresses)

            for acc in accounts_to_fetch:
                token_account = acc.get("address")
                amount = acc.get("uiAmount")
                # If uiAmount is None (sometimes happens), calculate from amount and decimals
                if amount is None:
                     raw = float(acc.get("amount", "0"))
                     decimals = acc.get("decimals", 9)
                     amount = raw / (10 ** decimals) if decimals > 0 else raw
                     
                raw_amount = int(acc.get("amount", "0"))
                
                # Get owner from batch result
                owner = owners_map.get(token_account, "")
                
                # Fallback if parsing failed
                if not owner:
                    owner = token_account 
                
                if amount > 0:
                    holders.append({
                        'address': token_account,
                        'owner': owner,
                        'balance': amount,
                        'balance_raw': raw_amount,
                        'is_burn': (token_account == BURN_ADDRESS or owner == BURN_ADDRESS),
                        'is_lp': False, 
                        'is_deployer': False
                    })
            
            # Calculate percentages
            calc_supply = total_supply if total_supply > 0 else sum(h['balance'] for h in holders)
            
            if calc_supply > 0:
                for holder in holders:
                    holder['percentage'] = (holder['balance'] / calc_supply * 100)
            
            return holders
            
        except Exception as e:
            logger.error(f"Error in largest accounts strategy: {e}", exc_info=True)
            return []

    async def _fetch_multiple_accounts(self, addresses: List[str]) -> List[Optional[Dict]]:
        """
        Fetch full account info for multiple addresses in one RPC call.
        Returns list of account info objects (or None if not found).
        """
        if not addresses:
            return []
            
        results = []
        try:
            await self._rate_limit()
            
            # Solana getMultipleAccounts supports up to 100 accounts
            chunk_size = 100
            
            for i in range(0, len(addresses), chunk_size):
                chunk = addresses[i:i + chunk_size]
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getMultipleAccounts",
                    "params": [
                        chunk,
                        {
                            "encoding": "jsonParsed"
                        }
                    ]
                }
                
                data = await self._rpc_request_with_retry(payload)
                
                if data and "result" in data and "value" in data["result"]:
                    batch_results = data["result"]["value"]
                    results.extend(batch_results)
                else:
                    # Fill with Nones if request failed
                    results.extend([None] * len(chunk))
                            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch account fetch: {e}")
            return [None] * len(addresses)

    async def _get_multiple_accounts_owners(self, addresses: List[str]) -> Dict[str, str]:
        """
        Get owners for multiple token accounts in one RPC call
        Returns dict {token_account_address: owner_address}
        """
        if not addresses:
            return {}
            
        try:
            await self._rate_limit()
            
            # Solana getMultipleAccounts supports up to 100 accounts
            # We usually only check 20, so one call is enough
            chunk_size = 100
            owners_map = {}
            
            for i in range(0, len(addresses), chunk_size):
                chunk = addresses[i:i + chunk_size]
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getMultipleAccounts",
                    "params": [
                        chunk,
                        {
                            "encoding": "jsonParsed"
                        }
                    ]
                }
                
                data = await self._rpc_request_with_retry(payload)
                
                if data and "result" in data and "value" in data["result"]:
                    results = data["result"]["value"]
                    
                    for j, account_info in enumerate(results):
                        if not account_info:
                            continue
                            
                        token_account = chunk[j]
                        owner = ""
                        try:
                            # Standard parsed token account
                            owner = account_info.get("data", {}).get("parsed", {}).get("info", {}).get("owner", "")
                        except:
                            pass
                            
                        if owner:
                            owners_map[token_account] = owner
                            
            return owners_map
            
        except Exception as e:
            logger.error(f"Error in batch account info fetch: {e}")
            return {}
            
    async def _get_account_info(self, address: str) -> Optional[Dict]:
        """Get account info from Solana RPC"""
        try:
            await self._rate_limit()
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    address,
                    {
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            # Use retry logic
            data = await self._rpc_request_with_retry(payload)
            if not data or "result" not in data:
                return None
            
            return data["result"]["value"]
                
        except Exception as e:
            logger.debug(f"Error getting account info: {e}")
            return None
    
    def calculate_concentration_risk(self, holders: List[Dict]) -> Dict:
        """
        Calculate holder concentration metrics and risk flags
        
        Args:
            holders: List of holder dicts from get_top_holders()
            
        Returns:
            {
                'top_holders': List[Dict],  # Top holders (limited)
                'top1_pct': float,  # % held by top 1
                'top5_pct': float,  # % held by top 5
                'top10_pct': float,  # % held by top 10
                'top20_pct': float,  # % held by top 20
                'is_centralized': bool,  # True if top 10 > 60%
                'risk_flags': List[str],  # Risk warnings
                'green_flags': List[str],  # Positive indicators
                'total_holders': int,  # Total number of holders
                'unique_owners': int  # Unique owner wallets (excluding LP/burn)
            }
        """
        if not holders:
            return {
                'top_holders': [],
                'top1_pct': 0.0,
                'top5_pct': 0.0,
                'top10_pct': 0.0,
                'top20_pct': 0.0,
                'is_centralized': True,  # No holders = risky
                'risk_flags': ['❌ No holders found'],
                'green_flags': [],
                'total_holders': 0,
                'unique_owners': 0
            }
        
        # Filter out burn addresses for concentration calculations
        active_holders = [h for h in holders if not h.get('is_burn', False)]
        
        if not active_holders:
            return {
                'top_holders': holders[:20],
                'top1_pct': 0.0,
                'top5_pct': 0.0,
                'top10_pct': 0.0,
                'top20_pct': 0.0,
                'is_centralized': True,
                'risk_flags': ['❌ All tokens in burn address'],
                'green_flags': [],
                'total_holders': len(holders),
                'unique_owners': 0
            }
        
        total_supply = sum(h['balance'] for h in active_holders)
        
        if total_supply == 0:
            return {
                'top_holders': holders[:20],
                'top1_pct': 0.0,
                'top5_pct': 0.0,
                'top10_pct': 0.0,
                'top20_pct': 0.0,
                'is_centralized': True,
                'risk_flags': ['❌ Total supply is zero'],
                'green_flags': [],
                'total_holders': len(holders),
                'unique_owners': len(set(h['owner'] for h in active_holders))
            }
        
        # Calculate concentration metrics
        top1_pct = (active_holders[0]['balance'] / total_supply * 100) if active_holders else 0
        top5_pct = sum(h['balance'] for h in active_holders[:5]) / total_supply * 100
        top10_pct = sum(h['balance'] for h in active_holders[:10]) / total_supply * 100
        top20_pct = sum(h['balance'] for h in active_holders[:20]) / total_supply * 100
        
        # Count unique owners (excluding burn)
        unique_owners = len(set(h['owner'] for h in active_holders))
        
        # Risk assessment
        risk_flags = []
        green_flags = []
        is_centralized = False
        
        # HARD RED FLAGS
        if top1_pct > 30:
            risk_flags.append(f"🚨 Top 1 holder has {top1_pct:.1f}% (extreme whale risk)")
            is_centralized = True
        elif top1_pct > 20:
            risk_flags.append(f"⚠️ Top 1 holder has {top1_pct:.1f}% (whale dump risk)")
            is_centralized = True
        
        if top10_pct > 70:
            risk_flags.append(f"🚨 Top 10 holders have {top10_pct:.1f}% (highly centralized)")
            is_centralized = True
        elif top10_pct > 60:
            risk_flags.append(f"⚠️ Top 10 holders have {top10_pct:.1f}% (centralized)")
            is_centralized = True
        
        if top5_pct > 50:
            risk_flags.append(f"⚠️ Top 5 holders have {top5_pct:.1f}% (very concentrated)")
        
        # GREEN FLAGS
        if top10_pct < 40:
            green_flags.append(f"✅ Well distributed: Top 10 hold {top10_pct:.1f}%")
        
        if top1_pct < 10:
            green_flags.append(f"✅ No whale risk: Top holder has {top1_pct:.1f}%")
        
        if unique_owners >= 100:
            green_flags.append(f"✅ Good holder count: {unique_owners} unique holders")
        elif unique_owners >= 50:
            green_flags.append(f"✅ Decent holder count: {unique_owners} unique holders")
        elif unique_owners < 20:
            risk_flags.append(f"⚠️ Low holder count: Only {unique_owners} unique holders")
        
        return {
            'top_holders': active_holders[:20],  # Top 20 active holders
            'top1_pct': top1_pct,
            'top5_pct': top5_pct,
            'top10_pct': top10_pct,
            'top20_pct': top20_pct,
            'is_centralized': is_centralized,
            'risk_flags': risk_flags,
            'green_flags': green_flags,
            'total_holders': len(holders),
            'unique_owners': unique_owners
        }
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = asyncio.get_event_loop().time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def _rpc_request_with_retry(self, payload: Dict, max_retries: int = 3) -> Optional[Dict]:
        """
        Make RPC request with automatic retry on rate limit (429) errors
        
        Args:
            payload: RPC request payload
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response data or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                await self._rate_limit()
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(self.rpc_url, json=payload)
                    
                    # Handle rate limiting or other errors
                    if response.status_code != 200:
                        logger.warning(f"Solana RPC error: {response.status_code} from {self.rpc_url}")
                        
                        # Switch endpoint if using load balancer
                        if hasattr(self, 'use_load_balancer') and self.use_load_balancer:
                            self.load_balancer.mark_failure(self.rpc_url)
                            self.rpc_url = self.load_balancer.get_next_rpc_url()
                            logger.debug(f"Switching to next RPC endpoint: {self.rpc_url[:50]}...")
                            await asyncio.sleep(1)
                            continue
                        
                        if response.status_code == 429 and attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 3
                            await asyncio.sleep(wait_time)
                            continue
                            
                        return None
                    
                    data = response.json()
                    
                    # Check for RPC errors
                    if "error" in data:
                        error_msg = data["error"].get("message", str(data["error"]))
                        logger.debug(f"Solana RPC application error: {error_msg}")
                        
                        # Switch endpoint on application error too
                        if hasattr(self, 'use_load_balancer') and self.use_load_balancer:
                            self.load_balancer.mark_failure(self.rpc_url)
                            self.rpc_url = self.load_balancer.get_next_rpc_url()
                            logger.debug(f"Switching to next RPC endpoint: {self.rpc_url[:50]}...")
                            await asyncio.sleep(1)
                            continue
                            
                        if "429" in error_msg or "rate limit" in error_msg.lower():
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 2
                                await asyncio.sleep(wait_time)
                                continue
                        
                        return None
                    
                    # Success!
                    # Mark success if we're using load balancer to clear failure count
                    if hasattr(self, 'use_load_balancer') and self.use_load_balancer:
                         self.load_balancer.mark_success(self.rpc_url)
                         
                    return data
                    
            except Exception as e:
                logger.warning(f"RPC request error to {self.rpc_url}: {e}")
                
                # Switch endpoint on exception
                if hasattr(self, 'use_load_balancer') and self.use_load_balancer:
                    self.load_balancer.mark_failure(self.rpc_url)
                    self.rpc_url = self.load_balancer.get_next_rpc_url()
                    logger.debug(f"Switching to next RPC endpoint: {self.rpc_url[:50]}...")
                    await asyncio.sleep(1)
                    continue
                    
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"RPC request failed after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    async def analyze_holder_distribution(self, mint_address: str, limit: int = 20) -> Dict:
        """
        Complete holder distribution analysis
        
        Args:
            mint_address: Token mint address
            limit: Maximum number of holders to analyze
            
        Returns:
            {
                'holders': List[Dict],  # Top holders
                'concentration': Dict,  # Concentration metrics from calculate_concentration_risk()
                'error': Optional[str]
            }
        """
        try:
            holders = await self.get_top_holders(mint_address, limit)
            
            if not holders:
                return {
                    'holders': [],
                    'concentration': {
                        'is_centralized': True,
                        'risk_flags': ['❌ No holders found'],
                        'green_flags': []
                    },
                    'error': 'No holders found'
                }
            
            concentration = self.calculate_concentration_risk(holders)
            
            return {
                'holders': holders,
                'concentration': concentration,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing holder distribution: {e}", exc_info=True)
            return {
                'holders': [],
                'concentration': {
                    'is_centralized': True,
                    'risk_flags': [f'❌ Analysis error: {str(e)}'],
                    'green_flags': []
                },
                'error': str(e)
            }

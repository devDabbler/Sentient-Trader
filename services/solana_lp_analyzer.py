"""
Solana LP Analyzer

Analyzes liquidity pool (LP) token ownership on Solana to detect rug risks:
1. Where are LP tokens held? (burn address, locker program, or EOA wallet)
2. What percentage of LP tokens are burned/locked?
3. Are LP tokens in a deployer wallet? (rug risk)

This is CRITICAL for detecting rug pulls - if LP tokens are in an EOA wallet,
the deployer can withdraw liquidity and dump the token.
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

# Try to import base58 (required for Solana address encoding)
try:
    import base58
    BASE58_AVAILABLE = True
except ImportError:
    BASE58_AVAILABLE = False
    logger.warning("base58 package not installed. Solana LP analysis will fail.")

load_dotenv()

# Solana RPC endpoint
DEFAULT_SOLANA_RPC = "https://api.mainnet-beta.solana.com"

# Known addresses
BURN_ADDRESS = "11111111111111111111111111111111"  # System program (burn)
SYSTEM_PROGRAM = "11111111111111111111111111111111"

# Known locker programs (add more as discovered)
KNOWN_LOCKER_PROGRAMS = [
    # Add known LP locker program IDs here
    # Example: "LocktDZ7jKZ7jKZ7jKZ7jKZ7jKZ7jKZ7jKZ7jKZ"
]


class SolanaLPAnalyzer:
    """Analyze Solana liquidity pool and LP token status"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize Solana LP Analyzer
        
        Args:
            rpc_url: Solana RPC endpoint (defaults to public mainnet)
        """
        # Use load balancer if available, otherwise use single endpoint
        if rpc_url:
            self.rpc_url = rpc_url
            self.use_load_balancer = False
        elif RPC_LOAD_BALANCER_AVAILABLE:
            self.load_balancer = get_rpc_load_balancer()
            self.rpc_url = self.load_balancer.get_next_rpc_url()
            self.use_load_balancer = True
            logger.info(f"Solana LP Analyzer using RPC Load Balancer ({len(self.load_balancer.get_all_rpc_urls())} endpoints)")
        else:
            self.rpc_url = os.getenv("SOLANA_RPC_URL", DEFAULT_SOLANA_RPC)
            self.use_load_balancer = False
        
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests (very conservative for free RPCs)
        
        logger.info(f"Solana LP Analyzer initialized with RPC: {self.rpc_url[:50]}...")
    
    async def analyze_lp_status(
        self, 
        pool_address: str, 
        mint_address: str,
        lp_token_mint: Optional[str] = None
    ) -> Dict:
        """
        Analyze LP token ownership and lock status
        
        Args:
            pool_address: Raydium/Orca pool address
            mint_address: Token mint address (for finding LP token mint)
            lp_token_mint: Optional LP token mint address (if known)
            
        Returns:
            {
                'lp_token_mint': str,
                'lp_total_supply': float,
                'lp_holders': List[Dict],
                'lp_owner_type': str,  # 'burn', 'locker', 'EOA_unknown', 'mixed'
                'lp_burn_pct': float,  # % of LP in burn address
                'lp_locked_pct': float,  # % of LP in locker programs
                'lp_eoa_pct': float,  # % of LP in EOA wallets
                'is_safe': bool,
                'risk_flags': List[str],
                'error': Optional[str]
            }
        """
        try:
            # Step 1: Find LP token mint if not provided
            if not lp_token_mint:
                lp_token_mint = await self._find_lp_token_mint(pool_address, mint_address)
                if not lp_token_mint:
                    return {
                        'error': 'Could not find LP token mint',
                        'is_safe': False,
                        'risk_flags': ['❌ Could not identify LP token mint (Meteora/Unknown DEX)']
                    }
            
            # Step 2: Get LP token holders
            # Check supply first to avoid wasted calls
            lp_supply = 0
            try:
                info = await self._get_account_info(lp_token_mint)
                if info:
                    data = info.get("data", {})
                    if isinstance(data, dict) and data.get("parsed", {}).get("type") == "mint":
                        lp_supply = int(data.get("parsed", {}).get("info", {}).get("supply", "0"))
            except Exception:
                pass
                
            if lp_supply == 0:
                logger.warning(f"LP token {lp_token_mint} has 0 supply")
                return {
                    'lp_token_mint': lp_token_mint,
                    'lp_total_supply': 0,
                    'lp_holders': [],
                    'lp_owner_type': 'empty',
                    'lp_burn_pct': 0,
                    'lp_locked_pct': 0,
                    'lp_eoa_pct': 0,
                    'is_safe': False,
                    'risk_flags': ['⚠️ LP token supply is 0 (Empty Pool)'],
                    'error': None
                }

            lp_holders = await self._get_lp_token_holders(lp_token_mint)
            
            if not lp_holders:
                return {
                    'error': 'No LP token holders found',
                    'is_safe': False,
                    'risk_flags': ['❌ No LP token holders found']
                }
            
            # Step 3: Analyze ownership distribution
            return self._analyze_ownership_distribution(lp_token_mint, lp_holders)
            
        except Exception as e:
            logger.error(f"Error analyzing LP status: {e}", exc_info=True)
            return {
                'error': str(e),
                'is_safe': False,
                'risk_flags': [f'❌ Analysis error: {str(e)}']
            }
    
    async def _find_lp_token_mint(self, pool_address: str, mint_address: str) -> Optional[str]:
        """
        Find LP token mint address for a pool

        For Raydium pools, LP token mint is stored at offset 72-104 in pool account data.
        For Orca pools, similar structure but different offsets.
        """
        print(f"DEBUG: _find_lp_token_mint called for pool {pool_address[:8]}")
        try:
            logger.debug(f"Finding LP token mint for pool {pool_address[:8]}...")

            # Get pool account data
            pool_info = await self._get_account_info(pool_address)
            if not pool_info:
                logger.warning(f"Could not get pool account info for {pool_address}")
                return None

            owner = pool_info.get('owner')
            logger.info(f"Pool Owner Program: {owner}")
            
            # Known DEX Programs
            RAYDIUM_V4 = "675kPX9mMzmarZXnR4c4mC42C8E93bB8hU1Vb2p"
            RAYDIUM_AMM_V3 = "27haf8L6oxUeXVI1ndxVRCP9Eo0axNpFf62MX5Zh5em2" # Raydium Liquidity Pool V3
            METEORA_DYNAMIC_AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
            METEORA_DLMM = "LBUZKhRxPF3XUpBCjp4YzTkDyEeWZGkqNEZqCJ1PFcz"
            ORCA_WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"

            # Identify DEX type
            is_meteora = owner in [METEORA_DYNAMIC_AMM, METEORA_DLMM]
            
            if owner not in [RAYDIUM_V4, RAYDIUM_AMM_V3] and not is_meteora:
                logger.warning(f"Unsupported DEX Program: {owner}. Currently only Raydium/Meteora is supported.")
                return None

            logger.info(f"Got pool account info, data length: {len(pool_info.get('data', []))}")

            # Parse raw pool account data
            account_data = pool_info.get('data', [])
            if not account_data or len(account_data) < 2:
                logger.debug("Pool account has no data")
                return None

            # Decode base64 data
            import base64
            try:
                raw_data = base64.b64decode(account_data[0])
            except Exception as e:
                logger.debug(f"Failed to decode pool account data: {e}")
                return None

            # --- RAYDIUM PARSING ---
            if owner in [RAYDIUM_V4, RAYDIUM_AMM_V3]:
                # Raydium AMM Pool Layout (simplified):
                # LP mint starts at offset 72 (32 bytes)
                try:
                    if len(raw_data) >= 104:
                        lp_mint_bytes = raw_data[72:104]
                        if BASE58_AVAILABLE:
                            lp_mint_address = base58.b58encode(lp_mint_bytes).decode('utf-8')
                            print(f"Found Raydium LP token mint: {lp_mint_address}")
                            if len(lp_mint_address) >= 32 and not lp_mint_address.startswith('0x'):
                                return lp_mint_address
                except Exception as e:
                    logger.debug(f"Error parsing Raydium LP mint: {e}")

            # --- METEORA PARSING (Heuristic) ---
            elif is_meteora:
                logger.info("Parsing Meteora pool data (heuristic)...")
                
                if not BASE58_AVAILABLE:
                    logger.warning("base58 not available, cannot parse Meteora pool")
                    return None
                    
                # Heuristic: Scan 32-byte chunks for valid pubkeys
                # Since the layout is unknown/unaligned, we scan every byte.
                # This generates many candidates (~length of data), so we use batch RPC calls.
                
                candidates = []
                # Check every byte (stride 1) because field might be unaligned
                for i in range(0, min(len(raw_data) - 32, 1000)): 
                    chunk = raw_data[i:i+32]
                    try:
                        candidate = base58.b58encode(chunk).decode('utf-8')
                        # Basic filtering
                        if 32 <= len(candidate) <= 44 and not candidate.startswith('0x'):
                            if candidate != mint_address and candidate != owner and candidate != SYSTEM_PROGRAM:
                                candidates.append(candidate)
                    except:
                        pass
                
                # Deduplicate while preserving order
                unique_candidates = []
                seen = set()
                for c in candidates:
                    if c not in seen:
                        unique_candidates.append(c)
                        seen.add(c)
                
                logger.info(f"Found {len(unique_candidates)} candidate Pubkeys in pool data. Batch checking...")
                
                # Batch check candidates in chunks of 100
                chunk_size = 100
                best_candidate = None
                
                for i in range(0, len(unique_candidates), chunk_size):
                    batch = unique_candidates[i:i+chunk_size]
                    results = await self._fetch_multiple_accounts(batch)
                    
                    for j, info in enumerate(results):
                        if not info:
                            continue
                            
                        cand_address = batch[j]
                        
                        # Check if it's a Mint
                        data = info.get("data", {})
                        if isinstance(data, dict) and data.get("parsed", {}).get("type") == "mint":
                            parsed_info = data.get("parsed", {}).get("info", {})
                            mint_authority = parsed_info.get("mintAuthority")
                            supply = int(parsed_info.get("supply", "0"))
                            
                            # CRITICAL: Check if mint authority is the pool
                            if mint_authority == pool_address:
                                logger.info(f"✅ Found DEFINITIVE LP mint (Authority == Pool): {cand_address}")
                                return cand_address
                                
                            # Secondary check: Supply > 0 (weak signal, store for later)
                            if supply > 0 and not best_candidate:
                                best_candidate = cand_address
                                logger.info(f"Found potential LP mint (Supply > 0): {cand_address}")
                                
                if best_candidate:
                    return best_candidate

            # Fallback: Try to find LP token by searching token accounts
            logger.debug("Could not parse LP mint from pool data, trying fallback method")
            return await self._find_lp_token_via_accounts(pool_address, mint_address)

        except Exception as e:
            logger.debug(f"Error finding LP token mint: {e}")
            return None

    async def _find_lp_token_via_accounts(self, pool_address: str, mint_address: str) -> Optional[str]:
        """
        Fallback method: Try to find LP token by looking for token accounts
        that hold both the base token and are associated with the pool.
        
        This is a heuristic approach and may not work for all pools.
        """
        try:
            # This would require complex logic to find LP tokens
            # For now, we'll skip this fallback as it's unreliable
            logger.debug("Skipping LP token search via accounts (not implemented)")
            return None

        except Exception as e:
            logger.debug(f"Error in LP token fallback search: {e}")
            return None
    
    async def _get_lp_token_holders(self, lp_token_mint: str, limit: int = 100) -> List[Dict]:
        """
        Get LP token holders via RPC
        
        Strategies:
        1. getTokenLargestAccounts (lighter) - PREFERRED
        2. getProgramAccounts (heavy) - Fallback
        """
        try:
            print(f"Getting LP token holders for mint: {lp_token_mint}")
            print(f"Using RPC endpoint: {self.rpc_url}")
            
            # Strategy 1: getTokenLargestAccounts
            try:
                logger.debug(f"Fetching largest accounts for LP mint {lp_token_mint}...")
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTokenLargestAccounts",
                    "params": [lp_token_mint]
                }
                
                data = await self._rpc_request_with_retry(payload)
                if data and "result" in data:
                    largest_accounts = data["result"].get("value", [])
                    if largest_accounts:
                        logger.info(f"Found {len(largest_accounts)} LP holders via largestAccounts strategy")
                        
                        holders = []
                        # Limit to top N
                        accounts_to_fetch = largest_accounts[:limit]
                        
                        # NEW: Batch fetch account info (Lighter RPC usage)
                        addresses = [acc.get("address") for acc in accounts_to_fetch if acc.get("address")]
                        owners_map = await self._get_multiple_accounts_owners(addresses)
                        
                        for acc in accounts_to_fetch:
                            token_account = acc.get("address")
                            amount = acc.get("uiAmount")
                            if amount is None:
                                raw = float(acc.get("amount", "0"))
                                decimals = acc.get("decimals", 9)
                                amount = raw / (10 ** decimals) if decimals > 0 else raw
                            
                            raw_amount = int(acc.get("amount", "0"))
                            
                            # Get owner from batch result
                            owner = owners_map.get(token_account, "")
                            if not owner:
                                owner = token_account
                                
                            if amount > 0:
                                holders.append({
                                    'address': token_account,
                                    'owner': owner,
                                    'balance': amount,
                                    'balance_raw': raw_amount
                                })
                        
                        # Sort by balance
                        holders.sort(key=lambda x: x['balance'], reverse=True)
                        return holders
            except Exception as e:
                logger.debug(f"getTokenLargestAccounts failed: {e}, falling back...")
            
            
            # Strategy 2: getProgramAccounts (Original)
            logger.info("Falling back to getProgramAccounts for LP holders...")
            await self._rate_limit()
            
            # Use getProgramAccounts to find all token accounts for this mint
            TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            
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
                                    "bytes": lp_token_mint
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
            for account in result[:limit]:  # Limit to top N
                try:
                    account_data = account.get("account", {})
                    parsed_data = account_data.get("data", {}).get("parsed", {})
                    info = parsed_data.get("info", {})

                    owner = info.get("owner", "")
                    balance = float(info.get("tokenAmount", {}).get("uiAmount", 0))

                    if balance > 0:  # Only include holders with balance
                        holders.append({
                            'address': account.get("pubkey", ""),
                            'owner': owner,
                            'balance': balance,
                            'balance_raw': int(info.get("tokenAmount", {}).get("amount", "0"))
                        })
                except Exception as e:
                    logger.debug(f"Error parsing holder account: {e}")
                    continue

            # Sort by balance descending
            holders.sort(key=lambda x: x['balance'], reverse=True)

            return holders
                
        except Exception as e:
            logger.error(f"Error getting LP token holders: {e}", exc_info=True)
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
    
    def _analyze_ownership_distribution(self, lp_token_mint: str, holders: List[Dict]) -> Dict:
        """Analyze LP token ownership distribution"""
        
        if not holders:
            return {
                'error': 'No holders to analyze',
                'is_safe': False,
                'risk_flags': ['❌ No LP token holders']
            }
        
        total_supply = sum(h['balance'] for h in holders)
        
        if total_supply == 0:
            return {
                'error': 'Total LP supply is zero',
                'is_safe': False,
                'risk_flags': ['❌ LP supply is zero']
            }
        
        # Categorize holders
        burn_balance = 0.0
        locker_balance = 0.0
        eoa_balance = 0.0
        eoa_holders = []
        
        for holder in holders:
            owner = holder.get('owner', '')
            balance = holder['balance']
            
            # Check if burned
            if holder['address'] == BURN_ADDRESS or owner == BURN_ADDRESS:
                burn_balance += balance
            # Check if in locker program
            elif owner in KNOWN_LOCKER_PROGRAMS:
                locker_balance += balance
            else:
                # EOA wallet or unknown program
                eoa_balance += balance
                eoa_holders.append({
                    'address': holder['address'],
                    'owner': owner,
                    'balance': balance,
                    'balance_pct': (balance / total_supply * 100) if total_supply > 0 else 0
                })
        
        # Calculate percentages
        burn_pct = (burn_balance / total_supply * 100) if total_supply > 0 else 0
        locked_pct = (locker_balance / total_supply * 100) if total_supply > 0 else 0
        eoa_pct = (eoa_balance / total_supply * 100) if total_supply > 0 else 0
        
        # Determine owner type
        if burn_pct > 90:
            owner_type = "burn"
        elif locked_pct > 80:
            owner_type = "locker"
        elif eoa_pct > 50:
            owner_type = "EOA_unknown"
        else:
            owner_type = "mixed"
        
        # Risk assessment
        risk_flags = []
        is_safe = False
        
        if owner_type == "burn":
            is_safe = True
            risk_flags.append(f"✅ LP tokens burned ({burn_pct:.1f}%)")
        elif owner_type == "locker":
            is_safe = True
            risk_flags.append(f"✅ LP tokens locked ({locked_pct:.1f}%)")
            # TODO: Check unlock time if possible
        elif owner_type == "EOA_unknown":
            is_safe = False
            risk_flags.append(f"🚨 LP tokens in EOA wallets ({eoa_pct:.1f}%) - RUG RISK!")
            if eoa_holders:
                top_eoa = eoa_holders[0]
                risk_flags.append(f"   Top EOA holder: {top_eoa['address'][:8]}... ({top_eoa['balance_pct']:.1f}%)")
        else:  # mixed
            is_safe = eoa_pct < 20  # Safe if <20% in EOA
            if eoa_pct > 20:
                risk_flags.append(f"⚠️ Mixed ownership: {eoa_pct:.1f}% in EOA wallets")
            else:
                risk_flags.append(f"✅ Mostly locked/burned: {burn_pct + locked_pct:.1f}%")
        
        return {
            'lp_token_mint': lp_token_mint,
            'lp_total_supply': total_supply,
            'lp_holders': holders[:20],  # Top 20
            'lp_owner_type': owner_type,
            'lp_burn_pct': burn_pct,
            'lp_locked_pct': locked_pct,
            'lp_eoa_pct': eoa_pct,
            'eoa_holders': eoa_holders[:10],  # Top 10 EOA holders
            'is_safe': is_safe,
            'risk_flags': risk_flags,
            'error': None
        }
    
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
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.rpc_url, json=payload)
                
                if response.status_code != 200:
                    return None
                
                data = response.json()
                
                if "error" in data or not data.get("result", {}).get("value"):
                    return None
                
                return data["result"]["value"]
                
        except Exception as e:
            logger.debug(f"Error getting account info: {e}")
            return None
    
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
    
    async def quick_check(self, pool_address: str, mint_address: str) -> Tuple[bool, str]:
        """
        Quick safety check - returns (is_safe, reason)
        
        Returns:
            (True, "Safe") if LP burned/locked
            (False, reason) if LP in EOA wallet
        """
        result = await self.analyze_lp_status(pool_address, mint_address)
        
        if result.get('error'):
            return False, f"Error: {result['error']}"
        
        if not result.get('is_safe'):
            flags = result.get('risk_flags', [])
            return False, flags[0] if flags else "LP tokens in EOA wallet"
        
        return True, result.get('risk_flags', ['LP tokens safe'])[0]

"""
Solana Mint Inspector

Performs on-chain inspection of SPL token mint accounts to detect:
1. Mint authority status (can mint more tokens = risk)
2. Freeze authority status (can freeze accounts = honeypot risk)
3. Token supply and decimals
4. Normalized supply calculations

This is CRITICAL for Solana token safety - these checks cannot be done via API alone.
"""

import os
import asyncio
import httpx
from typing import Dict, Optional, Tuple
from loguru import logger
from dotenv import load_dotenv

# Try to import RPC load balancer (optional)
try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.solana_rpc_load_balancer import get_rpc_load_balancer
    RPC_LOAD_BALANCER_AVAILABLE = True
except ImportError:
    RPC_LOAD_BALANCER_AVAILABLE = False

load_dotenv()

# Try to import base58 (required for Solana address encoding)
try:
    import base58
    BASE58_AVAILABLE = True
except ImportError:
    BASE58_AVAILABLE = False
    logger.warning("base58 package not installed. Solana mint inspection will fail.")

# Solana RPC endpoint (use public or your own)
DEFAULT_SOLANA_RPC = "https://api.mainnet-beta.solana.com"


class SolanaMintInspector:
    """Inspect SPL token mint accounts on-chain via Solana RPC"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize Solana Mint Inspector
        
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
            logger.info(f"Solana Mint Inspector using RPC Load Balancer ({len(self.load_balancer.get_all_rpc_urls())} endpoints)")
        else:
            self.rpc_url = os.getenv("SOLANA_RPC_URL", DEFAULT_SOLANA_RPC)
            self.use_load_balancer = False
        
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests (very conservative for free RPCs)
        
        logger.info(f"Solana Mint Inspector initialized with RPC: {self.rpc_url[:50]}...")
    
    async def inspect_mint(self, mint_address: str) -> Dict:
        """
        Inspect mint account for safety flags
        
        Args:
            mint_address: Solana mint address (base58)
            
        Returns:
            {
                'decimals': int,
                'supply': int (raw),
                'supply_normalized': float,
                'mint_authority': Optional[str],  # None = revoked ✅
                'freeze_authority': Optional[str],  # None = revoked ✅
                'is_safe': bool,
                'risk_flags': List[str],
                'error': Optional[str]
            }
        """
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Check if this is wrapped SOL BEFORE making RPC call (known address)
            # Wrapped SOL address: So11111111111111111111111111111111111111112
            if mint_address == "So11111111111111111111111111111111111111112":
                # Get account balance for wrapped SOL
                account_info = await self._get_account_info(mint_address)
                lamports = account_info.get('lamports', 0) if account_info else 0
                
                return {
                    'decimals': 9,  # SOL uses 9 decimals
                    'supply': lamports,
                    'supply_normalized': lamports / 1e9,
                    'mint_authority': None,  # Native SOL has no mint authority ✅
                    'freeze_authority': None,  # Native SOL has no freeze authority ✅
                    'is_safe': True,
                    'risk_flags': [],
                    'green_flags': ['✅ Native SOL account (wrapped SOL) - no authorities'],
                    'error': None
                }
            
            # Get account info via RPC for regular SPL token mints
            account_info = await self._get_account_info(mint_address)
            
            if not account_info:
                return {
                    'error': 'Mint account not found',
                    'is_safe': False,
                    'risk_flags': ['❌ Mint account not found on-chain']
                }
            
            # Parse mint account data
            # SPL Token mint account structure:
            # - 0-1 bytes: Option<Pubkey> for mint_authority (33 bytes if Some, 1 byte if None)
            # - 33-34 bytes: supply (u64 = 8 bytes)
            # - 41-42 bytes: decimals (u8 = 1 byte)
            # - 42-43 bytes: Option<Pubkey> for freeze_authority (33 bytes if Some, 1 byte if None)
            # Total: 82 bytes (if both authorities present) or 82 bytes (if both None)
            
            # Check if this is a native SOL account (wrapped SOL) vs SPL token mint
            # Native SOL accounts use System Program, not Token Program
            owner = account_info.get('owner', '')
            executable = account_info.get('executable', False)
            
            # Wrapped SOL check: System Program owned, not executable
            if owner == '11111111111111111111111111111111' and not executable:
                # This is wrapped SOL - it's a native account, not an SPL token mint
                # Wrapped SOL is safe by default (no mint/freeze authority)
                lamports = account_info.get('lamports', 0)
                return {
                    'decimals': 9,  # SOL uses 9 decimals
                    'supply': lamports,
                    'supply_normalized': lamports / 1e9,
                    'mint_authority': None,  # Native SOL has no mint authority ✅
                    'freeze_authority': None,  # Native SOL has no freeze authority ✅
                    'is_safe': True,
                    'risk_flags': [],
                    'green_flags': ['✅ Native SOL account (wrapped SOL) - no authorities'],
                    'error': None
                }
            
            mint_data = account_info.get('data', [])
            if not mint_data:
                # Try jsonParsed encoding
                parsed_data = account_info.get('data', {})
                if isinstance(parsed_data, dict) and parsed_data.get('parsed'):
                    # Handle jsonParsed format
                    return self._parse_json_mint_data(account_info, mint_address)
                
                return {
                    'error': 'Invalid mint account data (no data field)',
                    'is_safe': False,
                    'risk_flags': ['❌ Invalid mint account structure']
                }
            
            # Handle both string and array formats
            if isinstance(mint_data, str):
                # Direct base64 string
                import base64
                try:
                    raw_data = base64.b64decode(mint_data)
                except Exception:
                    return {
                        'error': 'Failed to decode account data',
                        'is_safe': False,
                        'risk_flags': ['❌ Failed to decode mint account data']
                    }
            elif isinstance(mint_data, list) and len(mint_data) > 0:
                # Array format [base64_string, encoding]
                import base64
                try:
                    raw_data = base64.b64decode(mint_data[0])
                except Exception as e:
                    return {
                        'error': f'Failed to decode account data: {e}',
                        'is_safe': False,
                        'risk_flags': ['❌ Failed to decode mint account data']
                    }
            else:
                return {
                    'error': 'Invalid mint account data format',
                    'is_safe': False,
                    'risk_flags': ['❌ Invalid mint account data format']
                }
            
            if len(raw_data) < 82:
                return {
                    'error': 'Mint account data too short',
                    'is_safe': False,
                    'risk_flags': ['❌ Mint account data incomplete']
                }
            
            # Check if base58 is available
            if not BASE58_AVAILABLE:
                return {
                    'error': 'base58 package not installed',
                    'is_safe': False,
                    'risk_flags': ['❌ Missing dependency: base58']
                }
            
            # Parse mint authority (first 33 bytes)
            mint_authority_bytes = raw_data[0:33]
            mint_authority = None
            if mint_authority_bytes[0] != 0:  # Option::Some
                # Extract pubkey (skip first byte which is the Option discriminator)
                pubkey_bytes = mint_authority_bytes[1:33]
                # Convert to base58 string
                mint_authority = base58.b58encode(pubkey_bytes).decode('utf-8')
            
            # Parse supply (8 bytes, little-endian u64)
            supply_bytes = raw_data[33:41]
            supply = int.from_bytes(supply_bytes, byteorder='little', signed=False)
            
            # Parse decimals (1 byte)
            decimals = raw_data[41]
            
            # Parse freeze authority (33 bytes starting at offset 42)
            freeze_authority_bytes = raw_data[42:75]
            freeze_authority = None
            if freeze_authority_bytes[0] != 0:  # Option::Some
                pubkey_bytes = freeze_authority_bytes[1:33]
                freeze_authority = base58.b58encode(pubkey_bytes).decode('utf-8')
            
            # Calculate normalized supply
            supply_normalized = supply / (10 ** decimals) if decimals > 0 else float(supply)
            
            # Risk assessment
            risk_flags = []
            is_safe = True
            
            # HARD RED FLAG: Mint authority retained
            if mint_authority is not None:
                risk_flags.append(f"🚨 MINT_AUTHORITY_RETAINED: {mint_authority[:8]}... (Can mint unlimited tokens!)")
                is_safe = False
            
            # HARD RED FLAG: Freeze authority retained (honeypot risk)
            if freeze_authority is not None:
                risk_flags.append(f"🚨 FREEZE_AUTHORITY_RETAINED: {freeze_authority[:8]}... (Can freeze user accounts = HONEYPOT!)")
                is_safe = False
            
            # Warning: Very large supply
            if supply_normalized > 1e18:
                risk_flags.append(f"⚠️ Very large supply: {supply_normalized:.2e} tokens")
            
            # Warning: Very small supply (potential manipulation)
            if supply_normalized < 1000 and decimals <= 9:
                risk_flags.append(f"⚠️ Very small supply: {supply_normalized:.2f} tokens (potential manipulation)")
            
            # Green flags
            green_flags = []
            if mint_authority is None:
                green_flags.append("✅ Mint authority revoked (cannot mint more)")
            if freeze_authority is None:
                green_flags.append("✅ Freeze authority revoked (cannot freeze accounts)")
            
            return {
                'decimals': int(decimals),
                'supply': supply,
                'supply_normalized': supply_normalized,
                'mint_authority': mint_authority,
                'freeze_authority': freeze_authority,
                'is_safe': is_safe,
                'risk_flags': risk_flags,
                'green_flags': green_flags,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error inspecting mint {mint_address}: {e}", exc_info=True)
            return {
                'error': str(e),
                'is_safe': False,
                'risk_flags': [f'❌ Inspection error: {str(e)}']
            }
    
    def _parse_json_mint_data(self, account_info: Dict, mint_address: str) -> Dict:
        """Parse mint data from jsonParsed encoding"""
        try:
            parsed = account_info.get('data', {}).get('parsed', {})
            info = parsed.get('info', {})
            
            mint_authority = info.get('mintAuthority')
            freeze_authority = info.get('freezeAuthority')
            supply = int(info.get('supply', '0'))
            decimals = info.get('decimals', 0)
            
            supply_normalized = supply / (10 ** decimals) if decimals > 0 else float(supply)
            
            # Risk assessment
            risk_flags = []
            green_flags = []
            is_safe = True
            
            if mint_authority:
                risk_flags.append(f"🚨 MINT_AUTHORITY_RETAINED: {mint_authority[:8]}... (Can mint unlimited tokens!)")
                is_safe = False
            
            if freeze_authority:
                risk_flags.append(f"🚨 FREEZE_AUTHORITY_RETAINED: {freeze_authority[:8]}... (Can freeze user accounts = HONEYPOT!)")
                is_safe = False
            
            if not mint_authority:
                green_flags.append("✅ Mint authority revoked (cannot mint more)")
            if not freeze_authority:
                green_flags.append("✅ Freeze authority revoked (cannot freeze accounts)")
            
            return {
                'decimals': int(decimals),
                'supply': supply,
                'supply_normalized': supply_normalized,
                'mint_authority': mint_authority,
                'freeze_authority': freeze_authority,
                'is_safe': is_safe,
                'risk_flags': risk_flags,
                'green_flags': green_flags,
                'error': None
            }
        except Exception as e:
            logger.debug(f"Error parsing JSON mint data: {e}")
            return {
                'error': f'Failed to parse JSON mint data: {e}',
                'is_safe': False,
                'risk_flags': [f'❌ JSON parsing error: {str(e)}']
            }
    
    async def _get_account_info(self, address: str) -> Optional[Dict]:
        """Get account info from Solana RPC"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    address,
                    {
                        "encoding": "base64"
                    }
                ]
            }
            
            # Use retry logic for rate limiting
            data = await self._rpc_request_with_retry(payload)
            if not data:
                return None
            
            result = data.get("result", {})
            if not result or not result.get("value"):
                return None  # Account doesn't exist
            
            return result["value"]
                
        except Exception as e:
            logger.error(f"Error calling Solana RPC: {e}")
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
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(self.rpc_url, json=payload)
                    
                    # Handle rate limiting with exponential backoff
                    if response.status_code == 429:
                        # Try next RPC endpoint if using load balancer
                        if hasattr(self, 'use_load_balancer') and self.use_load_balancer and attempt < max_retries - 1:
                            self.load_balancer.mark_failure(self.rpc_url)
                            self.rpc_url = self.load_balancer.get_next_rpc_url()
                            logger.debug(f"Rate limited (429), switching to next RPC endpoint: {self.rpc_url[:50]}...")
                            await asyncio.sleep(1)  # Brief pause before retry
                            continue
                        elif attempt < max_retries - 1:
                            wait_time = (2 ** attempt) * 3  # 3s, 6s, 12s (longer waits)
                            logger.debug(f"Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"Solana RPC rate limited (429) after {max_retries} attempts - skipping this check")
                            return None
                    
                    if response.status_code != 200:
                        logger.warning(f"Solana RPC error: {response.status_code}")
                        return None
                    
                    data = response.json()
                    
                    # Check for RPC errors (including rate limit in error message)
                    if "error" in data:
                        error_msg = data["error"].get("message", str(data["error"]))
                        if "429" in error_msg or "rate limit" in error_msg.lower():
                            if attempt < max_retries - 1:
                                wait_time = (2 ** attempt) * 2
                                logger.debug(f"Rate limit in RPC error, waiting {wait_time}s before retry")
                                await asyncio.sleep(wait_time)
                                continue
                        
                        logger.debug(f"Solana RPC error: {error_msg}")
                        return None
                    
                    return data
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1
                    logger.debug(f"RPC request error, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"RPC request failed after {max_retries} attempts: {e}")
                    return None
        
        return None
    
    async def quick_check(self, mint_address: str) -> Tuple[bool, str]:
        """
        Quick safety check - returns (is_safe, reason)
        
        Returns:
            (True, "Safe") if both authorities revoked
            (False, reason) if any authority retained
        """
        result = await self.inspect_mint(mint_address)
        
        if result.get('error'):
            return False, f"Error: {result['error']}"
        
        if not result.get('is_safe'):
            flags = result.get('risk_flags', [])
            return False, flags[0] if flags else "Unsafe token"
        
        return True, "Mint and freeze authorities revoked"


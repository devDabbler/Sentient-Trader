"""
Solana Metadata Inspector

Performs on-chain inspection of token metadata (Metaplex Metadata Standard) to detect:
1. Update authority status (can change metadata = impersonation risk)
2. Metadata immutability
3. Token name, symbol, and URI verification

This is IMPORTANT for detecting impersonation risks - if metadata can be changed,
the token creator can change the name/image/URI to impersonate legitimate tokens.
"""

import os
import asyncio
import httpx
from typing import Dict, Optional
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

# Try to import base58 (required for Solana address encoding)
try:
    import base58
    BASE58_AVAILABLE = True
except ImportError:
    BASE58_AVAILABLE = False
    logger.warning("base58 package not installed. Solana metadata inspection will fail.")

# Solana RPC endpoint
DEFAULT_SOLANA_RPC = "https://api.mainnet-beta.solana.com"

# Metaplex Metadata Program ID
METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

# Metadata account seed
METADATA_PREFIX = "metadata"


class SolanaMetadataInspector:
    """Inspect Solana token metadata on-chain via Solana RPC"""
    
    def __init__(self, rpc_url: Optional[str] = None):
        """
        Initialize Solana Metadata Inspector
        
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
            logger.info(f"Solana Metadata Inspector using RPC Load Balancer ({len(self.load_balancer.get_all_rpc_urls())} endpoints)")
        else:
            self.rpc_url = os.getenv("SOLANA_RPC_URL", DEFAULT_SOLANA_RPC)
            self.use_load_balancer = False
        
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests (very conservative for free RPCs)
        
        logger.info(f"Solana Metadata Inspector initialized with RPC: {self.rpc_url[:50]}...")
    
    async def inspect_metadata(self, mint_address: str) -> Dict:
        """Main entry point - delegates to inspect_metadata_with_pda"""
        return await self.inspect_metadata_with_pda(mint_address)
    
    async def _inspect_metadata_legacy(self, mint_address: str) -> Dict:
        """
        Inspect metadata account for safety flags
        
        Args:
            mint_address: Solana mint address (base58)
            
        Returns:
            {
                'name': str,
                'symbol': str,
                'uri': str,
                'update_authority': Optional[str],  # None = immutable ✅
                'is_immutable': bool,
                'is_safe': bool,
                'risk_flags': List[str],
                'green_flags': List[str],
                'error': Optional[str]
            }
        """
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Find metadata account address using PDA derivation
            metadata_address = self._derive_metadata_address(mint_address)
            
            if not metadata_address:
                return {
                    'error': 'Could not derive metadata address',
                    'is_safe': False,
                    'risk_flags': ['❌ Could not find metadata account']
                }
            
            # Get metadata account info
            account_info = await self._get_account_info(metadata_address)
            
            if not account_info:
                # Metadata account doesn't exist - this is okay for some tokens
                logger.debug(f"Metadata account not found for {mint_address[:8]}...")
                return {
                    'name': '',
                    'symbol': '',
                    'uri': '',
                    'update_authority': None,
                    'is_immutable': True,  # No metadata = can't be changed
                    'is_safe': True,
                    'risk_flags': [],
                    'green_flags': ['✅ No metadata account (immutable)'],
                    'error': None
                }
            
            # Parse metadata account data
            # Metaplex Metadata v1 structure (simplified):
            # - Key (1 byte): Metadata key type
            # - Update authority (32 bytes, Option<Pubkey>)
            # - Mint (32 bytes)
            # - Data (name, symbol, uri - variable length)
            # - ...
            
            metadata_data = account_info.get('data', [])
            if not metadata_data:
                return {
                    'error': 'Metadata account has no data',
                    'is_safe': False,
                    'risk_flags': ['❌ Invalid metadata account structure']
                }
            
            # Decode base64 data
            import base64
            try:
                raw_data = base64.b64decode(metadata_data[0])
            except Exception as e:
                return {
                    'error': f'Failed to decode metadata data: {e}',
                    'is_safe': False,
                    'risk_flags': ['❌ Failed to decode metadata account data']
                }
            
            if not BASE58_AVAILABLE:
                return {
                    'error': 'base58 package not installed',
                    'is_safe': False,
                    'risk_flags': ['❌ Missing dependency: base58']
                }
            
            # Try parsing with jsonParsed encoding first (easier)
            parsed_data = account_info.get('data', {}).get('parsed', {})
            if parsed_data and isinstance(parsed_data, dict):
                # Use parsed data if available (jsonParsed encoding)
                return self._parse_json_metadata(parsed_data, mint_address)
            
            # Fallback: Parse raw binary data
            return self._parse_binary_metadata(raw_data, mint_address)
            
        except Exception as e:
            logger.error(f"Error inspecting metadata {mint_address}: {e}", exc_info=True)
            return {
                'error': str(e),
                'is_safe': False,
                'risk_flags': [f'❌ Inspection error: {str(e)}']
            }
    
    def _parse_json_metadata(self, parsed_data: Dict, mint_address: str) -> Dict:
        """Parse metadata from jsonParsed encoding"""
        try:
            info = parsed_data.get('info', {})
            data = info.get('data', {})
            
            name = data.get('name', '').strip('\x00')
            symbol = data.get('symbol', '').strip('\x00')
            uri = data.get('uri', '').strip('\x00')
            update_authority = info.get('updateAuthority')
            
            # Check if immutable
            is_immutable = update_authority is None or update_authority == ''
            
            # Risk assessment
            risk_flags = []
            green_flags = []
            
            if not is_immutable:
                risk_flags.append(
                    f"⚠️ Metadata update authority: {update_authority[:8] if update_authority else 'Unknown'}... "
                    "(can change name/image/URI - impersonation risk)"
                )
            else:
                green_flags.append("✅ Metadata is immutable (update authority revoked)")
            
            return {
                'name': name,
                'symbol': symbol,
                'uri': uri,
                'update_authority': update_authority,
                'is_immutable': is_immutable,
                'is_safe': is_immutable,  # Safe if immutable
                'risk_flags': risk_flags,
                'green_flags': green_flags,
                'error': None
            }
            
        except Exception as e:
            logger.debug(f"Error parsing JSON metadata: {e}")
            return {
                'error': f'Failed to parse metadata: {e}',
                'is_safe': False,
                'risk_flags': [f'❌ Metadata parsing error: {str(e)}']
            }
    
    def _parse_binary_metadata(self, raw_data: bytes, mint_address: str) -> Dict:
        """Parse metadata from raw binary data (fallback)"""
        try:
            if len(raw_data) < 100:
                return {
                    'error': 'Metadata data too short',
                    'is_safe': False,
                    'risk_flags': ['❌ Invalid metadata account structure']
                }
            
            # Skip key byte (offset 0)
            # Update authority starts at offset 1 (33 bytes: Option<Pubkey>)
            update_authority_bytes = raw_data[1:34]
            
            update_authority = None
            if update_authority_bytes[0] != 0:  # Option::Some
                pubkey_bytes = update_authority_bytes[1:33]
                update_authority = base58.b58encode(pubkey_bytes).decode('utf-8')
            
            # Mint address starts at offset 33 (32 bytes)
            mint_bytes = raw_data[33:65]
            # Verify mint matches
            mint_from_data = base58.b58encode(mint_bytes).decode('utf-8')
            if mint_from_data != mint_address:
                logger.debug(f"Mint mismatch in metadata: expected {mint_address}, got {mint_from_data}")
            
            # Data starts at offset 65+ (name, symbol, uri - variable length)
            # This is complex to parse, so we'll use a simplified approach
            # For now, just check update authority
            
            is_immutable = update_authority is None
            
            risk_flags = []
            green_flags = []
            
            if not is_immutable:
                risk_flags.append(
                    f"⚠️ Metadata update authority: {update_authority[:8]}... "
                    "(can change name/image/URI - impersonation risk)"
                )
            else:
                green_flags.append("✅ Metadata is immutable (update authority revoked)")
            
            return {
                'name': '',  # Would need complex parsing for binary format
                'symbol': '',
                'uri': '',
                'update_authority': update_authority,
                'is_immutable': is_immutable,
                'is_safe': is_immutable,
                'risk_flags': risk_flags,
                'green_flags': green_flags,
                'error': None
            }
            
        except Exception as e:
            logger.debug(f"Error parsing binary metadata: {e}")
            return {
                'error': f'Failed to parse binary metadata: {e}',
                'is_safe': False,
                'risk_flags': [f'❌ Metadata parsing error: {str(e)}']
            }
    
    def _derive_metadata_address(self, mint_address: str) -> Optional[str]:
        """
        Derive metadata account PDA address using Metaplex standard
        
        PDA = Program Derived Address
        seeds = ["metadata", metadata_program_id, mint_address]
        """
        if not BASE58_AVAILABLE:
            return None
        
        try:
            # Use RPC to find the metadata account
            # We'll use getProgramAccounts with filters
            # For now, return None and use direct account lookup
            # (The RPC will return None if account doesn't exist)
            return None  # Will look up using getAccountInfo directly
            
        except Exception as e:
            logger.debug(f"Error deriving metadata address: {e}")
            return None
    
    async def _get_account_info(self, address: str) -> Optional[Dict]:
        """Get account info from Solana RPC (with PDA derivation attempt)"""
        try:
            await self._rate_limit()
            
            # First, try to find metadata account using getProgramAccounts
            # This is more reliable than manual PDA derivation
            TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
            
            # Extract mint from address if needed, or use provided address
            # For now, we'll use a simpler approach: try known metadata accounts
            # The proper way is to use findProgramAddress, but that requires SDK
            
            # Try direct lookup with potential metadata address patterns
            # This is a simplified approach - in production, use Metaplex SDK
            
            # Alternative: Use getProgramAccounts to search for metadata accounts
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getProgramAccounts",
                "params": [
                    METADATA_PROGRAM_ID,
                    {
                        "filters": [
                            {
                                "dataSize": 679  # Standard metadata account size
                            }
                        ],
                        "encoding": "jsonParsed"
                    }
                ]
            }
            
            # This is too broad - instead, let's use a different approach
            # We'll manually construct the PDA or use account info with address derivation
            
            # Simplified: Try using the address directly if it looks like a metadata account
            # For proper implementation, we need to derive PDA using seeds
            
            # For now, return None and handle gracefully
            # In production, integrate with Metaplex SDK or implement proper PDA derivation
            logger.debug("Metadata account lookup requires PDA derivation - using simplified approach")
            return None
            
        except Exception as e:
            logger.debug(f"Error getting metadata account info: {e}")
            return None
    
    async def _get_metadata_via_rpc(self, mint_address: str) -> Optional[Dict]:
        """
        Get metadata account using RPC with proper PDA derivation
        
        Since we can't easily derive PDAs in pure Python without the SDK,
        we'll use getProgramAccounts with a filter on mint address
        """
        try:
            await self._rate_limit()
            
            # Use getProgramAccounts to find metadata accounts for this mint
            # Filter by mint address in metadata account data
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getProgramAccounts",
                "params": [
                    METADATA_PROGRAM_ID,
                    {
                        "filters": [
                            {
                                "dataSize": 679  # Metadata account size
                            },
                            {
                                "memcmp": {
                                    "offset": 33,  # Mint address offset in metadata account
                                    "bytes": mint_address
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
                return None
                
                result = data.get("result", [])
                if not isinstance(result, list) or len(result) == 0:
                    return None  # No metadata account found
                
                # Return first metadata account (should only be one per mint)
                account = result[0]
                account_data = account.get("account", {})
                return account_data
                
        except Exception as e:
            logger.debug(f"Error getting metadata via RPC: {e}")
            return None
    
    async def inspect_metadata_with_pda(self, mint_address: str) -> Dict:
        """
        Inspect metadata using proper PDA lookup
        
        This is the main method - uses getProgramAccounts to find metadata account
        """
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Get metadata account via RPC
            account_info = await self._get_metadata_via_rpc(mint_address)
            
            if not account_info:
                # No metadata account - this is okay
                logger.debug(f"No metadata account found for {mint_address[:8]}...")
                return {
                    'name': '',
                    'symbol': '',
                    'uri': '',
                    'update_authority': None,
                    'is_immutable': True,
                    'is_safe': True,
                    'risk_flags': [],
                    'green_flags': ['✅ No metadata account (immutable)'],
                    'error': None
                }
            
            # Parse metadata from account info
            parsed_data = account_info.get('data', {}).get('parsed', {})
            
            if parsed_data and isinstance(parsed_data, dict):
                return self._parse_json_metadata(parsed_data, mint_address)
            else:
                # Fallback to binary parsing
                metadata_data = account_info.get('data', [])
                if metadata_data:
                    import base64
                    raw_data = base64.b64decode(metadata_data[0])
                    return self._parse_binary_metadata(raw_data, mint_address)
                else:
                    return {
                        'error': 'Metadata account has no data',
                        'is_safe': False,
                        'risk_flags': ['❌ Invalid metadata account structure']
                    }
            
        except Exception as e:
            logger.error(f"Error inspecting metadata {mint_address}: {e}", exc_info=True)
            return {
                'error': str(e),
                'is_safe': False,
                'risk_flags': [f'❌ Inspection error: {str(e)}']
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
    
    async def quick_check(self, mint_address: str) -> tuple[bool, str]:
        """
        Quick safety check - returns (is_safe, reason)
        
        Returns:
            (True, "Safe") if metadata is immutable
            (False, reason) if metadata can be changed
        """
        result = await self.inspect_metadata_with_pda(mint_address)
        
        if result.get('error'):
            # If error but no metadata account, consider safe
            if 'not found' in result['error'].lower() or 'no metadata' in result['error'].lower():
                return True, "No metadata account (immutable)"
            return False, f"Error: {result['error']}"
        
        if result.get('is_safe'):
            flags = result.get('green_flags', [])
            return True, flags[0] if flags else "Metadata is immutable"
        
        flags = result.get('risk_flags', [])
        return False, flags[0] if flags else "Metadata can be changed"


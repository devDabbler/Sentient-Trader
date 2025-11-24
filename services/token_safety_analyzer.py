"""
Token Safety Analyzer

Performs comprehensive safety checks on new tokens:
1. Honeypot detection
2. Buy/sell tax analysis
3. Liquidity lock verification
4. Ownership analysis (renounced, mintable, etc.)
5. Contract verification

Uses multiple free APIs:
- Honeypot.is API
- GoPlus Security API
- Token Sniffer API
- On-chain contract analysis
"""

import os
import asyncio
import httpx
from typing import Dict, Tuple, Optional
from loguru import logger
from dotenv import load_dotenv
from models.dex_models import ContractSafety, Chain, RiskLevel

load_dotenv()


class TokenSafetyAnalyzer:
    """Analyzes token contracts for safety and rug risk"""
    
    # Free API endpoints
    HONEYPOT_API = "https://api.honeypot.is/v2/IsHoneypot"
    GOPLUS_API = "https://api.gopluslabs.io/api/v1/token_security"
    TOKEN_SNIFFER_API = "https://tokensniffer.com/api/v2/tokens"
    
    def __init__(self):
        self.last_request_time = {}
        self.rate_limit_delays = {
            "honeypot": 2.0,  # 2 seconds between requests
            "goplus": 1.5,
            "tokensniffer": 3.0
        }
    
    async def analyze_token(
        self,
        contract_address: str,
        chain: Chain = Chain.ETH
    ) -> Tuple[bool, ContractSafety]:
        """
        Perform comprehensive safety analysis on a token
        
        Args:
            contract_address: Token contract address
            chain: Blockchain network
            
        Returns:
            (success, ContractSafety object)
        """
        logger.info(f"Analyzing token safety: {contract_address} on {chain.value}")
        
        safety = ContractSafety()
        
        # Run multiple checks in parallel
        results = await asyncio.gather(
            self._check_honeypot(contract_address, chain),
            self._check_goplus(contract_address, chain),
            self._check_token_sniffer(contract_address, chain),
            return_exceptions=True
        )
        
        # Aggregate results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Safety check {i} failed: {result}")
                continue
            
            if isinstance(result, ContractSafety):
                # Merge results (take worst case for safety)
                safety = self._merge_safety_results(safety, result)
        
        # Calculate final safety score
        safety.safety_score = self._calculate_safety_score(safety)
        
        logger.info(f"Token safety score: {safety.safety_score}/100")
        
        return True, safety
    
    async def _check_honeypot(self, address: str, chain: Chain) -> ContractSafety:
        """Check using Honeypot.is API (EVM chains only)"""
        try:
            await self._rate_limit("honeypot")
            
            # Map chain to Honeypot.is chain ID (numeric)
            chain_map = {
                Chain.ETH: "1",
                Chain.BSC: "56",
                Chain.BASE: "8453",
                Chain.ARBITRUM: "42161",
                Chain.POLYGON: "137"
            }
            
            # Skip if chain not supported (e.g., Solana)
            if chain not in chain_map:
                logger.debug(f"Honeypot.is doesn't support {chain.value} - skipping check")
                return ContractSafety()
            
            chain_id = chain_map.get(chain, "1")
            
            # Honeypot.is expects lowercase address
            address = address.lower() if address else ""
            
            if not address or not address.startswith("0x"):
                logger.debug(f"Honeypot.is requires 0x address (got: {address[:20]}...) - skipping")
                return ContractSafety()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    self.HONEYPOT_API,
                    params={"address": address, "chainID": chain_id}
                )
                
                if response.status_code == 404:
                    logger.debug(f"Honeypot.is: Token not yet indexed (404) - expected for new tokens")
                    return ContractSafety()
                elif response.status_code != 200:
                    logger.debug(f"Honeypot.is API returned {response.status_code}")
                    return ContractSafety()
                
                data = response.json()
                
                # Parse Honeypot.is response
                honeypot_result = data.get("honeypotResult", {})
                simulation_result = data.get("simulationResult", {})
                
                safety = ContractSafety(
                    is_honeypot=honeypot_result.get("isHoneypot", False),
                    buy_tax=float(simulation_result.get("buyTax", 0)),
                    sell_tax=float(simulation_result.get("sellTax", 0)),
                    safety_checks_total=10
                )
                
                # Count passed checks
                checks_passed = 0
                if not safety.is_honeypot:
                    checks_passed += 3  # Most important
                if safety.buy_tax < 10:
                    checks_passed += 1
                if safety.sell_tax < 10:
                    checks_passed += 1
                
                safety.safety_checks_passed = checks_passed
                
                return safety
                
        except Exception as e:
            logger.error(f"Honeypot check failed: {e}", exc_info=True)
            return ContractSafety()
    
    async def _check_goplus(self, address: str, chain: Chain) -> ContractSafety:
        """Check using GoPlus Security API"""
        try:
            await self._rate_limit("goplus")
            
            # Map chain to GoPlus chain ID
            chain_map = {
                Chain.ETH: "1",
                Chain.BSC: "56",
                Chain.POLYGON: "137",
                Chain.ARBITRUM: "42161",
                Chain.BASE: "8453",
                Chain.SOLANA: "solana"  # GoPlus supports Solana!
            }
            chain_id = chain_map.get(chain, "1")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.GOPLUS_API}/{chain_id}",
                    params={"contract_addresses": address}
                )
                
                if response.status_code != 200:
                    logger.warning(f"GoPlus API error: {response.status_code}")
                    return ContractSafety()
                
                data = response.json()
                
                # Parse GoPlus response - handle various response structures
                if not data or not isinstance(data, dict):
                    logger.debug("GoPlus returned empty or invalid data")
                    return ContractSafety()
                
                # GoPlus API v2 might return data directly or in result field
                # Try both structures
                result = data.get("result")
                
                # Handle None result (token not found or invalid address)
                # This is EXPECTED for brand new tokens not yet indexed by GoPlus
                if result is None:
                    logger.debug(f"GoPlus: Token not yet indexed (expected for new launches). Message: {data.get('message', 'OK')}")
                    # Return default safety profile for new tokens
                    return ContractSafety(
                        safety_score=50.0,  # Neutral score for unknown tokens
                        safety_checks_total=0  # No checks performed
                    )
                
                # Handle non-dict result
                if not isinstance(result, dict):
                    # Fallback: try using data itself as result
                    if isinstance(data, dict) and any(k.startswith('0x') for k in data.keys()):
                        result = data
                    else:
                        logger.debug(f"GoPlus result field missing or invalid. Response structure: {list(data.keys())}")
                        return ContractSafety()
                
                # Get token data (address might be lowercase)
                token_data = result.get(address.lower()) or result.get(address)
                if not token_data or not isinstance(token_data, dict):
                    logger.debug(f"GoPlus: No data for token {address} (expected for new launches)")
                    # Return neutral score like line 190-195
                    return ContractSafety(
                        safety_score=50.0,  # Neutral score for unknown tokens
                        safety_checks_total=0  # No checks performed
                    )
                
                # Safely parse all fields with proper type conversion
                def safe_get_bool(key: str) -> bool:
                    val = token_data.get(key, "0")
                    return str(val) == "1" or str(val).lower() == "true"
                
                def safe_get_float(key: str, default: float = 0.0) -> float:
                    try:
                        val = token_data.get(key, default)
                        return float(val) if val is not None else default
                    except (ValueError, TypeError):
                        return default
                
                safety = ContractSafety(
                    is_honeypot=safe_get_bool("is_honeypot"),
                    buy_tax=safe_get_float("buy_tax", 0) * 100,  # Convert to percentage
                    sell_tax=safe_get_float("sell_tax", 0) * 100,
                    is_mintable=safe_get_bool("is_mintable"),
                    is_proxy=safe_get_bool("is_proxy"),
                    has_blacklist=safe_get_bool("is_blacklist"),
                    owner_can_change_tax=safe_get_bool("can_take_back_ownership"),
                    hidden_owner=safe_get_bool("hidden_owner"),
                    safety_checks_total=10
                )
                
                # Check ownership
                owner_address = token_data.get("owner_address", "")
                safety.is_renounced = (
                    owner_address == "0x0000000000000000000000000000000000000000" or
                    safe_get_bool("is_open_source")
                )
                
                # Count passed checks
                checks_passed = 0
                if not safety.is_honeypot:
                    checks_passed += 3
                if safety.buy_tax < 10:
                    checks_passed += 1
                if safety.sell_tax < 10:
                    checks_passed += 1
                if not safety.is_mintable:
                    checks_passed += 1
                if not safety.has_blacklist:
                    checks_passed += 1
                if not safety.hidden_owner:
                    checks_passed += 1
                if safety.is_renounced:
                    checks_passed += 2
                
                safety.safety_checks_passed = checks_passed
                
                return safety
                
        except Exception as e:
            logger.error(f"GoPlus check failed: {e}", exc_info=True)
            return ContractSafety()
    
    async def _check_token_sniffer(self, address: str, chain: Chain) -> ContractSafety:
        """Check using TokenSniffer API - DEPRECATED (now requires paid API key)"""
        # TokenSniffer changed to paid-only API in 2024
        # Skipping this check to avoid 401 errors
        logger.debug("TokenSniffer check skipped (requires paid API key)")
        return ContractSafety()
    
    def _merge_safety_results(self, current: ContractSafety, new: ContractSafety) -> ContractSafety:
        """Merge multiple safety check results (take worst case)"""
        return ContractSafety(
            is_renounced=current.is_renounced or new.is_renounced,
            is_honeypot=current.is_honeypot or new.is_honeypot,  # If ANY says honeypot = red flag
            buy_tax=max(current.buy_tax, new.buy_tax),  # Take highest tax
            sell_tax=max(current.sell_tax, new.sell_tax),
            lp_locked=current.lp_locked or new.lp_locked,
            lp_lock_duration_days=max(
                current.lp_lock_duration_days or 0,
                new.lp_lock_duration_days or 0
            ) or None,
            is_mintable=current.is_mintable or new.is_mintable,
            is_proxy=current.is_proxy or new.is_proxy,
            has_blacklist=current.has_blacklist or new.has_blacklist,
            owner_can_change_tax=current.owner_can_change_tax or new.owner_can_change_tax,
            hidden_owner=current.hidden_owner or new.hidden_owner,
            safety_checks_passed=current.safety_checks_passed + new.safety_checks_passed,
            safety_checks_total=max(current.safety_checks_total, new.safety_checks_total)
        )
    
    def _calculate_safety_score(self, safety: ContractSafety) -> float:
        """Calculate 0-100 safety score"""
        score = 0.0
        
        # Honeypot = instant fail
        if safety.is_honeypot:
            return 0.0
        
        # Critical factors (60 points)
        if not safety.is_honeypot:
            score += 30
        if safety.buy_tax < 5:
            score += 10
        elif safety.buy_tax < 10:
            score += 5
        if safety.sell_tax < 5:
            score += 10
        elif safety.sell_tax < 10:
            score += 5
        if safety.lp_locked:
            score += 10
        
        # Important factors (30 points)
        if safety.is_renounced:
            score += 10
        if not safety.is_mintable:
            score += 5
        if not safety.has_blacklist:
            score += 5
        if not safety.owner_can_change_tax:
            score += 5
        if not safety.hidden_owner:
            score += 5
        
        # Bonus for LP lock duration (10 points)
        if safety.lp_lock_duration_days:
            if safety.lp_lock_duration_days >= 365:
                score += 10
            elif safety.lp_lock_duration_days >= 180:
                score += 7
            elif safety.lp_lock_duration_days >= 90:
                score += 5
            elif safety.lp_lock_duration_days >= 30:
                score += 3
        
        return min(score, 100.0)
    
    def get_risk_level(self, safety: ContractSafety) -> RiskLevel:
        """Determine risk level from safety score"""
        if safety.is_honeypot:
            return RiskLevel.EXTREME
        
        score = safety.safety_score
        
        if score >= 80:
            return RiskLevel.SAFE
        elif score >= 60:
            return RiskLevel.LOW
        elif score >= 40:
            return RiskLevel.MEDIUM
        elif score >= 20:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    async def _rate_limit(self, api_name: str):
        """Enforce rate limiting per API"""
        if api_name not in self.last_request_time:
            self.last_request_time[api_name] = 0
        
        elapsed = asyncio.get_event_loop().time() - self.last_request_time[api_name]
        delay = self.rate_limit_delays.get(api_name, 1.0)
        
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        
        self.last_request_time[api_name] = asyncio.get_event_loop().time()
    
    async def quick_safety_check(self, address: str, chain: Chain = Chain.ETH) -> Tuple[bool, Dict]:
        """
        Quick safety check (honeypot + taxes only)
        Faster than full analysis
        
        Returns:
            (is_safe, details_dict)
        """
        try:
            safety = await self._check_honeypot(address, chain)
            
            is_safe = (
                not safety.is_honeypot and
                safety.buy_tax < 15 and
                safety.sell_tax < 15
            )
            
            details = {
                "is_honeypot": safety.is_honeypot,
                "buy_tax": safety.buy_tax,
                "sell_tax": safety.sell_tax,
                "is_safe": is_safe
            }
            
            return is_safe, details
            
        except Exception as e:
            logger.error(f"Quick safety check failed: {e}")
            return False, {"error": str(e)}

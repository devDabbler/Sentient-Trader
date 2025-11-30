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
from models.dex_models import ContractSafety, Chain, RiskLevel, HolderDistribution

load_dotenv()

# Optional Solana inspectors (only import if needed)
try:
    from services.solana_mint_inspector import SolanaMintInspector
    from services.solana_lp_analyzer import SolanaLPAnalyzer
    from services.solana_holder_analyzer import SolanaHolderAnalyzer
    from services.solana_metadata_inspector import SolanaMetadataInspector
    SOLANA_INSPECTORS_AVAILABLE = True
except ImportError:
    SOLANA_INSPECTORS_AVAILABLE = False
    logger.warning("Solana inspectors not available - install base58 package")


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
        
        # Initialize Solana inspectors if available
        self.solana_mint_inspector = None
        self.solana_lp_analyzer = None
        self.solana_holder_analyzer = None
        self.solana_metadata_inspector = None
        if SOLANA_INSPECTORS_AVAILABLE:
            try:
                self.solana_mint_inspector = SolanaMintInspector()
                self.solana_lp_analyzer = SolanaLPAnalyzer()
                self.solana_holder_analyzer = SolanaHolderAnalyzer()
                self.solana_metadata_inspector = SolanaMetadataInspector()
                logger.info("✅ Solana on-chain inspectors initialized (mint, LP, holder, metadata)")
            except Exception as e:
                logger.warning(f"Failed to initialize Solana inspectors: {e}")
    
    async def analyze_token(
        self,
        contract_address: str,
        chain: Chain = Chain.ETH,
        pool_address: Optional[str] = None
    ) -> Tuple[bool, ContractSafety]:
        """
        Perform comprehensive safety analysis on a token
        
        Args:
            contract_address: Token contract address
            chain: Blockchain network
            pool_address: Optional pool address (required for Solana LP analysis)
            
        Returns:
            (success, ContractSafety object)
        """
        logger.info(f"Analyzing token safety: {contract_address} on {chain.value}")
        
        safety = ContractSafety()
        
        # For Solana: Run on-chain checks FIRST (critical for safety)
        if chain == Chain.SOLANA and SOLANA_INSPECTORS_AVAILABLE:
            solana_safety = await self._check_solana_onchain(contract_address, pool_address)
            if solana_safety:
                # Merge Solana results (these are hard red flags)
                safety = self._merge_safety_results(safety, solana_safety)
                
                # HARD REJECT: If Solana checks fail (safety_score = 0), return immediately
                # This happens when: mint authority retained, freeze authority retained, or LP in EOA
                if solana_safety.safety_score == 0.0 and (solana_safety.is_mintable or solana_safety.is_honeypot):
                    logger.warning(f"🚨 Solana token FAILED on-chain checks: {contract_address}")
                    safety.safety_score = 0.0
                    return True, safety
        
        # Run multiple checks in parallel (for EVM chains or as backup for Solana)
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
        
        # Calculate final safety score (preserve pre-calculated scores like native SOL = 70)
        # Check if this is native SOL by looking for it in risk flags (green flags)
        is_native_sol = any('Native SOL' in str(flag) or 'wrapped SOL' in str(flag).lower() for flag in safety.solana_risk_flags)
        
        if not is_native_sol or safety.safety_score == 0.0:
            calculated_score = self._calculate_safety_score(safety)
            # Preserve native SOL score of 70, otherwise use calculated
            if is_native_sol and safety.safety_score == 70.0:
                pass  # Keep the 70 score for native SOL
            else:
                safety.safety_score = calculated_score
        
        logger.info(f"Token safety score: {safety.safety_score}/100")
        
        return True, safety
    
    async def _check_solana_onchain(
        self, 
        mint_address: str, 
        pool_address: Optional[str] = None
    ) -> Optional[ContractSafety]:
        """
        Perform on-chain Solana safety checks
        
        Args:
            mint_address: Solana token mint address
            pool_address: Optional pool address for LP analysis
            
        Returns:
            ContractSafety object with Solana-specific flags, or None if checks failed
        """
        if not self.solana_mint_inspector:
            return None
        
        try:
            # 1. Check mint authority and freeze authority
            mint_data = await self.solana_mint_inspector.inspect_mint(mint_address)
            
            if mint_data.get('error'):
                logger.warning(f"Solana mint inspection error: {mint_data['error']}")
                return None
            
            # Create safety object from mint inspection
            safety = ContractSafety()
            
            # Store Solana-specific flags
            safety.solana_mint_authority_revoked = mint_data.get('mint_authority') is None
            safety.solana_freeze_authority_revoked = mint_data.get('freeze_authority') is None
            safety.solana_risk_flags = mint_data.get('risk_flags', [])
            
            # For native SOL (wrapped SOL), set defaults appropriately
            # Native SOL accounts don't have mint/freeze authority (they're safe)
            green_flags = mint_data.get('green_flags', [])
            is_native_sol = any('Native SOL' in str(flag) or 'wrapped SOL' in str(flag).lower() for flag in green_flags) if green_flags else False
            
            if is_native_sol:
                safety.is_mintable = False  # Native SOL can't be minted
                safety.is_honeypot = False  # Native SOL is not a honeypot
                safety.is_renounced = True  # Native SOL has no owner
                safety.buy_tax = 0.0  # No taxes on native SOL
                safety.sell_tax = 0.0
                # Give it a decent base score for native assets (will be used by _calculate_safety_score)
                safety.safety_score = 70.0  # Native SOL is inherently safe
                logger.info(f"✅ Native SOL detected - setting safety score to 70")
            
            # HARD RED FLAGS from mint inspection
            mint_authority = mint_data.get('mint_authority')
            if mint_authority:
                safety.is_mintable = True  # Can mint more = risk
                safety.safety_score = 0.0  # Instant fail
                safety.solana_risk_flags.append(f"MINT_AUTHORITY_RETAINED: {mint_authority[:8]}...")
                logger.warning(f"🚨 MINT AUTHORITY RETAINED: {mint_address[:8]}...")
                return safety
            
            freeze_authority = mint_data.get('freeze_authority')
            if freeze_authority:
                safety.is_honeypot = True  # Can freeze accounts = honeypot
                safety.safety_score = 0.0  # Instant fail
                safety.solana_risk_flags.append(f"FREEZE_AUTHORITY_RETAINED: {freeze_authority[:8]}... (HONEYPOT!)")
                logger.warning(f"🚨 FREEZE AUTHORITY RETAINED: {mint_address[:8]}... (HONEYPOT!)")
                return safety
            
            # Green flags
            if not mint_data.get('mint_authority') and not mint_data.get('freeze_authority'):
                safety.is_mintable = False  # Cannot mint more = safe
                safety.solana_risk_flags.extend(mint_data.get('green_flags', []))
                logger.info(f"✅ Mint and freeze authorities revoked: {mint_address[:8]}...")
            
            # 2. Check LP token ownership (if pool address provided)
            if pool_address and self.solana_lp_analyzer:
                lp_data = await self.solana_lp_analyzer.analyze_lp_status(
                    pool_address, 
                    mint_address
                )
                
                if not lp_data.get('error'):
                    # Store LP owner type
                    safety.solana_lp_owner_type = lp_data.get('lp_owner_type')
                    safety.solana_risk_flags.extend(lp_data.get('risk_flags', []))
                    
                    # Set LP lock status based on ownership
                    if lp_data.get('lp_owner_type') in ['burn', 'locker']:
                        safety.lp_locked = True
                        logger.info(f"✅ LP tokens safe: {lp_data.get('lp_owner_type')}")
                    elif lp_data.get('lp_owner_type') == 'EOA_unknown':
                        # HARD RED FLAG: LP in EOA wallet = rug risk
                        safety.safety_score = 0.0
                        safety.solana_risk_flags.append("LP_TOKENS_IN_EOA_WALLET - RUG RISK!")
                        logger.warning(f"🚨 LP tokens in EOA wallet - RUG RISK!")
                        return safety
                    
                    # Set LP lock duration if available
                    if lp_data.get('lp_lock_duration_days'):
                        safety.lp_lock_duration_days = lp_data.get('lp_lock_duration_days')
            
            # 3. Check holder distribution (for dump risk assessment)
            # Add delay between inspector calls to avoid rate limits
            await asyncio.sleep(1.0)  # 1 second delay between major RPC calls
            
            if self.solana_holder_analyzer:
                holder_data = await self.solana_holder_analyzer.analyze_holder_distribution(
                    mint_address,
                    limit=20
                )
                
                if not holder_data.get('error'):
                    concentration = holder_data.get('concentration', {})
                    
                    # Create HolderDistribution object
                    holder_dist = HolderDistribution(
                        top_holders=holder_data.get('holders', [])[:20],
                        top1_pct=concentration.get('top1_pct', 0.0),
                        top5_pct=concentration.get('top5_pct', 0.0),
                        top10_pct=concentration.get('top10_pct', 0.0),
                        top20_pct=concentration.get('top20_pct', 0.0),
                        is_centralized=concentration.get('is_centralized', False),
                        total_holders=concentration.get('total_holders', 0),
                        unique_owners=concentration.get('unique_owners', 0),
                        risk_flags=concentration.get('risk_flags', []),
                        green_flags=concentration.get('green_flags', [])
                    )
                    
                    # Store in safety object
                    safety.solana_holder_distribution = holder_dist
                    
                    # Add risk flags to safety
                    safety.solana_risk_flags.extend(concentration.get('risk_flags', []))
                    
                    # SOFT FLAG: High concentration raises risk (but not hard reject)
                    if concentration.get('top1_pct', 0) > 30:
                        # Extreme whale risk - significantly lower safety score
                        if safety.safety_score > 0:
                            safety.safety_score = max(0, safety.safety_score - 30)
                        logger.warning(f"⚠️ Extreme whale concentration: {concentration.get('top1_pct', 0):.1f}%")
                    elif concentration.get('top10_pct', 0) > 70:
                        # Highly centralized - lower safety score
                        if safety.safety_score > 0:
                            safety.safety_score = max(0, safety.safety_score - 20)
                        logger.warning(f"⚠️ Highly centralized: Top 10 hold {concentration.get('top10_pct', 0):.1f}%")
                    elif concentration.get('top10_pct', 0) > 60:
                        # Centralized - moderate penalty
                        if safety.safety_score > 0:
                            safety.safety_score = max(0, safety.safety_score - 10)
                        logger.info(f"⚠️ Centralized: Top 10 hold {concentration.get('top10_pct', 0):.1f}%")
                    else:
                        # Good distribution - add green flags
                        safety.solana_risk_flags.extend(concentration.get('green_flags', []))
                        logger.info(f"✅ Good holder distribution: Top 10 hold {concentration.get('top10_pct', 0):.1f}%")
            
            # 4. Check metadata immutability (for impersonation risk)
            # Add delay before metadata check to avoid rate limits
            await asyncio.sleep(1.0)  # 1 second delay between major RPC calls
            
            if self.solana_metadata_inspector:
                metadata_data = await self.solana_metadata_inspector.inspect_metadata_with_pda(mint_address)
                
                if not metadata_data.get('error'):
                    # Store metadata immutability status
                    safety.solana_metadata_immutable = metadata_data.get('is_immutable', False)
                    safety.solana_metadata_update_authority = metadata_data.get('update_authority')
                    
                    # Add risk flags (soft flag - not hard reject, but important to know)
                    safety.solana_risk_flags.extend(metadata_data.get('risk_flags', []))
                    safety.solana_risk_flags.extend(metadata_data.get('green_flags', []))
                    
                    # Minor penalty for mutable metadata (impersonation risk, but not a rug)
                    if not metadata_data.get('is_immutable'):
                        # Reduce safety score by 5 points (not catastrophic, but notable)
                        if safety.safety_score > 0:
                            safety.safety_score = max(0, safety.safety_score - 5)
                        logger.info(f"⚠️ Metadata is mutable: {mint_address[:8]}... (impersonation risk)")
                    else:
                        logger.info(f"✅ Metadata is immutable: {mint_address[:8]}...")
            
            return safety
            
        except Exception as e:
            logger.error(f"Error in Solana on-chain check: {e}", exc_info=True)
            return None
    
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
        # Merge holder distribution (prefer the one with more data)
        merged_holder_dist = None
        if new.solana_holder_distribution:
            merged_holder_dist = new.solana_holder_distribution
        elif current.solana_holder_distribution:
            merged_holder_dist = current.solana_holder_distribution
        
        # Merge Solana risk flags
        merged_solana_flags = list(set(current.solana_risk_flags + new.solana_risk_flags))
        
        # Preserve safety_score from new (Solana) if it's higher or specifically set (like native SOL = 70)
        # If new has a pre-calculated score (e.g., native SOL = 70), preserve it
        merged_safety_score = max(current.safety_score, new.safety_score)
        # If new has a specific score like 70 (native SOL), prefer it
        if new.safety_score >= 70.0:
            merged_safety_score = new.safety_score
        
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
            safety_checks_total=max(current.safety_checks_total, new.safety_checks_total),
            safety_score=merged_safety_score,  # Preserve pre-calculated scores
            # Solana-specific fields
            solana_mint_authority_revoked=current.solana_mint_authority_revoked or new.solana_mint_authority_revoked,
            solana_freeze_authority_revoked=current.solana_freeze_authority_revoked or new.solana_freeze_authority_revoked,
            solana_lp_owner_type=new.solana_lp_owner_type or current.solana_lp_owner_type,
            solana_risk_flags=merged_solana_flags,
            solana_holder_distribution=merged_holder_dist,
            # Metadata fields (prefer immutable/None over mutable)
            solana_metadata_immutable=new.solana_metadata_immutable if new.solana_metadata_immutable is not None else current.solana_metadata_immutable,
            solana_metadata_update_authority=new.solana_metadata_update_authority or current.solana_metadata_update_authority
        )
    
    def _calculate_safety_score(self, safety: ContractSafety) -> float:
        """Calculate 0-100 safety score"""
        # If safety_score was pre-calculated (e.g., for native SOL), use it
        if safety.safety_score > 0.0 and safety.safety_score != 0.0:
            return safety.safety_score
        
        # If already set to 0 (hard reject from Solana checks), return 0
        if safety.safety_score == 0.0 and (safety.is_mintable or safety.is_honeypot):
            return 0.0
        
        score = 0.0
        
        # Honeypot = instant fail
        if safety.is_honeypot:
            return 0.0
        
        # Mintable = instant fail (for Solana, this means mint authority retained)
        if safety.is_mintable and safety.safety_score == 0.0:
            return 0.0  # Already set to 0 by Solana inspector
        
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
        
        # Apply holder concentration penalties (if Solana holder data available)
        if safety.solana_holder_distribution:
            holder_dist = safety.solana_holder_distribution
            if holder_dist.is_centralized:
                # Penalize based on concentration level
                if holder_dist.top1_pct > 30:
                    score = max(0, score - 30)  # Extreme whale risk
                elif holder_dist.top10_pct > 70:
                    score = max(0, score - 20)  # Highly centralized
                elif holder_dist.top10_pct > 60:
                    score = max(0, score - 10)  # Centralized
                elif holder_dist.top1_pct > 20:
                    score = max(0, score - 5)  # Moderate whale risk
            else:
                # Bonus for good distribution
                if holder_dist.top10_pct < 40 and holder_dist.unique_owners >= 50:
                    score = min(100, score + 5)  # Well distributed
        
        # If safety_score was already calculated (from Solana checks), use that as base
        if safety.safety_score > 0:
            score = max(score, safety.safety_score)
        
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

"""
DEX Launch Hunter - Main Service

Orchestrates all components to find and alert on early token launches:
1. Monitor DexScreener for new pairs
2. Check token safety (honeypot, taxes, etc.)
3. Track smart money wallet activity
4. Aggregate social sentiment
5. Score launch potential
6. Send alerts for promising launches

This is the core engine for catching pump opportunities early.
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import asdict

from models.dex_models import (
    TokenLaunch, LaunchAlert, HunterConfig, Chain, RiskLevel,
    LaunchStage, ContractSafety, SmartMoneyActivity, SocialSignal, DexPair,
    VerificationChecklist
)
from clients.dexscreener_client import DexScreenerClient
from services.token_safety_analyzer import TokenSafetyAnalyzer
from services.smart_money_tracker import SmartMoneyTracker
from services.price_validator import PriceValidator
from services.crypto_whale_tracker import CryptoWhaleTracker
from services.dex_execution_webhook import get_dex_execution_webhook, ExecutionStrategy
from windows_services.runners.service_config_loader import save_analysis_results

# Optional X Sentiment integration
try:
    from services.x_sentiment_service import XSentimentService, get_x_sentiment_service
    from services.llm_budget_manager import get_llm_budget_manager
    X_SENTIMENT_AVAILABLE = True
except ImportError:
    X_SENTIMENT_AVAILABLE = False
    logger.warning("X Sentiment Service not available")


class DexLaunchHunter:
    """Main orchestrator for DEX launch hunting"""
    
    def __init__(self, config: Optional[HunterConfig] = None):
        """
        Initialize DEX Launch Hunter
        
        Args:
            config: Configuration settings (uses defaults if not provided)
        """
        self.config = config or HunterConfig()
        
        # Initialize services
        self.dex_client = DexScreenerClient()
        self.safety_analyzer = TokenSafetyAnalyzer()
        self.smart_money_tracker = SmartMoneyTracker()
        self.price_validator = PriceValidator()
        self.whale_tracker = CryptoWhaleTracker()
        
        # Execution webhook for future bundler integration
        self.execution_webhook = get_dex_execution_webhook()
        logger.info("✅ DEX Execution Webhook initialized (ready for future bundler integration)")
        
        # X Sentiment integration (optional - catches early pump coins)
        # Uses local LLM (Ollama) - no budget limits needed
        self.x_sentiment_service = None
        self.enable_x_sentiment = True  # Toggle for X scraping
        if X_SENTIMENT_AVAILABLE and self.enable_x_sentiment:
            try:
                self.x_sentiment_service = get_x_sentiment_service(
                    use_llm=True,  # Uses local Ollama LLM (FREE)
                    llm_budget_manager=None  # No budget limits with local LLM
                )
                print("[DEX] ✅ X Sentiment Service enabled (uses local Ollama LLM)", flush=True)
                logger.info("✅ X Sentiment Service integrated with DEX Hunter (local LLM)")
            except Exception as e:
                logger.warning(f"Could not initialize X Sentiment Service: {e}")
        
        # State
        self.discovered_tokens: Dict[str, TokenLaunch] = {}  # contract_address -> TokenLaunch
        self.recent_alerts: List[LaunchAlert] = []
        self.blacklisted_tokens: set = set()
        
        # Stats
        self.total_scanned = 0
        self.total_alerts = 0
        self.is_running = False
        
        # Log configuration
        logger.info("DEX Launch Hunter initialized")
        logger.info(f"  ├─ Chains: {[c.value for c in self.config.enabled_chains]}")
        logger.info(f"  ├─ Liquidity: ${self.config.min_liquidity_usd:,.0f} - ${self.config.max_liquidity_usd:,.0f}")
        logger.info(f"  ├─ Max age: {self.config.max_age_hours}h")
        logger.info(f"  ├─ Lenient Solana mode: {getattr(self.config, 'lenient_solana_mode', True)}")
        logger.info(f"  └─ Discovery mode: {getattr(self.config, 'discovery_mode', 'aggressive')}")
        
        # Print config summary
        lenient = getattr(self.config, 'lenient_solana_mode', True)
        discovery = getattr(self.config, 'discovery_mode', 'aggressive')
        print(f"[DEX] Config: lenient_solana_mode={lenient}, discovery_mode={discovery}", flush=True)
        if lenient:
            print("[DEX] ⚠️ LENIENT MODE ON: Allowing tokens with mint/freeze authority (higher risk)", flush=True)
    
    async def start_monitoring(self, continuous: bool = True):
        """
        Start continuous monitoring for new launches
        
        Args:
            continuous: If True, runs forever. If False, runs once.
        """
        logger.info("Starting DEX Launch Hunter monitoring...")
        self.is_running = True
        
        while self.is_running:
            try:
                # 1. Scan for new launches
                await self._scan_for_launches()
                
                # 2. Check smart money activity
                await self._check_smart_money()
                
                # 3. Process and score tokens
                await self._process_discovered_tokens()
                
                # 4. Generate alerts for promising tokens
                await self._generate_alerts()
                
                # 5. Cleanup old data
                self._cleanup_old_data()
                
                if not continuous:
                    break
                
                # Wait before next scan
                logger.info(f"Scan complete. Waiting {self.config.scan_interval_seconds}s...")
                await asyncio.sleep(self.config.scan_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(10)  # Short delay before retry
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        logger.info("Stopping DEX Launch Hunter...")
        self.is_running = False
    
    async def _scan_for_launches(self):
        """Scan for new token launches across enabled chains using FREE DexScreener API"""
        import sys
        logger.info("Scanning for new launches...")
        
        try:
            # Get chains to scan
            chain_ids = [chain.value for chain in self.config.enabled_chains]
            
            # Scan using FREE DexScreener API endpoints
            # Use discovery_mode from config (aggressive finds more tokens, conservative is safer)
            discovery_mode = getattr(self.config, 'discovery_mode', 'aggressive')
            
            success, new_pairs = await self.dex_client.get_new_pairs(
                chains=chain_ids,
                min_liquidity=self.config.min_liquidity_usd,
                max_liquidity=self.config.max_liquidity_usd,
                max_age_hours=self.config.max_age_hours,
                limit=100,  # Scan top 100 new tokens per cycle (was 50)
                discovery_mode=discovery_mode
            )
            
            if not success:
                logger.warning("Failed to fetch new pairs from DexScreener")
                return
            
            logger.info(f"Found {len(new_pairs)} new pairs to analyze")
            print(f"[DEX] Found {len(new_pairs)} pairs from DexScreener", flush=True)
            
            # Collect results for control panel
            results_for_panel = []
            
            # Limit pairs to analyze per cycle (prevent slow cycles)
            max_pairs_per_cycle = 20
            pairs_to_analyze = new_pairs[:max_pairs_per_cycle]
            
            analyzed_count = 0
            skipped_already = 0
            skipped_blacklist = 0
            failed_analysis = 0
            
            # Analyze each new pair
            for pair in pairs_to_analyze:
                try:
                    # Skip if already discovered
                    if pair.base_token_address.lower() in self.discovered_tokens:
                        skipped_already += 1
                        continue
                    
                    # Skip if blacklisted
                    if pair.base_token_address.lower() in self.blacklisted_tokens:
                        skipped_blacklist += 1
                        continue
                    
                    self.total_scanned += 1
                    analyzed_count += 1
                    
                    logger.info(f"[DEX] Analyzing: {pair.base_token_symbol} ({pair.chain.value})...")
                    print(f"[DEX] Analyzing: {pair.base_token_symbol} ({pair.chain.value})...", flush=True)
                    
                    # Add timeout to prevent hanging on individual token analysis
                    try:
                        success, token = await asyncio.wait_for(
                            self._analyze_pair_directly(pair),
                            timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Analysis timeout for {pair.base_token_symbol}")
                        failed_analysis += 1
                        continue
                    
                    if success and token:
                        # Detailed score breakdown - LOG it so we see in file
                        score_msg = f"[DEX] ✓ {token.symbol}: Score={token.composite_score:.1f}/100, Risk={token.risk_level.value}"
                        breakdown_msg = f"    └─ Pump:{token.pump_potential_score:.0f} Velocity:{token.velocity_score:.0f} Safety:{token.safety_score:.0f} Liq:{token.liquidity_score:.0f}"
                        price_msg = f"    └─ Price=${token.price_usd:.6f} Liq=${token.liquidity_usd:,.0f} Vol=${token.volume_24h:,.0f}"
                        
                        logger.info(score_msg)
                        logger.info(breakdown_msg)
                        logger.info(price_msg)
                        print(score_msg, flush=True)
                        print(breakdown_msg, flush=True)
                        print(price_msg, flush=True)
                        
                        logger.info(
                            f"Discovered: {token.symbol} - "
                            f"Score: {token.composite_score:.1f}, "
                            f"Risk: {token.risk_level.value}"
                        )
                    else:
                        logger.warning(f"[DEX] ✗ {pair.base_token_symbol}: Analysis failed (success={success})")
                        print(f"[DEX] ✗ {pair.base_token_symbol}: Analysis failed (success={success})", flush=True)
                        failed_analysis += 1
                        
                        # Add to results for panel
                        results_for_panel.append({
                            'ticker': token.symbol,
                            'signal': f"LAUNCH ({token.risk_level.value})",
                            'confidence': int(token.composite_score),
                            'price': token.price_usd,
                            'change_24h': token.price_change_24h,
                            'volume_24h': token.volume_24h,
                            'timestamp': datetime.now().isoformat()
                        })
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error analyzing pair {pair.base_token_symbol}: {e}")
                    failed_analysis += 1
                    continue
            
            # Print scan summary - also log it
            scan_summary = f"[DEX] Scan complete: Analyzed={analyzed_count}, Skipped(seen)={skipped_already}, Skipped(blacklist)={skipped_blacklist}, Failed={failed_analysis}"
            discovered_summary = f"[DEX] Total discovered this session: {len(self.discovered_tokens)}"
            logger.info(scan_summary)
            logger.info(discovered_summary)
            print(scan_summary, flush=True)
            print(discovered_summary, flush=True)
            
            # Save results to control panel
            if results_for_panel:
                try:
                    save_analysis_results('DEX Launch Hunter', results_for_panel)
                    logger.debug(f"💾 Saved {len(results_for_panel)} results to control panel")
                except Exception as e:
                    logger.warning(f"Failed to save results to control panel: {e}")
                    
        except Exception as e:
            logger.error(f"Error in scan loop: {e}", exc_info=True)
    
    async def _check_smart_money(self):
        """Check for smart money wallet activity"""
        print("[WHALE] 🐋 Checking smart money wallet activity...", flush=True)
        try:
            activities = await self.smart_money_tracker.check_all_wallets()
            
            if not activities:
                print("[WHALE] No smart money activity detected", flush=True)
                return
            
            print(f"[WHALE] Found {len(activities)} smart money activities!", flush=True)
            logger.info(f"Found {len(activities)} smart money activities")
            
            # Process each activity
            for activity in activities:
                # If it's a buy, investigate the token
                if activity.action == "BUY" and activity.amount_usd >= self.config.min_whale_buy_usd:
                    logger.info(
                        f"🐋 Smart money BUY detected: {activity.wallet_name} "
                        f"bought ${activity.amount_usd:,.0f}"
                    )
                    
                    # TODO: Extract token address from transaction and analyze
                    
        except Exception as e:
            logger.error(f"Error checking smart money: {e}", exc_info=True)
    
    async def analyze_token(
        self,
        contract_address: str,
        chain: Chain = Chain.ETH,
        include_social: bool = True
    ) -> Tuple[bool, Optional[TokenLaunch]]:
        """
        Perform comprehensive analysis on a token
        
        Args:
            contract_address: Token contract address
            chain: Blockchain
            include_social: Whether to fetch social sentiment
            
        Returns:
            (success, TokenLaunch object or None)
        """
        logger.info(f"Analyzing token: {contract_address} on {chain.value}")
        
        # Check if blacklisted
        if contract_address.lower() in self.blacklisted_tokens:
            logger.warning(f"Token is blacklisted: {contract_address}")
            return False, None
        
        try:
            # 1. Get DEX pair data
            success, pairs = await self.dex_client.get_token_pairs(contract_address, chain.value)
            
            if not success or not pairs:
                logger.warning(f"No DEX pairs found for {contract_address}")
                return False, None
            
            primary_pair = pairs[0]  # Use highest liquidity pair
            
            # 2. Safety analysis (pass pool address for Solana LP analysis)
            pool_address = primary_pair.pair_address if chain == Chain.SOLANA else None
            success, safety = await self.safety_analyzer.analyze_token(
                contract_address, 
                chain,
                pool_address=pool_address
            )
            
            if not success:
                logger.warning(f"Safety analysis failed for {contract_address}")
                safety = ContractSafety()
            
            # 3. Determine risk level
            risk_level = self.safety_analyzer.get_risk_level(safety)
            
            # 4. Check if it passes minimum safety requirements
            if self.config.verify_contract_before_alert:
                # For Solana tokens with mint/freeze authority
                if safety.safety_score == 0.0 and chain == Chain.SOLANA:
                    # In LENIENT MODE: Allow tokens with mint/freeze authority (common for new launches)
                    if self.config.lenient_solana_mode:
                        logger.warning(f"⚠️ Solana token has mint/freeze authority (lenient mode): {contract_address}")
                        if not safety.solana_mint_authority_revoked:
                            logger.warning(f"   Warning: Mint authority retained")
                        if not safety.solana_freeze_authority_revoked:
                            logger.warning(f"   Warning: Freeze authority retained")
                        # Adjust safety score to reflect risk but don't block
                        safety.safety_score = 25.0
                        safety.solana_risk_flags.append("LENIENT_MODE: Mint/freeze authority not revoked")
                    else:
                        # STRICT MODE: Blacklist tokens with mint/freeze authority
                        logger.warning(f"🚨 Blacklisting Solana token (on-chain check failed): {contract_address}")
                        if not safety.solana_mint_authority_revoked:
                            logger.warning(f"   Reason: Mint authority retained")
                        if not safety.solana_freeze_authority_revoked:
                            logger.warning(f"   Reason: Freeze authority retained")
                        if safety.solana_lp_owner_type == 'EOA_unknown':
                            logger.warning(f"   Reason: LP tokens in EOA wallet (rug risk)")
                        self.blacklisted_tokens.add(contract_address.lower())
                        return False, None
                
                if safety.is_honeypot and self.config.auto_blacklist_honeypots:
                    # In lenient mode for Solana, allow if "honeypot" is just due to freeze authority
                    is_freeze_authority_honeypot = (
                        chain == Chain.SOLANA and 
                        not safety.solana_freeze_authority_revoked
                    )
                    
                    if self.config.lenient_solana_mode and is_freeze_authority_honeypot:
                        logger.warning(f"⚠️ Freeze authority retained (risky but allowing in lenient mode): {contract_address}")
                    else:
                        logger.warning(f"Blacklisting honeypot: {contract_address}")
                        self.blacklisted_tokens.add(contract_address.lower())
                        return False, None
                
                if (safety.buy_tax > 15 or safety.sell_tax > 15) and self.config.auto_blacklist_high_tax:
                    logger.warning(f"Blacklisting high tax token: {contract_address} (buy: {safety.buy_tax}%, sell: {safety.sell_tax}%)")
                    self.blacklisted_tokens.add(contract_address.lower())
                    return False, None
            
            # 5. Validate price data across sources (for Solana, compare with Birdeye)
            price_validation = await self.price_validator.validate_price_data(
                contract_address=contract_address,
                chain=chain,
                dexscreener_price=primary_pair.price_usd,
                dexscreener_liquidity=primary_pair.liquidity_usd,
                dexscreener_volume_24h=primary_pair.volume_24h
            )
            
            # 6. Build TokenLaunch object
            token = TokenLaunch(
                symbol=primary_pair.base_token_symbol,
                name=primary_pair.base_token_symbol,  # DexScreener might not have full name
                contract_address=contract_address,
                chain=chain,
                price_usd=primary_pair.price_usd,
                liquidity_usd=primary_pair.liquidity_usd,
                volume_24h=primary_pair.volume_24h,
                price_change_5m=primary_pair.price_change_5m,
                price_change_1h=primary_pair.price_change_1h,
                price_change_24h=primary_pair.price_change_24h,
                pairs=pairs,
                primary_dex=primary_pair.dex_name,
                contract_safety=safety,
                risk_level=risk_level,
                age_hours=primary_pair.pair_age_hours,
                launched_at=primary_pair.created_at,
                holder_distribution=safety.solana_holder_distribution if chain == Chain.SOLANA else None,
                price_consistency_score=price_validation.get('consistency_score'),
                price_validation_warnings=price_validation.get('warnings', [])
            )
            
            # Add price validation warnings to alert reasons
            if price_validation.get('warnings'):
                # Store warnings in token's alert reasons (if exists) or safety flags
                if safety:
                    safety.solana_risk_flags.extend(price_validation.get('warnings', []))
            
            # 7. Check smart money/whale activity for this token
            await self._enrich_with_whale_activity(token)
            
            # 8. Determine launch stage
            token.launch_stage = self._get_launch_stage(token.age_hours)
            
            # 9. Score the token
            token = self._score_token(token)
            
            # 10. Store in discovered tokens
            self.discovered_tokens[contract_address.lower()] = token
            
            logger.info(
                f"Token analyzed: {token.symbol} - "
                f"Score: {token.composite_score:.1f}, "
                f"Risk: {token.risk_level.value}, "
                f"Liquidity: ${token.liquidity_usd:,.0f}"
            )
            
            return True, token
            
        except Exception as e:
            logger.error(f"Error analyzing token {contract_address}: {e}", exc_info=True)
            return False, None
    
    async def _analyze_pair_directly(self, pair: 'DexPair') -> Tuple[bool, Optional[TokenLaunch]]:
        """
        Analyze a token using pair data we already have (skip redundant API call)
        
        Args:
            pair: DexPair object from search
            
        Returns:
            (success, TokenLaunch object or None)
        """
        try:
            contract_address = pair.base_token_address
            
            # VALIDATION 1: Check address format
            if not self._is_valid_address(contract_address, pair.chain):
                print(f"[DEX] ✗ {pair.base_token_symbol}: Invalid address format", flush=True)
                return False, None
            
            # Check if blacklisted
            if contract_address.lower() in self.blacklisted_tokens:
                print(f"[DEX] ✗ {pair.base_token_symbol}: Already blacklisted", flush=True)
                return False, None
            
            # Safety analysis (pass pool address for Solana LP analysis)
            pool_address = pair.pair_address if pair.chain == Chain.SOLANA else None
            try:
                success, safety = await asyncio.wait_for(
                    self.safety_analyzer.analyze_token(
                        contract_address, 
                        pair.chain,
                        pool_address=pool_address
                    ),
                    timeout=15.0
                )
            except asyncio.TimeoutError:
                print(f"[DEX] ✗ {pair.base_token_symbol}: Safety analysis timeout", flush=True)
                success = False
                safety = ContractSafety()
            
            if not success:
                print(f"[DEX] ⚠ {pair.base_token_symbol}: Safety check failed, using defaults", flush=True)
                safety = ContractSafety()
            
            # Determine risk level
            risk_level = self.safety_analyzer.get_risk_level(safety)
            
            # Check minimum safety requirements
            if self.config.verify_contract_before_alert:
                # For Solana tokens with mint/freeze authority
                if safety.safety_score == 0.0 and pair.chain == Chain.SOLANA:
                    # In LENIENT MODE: Allow tokens with mint/freeze authority (common for new launches)
                    if self.config.lenient_solana_mode:
                        # Log warning but DON'T blacklist - add risk flags instead
                        print(f"[DEX] ⚠️ {pair.base_token_symbol}: RISKY (mint/freeze authority) - allowing in lenient mode", flush=True)
                        # Adjust safety score to reflect risk but don't block
                        safety.safety_score = 25.0  # Low but not zero
                        safety.solana_risk_flags.append("LENIENT_MODE: Mint/freeze authority not revoked")
                    else:
                        # STRICT MODE: Blacklist tokens with mint/freeze authority
                        print(f"[DEX] 🚨 {pair.base_token_symbol}: BLACKLISTED (Solana on-chain check failed - mint/freeze authority)", flush=True)
                        self.blacklisted_tokens.add(contract_address.lower())
                        return False, None
                
                if safety.is_honeypot and self.config.auto_blacklist_honeypots:
                    # In lenient mode for Solana, allow if "honeypot" is just due to freeze authority
                    # (not a real honeypot detection from external APIs)
                    is_freeze_authority_honeypot = (
                        pair.chain == Chain.SOLANA and 
                        not safety.solana_freeze_authority_revoked  # Freeze authority retained = why it's marked honeypot
                    )
                    
                    if self.config.lenient_solana_mode and is_freeze_authority_honeypot:
                        print(f"[DEX] ⚠️ {pair.base_token_symbol}: FREEZE AUTHORITY RETAINED (risky but allowing in lenient mode)", flush=True)
                        # Don't blacklist, continue with analysis
                    else:
                        print(f"[DEX] 🚨 {pair.base_token_symbol}: BLACKLISTED (honeypot detected)", flush=True)
                        self.blacklisted_tokens.add(contract_address.lower())
                        return False, None
                
                if (safety.buy_tax > 15 or safety.sell_tax > 15) and self.config.auto_blacklist_high_tax:
                    print(f"[DEX] 🚨 {pair.base_token_symbol}: BLACKLISTED (high tax: buy={safety.buy_tax}%, sell={safety.sell_tax}%)", flush=True)
                    self.blacklisted_tokens.add(contract_address.lower())
                    return False, None
            
            # Build TokenLaunch object from pair data
            token = TokenLaunch(
                symbol=pair.base_token_symbol,
                name=pair.base_token_symbol,
                contract_address=contract_address,
                chain=pair.chain,
                price_usd=pair.price_usd,
                liquidity_usd=pair.liquidity_usd,
                volume_24h=pair.volume_24h,
                price_change_5m=pair.price_change_5m,
                price_change_1h=pair.price_change_1h,
                price_change_24h=pair.price_change_24h,
                pairs=[pair],  # We have this one pair
                primary_dex=pair.dex_name,
                contract_safety=safety,
                risk_level=risk_level,
                age_hours=pair.pair_age_hours,
                launched_at=pair.created_at,
                holder_distribution=safety.solana_holder_distribution if pair.chain == Chain.SOLANA else None
            )
            
            # Check smart money/whale activity for this token
            await self._enrich_with_whale_activity(token)
            
            # Determine launch stage
            token.launch_stage = self._get_launch_stage(token.age_hours)
            
            # Score the token
            token = self._score_token(token)
            
            # Store in discovered tokens
            self.discovered_tokens[contract_address.lower()] = token
            
            logger.info(
                f"Token analyzed (from search): {token.symbol} - "
                f"Score: {token.composite_score:.1f}, "
                f"Risk: {token.risk_level.value}, "
                f"Liquidity: ${token.liquidity_usd:,.0f}"
            )
            
            return True, token
            
        except Exception as e:
            logger.error(f"Error analyzing pair {pair.base_token_symbol}: {e}", exc_info=True)
            return False, None
    
    def _is_valid_address(self, address: str, chain: Chain) -> bool:
        """Validate address format for chain"""
        if not address or len(address) < 10:
            return False
        
        # EVM chains: Must start with 0x and be 42 chars
        if chain in [Chain.ETH, Chain.BSC, Chain.BASE, Chain.ARBITRUM, Chain.POLYGON]:
            return address.startswith("0x") and len(address) == 42
        
        # Solana: Base58, 32-44 chars, no 0x
        if chain == Chain.SOLANA:
            return not address.startswith("0x") and 32 <= len(address) <= 44
        
        return True  # Other chains: accept for now
    
    async def _verify_token_exists(self, address: str, chain: Chain) -> bool:
        """Verify token still exists on DexScreener by fetching fresh data"""
        try:
            # Try to fetch token data directly
            success, pairs = await self.dex_client.get_token_pairs(address, chain.value)
            
            if not success or not pairs:
                return False
            
            # Check if any pair has meaningful liquidity (not dead)
            has_active_pair = any(p.liquidity_usd > 100 for p in pairs)
            return has_active_pair
            
        except Exception as e:
            logger.debug(f"Error verifying token {address}: {e}")
            return False
    
    async def _enrich_with_whale_activity(self, token: TokenLaunch):
        """
        Enrich token with whale activity and smart money data
        
        Combines:
        1. Smart money tracker (specific wallets buying this token)
        2. CryptoWhaleTracker (general whale transaction insights)
        
        Updates:
        - token.smart_money: List of SmartMoneyActivity
        - token.whale_activity_score: Combined score 0-100
        """
        try:
            smart_money_activities = []
            whale_score = 0.0
            
            # 1. Check if tracked wallets are buying this token
            # Look through recent smart money activities to find this token
            recent_activities = self.smart_money_tracker.recent_activities
            
            # Filter activities that might relate to this token
            # Note: SmartMoneyActivity doesn't store token address directly,
            # so we'd need to check transaction details (would require parsing tx data)
            # For now, we'll use a simplified approach: check all recent BUY activities
            # In production, you'd parse transaction data to extract token addresses
            
            # Count recent smart money buy activities as a signal
            recent_buy_activities = [
                a for a in recent_activities
                if a.action == "BUY" and a.amount_usd >= self.config.min_whale_buy_usd
            ]
            
            if recent_buy_activities:
                # Smart money is active (even if we can't confirm it's this exact token)
                # This boosts confidence that whales are active in the market
                smart_money_boost = min(len(recent_buy_activities) * 5, 30)  # Max 30 points
                whale_score += smart_money_boost
                logger.debug(f"🐋 Smart money activity detected: {len(recent_buy_activities)} recent buys")
            
            # 2. Get general whale activity insights via CryptoWhaleTracker
            try:
                # Convert chain enum to string for whale tracker
                chain_str = token.chain.value if hasattr(token.chain, 'value') else str(token.chain)
                
                # Get whale insights (last 24 hours)
                whale_insights = await self.whale_tracker.get_whale_insights(
                    symbol=token.symbol,
                    chain=chain_str,
                    hours=24
                )
                
                if whale_insights and whale_insights.get('whale_activity_score', 0) > 0:
                    # Combine general whale activity score
                    general_whale_score = whale_insights.get('whale_activity_score', 0)
                    
                    # Weight: 70% general whale activity, 30% smart money boost
                    if whale_score > 0:
                        whale_score = (general_whale_score * 0.7) + (whale_score * 0.3)
                    else:
                        whale_score = general_whale_score
                    
                    # Store smart money activities if any found
                    if whale_insights.get('total_transactions', 0) > 0:
                        logger.info(
                            f"🐋 Whale activity for {token.symbol}: "
                            f"score={whale_score:.1f}, "
                            f"txns={whale_insights.get('total_transactions', 0)}"
                        )
                        
                        # Add alert reason if significant whale activity
                        if whale_score >= 50:
                            token.alert_reasons.append(
                                f"🐋 High whale activity (score: {whale_score:.0f})"
                            )
                            
            except Exception as e:
                logger.debug(f"Could not get whale insights for {token.symbol}: {e}")
            
            # Store results
            token.smart_money = recent_buy_activities[:10]  # Store top 10
            token.whale_activity_score = min(whale_score, 100.0)
            
            if token.whale_activity_score > 0:
                logger.debug(
                    f"📊 {token.symbol} whale activity score: {token.whale_activity_score:.1f}/100"
                )
                
        except Exception as e:
            logger.error(f"Error enriching whale activity for {token.symbol}: {e}", exc_info=True)
            # Set defaults on error
            if not hasattr(token, 'smart_money') or not token.smart_money:
                token.smart_money = []
            token.whale_activity_score = 0.0
    
    def _score_token(self, token: TokenLaunch) -> TokenLaunch:
        """Calculate scoring for token launch potential"""
        
        # FIRST: Calculate launch timing indicators
        token = self._calculate_launch_timing(token)
        
        # 1. Pump Potential Score (0-100)
        pump_score = 0.0
        
        # Age factor (younger = more potential)
        if token.age_hours < 1:
            pump_score += 30  # Very early
        elif token.age_hours < 6:
            pump_score += 20
        elif token.age_hours < 24:
            pump_score += 10
        
        # Liquidity factor (sweet spot: $10k-$500k)
        if 10000 <= token.liquidity_usd <= 500000:
            pump_score += 20
        elif token.liquidity_usd < 10000:
            pump_score += 5  # Too low = risky
        
        # Volume factor
        if token.volume_24h > token.liquidity_usd * 2:
            pump_score += 15  # High volume/liquidity ratio = interest
        
        # Price momentum
        if token.price_change_1h > 10:
            pump_score += 15  # Already moving
        elif token.price_change_1h > 0:
            pump_score += 5
        
        # Safety factor
        if token.contract_safety:
            safety_boost = token.contract_safety.safety_score / 10  # 0-10 points
            pump_score += safety_boost
        
        token.pump_potential_score = min(pump_score, 100.0)
        
        # 2. Velocity Score (price momentum)
        velocity = 0.0
        if token.price_change_5m > 5:
            velocity += 40
        if token.price_change_1h > 10:
            velocity += 30
        if token.price_change_24h > 50:
            velocity += 30
        
        token.velocity_score = min(velocity, 100.0)
        
        # 3. Social Buzz Score (from X sentiment if available)
        # Note: social_buzz_score is populated by _enrich_with_x_sentiment()
        # Don't reset to 0 if already set!
        if not hasattr(token, 'social_buzz_score') or token.social_buzz_score is None:
            token.social_buzz_score = 0.0
        
        # 4. Whale Activity Score (calculated in _enrich_with_whale_activity)
        # If not set yet, default to 0.0
        if not hasattr(token, 'whale_activity_score') or token.whale_activity_score is None:
            token.whale_activity_score = 0.0
        
        # 2.5. VOLUME SPIKE DETECTION (NEW - Catch early momentum!)
        volume_spike_score = 0.0
        volume_surge_detected = False
        
        # Check for transaction surge (5min window)
        if hasattr(token, 'pairs') and token.pairs:
            pair = token.pairs[0]
            
            # High transaction count in 5 min = activity spike
            if pair.txn_count_5m > 50:
                volume_spike_score += 25
                volume_surge_detected = True
            elif pair.txn_count_5m > 30:
                volume_spike_score += 15
            
            # Buy pressure (more buys than sells)
            if pair.buys_5m > pair.sells_5m * 2:  # 2x more buys!
                volume_spike_score += 25
                volume_surge_detected = True
                logger.info(f" BUY PRESSURE: {token.symbol} - {pair.buys_5m} buys vs {pair.sells_5m} sells")
            elif pair.buys_5m > pair.sells_5m * 1.5:
                volume_spike_score += 15
            
            # Volume surge (comparing to liquidity)
            volume_to_liq_ratio = token.volume_24h / max(token.liquidity_usd, 1)
            if volume_to_liq_ratio > 5:  # 5x volume vs liquidity = HOT!
                volume_spike_score += 20
                volume_surge_detected = True
            elif volume_to_liq_ratio > 3:
                volume_spike_score += 10
        
        token.volume_spike_score = min(volume_spike_score, 100.0)
        
        # Mark as priority if volume surge detected
        if volume_surge_detected:
            logger.warning(f" VOLUME SURGE: {token.symbol} - Spike score: {token.volume_spike_score:.0f}")
        
        # 5. Composite Score (weighted average with X sentiment)
        safety_score = token.contract_safety.safety_score if token.contract_safety else 0
        
        # Calculate composite score with available data
        # Weight adjustments based on what data we have
        has_social = token.social_buzz_score > 0
        has_whale = token.whale_activity_score > 0
        
        if has_social and has_whale:
            # Full scoring with all signals
            token.composite_score = (
                token.pump_potential_score * 0.20 +    # 20% pump potential
                token.velocity_score * 0.15 +          # 15% momentum
                token.volume_spike_score * 0.15 +      # 15% volume surge
                token.social_buzz_score * 0.15 +       # 15% X sentiment
                token.whale_activity_score * 0.15 +    # 15% whale activity
                safety_score * 0.20                    # 20% safety
            )
            logger.debug(f"📊 {token.symbol} full score: social={token.social_buzz_score:.1f}, whale={token.whale_activity_score:.1f}")
        elif has_social:
            # X sentiment but no whale data
            token.composite_score = (
                token.pump_potential_score * 0.25 +    # 25% pump potential
                token.velocity_score * 0.20 +          # 20% momentum
                token.volume_spike_score * 0.15 +      # 15% volume surge
                token.social_buzz_score * 0.20 +       # 20% X sentiment
                safety_score * 0.20                    # 20% safety
            )
        elif has_whale:
            # Whale activity but no social data
            token.composite_score = (
                token.pump_potential_score * 0.25 +    # 25% pump potential
                token.velocity_score * 0.20 +          # 20% momentum
                token.volume_spike_score * 0.15 +      # 15% volume surge
                token.whale_activity_score * 0.15 +    # 15% whale activity
                safety_score * 0.25                    # 25% safety
            )
        else:
            # No X or whale data, use original weights
            token.composite_score = (
                token.pump_potential_score * 0.30 +    # 30% pump potential
                token.velocity_score * 0.25 +          # 25% momentum
                token.volume_spike_score * 0.20 +      # 20% volume surge
                safety_score * 0.25                    # 25% safety
            )
        
        # NEW: Calculate profitability score and penalize high-cost tokens
        token = self._calculate_profitability_score(token)
        
        return token
    
    def _calculate_profitability_score(self, token: TokenLaunch, trade_size_usd: float = 100.0) -> TokenLaunch:
        """
        Calculate REAL profitability after slippage, price impact, and fees.
        Penalizes tokens where breakeven requires extreme gains.
        
        Args:
            token: TokenLaunch to analyze
            trade_size_usd: Standard trade size for calculation
            
        Returns:
            TokenLaunch with profitability data added
        """
        liquidity_usd = token.liquidity_usd
        
        # Realistic slippage for meme coins (based on liquidity tier)
        if liquidity_usd < 5000:
            buy_slippage_pct = 12.0  # Very low liq = brutal slippage
            sell_slippage_pct = 15.0
            liquidity_tier = "MICRO (High Risk)"
        elif liquidity_usd < 20000:
            buy_slippage_pct = 8.0
            sell_slippage_pct = 10.0
            liquidity_tier = "LOW"
        elif liquidity_usd < 100000:
            buy_slippage_pct = 5.0
            sell_slippage_pct = 7.0
            liquidity_tier = "MEDIUM"
        elif liquidity_usd < 500000:
            buy_slippage_pct = 3.0
            sell_slippage_pct = 4.0
            liquidity_tier = "GOOD"
        else:
            buy_slippage_pct = 2.0
            sell_slippage_pct = 2.5
            liquidity_tier = "HIGH (Safer)"
        
        # DEX fees (Raydium/Jupiter ~0.25-0.3%)
        dex_fee_pct = 0.3
        
        # Priority fees (Solana)
        priority_fee_usd = 0.50
        
        # 1. Price Impact (AMM constant product approximation)
        buy_price_impact_pct = (trade_size_usd / (2 * max(liquidity_usd, 1))) * 100
        
        # 2. Total buy-side costs
        buy_costs_pct = buy_slippage_pct + buy_price_impact_pct + dex_fee_pct
        
        # 3. Effective entry (what you actually paid per token)
        effective_entry_value = trade_size_usd * (1 - buy_costs_pct / 100)
        
        # 4. Value after 50% displayed gain (test scenario)
        displayed_gain_pct = 50.0
        value_before_sell = effective_entry_value * (1 + displayed_gain_pct / 100)
        
        # 5. Sell-side costs (often WORSE - everyone trying to exit)
        sell_price_impact_pct = (value_before_sell / (2 * max(liquidity_usd, 1))) * 100
        sell_costs_pct = sell_slippage_pct + sell_price_impact_pct + dex_fee_pct
        
        # 6. Final value after selling
        final_value = value_before_sell * (1 - sell_costs_pct / 100) - (priority_fee_usd * 2)
        
        # 7. REAL profit
        real_profit_usd = final_value - trade_size_usd
        real_profit_pct = (real_profit_usd / trade_size_usd) * 100
        
        # 8. Minimum gain needed to break even
        breakeven_multiplier = 1 / ((1 - buy_costs_pct/100) * (1 - sell_costs_pct/100))
        breakeven_gain_pct = (breakeven_multiplier - 1) * 100
        
        # Store profitability data on token
        token.breakeven_gain_needed = round(breakeven_gain_pct, 2)
        token.real_profit_potential = real_profit_pct > 5  # Need at least 5% REAL profit
        token.liquidity_tier = liquidity_tier
        token.total_round_trip_cost_pct = round(buy_costs_pct + sell_costs_pct, 2)
        token.estimated_real_profit_pct = round(real_profit_pct, 2)
        
        # Apply profitability penalty to composite score
        if breakeven_gain_pct > 30:
            # Need 30%+ just to break even = very risky
            old_score = token.composite_score
            token.composite_score *= 0.5
            token.alert_reasons.append(f"⚠️ HIGH COSTS: Need {breakeven_gain_pct:.0f}% to break even")
            logger.warning(f"💸 {token.symbol}: Score penalty 50% (breakeven={breakeven_gain_pct:.0f}%)")
        elif breakeven_gain_pct > 20:
            token.composite_score *= 0.7
            token.alert_reasons.append(f"⚠️ Need {breakeven_gain_pct:.0f}% to break even")
            logger.info(f"💸 {token.symbol}: Score penalty 30% (breakeven={breakeven_gain_pct:.0f}%)")
        elif breakeven_gain_pct > 15:
            token.composite_score *= 0.85
            token.alert_reasons.append(f"ℹ️ Need {breakeven_gain_pct:.0f}% to break even")
        else:
            # Good profitability - boost score slightly
            token.composite_score = min(token.composite_score * 1.05, 100.0)
            token.alert_reasons.append(f"✅ Good profitability (breakeven: {breakeven_gain_pct:.0f}%)")
        
        return token
    
    def _calculate_launch_timing(self, token: TokenLaunch) -> TokenLaunch:
        """Calculate launch timing indicators - how new, missed pump?, breakout potential"""
        
        # Calculate minutes since launch
        token.minutes_since_launch = token.age_hours * 60
        
        # 1. Launch Timing Classification
        if token.age_hours < 0.5:  # < 30 min
            token.launch_timing = "ULTRA_FRESH"
        elif token.age_hours < 2:  # 30min-2hr
            token.launch_timing = "FRESH"
        elif token.age_hours < 12:  # 2-12hr
            token.launch_timing = "EARLY"
        elif token.age_hours < 48:  # 12-48hr
            token.launch_timing = "LATE"
        else:
            token.launch_timing = "MISSED_PUMP"
        
        # 2. Time to Pump Status (based on price action + age)
        if token.price_change_1h > 50 and token.age_hours < 6:
            token.time_to_pump = "DUMPED"  # Already pumped hard, likely dumping
            token.missed_pump_likely = True
        elif token.price_change_1h > 20 and token.age_hours < 3:
            token.time_to_pump = "HEATING"  # Pumping now
        elif token.price_change_1h < 5 and token.age_hours < 1 and token.liquidity_usd > 5000:
            token.time_to_pump = "PRIME"  # Fresh with liquidity, not pumped yet
        elif token.price_change_24h < -30:
            token.time_to_pump = "DUMPED"  # Dumping
            token.missed_pump_likely = True
        else:
            token.time_to_pump = "COOLING"  # Neutral/cooling off
        
        # 3. Timing Advantage Score (0-100, how early you are)
        timing_score = 100.0
        
        # Penalize based on age
        if token.age_hours < 0.25:  # < 15 min
            timing_score = 100  # Perfect timing!
        elif token.age_hours < 1:
            timing_score = 90
        elif token.age_hours < 3:
            timing_score = 75
        elif token.age_hours < 6:
            timing_score = 60
        elif token.age_hours < 12:
            timing_score = 40
        elif token.age_hours < 24:
            timing_score = 20
        else:
            timing_score = 5  # Very late
        
        # Penalize if already pumped
        if token.price_change_1h > 100:
            timing_score *= 0.3  # 70% penalty for massive pump
        elif token.price_change_1h > 50:
            timing_score *= 0.5  # 50% penalty
        elif token.price_change_1h > 20:
            timing_score *= 0.7  # 30% penalty
        
        token.timing_advantage_score = timing_score
        
        # 5. Entry Recommendation (NEW - December 2025)
        # Determines if this is a good entry point based on coin age and pump status
        token.entry_recommendation = self._calculate_entry_recommendation(token)
        
        # 4. Breakout Potential (for super fresh coins)
        breakout_score = 0.0
        
        if token.launch_timing in ["ULTRA_FRESH", "FRESH"]:
            # Base score for being fresh
            breakout_score = 50
            
            # Boost for good liquidity
            if token.liquidity_usd >= 10000:
                breakout_score += 20
            elif token.liquidity_usd >= 5000:
                breakout_score += 10
            
            # Boost for volume activity
            if token.volume_24h > token.liquidity_usd:
                breakout_score += 15
            
            # Boost if price stable/not dumping
            if -5 < token.price_change_1h < 15:
                breakout_score += 10  # Stable = accumulation phase
            
            # Boost for not pumped yet
            if token.time_to_pump == "PRIME":
                breakout_score += 15
            
            # Safety boost
            if token.contract_safety and token.contract_safety.safety_score > 70:
                breakout_score += 10
        
        token.breakout_potential = min(breakout_score, 100.0)
        
        return token
    
    def _calculate_entry_recommendation(self, token: TokenLaunch) -> Dict:
        """
        Calculate entry recommendation based on coin age and pump status.
        
        Philosophy:
        - If initial trading has passed, safer to enter (fomo/rug risk reduced)
        - If there's indication of another pump forming, good to enter
        - Ultra-fresh coins are higher risk but higher reward
        
        Returns:
            Dict with recommendation, reason, confidence, and coin age details
        """
        age_hours = token.age_hours
        age_minutes = age_hours * 60
        
        # Calculate time-based status
        if age_minutes < 5:
            age_status = "🔥 BRAND NEW (< 5 min)"
            initial_trading_passed = False
        elif age_minutes < 15:
            age_status = "⚡ VERY FRESH (5-15 min)"
            initial_trading_passed = False
        elif age_minutes < 30:
            age_status = "🆕 FRESH (15-30 min)"
            initial_trading_passed = True  # Initial FOMO likely settling
        elif age_hours < 1:
            age_status = "📊 SETTLING (30-60 min)"
            initial_trading_passed = True
        elif age_hours < 3:
            age_status = "✅ ESTABLISHED (1-3 hr)"
            initial_trading_passed = True
        elif age_hours < 6:
            age_status = "📈 MATURING (3-6 hr)"
            initial_trading_passed = True
        elif age_hours < 24:
            age_status = "🕐 AGED (6-24 hr)"
            initial_trading_passed = True
        else:
            age_status = f"⏰ OLD ({age_hours:.0f}+ hr)"
            initial_trading_passed = True
        
        # Detect pump patterns for second wave entry
        pump_forming = False
        pump_reason = ""
        
        # Check for accumulation pattern (price stable, volume increasing)
        if hasattr(token, 'pairs') and token.pairs:
            pair = token.pairs[0]
            volume_to_liq = token.volume_24h / max(token.liquidity_usd, 1)
            
            # Accumulation: price relatively stable but volume picking up
            if -10 < token.price_change_1h < 20 and volume_to_liq > 2:
                pump_forming = True
                pump_reason = "Accumulation pattern - volume building while price stable"
            
            # Buy pressure building
            if hasattr(pair, 'buys_5m') and hasattr(pair, 'sells_5m'):
                if pair.buys_5m > pair.sells_5m * 1.5 and pair.buys_5m > 10:
                    pump_forming = True
                    pump_reason = f"Buy pressure building ({pair.buys_5m} buys vs {pair.sells_5m} sells in 5m)"
            
            # Recovery from dip (potential second wave)
            if token.price_change_24h < -20 and token.price_change_1h > 10:
                pump_forming = True
                pump_reason = "Recovery bounce - potential second wave forming"
            
            # Breakout from consolidation
            if token.price_change_5m > 5 and abs(token.price_change_1h) < 15:
                pump_forming = True
                pump_reason = "Breakout from consolidation detected"
        
        # Generate recommendation
        if age_minutes < 5:
            recommendation = "⚠️ HIGH RISK ENTRY"
            reason = "Ultra-fresh coin - maximum volatility, high rug risk, but highest potential gains"
            confidence = 40
        elif not initial_trading_passed and not pump_forming:
            recommendation = "⏳ WAIT FOR SETTLING"
            reason = f"Initial trading still active ({age_minutes:.0f} min old) - wait for FOMO to cool"
            confidence = 30
        elif initial_trading_passed and pump_forming:
            recommendation = "🚀 GOOD ENTRY - PUMP FORMING"
            reason = f"Initial trading passed + {pump_reason}"
            confidence = 75
        elif initial_trading_passed and token.price_change_1h < -10:
            recommendation = "💰 POTENTIAL DIP BUY"
            reason = f"Initial trading passed, price down {token.price_change_1h:.1f}% - could be accumulation zone"
            confidence = 60
        elif initial_trading_passed:
            recommendation = "✅ SAFER ENTRY"
            reason = f"Initial trading passed ({age_minutes:.0f} min old) - reduced FOMO/rug risk"
            confidence = 55
        elif token.time_to_pump == "PRIME":
            recommendation = "🎯 PRIME ENTRY"
            reason = "Fresh coin with good liquidity, hasn't pumped yet"
            confidence = 70
        else:
            recommendation = "📊 EVALUATE"
            reason = "Mixed signals - review other metrics"
            confidence = 45
        
        # Build entry details
        entry_info = {
            "recommendation": recommendation,
            "reason": reason,
            "confidence": confidence,
            "age_status": age_status,
            "age_minutes": round(age_minutes, 1),
            "age_hours": round(age_hours, 2),
            "initial_trading_passed": initial_trading_passed,
            "pump_forming": pump_forming,
            "pump_reason": pump_reason if pump_forming else None,
            "timing_summary": self._get_timing_summary(token, initial_trading_passed, pump_forming)
        }
        
        return entry_info
    
    def _get_timing_summary(self, token: TokenLaunch, initial_passed: bool, pump_forming: bool) -> str:
        """Generate a human-readable timing summary for entry decision"""
        age_minutes = token.age_hours * 60
        
        if age_minutes < 5:
            return "⚡ ULTRA-FRESH: High risk/reward. Only for experienced degen traders."
        elif not initial_passed:
            return f"⏳ SETTLING: Coin is {age_minutes:.0f}m old. Wait for initial FOMO to cool (15-30m mark)."
        elif pump_forming:
            return "🚀 PUMP SIGNAL: Initial trading passed AND new pump indicators detected. Good entry window."
        elif token.price_change_1h < -20:
            return "📉 DIP OPPORTUNITY: Price down but coin survived initial phase. Potential accumulation zone."
        elif token.price_change_1h > 30:
            return "⚠️ ALREADY PUMPING: Consider waiting for pullback or smaller position."
        else:
            return "✅ STABLE: Initial chaos passed. Price action more predictable. Safer entry."
    
    def _get_launch_stage(self, age_hours: float) -> LaunchStage:
        """Determine launch stage based on age"""
        if age_hours < 1:
            return LaunchStage.JUST_LAUNCHED
        elif age_hours < 24:
            return LaunchStage.EARLY
        elif age_hours < 168:  # 7 days
            return LaunchStage.ESTABLISHING
        else:
            return LaunchStage.MATURE
    
    async def _process_discovered_tokens(self):
        """Process and update scores for discovered tokens"""
        for address, token in self.discovered_tokens.items():
            try:
                # Re-score based on latest data
                self.discovered_tokens[address] = self._score_token(token)
            except Exception as e:
                logger.error(f"Error processing token {address}: {e}")
        
        # Enrich top candidates with X sentiment (only for promising tokens)
        if self.x_sentiment_service:
            await self._enrich_with_x_sentiment()
    
    async def _enrich_with_x_sentiment(self):
        """
        Fetch X sentiment for top tokens by DEX score.
        Only enriches promising tokens to conserve API/scraping budget.
        """
        if not self.x_sentiment_service:
            print("[X] X Sentiment Service not available - skipping social enrichment", flush=True)
            return
        
        print("[X] 🐦 Starting X/Twitter sentiment enrichment...", flush=True)
        
        # Get top tokens by composite score (that don't have X data yet)
        tokens_needing_x = [
            token for token in self.discovered_tokens.values()
            if token.social_buzz_score == 0.0  # Not yet enriched
            and token.composite_score >= 30  # Only promising ones (min score to check X)
            and token.age_hours < 24  # Fresh tokens only
        ]
        
        print(f"[X] Found {len(tokens_needing_x)} tokens needing X sentiment (score>=30, age<24h)", flush=True)
        
        # Sort by score, take top 10 per cycle
        tokens_needing_x = sorted(
            tokens_needing_x, 
            key=lambda t: t.composite_score, 
            reverse=True
        )[:10]
        
        for token in tokens_needing_x:
            try:
                print(f"[X] 🐦 Fetching X/Twitter sentiment for ${token.symbol}...", flush=True)
                logger.info(f"\U0001F426 Fetching X sentiment for {token.symbol}...")
                
                snapshot = await self.x_sentiment_service.fetch_snapshot(
                    symbol=token.symbol,
                    max_tweets=50
                )
                
                if snapshot and snapshot.tweet_count > 0:
                    # Update social signals on token
                    token.social_signals = SocialSignal(
                        source="x_twitter",
                        mention_count=snapshot.tweet_count,
                        sentiment_score=snapshot.heuristic_sentiment,
                        engagement_score=snapshot.engagement_velocity * 100,
                    )
                    
                    # Update social buzz score (0-100)
                    token.social_buzz_score = snapshot.x_sentiment_score
                    
                    # Re-calculate composite score with social data
                    old_score = token.composite_score
                    token = self._score_token(token)
                    
                    # Update in storage
                    self.discovered_tokens[token.contract_address.lower()] = token
                    
                    print(f"[X] ✓ ${token.symbol}: {snapshot.tweet_count} tweets, sentiment={snapshot.heuristic_sentiment:.2f}, buzz={token.social_buzz_score:.1f}", flush=True)
                    print(f"[X]   └─ Score updated: {old_score:.1f} → {token.composite_score:.1f}", flush=True)
                    logger.info(
                        f"\u2705 X sentiment for {token.symbol}: "
                        f"tweets={snapshot.tweet_count}, "
                        f"sentiment={snapshot.heuristic_sentiment:.2f}, "
                        f"buzz_score={token.social_buzz_score:.1f}"
                    )
                else:
                    print(f"[X] ✗ ${token.symbol}: No tweets found", flush=True)
                    logger.debug(f"No X data found for {token.symbol}")
                
                # Small delay between X requests (be polite)
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"[X] ⚠ ${token.symbol}: Error fetching X data - {e}", flush=True)
                logger.warning(f"Failed to fetch X sentiment for {token.symbol}: {e}")
    
    async def _generate_alerts(self):
        """Generate alerts for promising tokens"""
        alert_candidates = len([t for t in self.discovered_tokens.values() if not t.notified])
        print(f"[ALERT] 🔔 Checking {alert_candidates} tokens for alert criteria (score>=60)...", flush=True)
        
        alerts_sent = 0
        for address, token in self.discovered_tokens.items():
            # Skip if already notified
            if token.notified:
                continue
            
            # Check if meets alert criteria
            should_alert, reasons = self._should_alert(token)
            
            if should_alert:
                # Create verification checklist with research links
                checklist = self._create_verification_checklist(token)
                
                alert = LaunchAlert(
                    token=token,
                    alert_type="NEW_LAUNCH",
                    priority=self._get_alert_priority(token),
                    message=self._generate_alert_message(token),
                    reasoning=reasons,
                    checklist=checklist
                )
                
                self.recent_alerts.append(alert)
                token.notified = True
                self.total_alerts += 1
                
                logger.info(
                    f"🚨 ALERT: {token.symbol} - {alert.priority} - "
                    f"Score: {token.composite_score:.1f}"
                )
                
                # Send to execution webhook if configured and priority is CRITICAL
                if token.chain == Chain.SOLANA and alert.priority in ["CRITICAL", "HIGH"]:
                    asyncio.create_task(
                        self._route_to_execution_service(token, alert)
                    )
    
    def _should_alert(self, token: TokenLaunch) -> Tuple[bool, List[str]]:
        """Determine if token should trigger an alert"""
        reasons = []
        
        # Check minimum score
        if token.composite_score < self.config.min_composite_score:
            return False, []
        
        # Check risk level
        risk_levels = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.EXTREME]
        if risk_levels.index(token.risk_level) > risk_levels.index(self.config.max_risk_level):
            return False, []
        
        # Check liquidity
        if token.liquidity_usd < self.config.min_liquidity_usd:
            return False, []
        
        if token.liquidity_usd > self.config.max_liquidity_usd:
            return False, []  # Too established
        
        # Build alert reasons
        if token.age_hours < 1:
            reasons.append("🆕 Just launched (< 1 hour)")
        
        if token.pump_potential_score >= 70:
            reasons.append(f"📈 High pump potential ({token.pump_potential_score:.0f}/100)")
        
        if token.velocity_score >= 60:
            reasons.append(f"🚀 Strong momentum ({token.velocity_score:.0f}/100)")
        
        if token.contract_safety and token.contract_safety.safety_score >= 70:
            reasons.append(f"✅ Safe contract ({token.contract_safety.safety_score:.0f}/100)")
        
        if token.liquidity_usd >= 50000:
            reasons.append(f"💰 Good liquidity (${token.liquidity_usd:,.0f})")
        
        # X Sentiment reasons (NEW!)
        if token.social_buzz_score >= 60:
            reasons.append(f"🐦 Strong X buzz ({token.social_buzz_score:.0f}/100)")
        elif token.social_buzz_score >= 40:
            reasons.append(f"🐦 Moderate X activity ({token.social_buzz_score:.0f}/100)")
        
        if token.social_signals and token.social_signals.mention_count >= 20:
            reasons.append(f"📢 High X mentions ({token.social_signals.mention_count})")
        
        return len(reasons) > 0, reasons
    
    def _get_alert_priority(self, token: TokenLaunch) -> str:
        """Determine alert priority"""
        if token.composite_score >= 80:
            return "CRITICAL"
        elif token.composite_score >= 65:
            return "HIGH"
        elif token.composite_score >= 50:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_alert_message(self, token: TokenLaunch) -> str:
        """Generate alert message with profitability warning"""
        # Get profitability data from token
        breakeven = getattr(token, 'breakeven_gain_needed', 0)
        liquidity_tier = getattr(token, 'liquidity_tier', 'UNKNOWN')
        total_costs = getattr(token, 'total_round_trip_cost_pct', 0)
        real_profit = getattr(token, 'estimated_real_profit_pct', 0)
        
        # Get entry recommendation (NEW - December 2025)
        entry_rec = getattr(token, 'entry_recommendation', None) or {}
        entry_status = entry_rec.get('recommendation', 'EVALUATE')
        entry_reason = entry_rec.get('reason', '')
        age_status = entry_rec.get('age_status', f"{token.age_hours:.1f} hours")
        timing_summary = entry_rec.get('timing_summary', '')
        initial_passed = entry_rec.get('initial_trading_passed', token.age_hours > 0.5)
        pump_forming = entry_rec.get('pump_forming', False)
        
        msg = (
            f"🚀 **{token.symbol}** on {token.chain.value}\n"
            f"💰 Price: ${token.price_usd:.8f}\n"
            f"💧 Liquidity: ${token.liquidity_usd:,.0f} ({liquidity_tier})\n"
            f"📊 Score: {token.composite_score:.1f}/100\n"
            f"⚠️ Risk: {token.risk_level.value}\n"
            f"\n"
            f"**⏰ COIN AGE & ENTRY TIMING:**\n"
            f"• Age: {age_status}\n"
            f"• Initial Trading Passed: {'✅ YES' if initial_passed else '⏳ NO'}\n"
            f"• Pump Signal: {'🚀 YES' if pump_forming else '❌ NO'}\n"
            f"• **{entry_status}**\n"
            f"• {timing_summary}\n"
            f"\n"
            f"**💸 REAL COST ANALYSIS ($100 trade):**\n"
            f"• Round-trip costs: ~{total_costs:.1f}%\n"
            f"• Breakeven requires: **{breakeven:.0f}%+ gain**\n"
            f"• If price +50%: Real profit = **{real_profit:.1f}%**\n"
        )
        
        # Add warning if breakeven is high
        if breakeven > 20:
            msg += f"\n⚠️ **WARNING**: Need {breakeven:.0f}%+ displayed gain just to break even!\n"
        
        return msg
    
    def _create_verification_checklist(self, token: TokenLaunch) -> VerificationChecklist:
        """Create verification checklist with research links for manual checks"""
        
        # Build chain-specific explorer URL
        if token.chain == Chain.ETH:
            explorer_base = "https://etherscan.io"
            holders_url = f"{explorer_base}/token/{token.contract_address}#balances"
        elif token.chain == Chain.BSC:
            explorer_base = "https://bscscan.com"
            holders_url = f"{explorer_base}/token/{token.contract_address}#balances"
        elif token.chain == Chain.SOLANA:
            explorer_base = "https://solscan.io"
            holders_url = f"{explorer_base}/token/{token.contract_address}#holders"
        else:
            explorer_base = ""
            holders_url = ""
        
        etherscan_url = f"{explorer_base}/address/{token.contract_address}" if explorer_base else ""
        
        # DexScreener URL (works for all chains)
        dexscreener_url = f"https://dexscreener.com/{token.chain.value}/{token.contract_address}"
        
        # Social media search URLs
        twitter_search_url = f"https://twitter.com/search?q=%24{token.symbol}+OR+{token.symbol}&f=live"
        telegram_search_url = f"https://t.me/s/uniswaplistings?q={token.symbol}"
        
        checklist = VerificationChecklist(
            dexscreener_url=dexscreener_url,
            etherscan_url=etherscan_url,
            telegram_search_url=telegram_search_url,
            twitter_search_url=twitter_search_url,
            holders_url=holders_url
        )
        
        return checklist
    
    async def _route_to_execution_service(self, token: TokenLaunch, alert: LaunchAlert):
        """
        Route high-priority alerts to execution service via webhook
        
        HIGH-LEVEL FLOW:
        1. Only routes Solana CRITICAL/HIGH priority tokens
        2. Sends token data to execution webhook
        3. External service (bundler, sniper) handles actual execution
        4. This function is non-blocking (fire-and-forget)
        
        Args:
            token: TokenLaunch object
            alert: LaunchAlert with priority info
        """
        try:
            # Only route Solana tokens
            if token.chain != Chain.SOLANA:
                logger.debug(f"[EXECUTION] Skipping non-Solana token {token.symbol}")
                return
            
            # Only route high-priority alerts
            if alert.priority not in ["CRITICAL", "HIGH"]:
                logger.debug(f"[EXECUTION] Skipping low-priority alert for {token.symbol}")
                return
            
            logger.info(
                f"[EXECUTION] 🚀 Routing {alert.priority} alert to execution service: {token.symbol}"
            )
            
            # Prepare metadata
            metadata = {
                'symbol': token.symbol,
                'chain': token.chain.value,
                'liquidity_usd': token.liquidity_usd,
                'age_hours': token.age_hours,
                'safety_score': token.safety_score,
                'composite_score': token.composite_score,
                'risk_level': token.risk_level.value,
                'dexscreener_url': f"https://dexscreener.com/{token.chain.value}/{token.contract_address}",
                'alert_reasons': alert.reasoning
            }
            
            # Suggest execution amount (conservative: 5-50 USD based on safety)
            if token.safety_score >= 80:
                suggested_amount = 50.0
            elif token.safety_score >= 60:
                suggested_amount = 25.0
            else:
                suggested_amount = 5.0
            
            # Route to execution webhook
            success, message, _ = await self.execution_webhook.execute_snipe(
                token_mint=token.contract_address,
                amount_usd=suggested_amount,
                slippage_bps=50,  # 0.5%
                request_id=f"dex_{token.contract_address[:8]}_{int(datetime.now().timestamp())}",
                metadata=metadata
            )
            
            if success:
                logger.info(
                    f"[EXECUTION] ✅ Successfully sent to execution service: {token.symbol} "
                    f"(${suggested_amount:.2f}) - {message}"
                )
            else:
                logger.warning(
                    f"[EXECUTION] ⚠️ Execution service returned warning: {token.symbol} - {message}"
                )
            
        except Exception as e:
            logger.error(
                f"[EXECUTION] Error routing {token.symbol} to execution service: {e}",
                exc_info=True
            )
    
    def _cleanup_old_data(self):
        """Remove old discovered tokens and alerts"""
        # Remove tokens older than 7 days
        cutoff = datetime.now() - timedelta(days=7)
        
        tokens_to_remove = []
        for address, token in self.discovered_tokens.items():
            if token.detected_at < cutoff:
                tokens_to_remove.append(address)
        
        for address in tokens_to_remove:
            del self.discovered_tokens[address]
        
        # Keep only last 500 alerts
        if len(self.recent_alerts) > 500:
            self.recent_alerts = self.recent_alerts[-500:]
    
    def get_stats(self) -> Dict:
        """Get hunter statistics"""
        return {
            "total_scanned": self.total_scanned,
            "total_discovered": len(self.discovered_tokens),
            "total_alerts": self.total_alerts,
            "blacklisted_tokens": len(self.blacklisted_tokens),
            "tracked_wallets": len(self.smart_money_tracker.watched_wallets),
            "is_running": self.is_running
        }
    
    def get_top_opportunities(self, limit: int = 20) -> List[TokenLaunch]:
        """Get top scored tokens"""
        sorted_tokens = sorted(
            self.discovered_tokens.values(),
            key=lambda t: t.composite_score,
            reverse=True
        )
        return sorted_tokens[:limit]
    
    def get_recent_alerts(self, limit: int = 50) -> List[LaunchAlert]:
        """Get recent alerts"""
        return self.recent_alerts[-limit:]


# Singleton instance
_hunter_instance: Optional[DexLaunchHunter] = None


def get_dex_launch_hunter(config: Optional[HunterConfig] = None) -> DexLaunchHunter:
    """Get or create DEX Launch Hunter singleton"""
    global _hunter_instance
    
    if _hunter_instance is None:
        _hunter_instance = DexLaunchHunter(config=config)
    
    return _hunter_instance

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
        
        logger.info("DEX Launch Hunter initialized")
    
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
        logger.info("Scanning for new launches...")
        
        try:
            # Get chains to scan
            chain_ids = [chain.value for chain in self.config.enabled_chains]
            
            # Scan using FREE DexScreener API endpoints
            success, new_pairs = await self.dex_client.get_new_pairs(
                chains=chain_ids,
                min_liquidity=self.config.min_liquidity_usd,
                max_liquidity=self.config.max_liquidity_usd,
                max_age_hours=self.config.max_age_hours,
                limit=50  # Scan top 50 new tokens per cycle
            )
            
            if not success:
                logger.warning("Failed to fetch new pairs from DexScreener")
                return
            
            logger.info(f"Found {len(new_pairs)} new pairs to analyze")
            
            # Analyze each new pair
            for pair in new_pairs:
                try:
                    # Skip if already discovered
                    if pair.base_token_address.lower() in self.discovered_tokens:
                        continue
                    
                    # Skip if blacklisted
                    if pair.base_token_address.lower() in self.blacklisted_tokens:
                        continue
                    
                    self.total_scanned += 1
                    
                    # We already have the pair from search, use it directly for faster analysis
                    success, token = await self._analyze_pair_directly(pair)
                    
                    if success and token:
                        logger.info(
                            f"Discovered: {token.symbol} - "
                            f"Score: {token.composite_score:.1f}, "
                            f"Risk: {token.risk_level.value}"
                        )
                    
                    # Small delay to avoid rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error analyzing pair {pair.base_token_symbol}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in scan loop: {e}", exc_info=True)
    
    async def _check_smart_money(self):
        """Check for smart money wallet activity"""
        try:
            activities = await self.smart_money_tracker.check_all_wallets()
            
            if not activities:
                return
            
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
            
            # 2. Safety analysis
            success, safety = await self.safety_analyzer.analyze_token(contract_address, chain)
            
            if not success:
                logger.warning(f"Safety analysis failed for {contract_address}")
                safety = ContractSafety()
            
            # 3. Determine risk level
            risk_level = self.safety_analyzer.get_risk_level(safety)
            
            # 4. Check if it passes minimum safety requirements
            if self.config.verify_contract_before_alert:
                if safety.is_honeypot and self.config.auto_blacklist_honeypots:
                    logger.warning(f"Blacklisting honeypot: {contract_address}")
                    self.blacklisted_tokens.add(contract_address.lower())
                    return False, None
                
                if (safety.buy_tax > 15 or safety.sell_tax > 15) and self.config.auto_blacklist_high_tax:
                    logger.warning(f"Blacklisting high tax token: {contract_address}")
                    self.blacklisted_tokens.add(contract_address.lower())
                    return False, None
            
            # 5. Build TokenLaunch object
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
                launched_at=primary_pair.created_at
            )
            
            # 6. Determine launch stage
            token.launch_stage = self._get_launch_stage(token.age_hours)
            
            # 7. Score the token
            token = self._score_token(token)
            
            # 8. Store in discovered tokens
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
                logger.debug(f"Invalid address format for {pair.chain.value}: {contract_address}")
                return False, None
            
            # VALIDATION 2: Verify token still exists on DexScreener (catch delisted/stale)
            verified = await self._verify_token_exists(contract_address, pair.chain)
            if not verified:
                logger.debug(f"Token not found on DexScreener (delisted?): {pair.base_token_symbol}")
                return False, None
            
            # Check if blacklisted
            if contract_address.lower() in self.blacklisted_tokens:
                return False, None
            
            #  Safety analysis
            success, safety = await self.safety_analyzer.analyze_token(contract_address, pair.chain)
            
            if not success:
                logger.warning(f"Safety analysis failed for {contract_address}")
                safety = ContractSafety()
            
            # Determine risk level
            risk_level = self.safety_analyzer.get_risk_level(safety)
            
            # Check minimum safety requirements
            if self.config.verify_contract_before_alert:
                if safety.is_honeypot and self.config.auto_blacklist_honeypots:
                    logger.warning(f"Blacklisting honeypot: {contract_address}")
                    self.blacklisted_tokens.add(contract_address.lower())
                    return False, None
                
                if (safety.buy_tax > 15 or safety.sell_tax > 15) and self.config.auto_blacklist_high_tax:
                    logger.warning(f"Blacklisting high tax token: {contract_address}")
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
                launched_at=pair.created_at
            )
            
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
        
        # 4. Whale Activity Score (placeholder - implement with smart money tracker)
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
        
        # If we have X sentiment data, include it in scoring
        if token.social_buzz_score > 0:
            token.composite_score = (
                token.pump_potential_score * 0.25 +    # 25% pump potential
                token.velocity_score * 0.20 +          # 20% momentum
                token.volume_spike_score * 0.15 +      # 15% volume surge
                token.social_buzz_score * 0.20 +       # 20% X sentiment (NEW!)
                safety_score * 0.20                    # 20% safety
            )
            logger.debug(f"\U0001F426 {token.symbol} score includes X sentiment: {token.social_buzz_score:.1f}")
        else:
            # No X data yet, use original weights
            token.composite_score = (
                token.pump_potential_score * 0.30 +    # 30% pump potential
                token.velocity_score * 0.25 +          # 25% momentum
                token.volume_spike_score * 0.20 +      # 20% volume surge
                safety_score * 0.25                    # 25% safety
            )
        
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
            return
        
        # Get top tokens by composite score (that don't have X data yet)
        tokens_needing_x = [
            token for token in self.discovered_tokens.values()
            if token.social_buzz_score == 0.0  # Not yet enriched
            and token.composite_score >= 30  # Only promising ones
            and token.age_hours < 24  # Fresh tokens only
        ]
        
        # Sort by score, take top 10 per cycle
        tokens_needing_x = sorted(
            tokens_needing_x, 
            key=lambda t: t.composite_score, 
            reverse=True
        )[:10]
        
        for token in tokens_needing_x:
            try:
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
                    token = self._score_token(token)
                    
                    # Update in storage
                    self.discovered_tokens[token.contract_address.lower()] = token
                    
                    logger.info(
                        f"\u2705 X sentiment for {token.symbol}: "
                        f"tweets={snapshot.tweet_count}, "
                        f"sentiment={snapshot.heuristic_sentiment:.2f}, "
                        f"buzz_score={token.social_buzz_score:.1f}"
                    )
                else:
                    logger.debug(f"No X data found for {token.symbol}")
                
                # Small delay between X requests (be polite)
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"Failed to fetch X sentiment for {token.symbol}: {e}")
    
    async def _generate_alerts(self):
        """Generate alerts for promising tokens"""
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
        """Generate alert message"""
        return (
            f"New Launch Detected: {token.symbol}\n"
            f"Chain: {token.chain.value}\n"
            f"Price: ${token.price_usd:.8f}\n"
            f"Liquidity: ${token.liquidity_usd:,.0f}\n"
            f"Age: {token.age_hours:.1f} hours\n"
            f"Score: {token.composite_score:.1f}/100\n"
            f"Risk: {token.risk_level.value}"
        )
    
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

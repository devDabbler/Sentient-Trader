"""
Crypto Breakout Background Monitor

Continuously monitors cryptocurrency markets for high-confidence breakout patterns
and sends Discord notifications for quick trade execution.

Features:
- Multi-indicator breakout detection (Volume, EMA, MACD, RSI, Bollinger Bands)
- Configurable scan intervals
- Discord webhook integration for instant alerts
- Comprehensive technical analysis in notifications
- Tracks recent alerts to avoid spam
- Watchlist integration
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from loguru import logger
import requests
from dotenv import load_dotenv

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from clients.kraken_client import KrakenClient
from clients.jupiter_client import get_jupiter_client
from services.crypto_scanner import CryptoOpportunityScanner
from services.ai_crypto_scanner import AICryptoScanner
from services.pre_listing_scanner import PreListingScanner
from services.crypto_watchlist_manager import CryptoWatchlistManager
from services.ticker_manager import TickerManager
from utils.crypto_pair_utils import normalize_crypto_pair, extract_base_asset
from clients.supabase_client import get_supabase_client
from windows_services.runners.service_config_loader import save_analysis_results
# NOTE: get_pending_analysis_requests and mark_analysis_complete are handled by run_analysis_queue_processor.py


load_dotenv()

# Configure logger to also output to console
logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/crypto_breakout_monitor.log", level="DEBUG", rotation="10 MB")


@dataclass
class BreakoutAlert:
    """Breakout alert data"""
    symbol: str
    alert_type: str  # 'BREAKOUT', 'BUZZING', 'HOTTEST'
    score: float
    confidence: str
    price: float
    change_24h: float
    volume_ratio: float
    
    # Technical indicators
    rsi: float
    ema_8: float
    ema_20: float
    ema_50: float
    macd_signal: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    bb_position: str  # 'UPPER', 'MIDDLE', 'LOWER'
    
    # AI analysis (if available)
    ai_confidence: Optional[str] = None
    ai_reasoning: Optional[str] = None
    ai_rating: Optional[float] = None
    
    # Jupiter DEX price validation (for crypto pairs)
    jupiter_validated: bool = False
    jupiter_price: Optional[float] = None
    jupiter_spread_pct: Optional[float] = None
    jupiter_liquidity_usd: Optional[float] = None
    jupiter_arbitrage_opportunity: bool = False
    
    # Meta
    timestamp: Optional[str] = None
    scan_duration: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class CryptoBreakoutMonitor:
    """
    Background service that continuously monitors crypto markets for breakouts
    and sends Discord alerts for high-confidence trading opportunities.
    """
    
    def __init__(
        self,
        scan_interval_seconds: int = 300,  # 5 minutes default
        min_score: float = 70.0,
        min_confidence: str = 'HIGH',
        use_ai: bool = True,
        use_watchlist: bool = True,
        alert_cooldown_minutes: int = 60,  # Don't alert same crypto within 60 min
        auto_add_to_watchlist: bool = True  # Auto-add detected coins to watchlist
    ):
        """
        Initialize crypto breakout monitor
        
        Args:
            scan_interval_seconds: How often to scan (default 5 minutes)
            min_score: Minimum score threshold for alerts (0-100)
            min_confidence: Minimum confidence level ('HIGH', 'MEDIUM', 'LOW')
            use_ai: Whether to use AI analysis (requires OpenRouter API key)
            use_watchlist: Whether to monitor user's watchlist
            alert_cooldown_minutes: Minutes to wait before alerting same crypto again
        """
        self.scan_interval = scan_interval_seconds
        self.min_score = min_score
        self.min_confidence = min_confidence
        self.use_ai = use_ai
        self.use_watchlist = use_watchlist
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        self.auto_add_to_watchlist = auto_add_to_watchlist
        
        # Track recent alerts to avoid spam
        self.recent_alerts: Dict[str, datetime] = {}
        
        # Track coins added to watchlist in this session
        self.watchlist_added: Set[str] = set()
        
        # Discord webhook (for simple embeds) - use channel routing
        try:
            from src.integrations.discord_channels import get_discord_webhook, AlertCategory
            self.discord_webhook = get_discord_webhook(AlertCategory.CRYPTO_ALERTS)
        except ImportError:
            self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        
        if not self.discord_webhook:
            logger.warning("⚠️ DISCORD_WEBHOOK not set - alerts will be logged only")
        
        # Discord bot for interactive buttons (Watch/Analyze/Dismiss)
        self.discord_bot_manager = None
        try:
            from services.discord_trade_approval import get_discord_approval_manager
            self.discord_bot_manager = get_discord_approval_manager()
            if self.discord_bot_manager and self.discord_bot_manager.enabled:
                logger.info("✅ Discord bot enabled - alerts will have interactive buttons")
            else:
                logger.info("ℹ️ Discord bot not configured - using webhook only (no buttons)")
                self.discord_bot_manager = None
        except Exception as e:
            logger.debug(f"Discord bot not available: {e}")
            self.discord_bot_manager = None
        
        # Initialize Kraken client
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set")
        
        logger.info("🔧 Initializing Kraken client...")
        self.kraken_client = KrakenClient(api_key=api_key, api_secret=api_secret)
        success, message = self.kraken_client.validate_connection()
        
        if not success:
            raise ConnectionError(f"Failed to connect to Kraken: {message}")
        
        logger.info(f"✅ {message}")
        
        # Initialize Jupiter client for multi-DEX price validation
        logger.info("🔧 Initializing Jupiter Aggregator client...")
        try:
            self.jupiter_client = get_jupiter_client()
            logger.info("✅ Jupiter client ready - multi-DEX price validation enabled")
        except Exception as e:
            logger.warning(f"⚠️ Jupiter client initialization failed: {e} - validation disabled")
            self.jupiter_client = None
        
        # Initialize scanners
        if self.use_ai:
            logger.info("🤖 Initializing AI-powered crypto scanner...")
            self.scanner = AICryptoScanner(self.kraken_client, config=None)
            logger.info("✅ AI scanner ready")
        else:
            logger.info("📊 Initializing quantitative crypto scanner...")
            self.scanner = CryptoOpportunityScanner(self.kraken_client, config=None)
            logger.info("✅ Quantitative scanner ready")
        
        # Initialize pre-listing scanner
        logger.info("🚀 Initializing pre-listing/new launch scanner...")
        self.pre_listing_scanner = PreListingScanner(self.kraken_client)
        logger.info("✅ Pre-listing scanner ready")
        
        # Initialize Supabase for watchlist (optional)
        self.supabase_client = None
        self.watchlist_manager = None
        if self.use_watchlist or self.auto_add_to_watchlist:
            try:
                self.supabase_client = get_supabase_client()
                self.watchlist_manager = CryptoWatchlistManager()
                logger.info("✅ Supabase client connected for watchlist monitoring")
                
                # Get existing watchlist symbols
                if self.watchlist_manager:
                    existing_symbols = self.watchlist_manager.get_watchlist_symbols()
                    logger.info(f"📋 Current watchlist has {len(existing_symbols)} coins")
            except Exception as e:
                logger.warning(f"⚠️ Could not connect to Supabase: {e}")
                logger.info("Continuing without watchlist integration")
                self.use_watchlist = False
                self.auto_add_to_watchlist = False
        
        # Statistics
        self.stats = {
            'scans_completed': 0,
            'alerts_sent': 0,
            'errors': 0,
            'start_time': datetime.now().isoformat()
        }
        
        logger.info("=" * 80)
        logger.info("🚀 Crypto Breakout Monitor Initialized")
        logger.info(f"   Scan Interval: {scan_interval_seconds}s")
        logger.info(f"   Min Score: {min_score}")
        logger.info(f"   Min Confidence: {min_confidence}")
        logger.info(f"   AI Analysis: {'Enabled' if use_ai else 'Disabled'}")
        logger.info(f"   Watchlist Monitoring: {'Enabled' if use_watchlist else 'Disabled'}")
        logger.info(f"   Auto-Add to Watchlist: {'Enabled' if auto_add_to_watchlist else 'Disabled'}")
        logger.info(f"   Alert Cooldown: {alert_cooldown_minutes} minutes")
        logger.info(f"   Discord Alerts: {'Enabled' if self.discord_webhook else 'Disabled'}")
        logger.info("=" * 80)
    
    def start(self):
        """Start the monitoring loop"""
        logger.info("🎬 Starting crypto breakout monitoring loop...")
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while True:
                scan_start = time.time()
                
                try:
                    # NOTE: Analysis queue processing is handled by run_analysis_queue_processor.py
                    # Do NOT process here to avoid duplicate analyses
                    
                    # Perform scan
                    self._scan_and_alert()
                    self.stats['scans_completed'] += 1
                    
                except Exception as e:
                    logger.error("❌ Error during scan: {}", str(e), exc_info=True)
                    self.stats['errors'] += 1
                
                # Calculate sleep time
                scan_duration = time.time() - scan_start
                sleep_time = max(0, self.scan_interval - scan_duration)
                
                # Log stats
                logger.info(f"\n📊 Session Stats:")
                logger.info(f"   Scans: {self.stats['scans_completed']}")
                logger.info(f"   Alerts: {self.stats['alerts_sent']}")
                logger.info(f"   Errors: {self.stats['errors']}")
                pass  # logger.info(f"   Uptime: {self._get_uptime(}")
                
                if sleep_time > 0:
                    next_scan = datetime.now() + timedelta(seconds=sleep_time)
                    pass  # logger.info("💤 Sleeping {}s... Next scan at {next_scan.strftime('%H:%M:%S')}\n", {sleep_time:.1f})
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Monitoring stopped by user")
            self._print_final_stats()
        
        except Exception as e:
            logger.error("❌ Fatal error: {}", str(e), exc_info=True)
            self._print_final_stats()
            raise
    
    def scan_and_alert(self):
        """Public method to run a single scan cycle"""
        # NOTE: Analysis queue processing is handled EXCLUSIVELY by run_analysis_queue_processor.py
        # This prevents duplicate analyses when user clicks "3" (ultimate) on a single coin
        return self._scan_and_alert()

    def _scan_and_alert(self):
        """Perform a single scan and send alerts for breakouts"""
        logger.info("=" * 80)
        logger.info("🔍 Starting scan at {}", str(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        scan_start = time.time()
        
        # Get watchlist symbols if enabled
        watchlist_symbols = []
        if self.use_watchlist and self.supabase_client:
            watchlist_symbols = self._get_watchlist_symbols()
            if watchlist_symbols:
                logger.info(f"📋 Monitoring {len(watchlist_symbols)} watchlist cryptos")
        
        # Scan for breakouts
        breakouts = self._detect_breakouts()
        scan_duration = time.time() - scan_start
        
        pass  # logger.info(f"✅ Scan complete in {}s - Found {len(breakouts))} potential breakouts {scan_duration:.2f}")
        
        if not breakouts:
            logger.info("   No breakouts detected above threshold")
            return []
        
        # Filter and send alerts
        alerts_sent = 0
        added_to_watchlist = 0
        results_for_panel = []
        
        for breakout in breakouts:
            # Validate with Jupiter if available (optional cross-DEX confirmation)
            if self.jupiter_client and breakout.symbol.endswith('/USD'):
                try:
                    # Try to validate price with Jupiter (non-blocking attempt)
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If already in async context, skip Jupiter validation
                            pass
                        else:
                            # Run Jupiter validation synchronously
                            breakout = loop.run_until_complete(
                                self._validate_with_jupiter(breakout)
                            )
                    except RuntimeError:
                        # No event loop, try to create one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        breakout = loop.run_until_complete(
                            self._validate_with_jupiter(breakout)
                        )
                except Exception as e:
                    logger.debug(f"[JUPITER] Skipped validation for {breakout.symbol}: {e}")
            
            # Store result for control panel (even if not alerted)
            results_for_panel.append({
                'ticker': breakout.symbol,
                'signal': breakout.alert_type,
                'confidence': int(breakout.score),
                'price': breakout.price,
                'change_24h': breakout.change_24h,
                'volume_24h': breakout.volume_ratio,  # Using volume_ratio as proxy
                'timestamp': datetime.now().isoformat(),
                'jupiter_validated': breakout.jupiter_validated,
                'jupiter_spread_pct': breakout.jupiter_spread_pct
            })
            
            if self._should_alert(breakout):
                # Send Discord alert
                self._send_alert(breakout)
                self.recent_alerts[breakout.symbol] = datetime.now()
                alerts_sent += 1
                self.stats['alerts_sent'] += 1
                
                # Auto-add to watchlist with tags
                if self.auto_add_to_watchlist and self.watchlist_manager:
                    if self._add_to_watchlist(breakout):
                        added_to_watchlist += 1
            else:
                logger.debug(f"   Skipping {breakout.symbol} - recently alerted or filtered out")
        
        logger.info(f"📤 Sent {alerts_sent} new alerts")
        if added_to_watchlist > 0:
            logger.info(f"📋 Added {added_to_watchlist} new coins to watchlist")
        
        # Save results to control panel
        if results_for_panel:
            try:
                save_analysis_results('Crypto Breakout Monitor', results_for_panel)
                logger.debug(f"💾 Saved {len(results_for_panel)} results to control panel")
            except Exception as e:
                logger.warning(f"Failed to save results to control panel: {e}")
        
        return results_for_panel
    
    def _detect_breakouts(self) -> List[BreakoutAlert]:
        """Detect breakout opportunities using scanner"""
        breakouts = []
        
        try:
            # Scan for different types
            scan_types = [
                ('BREAKOUT', 'scan_breakout_cryptos', 10),
                ('BUZZING', 'scan_buzzing_cryptos', 5),
                ('HOTTEST', 'scan_hottest_cryptos', 5),
                ('PRE_LISTING', 'scan_new_listings', 5),  # NEW: Pre-listing scanner
                ('PRE_IPO_BUZZ', 'scan_pre_ipo_buzz', 3)  # NEW: Pre-IPO buzz
            ]
            
            for alert_type, method_name, top_n in scan_types:
                logger.info(f"   Scanning for {alert_type} opportunities...")
                scan_type_start = time.time()
                
                opportunities = []
                
                # Check if this is a pre-listing scan
                if alert_type in ['PRE_LISTING', 'PRE_IPO_BUZZ']:
                    # Use pre-listing scanner
                    if hasattr(self.pre_listing_scanner, method_name):
                        scan_method = getattr(self.pre_listing_scanner, method_name)
                        # Pre-listing scanner returns PreListingSignal objects
                        pre_listing_results = scan_method(top_n=top_n)
                        
                        # Convert to BreakoutAlert format
                        for result in pre_listing_results:
                            alert = self._pre_listing_to_alert(result, alert_type)
                            if alert.score >= self.min_score * 0.7:  # Lower threshold for pre-listings
                                breakouts.append(alert)
                                logger.info(f"      ✓ {alert.symbol}: Social {result.social_score:.0f}, {alert.confidence} confidence")
                        
                        logger.info(f"   ✅ {alert_type} scan complete in {time.time() - scan_type_start:.1f}s")
                        continue  # Skip normal processing for pre-listing scans
                
                # Normal breakout/buzzing/hottest scanning
                # Only use AI for BREAKOUT scan to prevent cumulative timeouts
                # BUZZING and HOTTEST use quantitative scoring only (much faster)
                use_ai_for_this_scan = self.use_ai and alert_type == 'BREAKOUT'
                
                if use_ai_for_this_scan and hasattr(self.scanner, 'scan_with_ai_confidence'):
                    # AI scanner - only for BREAKOUT
                    logger.info(f"      🤖 Using AI analysis for {alert_type}...")
                    strategy_map = {
                        'BREAKOUT': 'ALL',
                        'BUZZING': 'MOMENTUM', 
                        'HOTTEST': 'SCALP'
                    }
                    opportunities = self.scanner.scan_with_ai_confidence(
                        strategy=strategy_map.get(alert_type, 'ALL'),
                        top_n=top_n,
                        min_score=self.min_score * 0.8,  # Slightly lower threshold
                        min_ai_confidence=self.min_confidence if self.min_confidence else None
                    )
                else:
                    # Quantitative scanner - faster, no AI timeout risk
                    logger.info(f"      📊 Using quantitative analysis for {alert_type}...")
                    if hasattr(self.scanner, 'base_scanner'):
                        base = self.scanner.base_scanner
                    elif hasattr(self.scanner, method_name):
                        base = self.scanner
                    else:
                        base = None
                    
                    if base and hasattr(base, method_name):
                        scan_method = getattr(base, method_name)
                        opportunities = scan_method(top_n=top_n)
                    elif hasattr(self.scanner, 'scan_opportunities'):
                        opportunities = self.scanner.scan_opportunities(
                            strategy='ALL',
                            top_n=top_n,
                            min_score=self.min_score * 0.8
                        )
                
                # Convert to BreakoutAlert
                for opp in opportunities:
                    if opp.score >= self.min_score:
                        alert = self._opportunity_to_alert(opp, alert_type)
                        breakouts.append(alert)
                        logger.info(f"      ✓ {alert.symbol}: Score {alert.score:.1f}, {alert.confidence} confidence")
                
                logger.info(f"   ✅ {alert_type} scan complete in {time.time() - scan_type_start:.1f}s - found {len(opportunities)} opportunities")
        
        except Exception as e:
            logger.error("Error detecting breakouts: {}", str(e), exc_info=True)
        
        # Sort by score
        breakouts.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"🔍 Total breakouts detected: {len(breakouts)}")
        return breakouts
    
    def _pre_listing_to_alert(self, signal, alert_type: str) -> BreakoutAlert:
        """Convert PreListingSignal to BreakoutAlert"""
        
        return BreakoutAlert(
            symbol=signal.symbol,
            alert_type=alert_type,
            score=signal.score,
            confidence=signal.confidence,
            price=signal.current_price,
            change_24h=signal.change_24h,
            volume_ratio=signal.volume_ratio,
            rsi=50.0,  # Default
            ema_8=signal.current_price,
            ema_20=signal.current_price,
            ema_50=signal.current_price,
            macd_signal='BULLISH' if signal.change_24h > 0 else 'NEUTRAL',
            bb_position='MIDDLE',
            ai_confidence=signal.confidence,
            ai_reasoning=signal.reasoning,
            ai_rating=signal.score / 10.0  # Convert to 0-10 scale
        )
    
    def _opportunity_to_alert(self, opp, alert_type: str) -> BreakoutAlert:
        """Convert opportunity to breakout alert"""
        
        # Determine MACD signal
        macd_signal = 'NEUTRAL'
        if hasattr(opp, 'macd_histogram') and opp.macd_histogram is not None:
            macd_signal = 'BULLISH' if opp.macd_histogram > 0 else 'BEARISH'
        
        # Determine Bollinger Band position
        bb_position = 'MIDDLE'
        if hasattr(opp, 'current_price') and hasattr(opp, 'bb_upper') and hasattr(opp, 'bb_lower'):
            if opp.current_price and opp.bb_upper and opp.bb_lower:
                bb_range = opp.bb_upper - opp.bb_lower
                if bb_range > 0:
                    position_pct = (opp.current_price - opp.bb_lower) / bb_range
                    if position_pct > 0.8:
                        bb_position = 'UPPER'
                    elif position_pct < 0.2:
                        bb_position = 'LOWER'
        
        # Extract AI fields if available
        ai_confidence = getattr(opp, 'ai_confidence', None)
        ai_reasoning = getattr(opp, 'ai_reasoning', None)
        ai_rating = getattr(opp, 'ai_rating', None)
        
        return BreakoutAlert(
            symbol=opp.symbol,
            alert_type=alert_type,
            score=opp.score,
            confidence=opp.confidence if hasattr(opp, 'confidence') else 'MEDIUM',
            price=opp.current_price if hasattr(opp, 'current_price') else 0.0,
            change_24h=opp.change_pct_24h if hasattr(opp, 'change_pct_24h') else 0.0,
            volume_ratio=getattr(opp, 'volume_ratio', 1.0),
            rsi=getattr(opp, 'rsi', 50.0),
            ema_8=getattr(opp, 'ema_8', 0.0),
            ema_20=getattr(opp, 'ema_20', 0.0),
            ema_50=getattr(opp, 'ema_50', 0.0),
            macd_signal=macd_signal,
            bb_position=bb_position,
            ai_confidence=ai_confidence,
            ai_reasoning=ai_reasoning,
            ai_rating=ai_rating
        )
    
    async def _validate_with_jupiter(self, breakout: BreakoutAlert) -> BreakoutAlert:
        """
        Validate breakout signal using Jupiter DEX prices
        Cross-validates Kraken prices with Jupiter to detect:
        - Price mismatches (potential arbitrage)
        - Liquidity depth (important for execution)
        - Multi-DEX confirmation
        
        Args:
            breakout: BreakoutAlert to validate
            
        Returns:
            Updated BreakoutAlert with Jupiter validation data
        """
        if not self.jupiter_client or not breakout.symbol.endswith('/USD'):
            # Only validate Solana/USD pairs for now
            return breakout
        
        try:
            # Extract base asset (BTC from BTC/USD)
            base_asset = breakout.symbol.split('/')[0]
            
            # Skip major assets that aren't on Solana (for now just try)
            logger.debug(f"[JUPITER] Validating {breakout.symbol} with Jupiter prices...")
            
            # Get Jupiter price for this asset
            # Note: Would need token mint mapping for different assets
            # For now, this is a placeholder that logs the attempt
            
            # In production, you'd have a mapping like:
            # {'SOL': 'So11111111111111111111111111111111111111112', 'USDC': '...', etc}
            
            jupiter_price = await self.jupiter_client.get_price(
                mint_id=base_asset  # This should be the actual token mint
            )
            
            if jupiter_price:
                # Calculate spread
                spread_pct = ((jupiter_price - breakout.price) / breakout.price) * 100 if breakout.price > 0 else 0
                
                # Flag arbitrage opportunities
                arbitrage_opp = abs(spread_pct) > 1.0
                
                logger.info(
                    f"[JUPITER] ✓ {breakout.symbol}: "
                    f"Jupiter=${jupiter_price:.6f} vs Kraken=${breakout.price:.6f} "
                    f"(spread={spread_pct:+.2f}%) "
                    f"{'⚡ ARB OPP' if arbitrage_opp else ''}"
                )
                
                # Update breakout with Jupiter data
                breakout.jupiter_validated = True
                breakout.jupiter_price = jupiter_price
                breakout.jupiter_spread_pct = spread_pct
                breakout.jupiter_arbitrage_opportunity = arbitrage_opp
                
                # Check liquidity depth
                try:
                    liquidity_info = await self.jupiter_client.get_liquidity_depth(
                        token_mint=base_asset,
                        depth_levels=[0.5, 1.0]  # Check at 0.5 and 1.0 USD
                    )
                    if liquidity_info:
                        total_liquidity = sum(
                            info.get('tokens_out', 0) * jupiter_price 
                            for info in liquidity_info.values()
                        ) if jupiter_price else 0
                        breakout.jupiter_liquidity_usd = total_liquidity
                        logger.debug(f"[JUPITER] Liquidity depth: ${total_liquidity:,.2f}")
                except Exception as e:
                    logger.debug(f"[JUPITER] Liquidity depth check failed: {e}")
            else:
                logger.debug(f"[JUPITER] Could not fetch price for {base_asset}")
            
            return breakout
            
        except Exception as e:
            logger.debug(f"[JUPITER] Validation failed for {breakout.symbol}: {e}")
            return breakout
    
    def _should_alert(self, breakout: BreakoutAlert) -> bool:
        """Determine if we should send an alert for this breakout"""
        
        # Check cooldown
        if breakout.symbol in self.recent_alerts:
            last_alert = self.recent_alerts[breakout.symbol]
            if datetime.now() - last_alert < self.alert_cooldown:
                logger.debug(f"   ⏳ {breakout.symbol} skipped - cooldown active (last alert: {last_alert})")
                return False
        
        # Check score threshold
        if breakout.score < self.min_score:
            logger.debug(f"   📊 {breakout.symbol} skipped - score {breakout.score:.1f} < min {self.min_score}")
            return False
        
        # Check confidence threshold
        confidence_levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        min_level = confidence_levels.get(self.min_confidence, 0)
        breakout_level = confidence_levels.get(breakout.confidence, 0)
        
        if breakout_level < min_level:
            logger.debug(f"   🎯 {breakout.symbol} skipped - confidence {breakout.confidence} < min {self.min_confidence}")
            return False
        
        # Check AI confidence if available - only filter if AI analysis was actually done
        if breakout.ai_confidence and self.use_ai:
            ai_level = confidence_levels.get(breakout.ai_confidence, 0)
            # If AI timed out and returned empty, ai_confidence might be None - don't filter in that case
            if breakout.ai_confidence and ai_level < min_level:
                logger.debug(f"   🤖 {breakout.symbol} skipped - AI confidence {breakout.ai_confidence} < min {self.min_confidence}")
                return False
        
        # 🧠 Check Signal Memory for historical pattern performance (RAG)
        try:
            from services.signal_memory import get_crypto_historical_performance
            
            history = get_crypto_historical_performance(
                symbol=breakout.symbol,
                strategy=breakout.alert_type,
                signal_type='BUY',
                price=breakout.price,
                rsi=breakout.rsi,
                volume_ratio=breakout.volume_ratio,
                change_24h=breakout.change_24h
            )
            
            if history.get('sample_size', 0) >= 5:
                recommendation = history.get('recommendation', 'NEUTRAL')
                win_rate = history.get('win_rate', 0)
                adjustment = history.get('confidence_adjustment', 1.0)
                
                # Apply confidence adjustment to score
                if adjustment != 1.0:
                    original_score = breakout.score
                    breakout.score = min(100, breakout.score * adjustment)
                    
                    if recommendation == 'BOOST':
                        logger.info(f"   📈 Historical boost for {breakout.symbol}: score {original_score:.1f} → {breakout.score:.1f} (win rate: {win_rate:.0%})")
                        breakout.ai_reasoning = f"{breakout.ai_reasoning or ''} [Historical: {win_rate:.0%} win rate]".strip()
                    elif recommendation == 'REDUCE':
                        logger.warning(f"   📉 Historical caution for {breakout.symbol}: score {original_score:.1f} → {breakout.score:.1f} (win rate: {win_rate:.0%})")
                        breakout.ai_reasoning = f"{breakout.ai_reasoning or ''} [⚠️ Historical: {win_rate:.0%} win rate]".strip()
                        
                        # If score dropped below threshold after adjustment, skip
                        if breakout.score < self.min_score:
                            logger.warning(f"   ⛔ {breakout.symbol} skipped - adjusted score {breakout.score:.1f} < min {self.min_score}")
                            return False
                            
        except Exception as e:
            logger.debug(f"   Could not check signal memory (non-critical): {e}")
        
        logger.info(f"   ✅ {breakout.symbol} PASSED all filters (score={breakout.score:.1f}, conf={breakout.confidence}, ai_conf={breakout.ai_confidence})")
        return True
        return True
    
    def _send_alert(self, breakout: BreakoutAlert):
        """Send Discord alert for breakout and queue to orchestrator"""
        
        # Log alert
        logger.info(f"\n🚨 BREAKOUT ALERT: {breakout.symbol}")
        logger.info(f"   Type: {breakout.alert_type}")
        logger.info(f"   Score: {breakout.score:.1f}")
        logger.info(f"   Confidence: {breakout.confidence} (AI: {breakout.ai_confidence or 'N/A'})")
        logger.info(f"   Price: ${breakout.price:,.4f}")
        logger.info(f"   24h Change: {breakout.change_24h:+.2f}%")
        
        # Queue to Service Orchestrator for review in Control Panel
        self._queue_to_orchestrator(breakout)
        
        # Get target channel ID for crypto alerts (for channel routing with buttons)
        target_channel_id = None
        try:
            from src.integrations.discord_channels import get_channel_id_for_category, AlertCategory
            target_channel_id = get_channel_id_for_category(AlertCategory.CRYPTO_ALERTS)
            if target_channel_id:
                logger.debug(f"   Using crypto alerts channel ID: {target_channel_id}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"   Could not get crypto channel ID: {e}")
        
        # Try Discord bot first (has interactive buttons for Watch/Analyze/Trade/Dismiss)
        bot_sent = False
        if self.discord_bot_manager and self.discord_bot_manager.is_running():
            try:
                import asyncio
                
                # Build message for bot
                type_emoji = {
                    'BREAKOUT': '💥', 'BUZZING': '🔥', 'HOTTEST': '🌶️',
                    'PRE_LISTING': '🚀', 'PRE_IPO_BUZZ': '⚡', 'RUNNING': '🏃', 'MOONING': '🌙'
                }
                emoji = type_emoji.get(breakout.alert_type, '📊')
                
                message_text = (
                    f"**Score:** {breakout.score:.0f}/100 | **Confidence:** {breakout.confidence}\n"
                    f"**Price:** ${breakout.price:,.4f} | **24h:** {breakout.change_24h:+.2f}%\n"
                    f"**Volume:** {breakout.volume_ratio:.1f}x avg"
                )
                
                # Add Jupiter validation info if available
                if breakout.jupiter_validated:
                    jupiter_info = f"\n\n📊 **Jupiter DEX:** ${breakout.jupiter_price:.6f}"
                    if breakout.jupiter_spread_pct:
                        spread_emoji = "⚡" if abs(breakout.jupiter_spread_pct) > 1.0 else "✓"
                        jupiter_info += f" ({spread_emoji} {breakout.jupiter_spread_pct:+.2f}% spread)"
                    if breakout.jupiter_arbitrage_opportunity:
                        jupiter_info += f" **ARB OPP DETECTED!**"
                    message_text += jupiter_info
                
                if breakout.ai_reasoning:
                    message_text += f"\n\n🤖 **AI:** {breakout.ai_reasoning[:200]}"
                
                message_text += "\n\n**Actions:** `1`=Standard `2`=Multi `3`=Ultimate | `W`=Watch | `🚀`=Trade | `X`=Dismiss"
                
                # Color based on score
                if breakout.score >= 90:
                    color = 0x00FF00  # Green
                elif breakout.score >= 80:
                    color = 0x32CD32  # Lime
                elif breakout.score >= 70:
                    color = 0xFFD700  # Gold
                else:
                    color = 0xFFA500  # Orange
                
                # Build alert data for Trade button functionality
                alert_data = {
                    'symbol': breakout.symbol,
                    'alert_type': breakout.alert_type,
                    'score': breakout.score,
                    'confidence': breakout.confidence,
                    'price': breakout.price,
                    'change_24h': breakout.change_24h,
                    'volume_ratio': breakout.volume_ratio,
                    'rsi': breakout.rsi,
                    'ai_confidence': breakout.ai_confidence,
                    'ai_reasoning': breakout.ai_reasoning
                }
                
                # Send via bot (has buttons including Trade for crypto)
                async def send_bot_alert():
                    return await self.discord_bot_manager.bot.send_alert_notification(
                        symbol=breakout.symbol,
                        alert_type=f"{emoji} {breakout.alert_type}",
                        message_text=message_text,
                        confidence=breakout.confidence,
                        color=color,
                        asset_type="crypto",
                        alert_data=alert_data,
                        target_channel_id=target_channel_id
                    )
                
                if self.discord_bot_manager.loop:
                    future = asyncio.run_coroutine_threadsafe(
                        send_bot_alert(),
                        self.discord_bot_manager.loop
                    )
                    bot_sent = future.result(timeout=10)
                    
                if bot_sent:
                    logger.info(f"   ✅ Discord alert sent via BOT (with buttons) to channel {target_channel_id or 'default'}")
                else:
                    logger.debug("   Bot send returned False, falling back to webhook")
                    
            except Exception as e:
                logger.debug(f"   Bot alert failed ({type(e).__name__}), falling back to webhook: {e}")
        
        # Fallback to webhook (no buttons, but still shows embed)
        if not bot_sent:
            if not self.discord_webhook:
                logger.warning("   ⚠️ Discord not configured - alert logged only\n")
                return
            
            try:
                # Build Discord embed
                embed = self._build_discord_embed(breakout)
                
                payload = {
                    'embeds': [embed],
                    'username': 'Crypto Breakout Monitor',
                    'avatar_url': 'https://cdn-icons-png.flaticon.com/512/6001/6001368.png'
                }
                
                logger.debug(f"   📤 Sending Discord webhook to: {self.discord_webhook[:50]}...")
                response = requests.post(self.discord_webhook, json=payload, timeout=15)
                response.raise_for_status()
                
                logger.info(f"   ✅ Discord alert sent via WEBHOOK (HTTP {response.status_code})")
                logger.info(f"      ⚠️ No buttons - reply with 1/2/3/W/X or configure Discord bot")
            
            except requests.exceptions.Timeout:
                logger.error(f"   ❌ Discord webhook timeout (15s)\n")
            except requests.exceptions.ConnectionError as e:
                logger.error(f"   ❌ Discord connection error: {e}\n")
            except requests.exceptions.HTTPError as e:
                logger.error(f"   ❌ Discord HTTP error: {e}\n")
            except Exception as e:
                logger.error(f"   ❌ Failed to send Discord alert: {type(e).__name__}: {e}\n")
    
    def _build_discord_embed(self, breakout: BreakoutAlert) -> Dict:
        """Build Discord embed message for breakout alert"""
        
        # Determine color based on score and confidence
        # Pre-listing alerts get special colors
        if breakout.alert_type in ['PRE_LISTING', 'PRE_IPO_BUZZ']:
            if breakout.score >= 80:
                color = 0xFF0000  # Bright red (urgent!)
            else:
                color = 0xFF6600  # Orange-red
        elif breakout.score >= 90 and breakout.confidence == 'HIGH':
            color = 0x00FF00  # Bright green
        elif breakout.score >= 80:
            color = 0x32CD32  # Lime green
        elif breakout.score >= 70:
            color = 0xFFD700  # Gold
        else:
            color = 0xFFA500  # Orange
        
        # Emoji for alert type
        type_emoji = {
            'BREAKOUT': '💥',
            'BUZZING': '🔥',
            'HOTTEST': '🌶️',
            'PRE_LISTING': '🚀',
            'PRE_IPO_BUZZ': '⚡',
            'RUNNING': '🏃',
            'MOONING': '🌙'
        }
        emoji = type_emoji.get(breakout.alert_type, '📊')
        
        # Title
        title = f"{emoji} {breakout.alert_type} DETECTED: {breakout.symbol}"
        
        # Description
        description = f"**Score:** {breakout.score:.1f}/100 | **Confidence:** {breakout.confidence}"
        
        if breakout.ai_confidence:
            description += f" | **AI:** {breakout.ai_confidence}"
        
        # Fields
        fields = []
        
        # Price & Change
        change_emoji = "📈" if breakout.change_24h > 0 else "📉"
        fields.append({
            'name': '💰 Price',
            'value': f'${breakout.price:,.4f}\n{change_emoji} {breakout.change_24h:+.2f}% (24h)',
            'inline': True
        })
        
        # Volume
        fields.append({
            'name': '📊 Volume',
            'value': f'{breakout.volume_ratio:.2f}x average\n{"🚀 High volume!" if breakout.volume_ratio > 2 else "Moderate"}',
            'inline': True
        })
        
        # RSI
        rsi_status = "Overbought" if breakout.rsi > 70 else "Oversold" if breakout.rsi < 30 else "Neutral"
        fields.append({
            'name': '🎯 RSI',
            'value': f'{breakout.rsi:.1f}\n{rsi_status}',
            'inline': True
        })
        
        # Moving Averages
        ema_alignment = ""
        if breakout.price > breakout.ema_8 > breakout.ema_20:
            ema_alignment = "✅ Bullish alignment"
        elif breakout.price < breakout.ema_8 < breakout.ema_20:
            ema_alignment = "❌ Bearish alignment"
        else:
            ema_alignment = "⚠️ Mixed signals"
        
        fields.append({
            'name': '📈 EMAs',
            'value': f'EMA8: ${breakout.ema_8:,.4f}\nEMA20: ${breakout.ema_20:,.4f}\n{ema_alignment}',
            'inline': True
        })
        
        # MACD
        macd_emoji = "🟢" if breakout.macd_signal == 'BULLISH' else "🔴" if breakout.macd_signal == 'BEARISH' else "⚪"
        fields.append({
            'name': '🌊 MACD',
            'value': f'{macd_emoji} {breakout.macd_signal}',
            'inline': True
        })
        
        # Bollinger Bands
        bb_emoji = "⬆️" if breakout.bb_position == 'UPPER' else "⬇️" if breakout.bb_position == 'LOWER' else "↔️"
        fields.append({
            'name': '📊 BB Position',
            'value': f'{bb_emoji} {breakout.bb_position}',
            'inline': True
        })
        
        # AI Reasoning (if available) or Social Buzz for pre-listings
        if breakout.ai_reasoning:
            # Truncate if too long
            reasoning = breakout.ai_reasoning
            if len(reasoning) > 500:
                reasoning = reasoning[:497] + "..."
            
            # For pre-listing alerts, emphasize the social aspect
            if breakout.alert_type in ['PRE_LISTING', 'PRE_IPO_BUZZ']:
                fields.append({
                    'name': '🚨 SOCIAL BUZZ DETECTED',
                    'value': f'**{reasoning}**\n\n⚠️ Pre-listing opportunity - act fast!',
                    'inline': False
                })
            else:
                fields.append({
                    'name': '🤖 AI Analysis',
                    'value': reasoning,
                    'inline': False
                })
        
        # Trading suggestion
        action_suggestion = self._get_action_suggestion(breakout)
        fields.append({
            'name': '💡 Suggested Action',
            'value': action_suggestion,
            'inline': False
        })
        
        # Build embed
        embed = {
            'title': title,
            'description': description,
            'color': color,
            'fields': fields,
            'timestamp': datetime.now().isoformat(),
            'footer': {
                'text': f'Scan #{self.stats["scans_completed"]} | Crypto Breakout Monitor'
            }
        }
        
        return embed
    
    def _get_action_suggestion(self, breakout: BreakoutAlert) -> str:
        """Get trading action suggestion based on breakout analysis"""
        
        suggestions = []
        
        # Score-based suggestion
        if breakout.score >= 90:
            suggestions.append("⚡ **STRONG BUY SIGNAL** - Very high confidence")
        elif breakout.score >= 80:
            suggestions.append("✅ **BUY** - High confidence setup")
        elif breakout.score >= 70:
            suggestions.append("👀 **MONITOR** - Good setup, confirm entry")
        else:
            suggestions.append("⚠️ **WATCH** - Moderate signal")
        
        # Volume consideration
        if breakout.volume_ratio > 3:
            suggestions.append("🚀 Massive volume surge - strong momentum")
        elif breakout.volume_ratio < 1.5:
            suggestions.append("⚠️ Low volume - wait for confirmation")
        
        # RSI consideration
        if breakout.rsi > 75:
            suggestions.append("⚠️ RSI overbought - consider waiting for pullback")
        elif breakout.rsi > 30 and breakout.rsi < 70:
            suggestions.append("✅ RSI in healthy range")
        
        # MACD consideration
        if breakout.macd_signal == 'BULLISH':
            suggestions.append("✅ MACD bullish crossover")
        elif breakout.macd_signal == 'BEARISH':
            suggestions.append("⚠️ MACD bearish - caution advised")
        
        return "\n".join(suggestions)
    
    def _queue_to_orchestrator(self, breakout: BreakoutAlert):
        """Queue breakout alert to Service Orchestrator for review in Control Panel"""
        try:
            from services.service_orchestrator import get_orchestrator
            orch = get_orchestrator()
            
            # Build reasoning from available data
            reasoning_parts = [f"{breakout.alert_type}: Score {breakout.score:.0f}"]
            if breakout.ai_reasoning:
                reasoning_parts.append(breakout.ai_reasoning[:150])
            else:
                reasoning_parts.append(f"24h: {breakout.change_24h:+.1f}%, Vol: {breakout.volume_ratio:.1f}x")
            
            orch.add_alert(
                symbol=breakout.symbol,
                alert_type=breakout.alert_type,
                source="crypto_breakout_scanner",
                asset_type="crypto",
                price=breakout.price,
                reasoning=" | ".join(reasoning_parts),
                confidence=breakout.confidence,
                expires_minutes=120,  # 2 hour expiry
                metadata={
                    "score": breakout.score,
                    "change_24h": breakout.change_24h,
                    "volume_ratio": breakout.volume_ratio,
                    "rsi": breakout.rsi,
                    "macd_signal": breakout.macd_signal,
                    "ai_confidence": breakout.ai_confidence,
                    "ai_rating": breakout.ai_rating
                }
            )
            logger.info(f"   📥 Queued to Control Panel alert queue")
        except Exception as e:
            logger.debug(f"   Could not queue to orchestrator: {e}")
    
    def _add_to_watchlist(self, breakout: BreakoutAlert) -> bool:
        """
        Add detected breakout to watchlist with appropriate tags
        
        Returns:
            True if added (new coin), False if already exists
        """
        try:
            # Check if already in watchlist or already added this session
            if breakout.symbol in self.watchlist_added:
                logger.debug(f"   {breakout.symbol} already added to watchlist this session")
                return False
            
            # Check if exists in database
            if self.watchlist_manager and self.watchlist_manager.is_in_watchlist(breakout.symbol):
                logger.debug(f"   {breakout.symbol} already in watchlist")
                self.watchlist_added.add(breakout.symbol)  # Track to avoid checking again
                return False
            
            # Prepare opportunity data
            opportunity_data = {
                'symbol': breakout.symbol,
                'current_price': breakout.price,
                'change_pct_24h': breakout.change_24h,
                'volume_24h': 0,  # Not always available
                'volume_ratio': breakout.volume_ratio,
                'volatility_24h': 0,
                'rsi': breakout.rsi,
                'score': breakout.score,
                'confidence': breakout.confidence,
                'risk_level': 'MEDIUM',  # Default
                'strategy': breakout.alert_type.lower(),
                'reason': f"{breakout.alert_type} detected by monitor"
            }
            
            # Add AI reasoning if available
            if breakout.ai_reasoning:
                opportunity_data['reason'] = breakout.ai_reasoning
            
            if self.watchlist_manager:
                success = self.watchlist_manager.add_crypto(breakout.symbol, opportunity_data)
            else:
                success = False
            
            if success:
                # Add tags based on alert type
                tags = [breakout.alert_type]
                
                # Add confidence tag
                tags.append(f"CONF_{breakout.confidence}")
                
                # Add special tags for pre-listings
                if breakout.alert_type in ['PRE_LISTING', 'PRE_IPO_BUZZ']:
                    tags.append('HOT_NEW')
                    tags.append('URGENT')
                
                # Add high score tag
                if breakout.score >= 90:
                    tags.append('HIGH_SCORE')
                
                # Add tags to database
                if self.watchlist_manager:
                    for tag in tags:
                        self.watchlist_manager.add_tag(breakout.symbol, tag)
                
                # Track that we added it
                self.watchlist_added.add(breakout.symbol)
                
                logger.info(f"   ✅ Added {breakout.symbol} to watchlist with tags: {', '.join(tags)}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"   ❌ Error adding {breakout.symbol} to watchlist: {e}")
            return False
    
    def _get_watchlist_symbols(self) -> List[str]:
        """Get symbols from user's watchlist via Supabase (CryptoWatchlistManager + TickerManager)"""
        symbols = set()
        
        # Source 1: CryptoWatchlistManager (Internal bot watchlist)
        try:
            if self.watchlist_manager:
                internal_symbols = self.watchlist_manager.get_watchlist_symbols()
                symbols.update(internal_symbols)
                pass  # logger.debug(f"Loaded {len(internal_symbols)} from CryptoWatchlistManager")
        except Exception as e:
            logger.debug(f"Error fetching from CryptoWatchlistManager: {e}")

        # Source 2: TickerManager (User's My Tickers)
        try:
            tm = TickerManager()
            if tm.test_connection():
                all_tickers = tm.get_all_tickers()
                # Filter for crypto types
                crypto_tickers = [
                    t['ticker'] for t in all_tickers 
                    if t.get('type') in ['crypto', 'coin', 'token'] or '/' in t['ticker']
                ]
                symbols.update(crypto_tickers)
                pass  # logger.debug(f"Loaded {len(crypto_tickers)} from TickerManager")
        except Exception as e:
            logger.debug(f"Error fetching from TickerManager: {e}")

        return list(symbols)
    
    def _get_uptime(self) -> str:
        """Get monitor uptime"""
        start = datetime.fromisoformat(self.stats['start_time'])
        uptime = datetime.now() - start
        
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        
        return f"{hours}h {minutes}m"
    
    def _print_final_stats(self):
        """Print final statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL SESSION STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total Scans: {self.stats['scans_completed']}")
        logger.info(f"Alerts Sent: {self.stats['alerts_sent']}")
        logger.info(f"Errors: {self.stats['errors']}")
        pass  # logger.info(f"Uptime: {self._get_uptime(}")
        logger.info(f"Started: {self.stats['start_time']}")
        logger.info("Ended: {}", str(datetime.now().isoformat()))
        logger.info("=" * 80)


def main():
    """Main entry point for background monitor"""
    
    # Configuration from environment variables or defaults
    # Strip commas from numeric values to handle formatting like '1,800'
    scan_interval = int(os.getenv('BREAKOUT_SCAN_INTERVAL', '300').replace(',', ''))  # 5 minutes
    min_score = float(os.getenv('BREAKOUT_MIN_SCORE', '70.0').replace(',', ''))
    min_confidence = os.getenv('BREAKOUT_MIN_CONFIDENCE', 'HIGH')
    use_ai = os.getenv('BREAKOUT_USE_AI', 'true').lower() == 'true'
    use_watchlist = os.getenv('BREAKOUT_USE_WATCHLIST', 'true').lower() == 'true'
    auto_add_watchlist = os.getenv('BREAKOUT_AUTO_ADD_WATCHLIST', 'true').lower() == 'true'
    alert_cooldown = int(os.getenv('BREAKOUT_ALERT_COOLDOWN', '60').replace(',', ''))
    
    try:
        # Initialize monitor
        monitor = CryptoBreakoutMonitor(
            scan_interval_seconds=scan_interval,
            min_score=min_score,
            min_confidence=min_confidence,
            use_ai=use_ai,
            use_watchlist=use_watchlist,
            auto_add_to_watchlist=auto_add_watchlist,
            alert_cooldown_minutes=alert_cooldown
        )
        
        # Start monitoring
        monitor.start()
    
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
    
    except Exception as e:
        logger.error("❌ Fatal error: {}", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


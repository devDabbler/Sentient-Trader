"""
Stock Informational Monitor Service - ENHANCED VERSION
Monitors stocks without executing trades - alerts only

Integrates with LLM Request Manager for cost-efficient scanning
Uses LOW priority LLM requests with aggressive caching

ENHANCEMENTS:
- Comprehensive stats tracking and health metrics
- Error recovery and resilience patterns
- Circuit breaker protection against cascading failures
- Automatic retry logic with exponential backoff
- State management and lifecycle control
- Dynamic watchlist synchronization
- Resource cleanup and graceful shutdown
"""
import time
import sys
_debug_file = open('DEBUG_TRACE.txt', 'a')
_debug_file.write("[TRACE] stock_informational_monitor.py: Starting module load...\n")
_debug_file.flush()
from typing import List, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import json
import os

from loguru import logger
import requests
_debug_file.write("[TRACE] stock_informational_monitor.py: Imported logger\n")
_debug_file.flush()
from services.llm_helper import get_llm_helper, LLMServiceMixin
_debug_file.write("[TRACE] stock_informational_monitor.py: Imported llm_helper\n")
_debug_file.flush()
from services.alert_system import get_alert_system
_debug_file.write("[TRACE] stock_informational_monitor.py: Imported alert_system\n")
_debug_file.flush()
from services.ai_confidence_scanner import AIConfidenceScanner
_debug_file.write("[TRACE] stock_informational_monitor.py: Imported AIConfidenceScanner\n")
_debug_file.flush()
from services.enhanced_stock_opportunity_detector import get_enhanced_stock_detector
_debug_file.write("[TRACE] stock_informational_monitor.py: Imported enhanced_stock_opportunity_detector\n")
_debug_file.flush()
from services.stock_discovery_universe import get_stock_discovery_universe
_debug_file.write("[TRACE] stock_informational_monitor.py: Imported stock_discovery_universe\n")
_debug_file.flush()
from windows_services.runners.service_config_loader import save_analysis_results, get_pending_analysis_requests, mark_analysis_complete

_debug_file.write("[TRACE] stock_informational_monitor.py: About to import config_stock_informational\n")
_debug_file.flush()
try:
    import config_stock_informational as cfg
    _debug_file.write("[TRACE] stock_informational_monitor.py: Imported config_stock_informational\n")
    _debug_file.flush()
    SCAN_INTERVAL_MINUTES = cfg.SCAN_INTERVAL_MINUTES
    CACHE_TTL_SECONDS = cfg.CACHE_TTL_SECONDS
    WATCHLIST = cfg.WATCHLIST
    MIN_ENSEMBLE_SCORE = cfg.MIN_ENSEMBLE_SCORE
    ENABLE_ALERTS = cfg.ENABLE_ALERTS
    LOG_OPPORTUNITIES = cfg.LOG_OPPORTUNITIES
    OPPORTUNITIES_LOG_PATH = cfg.OPPORTUNITIES_LOG_PATH
    _debug_file.write("[TRACE] stock_informational_monitor.py: Loaded config values\n")
    _debug_file.flush()
except ImportError:
    _debug_file.write("[TRACE] stock_informational_monitor.py: Config not found, using defaults\n")
    _debug_file.flush()
    # Fallback to default settings
    SCAN_INTERVAL_MINUTES = 30
    CACHE_TTL_SECONDS = 900
    WATCHLIST = []
    MIN_ENSEMBLE_SCORE = 60
    ENABLE_ALERTS = True
    LOG_OPPORTUNITIES = True
    OPPORTUNITIES_LOG_PATH = "logs/stock_opportunities.json"

_debug_file.write("[TRACE] stock_informational_monitor.py: About to define dataclasses and classes\n")
_debug_file.flush()

@dataclass
class StockOpportunity:
    """Detected stock opportunity"""
    symbol: str
    opportunity_type: str  # ENTRY, BREAKOUT, SETUP, WATCH, etc.
    ensemble_score: int  # 0-100
    confidence: float  # 0-1.0
    price: float
    reasoning: str
    technical_summary: str
    timeframe_alignment: str  # TRIPLE_THREAT, DUAL_ALIGN, SINGLE, NONE
    alert_priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MonitorStats:
    """Comprehensive monitoring statistics for health tracking"""
    scans_completed: int = 0
    alerts_sent: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    circuit_breaker_trips: int = 0
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    last_scan_time: Optional[str] = None
    last_error_time: Optional[str] = None
    total_opportunities_found: int = 0
    tickers_scanned: int = 0
    avg_scan_duration_seconds: float = 0.0
    last_queue_process_time: Optional[str] = None
    queue_requests_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'scans_completed': self.scans_completed,
            'alerts_sent': self.alerts_sent,
            'errors': self.errors,
            'consecutive_errors': self.consecutive_errors,
            'circuit_breaker_trips': self.circuit_breaker_trips,
            'start_time': self.start_time,
            'last_scan_time': self.last_scan_time,
            'last_error_time': self.last_error_time,
            'total_opportunities_found': self.total_opportunities_found,
            'tickers_scanned': self.tickers_scanned,
            'avg_scan_duration_seconds': self.avg_scan_duration_seconds,
            'queue_requests_processed': self.queue_requests_processed,
            'uptime_minutes': (datetime.fromisoformat(datetime.now().isoformat()) - 
                             datetime.fromisoformat(self.start_time)).total_seconds() / 60
        }
    
    def get_health_status(self) -> str:
        """Determine overall health status"""
        if self.circuit_breaker_trips > 5:
            return "CRITICAL"
        elif self.consecutive_errors > 3:
            return "DEGRADED"
        elif self.errors > 10:
            return "WARNING"
        else:
            return "HEALTHY"


class StockInformationalMonitor(LLMServiceMixin):
    """
    Monitors stocks without trading - alerts only
    
    Features:
    - Cost-efficient LLM usage (LOW priority, aggressive caching)
    - Multi-factor validation (Technical + ML + LLM)
    - Event monitoring (earnings, SEC, news)
    - Priority-based Discord alerts
    - No trade execution
    """
    
    def __init__(
        self,
        watchlist: Optional[List[str]] = None,
        scan_interval_minutes: Optional[int] = None,
        min_score: Optional[int] = None
    ):
        """
        Initialize Stock Informational Monitor
        
        Args:
            watchlist: List of tickers to monitor (or use config)
            scan_interval_minutes: Scan frequency (or use config)
            min_score: Minimum ensemble score for alerts (or use config)
        """
        print(f"[TRACE] StockInformationalMonitor.__init__: Starting initialization...", flush=True)
        super().__init__()
        print(f"[TRACE] StockInformationalMonitor.__init__: super().__init__() completed", flush=True)
        
        # Initialize LLM helper with LOW priority
        self._init_llm("stock_informational_monitor", default_priority="LOW")
        print(f"[TRACE] StockInformationalMonitor.__init__: LLM initialized", flush=True)
        
        # Configuration
        self.watchlist = watchlist
        
        # If no watchlist provided, try to load from TickerManager (Supabase)
        if not self.watchlist:
            try:
                from services.ticker_manager import TickerManager
                tm = TickerManager()
                # Only use if we can connect
                if tm.test_connection():
                    # IMPORTANT: Filter for stocks only to avoid crypto tickers
                    db_tickers = tm.get_all_tickers(ticker_type='stock')
                    if db_tickers:
                        # Additional validation: filter out any crypto-like tickers
                        valid_tickers = []
                        for t in db_tickers:
                            ticker = t.get('ticker', '')
                            # Skip crypto pairs (e.g., BTC/USD, TURBO/USD)
                            if '/' in ticker:
                                logger.debug(f"Skipping crypto ticker: {ticker}")
                                continue
                            # Skip if not alphanumeric (with dots allowed for BRK.B etc)
                            if not ticker.replace('.', '').isalnum():
                                logger.debug(f"Skipping invalid ticker: {ticker}")
                                continue
                            valid_tickers.append(ticker)
                        self.watchlist = valid_tickers
                        logger.info(f"Loaded {len(self.watchlist)} stock tickers from TickerManager (My Tickers)")
            except Exception as e:
                logger.warning(f"Failed to load from TickerManager: {e}")

        # Fallback to config if still empty
        if not self.watchlist:
            self.watchlist = WATCHLIST
        
        # Ensure watchlist is never None
        if self.watchlist is None:
            self.watchlist = []
            
        self.scan_interval = scan_interval_minutes or SCAN_INTERVAL_MINUTES
        self.min_score = min_score or MIN_ENSEMBLE_SCORE
        
        # Services
        print(f"[TRACE] StockInformationalMonitor.__init__: Getting alert system...", flush=True)
        self.alert_system = get_alert_system()
        print(f"[TRACE] StockInformationalMonitor.__init__: Creating AIConfidenceScanner...", flush=True)
        self.confidence_scanner = AIConfidenceScanner()
        print(f"[TRACE] StockInformationalMonitor.__init__: AIConfidenceScanner created", flush=True)
        
        # ENHANCED: Multi-pronged detector for superior opportunities
        print(f"[TRACE] StockInformationalMonitor.__init__: Initializing enhanced detector...", flush=True)
        self.enhanced_detector = get_enhanced_stock_detector(use_llm=True)
        print(f"[TRACE] StockInformationalMonitor.__init__: Enhanced detector initialized", flush=True)
        
        # DISCOVERY: Stock universe discovery (toggleable)
        print(f"[TRACE] StockInformationalMonitor.__init__: Initializing discovery universe...", flush=True)
        self.discovery_universe = get_stock_discovery_universe(detector=self.enhanced_detector)
        self.discovery_enabled = False  # Can be toggled via control panel
        self.discovery_modes = {}  # Tracks which modes are enabled
        print(f"[TRACE] StockInformationalMonitor.__init__: Discovery universe initialized", flush=True)
        
        # State
        print(f"[TRACE] StockInformationalMonitor.__init__: Setting up state variables...", flush=True)
        self.opportunities: List[StockOpportunity] = []
        self.last_scan_time: Optional[datetime] = None
        self.is_running = False
        
        # Resilience & Health Tracking (matching crypto breakout monitor robustness)
        self.stats = MonitorStats()
        self.recent_errors: Dict[str, datetime] = {}  # Track errors per ticker
        self.failed_tickers: Set[str] = set()  # Tickers in circuit breaker state
        self.circuit_breaker_threshold = 3  # Errors before circuit breaker trips
        self.circuit_breaker_duration_minutes = 30  # Minutes to keep ticker in circuit breaker
        self.max_consecutive_errors = 5  # Max consecutive errors before degrading service
        self.error_recovery_backoff_seconds = 60  # Start backoff after error
        self.watchlist_sync_interval_minutes = 60  # Sync with config every hour
        self.last_watchlist_sync: Optional[datetime] = None
        
        # Alert cooldown to prevent spam (like crypto breakout monitor)
        self.recent_alerts: Dict[str, datetime] = {}
        self.alert_cooldown_minutes = 60
        
        # Discord webhook for rich embeds (like crypto breakout monitor)
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        if not self.discord_webhook:
            logger.warning("⚠️ DISCORD_WEBHOOK_URL not set - alerts will use fallback system")
        
        # Discord bot for interactive buttons (Watch/Analyze/Dismiss) - same as crypto breakout monitor
        self.discord_bot_manager = None
        try:
            from services.discord_trade_approval import get_discord_approval_manager
            self.discord_bot_manager = get_discord_approval_manager()
            if self.discord_bot_manager and self.discord_bot_manager.enabled:
                logger.info("✅ Discord bot enabled - stock alerts will have interactive buttons")
            else:
                logger.info("ℹ️ Discord bot not configured - using webhook only (no buttons)")
                self.discord_bot_manager = None
        except Exception as e:
            logger.debug(f"Discord bot not available: {e}")
            self.discord_bot_manager = None
        
        print(f"[TRACE] StockInformationalMonitor.__init__: About to log initialization complete...", flush=True)
        logger.info(
            f"Stock Informational Monitor initialized "
            f"({len(self.watchlist)} tickers, {self.scan_interval}min interval)"
        )
        logger.info(f"   ✅ Health tracking enabled | Circuit breaker threshold: {self.circuit_breaker_threshold} errors")
        logger.info(f"   ✅ Alert cooldown: {self.alert_cooldown_minutes} minutes | Watchlist sync: {self.watchlist_sync_interval_minutes}min")
        print(f"[TRACE] StockInformationalMonitor.__init__: Constructor COMPLETE", flush=True)
    
    def scan_ticker(self, symbol: str, max_retries: int = 1) -> Optional[StockOpportunity]:
        """
        Scan a single ticker for opportunities with multi-pronged approach
        
        ENHANCED: Now uses comprehensive analysis instead of cache-only approach:
        - Technical indicators analysis
        - ML confidence scoring
        - Event/catalyst detection
        - LLM reasoning
        - Composite weighted scoring
        
        Args:
            symbol: Ticker symbol to scan
            max_retries: Maximum number of retry attempts
            
        Returns:
            StockOpportunity if found, None otherwise
        """
        # Validate ticker format - skip crypto and invalid tickers
        if '/' in symbol:
            logger.debug(f"  ⏭️  {symbol}: Skipped (crypto pair format)")
            return None
        
        if not symbol.replace('.', '').isalnum():
            logger.debug(f"  ⏭️  {symbol}: Skipped (invalid format)")
            return None
        
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if retry_count == 0:
                    logger.info(f"🔎 Scanning {symbol}...")
                else:
                    logger.debug(f"  🔄 Retrying {symbol} (attempt {retry_count + 1}/{max_retries + 1})...")
                
                # ENHANCED: Use multi-pronged detector for comprehensive analysis
                enhanced_opp = self.enhanced_detector.analyze_ticker(symbol)
                
                if enhanced_opp:
                    # Convert enhanced opportunity to StockOpportunity
                    return self._convert_enhanced_to_stock_opportunity(enhanced_opp)
                
                # Fallback: Try legacy analyzer if enhanced returns None
                logger.debug(f"  ↳ {symbol}: Falling back to legacy analyzer...")
                analysis = self.confidence_scanner.analyze_ticker(symbol)
                logger.debug(f"  ↳ {symbol}: Legacy analyzer returned: {analysis is not None}")
                
                if not analysis:
                    logger.debug(f"  ↳ {symbol}: Not in scanner cache, skipping")
                    return None
                
                ensemble_score = analysis.get('ensemble_score', 0)
                logger.debug(f"  ↳ {symbol}: Found in cache with score {ensemble_score}")
                
                # Filter by minimum score
                if ensemble_score < self.min_score:
                    logger.debug(f"  ↳ {symbol}: Score {ensemble_score} < {self.min_score} threshold, skipping")
                    return None
                
                # Skip the second LLM call for reasoning - use technical summary instead
                # This was causing hangs due to rate limiting with local Ollama
                logger.debug(f"  ↳ {symbol}: Score passed threshold, generating reasoning from technical data...")
                reasoning = f"Score {ensemble_score}/100 with {analysis.get('technical_setup', 'technical indicators')}. "
                if analysis.get('volume_surge'):
                    reasoning += "Volume surge detected. "
                confidence_pct = analysis.get('ml_confidence', 0.5) * 100
                reasoning += f"ML confidence: {confidence_pct:.0f}%."
                
                # Determine timeframe alignment
                timeframe_alignment = self._determine_alignment(analysis)
                
                # Determine alert priority
                alert_priority = self._determine_priority(
                    ensemble_score,
                    timeframe_alignment,
                    analysis
                )
                
                # Create opportunity
                opportunity = StockOpportunity(
                    symbol=symbol,
                    opportunity_type=analysis.get('setup_type', 'SETUP'),
                    ensemble_score=ensemble_score,
                    confidence=analysis.get('confidence', 0.5),
                    price=analysis.get('current_price', 0.0),
                    reasoning=reasoning.strip(),
                    technical_summary=analysis.get('technical_setup', ''),
                    timeframe_alignment=timeframe_alignment,
                    alert_priority=alert_priority,
                    timestamp=datetime.now().isoformat(),
                    metadata={
                        'ml_confidence': analysis.get('ml_confidence', 0),
                        'sentiment_score': analysis.get('sentiment_score', 50),
                        'volume_surge': analysis.get('volume_surge', False),
                        'volatility': analysis.get('volatility', 0),
                    }
                )
                
                logger.debug(
                    f"Found opportunity: {symbol} "
                    f"(Score: {ensemble_score}, Priority: {alert_priority})"
                )
                
                return opportunity
            
            except Exception as e:
                if retry_count < max_retries:
                    logger.debug(f"  ⚠️  {symbol} attempt {retry_count + 1} failed: {e}, retrying...")
                    retry_count += 1
                    time.sleep(0.5)  # Brief backoff before retry
                else:
                    logger.error(f"Error scanning {symbol} (all retries exhausted): {e}")
                    self._record_ticker_error(symbol)
                    return None
    
    def _determine_alignment(self, analysis: Dict) -> str:
        """Determine timeframe alignment level"""
        aligned_count = analysis.get('timeframes_aligned', 0)
        
        if aligned_count >= 3:
            return "TRIPLE_THREAT"
        elif aligned_count == 2:
            return "DUAL_ALIGN"
        elif aligned_count == 1:
            return "SINGLE"
        else:
            return "NONE"
    
    def _determine_priority(
        self,
        score: int,
        alignment: str,
        analysis: Dict
    ) -> str:
        """Determine alert priority based on multiple factors"""
        
        # CRITICAL: Triple threat or very high score
        if alignment == "TRIPLE_THREAT" or score >= 85:
            return "CRITICAL"
        
        # HIGH: Dual alignment or high score
        if alignment == "DUAL_ALIGN" or score >= 75:
            return "HIGH"
        
        # MEDIUM: Single alignment or moderate score
        if alignment == "SINGLE" or score >= 65:
            return "MEDIUM"
        
        # LOW: Below threshold but still valid
        return "LOW"
    
    # ============================================================
    # DISCOVERY: Stock Universe Discovery Control
    # ============================================================
    
    def set_discovery_enabled(self, enabled: bool):
        """
        Enable or disable stock discovery
        
        Args:
            enabled: True to discover stocks outside watchlist
        """
        self.discovery_enabled = enabled
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        logger.info(f"Stock discovery: {status}")
    
    def configure_discovery_modes(self, modes_config: Dict[str, bool]):
        """
        Configure which discovery modes are active
        
        Args:
            modes_config: Dict of mode_name -> enabled (e.g., {'top_gainers': True, 'most_active': False})
        """
        logger.info("📋 Configuring discovery modes...")
        for mode_name, enabled in modes_config.items():
            if mode_name in self.discovery_universe.modes:
                self.discovery_universe.set_discovery_enabled(mode_name, enabled)
                self.discovery_modes[mode_name] = enabled
        logger.info(f"   Active modes: {sum(1 for e in modes_config.values() if e)}")
    
    def get_discovery_config(self) -> Dict[str, Any]:
        """Get current discovery configuration"""
        return {
            'enabled': self.discovery_enabled,
            'modes': self.discovery_universe.get_discovery_config()
        }
    
    def set_discovery_config_from_panel(self, config: Dict[str, Any]):
        """Load discovery configuration from control panel"""
        try:
            if 'enabled' in config:
                self.set_discovery_enabled(config['enabled'])
            if 'modes' in config:
                self.discovery_universe.set_discovery_config(config['modes'])
            logger.info("✅ Discovery config loaded from control panel")
        except Exception as e:
            logger.error(f"Error loading discovery config: {e}")
    
    # ============================================================
    # ENHANCEMENT: Multi-Pronged Analysis Integration
    # ============================================================
    
    def _convert_enhanced_to_stock_opportunity(self, enhanced_opp) -> StockOpportunity:
        """Convert enhanced opportunity to StockOpportunity format"""
        try:
            # Map enhanced signals to stock opportunity
            ensemble_score = int(enhanced_opp.composite_score)
            
            # Determine timeframe alignment from signals
            timeframe_alignment = "SINGLE"
            if enhanced_opp.technical_signals.price_above_sma50 and enhanced_opp.technical_signals.price_above_sma200:
                timeframe_alignment = "TRIPLE_THREAT"
            elif enhanced_opp.technical_signals.price_above_sma50 or enhanced_opp.technical_signals.macd_bullish:
                timeframe_alignment = "DUAL_ALIGN"
            
            # Determine priority from enhanced confidence
            priority_map = {
                "CRITICAL": "CRITICAL",
                "HIGH": "HIGH",
                "MEDIUM": "MEDIUM",
                "LOW": "LOW"
            }
            alert_priority = priority_map.get(enhanced_opp.confidence, "MEDIUM")
            
            # Build reasoning from all signals
            reasoning_parts = []
            if enhanced_opp.composite_reasoning:
                reasoning_parts.append(enhanced_opp.composite_reasoning)
            if enhanced_opp.ml_reasoning:
                reasoning_parts.append(f"ML: {enhanced_opp.ml_reasoning[:100]}")
            
            reasoning = " | ".join(reasoning_parts) if reasoning_parts else "Multi-factor analysis opportunity"
            
            # Create opportunity
            opportunity = StockOpportunity(
                symbol=enhanced_opp.symbol,
                opportunity_type="MULTI_FACTOR",
                ensemble_score=ensemble_score,
                confidence=min(1.0, enhanced_opp.composite_score / 100.0),
                price=enhanced_opp.price,
                reasoning=reasoning,
                technical_summary=f"Tech: {enhanced_opp.technical_signals.score:.0f} | Events: {enhanced_opp.event_signals.score:.0f}",
                timeframe_alignment=timeframe_alignment,
                alert_priority=alert_priority,
                timestamp=enhanced_opp.timestamp,
                metadata={
                    'technical_score': enhanced_opp.technical_score,
                    'event_score': enhanced_opp.event_score,
                    'ml_score': enhanced_opp.ml_score,
                    'llm_score': enhanced_opp.llm_score,
                    'trend': enhanced_opp.technical_signals.trend,
                    'volume_ratio': enhanced_opp.technical_signals.volume_ratio,
                    'composite_reasoning': enhanced_opp.composite_reasoning,
                }
            )
            
            logger.debug(f"  ✅ Converted enhanced opportunity: {enhanced_opp.symbol} ({ensemble_score}/100)")
            return opportunity
        
        except Exception as e:
            logger.error(f"Error converting enhanced opportunity: {e}")
            return None
    
    # ============================================================
    # RESILIENCE & HEALTH MANAGEMENT METHODS
    # ============================================================
    
    def _is_ticker_in_circuit_breaker(self, symbol: str) -> bool:
        """Check if ticker is currently in circuit breaker state"""
        if symbol not in self.failed_tickers:
            return False
        
        # Check if circuit breaker has expired
        try:
            # We'd need to track when it was added, for now just check if in set
            # In a more robust version, store timestamps
            return True
        except:
            return False
    
    def _record_ticker_error(self, symbol: str):
        """Record an error for a ticker and potentially trip circuit breaker"""
        now = datetime.now()
        self.recent_errors[symbol] = now
        
        # Count recent errors (last 10 minutes)
        error_count = 0
        cutoff_time = now - timedelta(minutes=10)
        
        for error_time in self.recent_errors.values():
            if error_time > cutoff_time:
                error_count += 1
        
        # Trip circuit breaker if threshold exceeded
        if error_count >= self.circuit_breaker_threshold:
            if symbol not in self.failed_tickers:
                self.failed_tickers.add(symbol)
                self.stats.circuit_breaker_trips += 1
                logger.warning(f"⚠️  Circuit breaker TRIPPED for {symbol} ({error_count} errors)")
    
    def _is_alert_on_cooldown(self, symbol: str) -> bool:
        """Check if symbol has been recently alerted to prevent spam"""
        if symbol not in self.recent_alerts:
            return False
        
        last_alert = self.recent_alerts[symbol]
        cooldown_expired = datetime.now() > last_alert + timedelta(minutes=self.alert_cooldown_minutes)
        
        return not cooldown_expired
    
    def _record_alert(self, symbol: str):
        """Record that we sent an alert for this symbol"""
        self.recent_alerts[symbol] = datetime.now()
    
    def _sync_watchlist_from_config(self):
        """Periodically sync watchlist with Control Panel configuration"""
        try:
            now = datetime.now()
            if self.last_watchlist_sync:
                elapsed = (now - self.last_watchlist_sync).total_seconds() / 60
                if elapsed < self.watchlist_sync_interval_minutes:
                    return  # Not time to sync yet
            
            # Load from Control Panel if available
            try:
                from windows_services.runners.service_config_loader import load_service_watchlist
                custom_watchlist = load_service_watchlist('sentient-stock-monitor')
                if custom_watchlist and custom_watchlist != self.watchlist:
                    logger.info(f"📋 Syncing watchlist: {len(custom_watchlist)} tickers from Control Panel")
                    self.watchlist = custom_watchlist
                    self.last_watchlist_sync = now
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Could not sync watchlist: {e}")
        
        except Exception as e:
            logger.error(f"Error in _sync_watchlist_from_config: {e}")
    
    def _sync_discovery_config(self):
        """Periodically sync discovery configuration from Control Panel"""
        try:
            from windows_services.runners.service_discovery_config import load_discovery_config
            config = load_discovery_config()
            
            # Update discovery enabled state
            if config.get('enabled') != self.discovery_enabled:
                self.set_discovery_enabled(config['enabled'])
                logger.info(f"🔍 Discovery {'enabled' if config['enabled'] else 'disabled'} via Control Panel")
            
            # Update discovery modes
            modes_config = {
                name: settings['enabled']
                for name, settings in config.get('modes', {}).items()
            }
            self.configure_discovery_modes(modes_config)
            
            # Update universe sizes
            for mode_name, mode_settings in config.get('modes', {}).items():
                if mode_name in self.discovery_universe.modes:
                    max_size = mode_settings.get('max_universe_size')
                    if max_size:
                        self.discovery_universe.modes[mode_name].max_universe_size = max_size
            
        except ImportError:
            pass  # Discovery config not available
        except Exception as e:
            logger.debug(f"Could not sync discovery config: {e}")
    
    def _cleanup_old_errors(self):
        """Remove old error records to prevent memory growth"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            symbols_to_remove = [
                s for s, t in self.recent_errors.items()
                if t < cutoff_time
            ]
            for symbol in symbols_to_remove:
                del self.recent_errors[symbol]
            
            # Also cleanup old alerts
            alert_cutoff = datetime.now() - timedelta(hours=2)
            alert_symbols_to_remove = [
                s for s, t in self.recent_alerts.items()
                if t < alert_cutoff
            ]
            for symbol in alert_symbols_to_remove:
                del self.recent_alerts[symbol]
        
        except Exception as e:
            logger.debug(f"Error cleaning up old errors: {e}")
    
    def _get_health_summary(self) -> str:
        """Get a summary of monitor health"""
        health = self.stats.get_health_status()
        uptime_minutes = (datetime.now() - datetime.fromisoformat(self.stats.start_time)).total_seconds() / 60
        
        summary = (
            f"Health: {health} | "
            f"Uptime: {uptime_minutes:.0f}min | "
            f"Scans: {self.stats.scans_completed} | "
            f"Alerts: {self.stats.alerts_sent} | "
            f"Errors: {self.stats.errors} ({self.stats.consecutive_errors} consecutive) | "
            f"Circuit Breaks: {self.stats.circuit_breaker_trips}"
        )
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics (matching crypto breakout monitor)"""
        return self.stats.to_dict()
    
    def _print_final_stats(self):
        """Print final statistics when service stops (like crypto breakout monitor)"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 STOCK MONITOR FINAL STATISTICS")
        logger.info("=" * 80)
        for key, value in self.stats.to_dict().items():
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
        logger.info("=" * 80)
    
    def scan_all_tickers(self) -> List[StockOpportunity]:
        """
        Scan all tickers in watchlist with resilience features
        
        Returns:
            List of detected opportunities
        """
        scan_start = time.time()
        opportunities = []
        skipped_count = 0
        error_count = 0
        
        logger.info("=" * 80)
        logger.info(f"🔍 Starting scan of {len(self.watchlist) if self.watchlist else 0} tickers...")
        
        if not self.watchlist:
            logger.warning("No tickers in watchlist to scan")
            self.stats.scans_completed += 1
            return opportunities
        
        # Cleanup old error/alert records periodically
        self._cleanup_old_errors()
        
        # Sync watchlist from Control Panel if needed
        self._sync_watchlist_from_config()
        
        # BUILD SCAN UNIVERSE
        scan_universe = list(self.watchlist) if self.watchlist else []
        
        # Add discovered stocks if enabled
        if self.discovery_enabled:
            logger.info("🔍 Running stock discovery to expand universe...")
            try:
                discovered_dict = self.discovery_universe.discover_stocks(exclude_watchlist=scan_universe)
                
                # Flatten discovered tickers from all modes
                discovered_tickers = []
                for mode_tickers in discovered_dict.values():
                    discovered_tickers.extend(mode_tickers)
                
                # Deduplicate and validate (discovery should already filter, but double-check)
                discovered_tickers = list(set(discovered_tickers))
                # Filter out any invalid tickers (shouldn't be needed, but safety check)
                valid_discovered = [t for t in discovered_tickers if self.discovery_universe._is_valid_stock_ticker(t)]
                
                # Add to scan universe (limit to avoid overwhelming)
                max_discovered = 50  # Limit added tickers
                valid_discovered = valid_discovered[:max_discovered]
                scan_universe.extend(valid_discovered)
                
                logger.info(f"📈 Added {len(valid_discovered)} discovered stock tickers (total universe: {len(scan_universe)})")
            except Exception as e:
                logger.error(f"❌ Error in discovery: {e}")
        
        self.stats.tickers_scanned = 0
        
        for symbol in scan_universe:
            try:
                # Skip if in circuit breaker
                if self._is_ticker_in_circuit_breaker(symbol):
                    logger.debug(f"  ⏭️  {symbol}: Skipped (circuit breaker active)")
                    skipped_count += 1
                    continue
                
                self.stats.tickers_scanned += 1
                
                # Scan the ticker
                opportunity = self.scan_ticker(symbol)
                
                if opportunity:
                    opportunities.append(opportunity)
                    
                    # Check cooldown before sending alert
                    if ENABLE_ALERTS and not self._is_alert_on_cooldown(symbol):
                        self._send_alert(opportunity)
                        self._record_alert(symbol)
                        self.stats.alerts_sent += 1
                    elif ENABLE_ALERTS:
                        logger.debug(f"  🔕 {symbol}: Alert on cooldown")
                    
                    # Log opportunity
                    if LOG_OPPORTUNITIES:
                        self._log_opportunity(opportunity)
                    
                    self.stats.total_opportunities_found += 1
                    # Reset error counter on success
                    self.stats.consecutive_errors = 0
                
                # Small delay to respect rate limits
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                self._record_ticker_error(symbol)
                self.stats.errors += 1
                self.stats.consecutive_errors += 1
                self.stats.last_error_time = datetime.now().isoformat()
                error_count += 1
                continue
        
        # Update stats
        scan_duration = time.time() - scan_start
        self.stats.scans_completed += 1
        self.stats.last_scan_time = datetime.now().isoformat()
        
        # Calculate average scan duration
        if self.stats.scans_completed > 1:
            self.stats.avg_scan_duration_seconds = (
                (self.stats.avg_scan_duration_seconds * (self.stats.scans_completed - 1) + scan_duration) /
                self.stats.scans_completed
            )
        else:
            self.stats.avg_scan_duration_seconds = scan_duration
        
        logger.info(f"✅ Scan complete in {scan_duration:.1f}s")
        logger.info(f"   Found: {len(opportunities)} opportunities | Errors: {error_count} | Skipped: {skipped_count}")
        logger.info(f"   {self._get_health_summary()}")
        logger.info("")
        
        self.opportunities = opportunities
        self.last_scan_time = datetime.now()
        
        # Save results to control panel
        if opportunities:
            try:
                results_for_panel = []
                for opp in opportunities:
                    results_for_panel.append({
                        'ticker': opp.symbol,
                        'signal': f"{opp.opportunity_type}",
                        'confidence': int(opp.ensemble_score),
                        'price': opp.price,
                        'change_24h': 0.0, # Not readily available
                        'volume_24h': 0.0, # Not readily available
                        'timestamp': datetime.now().isoformat()
                    })
                
                save_analysis_results('Stock Info Monitor', results_for_panel)
                logger.debug(f"💾 Saved {len(results_for_panel)} results to control panel")
            except Exception as e:
                logger.warning(f"Failed to save results to control panel: {e}")
        
        return opportunities
    
    def _send_alert(self, opportunity: StockOpportunity):
        """Send Discord alert for opportunity and queue to orchestrator (crypto-style with buttons)"""
        
        # Log alert
        logger.info(f"\n🚨 STOCK ALERT: {opportunity.symbol}")
        logger.info(f"   Type: {opportunity.opportunity_type}")
        logger.info(f"   Score: {opportunity.ensemble_score}/100")
        logger.info(f"   Confidence: {opportunity.alert_priority} ({opportunity.confidence:.1%})")
        logger.info(f"   Price: ${opportunity.price:.2f}")
        logger.info(f"   Alignment: {opportunity.timeframe_alignment}")
        
        try:
            # Queue to Service Orchestrator for review in Control Panel
            self._queue_to_orchestrator(opportunity)
            
            # Try Discord bot first (has interactive buttons for Watch/Analyze/Dismiss)
            bot_sent = False
            if self.discord_bot_manager:
                try:
                    import asyncio
                    
                    # Build message for bot (matching crypto breakout monitor format)
                    emoji = self._get_opportunity_emoji(opportunity.opportunity_type, opportunity.ensemble_score)
                    
                    message_text = (
                        f"**Score:** {opportunity.ensemble_score}/100 | **Confidence:** {opportunity.alert_priority}\n"
                        f"**Price:** ${opportunity.price:.2f} | **Alignment:** {opportunity.timeframe_alignment}\n"
                        f"**Technical:** {opportunity.technical_summary[:100] if opportunity.technical_summary else 'N/A'}"
                    )
                    
                    # Add metadata from analysis
                    if opportunity.metadata:
                        tech_score = opportunity.metadata.get('technical_score', 0)
                        ml_score = opportunity.metadata.get('ml_score', 0)
                        trend = opportunity.metadata.get('trend', '')
                        volume_ratio = opportunity.metadata.get('volume_ratio', 0)
                        
                        if tech_score or ml_score:
                            message_text += f"\n**Scores:** Tech: {tech_score:.0f} | ML: {ml_score:.0f}"
                        if trend:
                            message_text += f"\n**Trend:** {trend}"
                        if volume_ratio > 0:
                            message_text += f" | **Volume:** {volume_ratio:.1f}x"
                    
                    # Add reasoning if available
                    if opportunity.reasoning:
                        message_text += f"\n\n🤖 **AI:** {opportunity.reasoning[:200]}"
                    
                    message_text += "\n\n**📊 Analysis:** `1`=Standard `2`=Multi `3`=Ultimate"
                    message_text += "\n**🎯 Actions:** `W`=Watch | `T`=Trade | `X`=Dismiss"
                    
                    # Color based on score (matching crypto breakout monitor)
                    if opportunity.ensemble_score >= 85:
                        color = 0x00FF00  # Green
                    elif opportunity.ensemble_score >= 75:
                        color = 0x32CD32  # Lime
                    elif opportunity.ensemble_score >= 65:
                        color = 0xFFD700  # Gold
                    else:
                        color = 0xFFA500  # Orange
                    
                    # Send via bot (has buttons) - same pattern as crypto breakout monitor
                    async def send_bot_alert():
                        return await self.discord_bot_manager.bot.send_alert_notification(
                            symbol=opportunity.symbol,
                            alert_type=f"{emoji} STOCK {opportunity.opportunity_type}",
                            message_text=message_text,
                            confidence=opportunity.alert_priority,
                            color=color
                        )
                    
                    if self.discord_bot_manager.loop:
                        future = asyncio.run_coroutine_threadsafe(
                            send_bot_alert(),
                            self.discord_bot_manager.loop
                        )
                        bot_sent = future.result(timeout=10)
                    
                    if bot_sent:
                        logger.info(f"   ✅ Discord alert sent via BOT (with Watch/Analyze/Dismiss buttons)")
                    else:
                        logger.debug("   Bot send returned False, falling back to webhook")
                
                except Exception as e:
                    logger.debug(f"   Bot alert failed ({type(e).__name__}), falling back to webhook: {e}")
            
            # Fallback to webhook (no buttons, but still shows embed)
            if not bot_sent:
                if self.discord_webhook:
                    try:
                        embed = self._build_discord_embed(opportunity)
                        
                        payload = {
                            'embeds': [embed],
                            'username': 'Stock Opportunity Monitor',
                            'avatar_url': 'https://cdn-icons-png.flaticon.com/512/2830/2830284.png'
                        }
                        
                        logger.debug(f"   📤 Sending Discord webhook...")
                        response = requests.post(self.discord_webhook, json=payload, timeout=15)
                        response.raise_for_status()
                        
                        logger.info(f"   ✅ Discord alert sent via WEBHOOK (HTTP {response.status_code})")
                        logger.info(f"      ⚠️ No buttons - reply with 1/2/3/W/X or configure Discord bot")
                    
                    except requests.exceptions.Timeout:
                        logger.error(f"   ❌ Discord webhook timeout (15s)")
                        # Fallback to alert system
                        self._send_fallback_alert(opportunity)
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"   ❌ Discord connection error: {e}")
                        self._send_fallback_alert(opportunity)
                    except Exception as e:
                        logger.error(f"   ❌ Failed to send Discord webhook: {type(e).__name__}: {e}")
                        self._send_fallback_alert(opportunity)
                else:
                    # Use fallback alert system
                    self._send_fallback_alert(opportunity)
        
        except Exception as e:
            logger.error(f"Error sending alert for {opportunity.symbol}: {e}")
    
    def _send_fallback_alert(self, opportunity: StockOpportunity):
        """Fallback to simple alert system when Discord fails"""
        try:
            alert_title = f"📊 {opportunity.symbol} - {opportunity.opportunity_type}"
            alert_message = f"""
**Score:** {opportunity.ensemble_score}/100
**Confidence:** {opportunity.confidence:.1%}
**Price:** ${opportunity.price:.2f}
**Alignment:** {opportunity.timeframe_alignment}

**Reasoning:**
{opportunity.reasoning}

**Technical:**
{opportunity.technical_summary}
            """.strip()
            
            self.alert_system.send_alert(
                title=alert_title,
                message=alert_message,
                priority=opportunity.alert_priority,
                metadata={
                    'symbol': opportunity.symbol,
                    'score': opportunity.ensemble_score,
                    'type': 'STOCK_OPPORTUNITY'
                }
            )
        except Exception as e:
            logger.error(f"Fallback alert also failed for {opportunity.symbol}: {e}")
    
    def _get_opportunity_emoji(self, opportunity_type: str, score: int) -> str:
        """Get emoji for opportunity type"""
        type_emoji = {
            'MULTI_FACTOR': '🎯',
            'SETUP': '📊',
            'BREAKOUT': '💥',
            'ENTRY': '🚀',
            'MOMENTUM': '⚡',
            'REVERSAL': '🔄',
            'WATCH': '👀'
        }
        emoji = type_emoji.get(opportunity_type, '📈')
        
        # Add fire for high scores
        if score >= 85:
            emoji = '🔥' + emoji
        elif score >= 75:
            emoji = '🌟' + emoji
        
        return emoji
    
    def _build_discord_embed(self, opportunity: StockOpportunity) -> Dict:
        """Build Discord embed message for stock opportunity (crypto-style)"""
        
        # Determine color based on score and priority
        if opportunity.ensemble_score >= 85 and opportunity.alert_priority == "CRITICAL":
            color = 0x00FF00  # Bright green
        elif opportunity.ensemble_score >= 75:
            color = 0x32CD32  # Lime green
        elif opportunity.ensemble_score >= 65:
            color = 0xFFD700  # Gold
        else:
            color = 0xFFA500  # Orange
        
        # Emoji for opportunity type
        emoji = self._get_opportunity_emoji(opportunity.opportunity_type, opportunity.ensemble_score)
        
        # Title
        title = f"{emoji} STOCK OPPORTUNITY: {opportunity.symbol}"
        
        # Description
        description = f"**Score:** {opportunity.ensemble_score}/100 | **Confidence:** {opportunity.alert_priority} ({opportunity.confidence:.1%})"
        
        # Fields
        fields = []
        
        # Price
        fields.append({
            'name': '💰 Price',
            'value': f'${opportunity.price:.2f}',
            'inline': True
        })
        
        # Timeframe Alignment
        alignment_emoji = {
            'TRIPLE_THREAT': '🎯',
            'DUAL_ALIGN': '✅',
            'SINGLE': '👀',
            'NONE': '⚪'
        }
        fields.append({
            'name': '📊 Alignment',
            'value': f"{alignment_emoji.get(opportunity.timeframe_alignment, '📊')} {opportunity.timeframe_alignment}",
            'inline': True
        })
        
        # Priority
        priority_emoji = {
            'CRITICAL': '🔴',
            'HIGH': '🟠',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }
        fields.append({
            'name': '⚡ Priority',
            'value': f"{priority_emoji.get(opportunity.alert_priority, '⚪')} {opportunity.alert_priority}",
            'inline': True
        })
        
        # Technical Summary
        if opportunity.technical_summary:
            fields.append({
                'name': '📈 Technical Analysis',
                'value': opportunity.technical_summary[:200] if len(opportunity.technical_summary) > 200 else opportunity.technical_summary,
                'inline': False
            })
        
        # Metadata scores if available
        if opportunity.metadata:
            tech_score = opportunity.metadata.get('technical_score', 0)
            ml_score = opportunity.metadata.get('ml_score', 0)
            llm_score = opportunity.metadata.get('llm_score', 0)
            
            if any([tech_score, ml_score, llm_score]):
                score_breakdown = f"Tech: {tech_score:.0f} | ML: {ml_score:.0f} | LLM: {llm_score:.0f}"
                fields.append({
                    'name': '🔬 Score Breakdown',
                    'value': score_breakdown,
                    'inline': False
                })
            
            # Volume ratio if available
            volume_ratio = opportunity.metadata.get('volume_ratio', 0)
            if volume_ratio > 0:
                vol_emoji = "🚀" if volume_ratio > 2 else "📊"
                fields.append({
                    'name': '📊 Volume',
                    'value': f'{vol_emoji} {volume_ratio:.1f}x average',
                    'inline': True
                })
            
            # Trend if available
            trend = opportunity.metadata.get('trend', '')
            if trend:
                trend_emoji = "📈" if trend == "UPTREND" else "📉" if trend == "DOWNTREND" else "➡️"
                fields.append({
                    'name': '📊 Trend',
                    'value': f'{trend_emoji} {trend}',
                    'inline': True
                })
        
        # AI Reasoning
        if opportunity.reasoning:
            reasoning = opportunity.reasoning
            if len(reasoning) > 400:
                reasoning = reasoning[:397] + "..."
            fields.append({
                'name': '🤖 AI Analysis',
                'value': reasoning,
                'inline': False
            })
        
        # Trading suggestion
        action_suggestion = self._get_action_suggestion(opportunity)
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
                'text': f'Scan #{self.stats.scans_completed} | Stock Opportunity Monitor | Uptime: {(datetime.now() - datetime.fromisoformat(self.stats.start_time)).total_seconds() / 60:.0f}min'
            }
        }
        
        return embed
    
    def _get_action_suggestion(self, opportunity: StockOpportunity) -> str:
        """Get trading action suggestion based on opportunity analysis"""
        
        suggestions = []
        
        # Score-based suggestion
        if opportunity.ensemble_score >= 85:
            suggestions.append("⚡ **STRONG SETUP** - High confidence, consider entry")
        elif opportunity.ensemble_score >= 75:
            suggestions.append("✅ **GOOD SETUP** - Solid technical profile")
        elif opportunity.ensemble_score >= 65:
            suggestions.append("👀 **MONITOR** - Developing setup, wait for confirmation")
        else:
            suggestions.append("⚠️ **WATCH** - Moderate signal, requires more validation")
        
        # Alignment consideration
        if opportunity.timeframe_alignment == "TRIPLE_THREAT":
            suggestions.append("🎯 Triple timeframe alignment - strongest signal")
        elif opportunity.timeframe_alignment == "DUAL_ALIGN":
            suggestions.append("✅ Dual timeframe alignment - good confirmation")
        
        # Metadata-based suggestions
        if opportunity.metadata:
            volume_ratio = opportunity.metadata.get('volume_ratio', 0)
            if volume_ratio > 3:
                suggestions.append("🚀 Massive volume surge - strong momentum")
            elif volume_ratio < 1.5 and volume_ratio > 0:
                suggestions.append("⚠️ Low volume - wait for confirmation")
            
            trend = opportunity.metadata.get('trend', '')
            if trend == "UPTREND":
                suggestions.append("📈 Confirmed uptrend")
            elif trend == "DOWNTREND":
                suggestions.append("⚠️ Downtrend - counter-trend play")
        
        return "\n".join(suggestions) if suggestions else "📊 Standard opportunity - evaluate entry"
    
    def _queue_to_orchestrator(self, opportunity: StockOpportunity):
        """Queue stock opportunity to Service Orchestrator for review in Control Panel (crypto-style)"""
        try:
            from services.service_orchestrator import get_orchestrator
            orch = get_orchestrator()
            
            # Map priority to confidence (matching crypto breakout monitor)
            priority_to_confidence = {
                "CRITICAL": "HIGH",
                "HIGH": "HIGH",
                "MEDIUM": "MEDIUM",
                "LOW": "LOW"
            }
            
            # Build comprehensive reasoning (like crypto breakout monitor)
            reasoning_parts = [f"{opportunity.opportunity_type}: Score {opportunity.ensemble_score}/100"]
            reasoning_parts.append(f"Alignment: {opportunity.timeframe_alignment}")
            
            if opportunity.reasoning:
                reasoning_parts.append(opportunity.reasoning[:150])
            
            if opportunity.metadata:
                trend = opportunity.metadata.get('trend', '')
                volume_ratio = opportunity.metadata.get('volume_ratio', 0)
                if trend:
                    reasoning_parts.append(f"Trend: {trend}")
                if volume_ratio > 0:
                    reasoning_parts.append(f"Vol: {volume_ratio:.1f}x")
            
            # Create comprehensive metadata for control panel
            full_metadata = {
                # Core scores
                "score": opportunity.ensemble_score,
                "ensemble_score": opportunity.ensemble_score,
                "confidence_pct": opportunity.confidence,
                
                # Technical
                "timeframe_alignment": opportunity.timeframe_alignment,
                "alert_priority": opportunity.alert_priority,
                "technical_summary": opportunity.technical_summary[:300] if opportunity.technical_summary else "",
                
                # For display in Control Panel
                "display_score": f"{opportunity.ensemble_score}/100",
                "display_confidence": f"{opportunity.confidence:.1%}",
                "display_alignment": opportunity.timeframe_alignment,
                
                # Action suggestions for Control Panel
                "suggested_action": self._get_action_suggestion(opportunity),
                
                # Full reasoning for analysis
                "full_reasoning": opportunity.reasoning,
                
                # Timestamp for tracking
                "detected_at": opportunity.timestamp,
                
                # Include all original metadata
                **opportunity.metadata
            }
            
            orch.add_alert(
                symbol=opportunity.symbol,
                alert_type=opportunity.opportunity_type,
                source="stock_monitor",
                asset_type="stock",
                price=opportunity.price,
                reasoning=" | ".join(reasoning_parts),
                confidence=priority_to_confidence.get(opportunity.alert_priority, "MEDIUM"),
                expires_minutes=240,  # 4 hour expiry for stocks
                metadata=full_metadata
            )
            logger.info(f"   📥 {opportunity.symbol} queued to Control Panel alert queue")
            
            # Also save to analysis results for the Results tab
            self._save_to_analysis_results(opportunity)
            
        except Exception as e:
            logger.debug(f"   Could not queue to orchestrator: {e}")
    
    def _save_to_analysis_results(self, opportunity: StockOpportunity):
        """Save opportunity to analysis results for display in Control Panel Results tab"""
        try:
            result = {
                'ticker': opportunity.symbol,
                'signal': opportunity.opportunity_type,
                'confidence': opportunity.ensemble_score,
                'price': opportunity.price,
                'change_24h': 0.0,  # Could be calculated from metadata if available
                'volume_24h': opportunity.metadata.get('volume_ratio', 0) if opportunity.metadata else 0,
                'timestamp': datetime.now().isoformat(),
                'alignment': opportunity.timeframe_alignment,
                'priority': opportunity.alert_priority,
                'reasoning': opportunity.reasoning[:200] if opportunity.reasoning else '',
                'technical_summary': opportunity.technical_summary[:150] if opportunity.technical_summary else '',
            }
            
            # Add to results
            save_analysis_results('Stock Monitor Alert', [result])
            logger.debug(f"   💾 Saved {opportunity.symbol} to analysis results")
        except Exception as e:
            logger.debug(f"   Could not save to analysis results: {e}")
    
    def _log_opportunity(self, opportunity: StockOpportunity):
        """Log opportunity to file"""
        try:
            log_path = OPPORTUNITIES_LOG_PATH
            
            # Create logs directory if needed
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            # Load existing logs
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new opportunity
            logs.append(opportunity.to_dict())
            
            # Keep last 1000 entries
            logs = logs[-1000:]
            
            # Save
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error logging opportunity: {e}")
    
    def _process_analysis_queue(self):
        """Process pending analysis requests"""
        try:
            requests = get_pending_analysis_requests()
            
            # Filter for stock requests
            my_requests = []
            for req in requests:
                preset = req.get('preset', '')
                tickers = req.get('tickers', [])
                is_stock = preset == 'stock_momentum'
                is_custom_stock = preset == 'custom' and tickers and not any('/' in t for t in tickers)
                
                if (is_stock or is_custom_stock) and tickers:
                    my_requests.append(req)
            
            if not my_requests:
                return
                
            for req in my_requests:
                logger.info(f"Processing request {req['id']}...")
                tickers = req.get('tickers', [])
                results = []
                
                for ticker in tickers:
                    try:
                        opp = self.scan_ticker(ticker)
                        if opp:
                            results.append({
                                'ticker': opp.symbol,
                                'signal': f"{opp.opportunity_type}",
                                'confidence': int(opp.ensemble_score),
                                'price': opp.price,
                                'change_24h': 0.0,
                                'volume_24h': 0.0,
                                'timestamp': datetime.now().isoformat()
                            })
                    except Exception as e:
                        logger.error(f"Error scanning {ticker}: {e}")
                
                save_analysis_results('Stock Info Monitor (On-Demand)', results)
                mark_analysis_complete(req['id'], {'count': len(results)})
                logger.info(f"Completed request {req['id']}")
                
        except Exception as e:
            logger.error(f"Error processing analysis queue: {e}")

    def run_continuous(self):
        """
        Run continuous monitoring loop with resilience and health tracking
        Matches the robustness pattern of crypto breakout monitor
        """
        self.is_running = True
        logger.info("\n" + "=" * 80)
        logger.info("🎬 STOCK MONITOR - STARTING CONTINUOUS LOOP")
        logger.info("=" * 80)
        logger.info("Press Ctrl+C to stop\n")
        
        try:
            while self.is_running:
                cycle_start = time.time()
                
                try:
                    # Check analysis queue from Control Panel
                    self._process_analysis_queue()
                    
                    # Sync watchlist and discovery config from Control Panel
                    self._sync_watchlist_from_config()
                    self._sync_discovery_config()
                    
                    # Perform scan with resilience features
                    opportunities = self.scan_all_tickers()
                    
                except Exception as e:
                    logger.error("❌ Error during scan: {}", str(e), exc_info=True)
                    self.stats.errors += 1
                    self.stats.consecutive_errors += 1
                    self.stats.last_error_time = datetime.now().isoformat()
                
                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, (self.scan_interval * 60) - cycle_duration)
                
                if sleep_time > 0:
                    next_scan = datetime.now() + timedelta(seconds=sleep_time)
                    logger.info(f"💤 Sleeping {sleep_time:.0f}s... Next scan at {next_scan.strftime('%H:%M:%S')}\n")
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("\n⏹️  Monitoring stopped by user")
            self._print_final_stats()
        
        except Exception as e:
            logger.error("❌ Fatal error: {}", str(e), exc_info=True)
            self._print_final_stats()
            raise
        
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop continuous monitoring"""
        self.is_running = False
        logger.info("Stopping monitoring...")
    
    def get_recent_opportunities(self, limit: int = 50) -> List[StockOpportunity]:
        """Get recent opportunities"""
        return self.opportunities[-limit:]
    
    def get_opportunities_by_priority(self, priority: str) -> List[StockOpportunity]:
        """Get opportunities by alert priority"""
        return [
            opp for opp in self.opportunities
            if opp.alert_priority == priority.upper()
        ]
    
    def update_watchlist(self, new_tickers: List[str]):
        """
        Update the watchlist with new tickers
        
        Args:
            new_tickers: List of ticker symbols to monitor
        """
        self.watchlist = list(set(new_tickers))  # Remove duplicates
        logger.info(f"Updated watchlist to {len(self.watchlist)} tickers: {self.watchlist[:10]}...")
    
    def add_tickers(self, tickers: List[str]):
        """
        Add tickers to existing watchlist
        
        Args:
            tickers: List of ticker symbols to add
        """
        if not tickers:
            return
        
        if not self.watchlist:
            self.watchlist = []
        
        existing = set(self.watchlist)
        for ticker in tickers:
            if ticker not in existing:
                self.watchlist.append(ticker)
                existing.add(ticker)
        logger.info(f"Added {len(tickers)} tickers. Total: {len(self.watchlist)}")
    
    def remove_ticker(self, ticker: str):
        """Remove a ticker from watchlist"""
        if self.watchlist and ticker in self.watchlist:
            self.watchlist.remove(ticker)
            logger.info(f"Removed {ticker} from watchlist")
    
    def sync_from_tiered_scanner(self, tiered_results: List[Dict]):
        """
        Sync watchlist from tiered scanner results
        
        Args:
            tiered_results: Results from TieredStockScanner (Tier 2 or 3)
        """
        new_tickers = []
        for result in tiered_results:
            ticker = result.get('ticker')
            if ticker:
                new_tickers.append(ticker)
        
        if new_tickers:
            self.add_tickers(new_tickers)
            logger.info(f"Synced {len(new_tickers)} tickers from tiered scanner")
    
    def get_watchlist(self) -> List[str]:
        """Get current watchlist"""
        return self.watchlist.copy() if self.watchlist else []


# Singleton instance
_monitor_instance = None


def get_stock_informational_monitor(
    watchlist: Optional[List[str]] = None,
    **kwargs
) -> StockInformationalMonitor:
    """
    Get singleton instance of Stock Informational Monitor
    
    Args:
        watchlist: Optional custom watchlist
        **kwargs: Additional configuration
        
    Returns:
        StockInformationalMonitor instance
    """
    print(f"[TRACE] get_stock_informational_monitor: Called with watchlist={watchlist}, kwargs={kwargs}", flush=True)
    global _monitor_instance
    
    if _monitor_instance is None:
        print(f"[TRACE] get_stock_informational_monitor: Creating NEW instance...", flush=True)
        _monitor_instance = StockInformationalMonitor(
            watchlist=watchlist,
            **kwargs
        )
        print(f"[TRACE] get_stock_informational_monitor: Instance created successfully", flush=True)
    else:
        print(f"[TRACE] get_stock_informational_monitor: Returning existing instance", flush=True)
    
    return _monitor_instance


_debug_file.write("[TRACE] stock_informational_monitor.py: Module load COMPLETE\n")
_debug_file.flush()
_debug_file.close()

if __name__ == "__main__":
    # Create monitor
    monitor = get_stock_informational_monitor()
    
    logger.info("Starting continuous monitoring...")
    logger.info(f"Watching {len(monitor.watchlist) if monitor.watchlist else 0} symbols")
    
    # Run continuous monitoring
    try:
        monitor.run_continuous()
    except KeyboardInterrupt:
        monitor.stop()
        logger.info("Monitor stopped")

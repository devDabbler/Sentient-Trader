"""
Bonding Curve Monitor - Real-time monitoring for early token launches

This catches tokens at LAUNCH, not hours later from DexScreener.

Supported Platforms:
- pump.fun (via PumpPortal WebSocket - FREE, real-time)
- LaunchLab/Raydium (via Raydium V3 API)
- moonshot (planned)

The key insight: DexScreener only shows tokens AFTER they graduate (hit 100% 
bonding curve and get liquidity on Raydium/etc). By then, it's often too late.

This monitor connects DIRECTLY to bonding curve platforms to catch:
1. Token creation events (within seconds of launch)
2. Trade activity (early momentum detection)
3. Migration/graduation events (100% bonding curve complete)

Author: Sentient Trader
Created: December 2025
"""

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import httpx
from loguru import logger

# Discord integration
try:
    from src.integrations.discord_channels import AlertCategory, get_discord_webhook
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("Discord channels not available")


class BondingPlatform(Enum):
    """Supported bonding curve platforms"""
    PUMP_FUN = "pump.fun"
    LAUNCHLAB = "launchlab"
    MOONSHOT = "moonshot"


class TokenStage(Enum):
    """Token lifecycle stage on bonding curve"""
    CREATED = "created"  # Just created, 0% progress
    BONDING = "bonding"  # On bonding curve, 0-100%
    GRADUATING = "graduating"  # About to graduate (90%+)
    GRADUATED = "graduated"  # Hit 100%, migrating to DEX


@dataclass
class BondingToken:
    """Token on a bonding curve platform"""
    # Identity
    mint: str  # Token mint address
    symbol: str
    name: str
    platform: BondingPlatform
    
    # Creation info
    created_at: datetime = field(default_factory=datetime.now)
    creator: str = ""
    
    # Bonding curve state
    stage: TokenStage = TokenStage.CREATED
    progress_pct: float = 0.0  # 0-100%
    market_cap_sol: float = 0.0
    market_cap_usd: float = 0.0
    
    # Trade activity
    total_trades: int = 0
    total_volume_sol: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    unique_traders: int = 0
    
    # Momentum signals
    trades_1m: int = 0  # Trades in last minute
    volume_1m_sol: float = 0.0
    buy_pressure: float = 0.5  # 0-1, ratio of buys
    
    # Links
    uri: str = ""  # Metadata URI
    image_uri: str = ""
    twitter: str = ""
    telegram: str = ""
    website: str = ""
    
    # Internal tracking
    last_update: datetime = field(default_factory=datetime.now)
    alerted: bool = False
    score: float = 0.0


@dataclass
class MigrationEvent:
    """Token graduation/migration event"""
    mint: str
    symbol: str
    platform: BondingPlatform
    migrated_at: datetime
    pool_address: str = ""  # New DEX pool address
    initial_liquidity_sol: float = 0.0
    
    
class BondingCurveMonitor:
    """
    Real-time monitor for bonding curve platforms.
    
    Catches tokens at launch instead of waiting for DexScreener.
    
    RELAXED MODE: When enabled, uses lower thresholds to catch more tokens faster.
    This increases noise but also catches opportunities earlier (like SAFEMARS).
    """
    
    # PumpPortal WebSocket endpoint (FREE!)
    PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"
    
    # Raydium LaunchLab API
    RAYDIUM_API = "https://api-v3.raydium.io"
    
    # Persistence
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    TOKENS_FILE = DATA_DIR / "bonding_tokens.json"
    
    def __init__(
        self,
        enable_pump_fun: bool = True,
        enable_launchlab: bool = True,
        min_trades_to_alert: int = 5,
        min_volume_sol_to_alert: float = 1.0,
        alert_on_creation: bool = True,
        alert_on_graduation: bool = True,
        launchlab_poll_interval: int = 10,  # seconds (fast mode for catching launches)
        relaxed_mode: bool = False,  # NEW: Lower thresholds for more opportunities
        creation_filter_mode: str = "all",  # "all", "with_socials", "promising_only"
    ):
        """
        Initialize Bonding Curve Monitor
        
        Args:
            enable_pump_fun: Monitor pump.fun via WebSocket
            enable_launchlab: Monitor LaunchLab via polling
            min_trades_to_alert: Minimum trades before alerting
            min_volume_sol_to_alert: Minimum volume (SOL) before alerting
            alert_on_creation: Alert immediately on token creation
            alert_on_graduation: Alert when token graduates (100%)
            launchlab_poll_interval: How often to poll LaunchLab API
            relaxed_mode: Enable lower thresholds to catch more opportunities faster
            creation_filter_mode: Filter for creation alerts
                - "all": Alert on ALL new tokens (high noise)
                - "with_socials": Only alert if token has at least 1 social link
                - "promising_only": Only alert if instant assessment is PROMISING or WORTH WATCHING
        """
        # Check for relaxed mode from environment
        self.relaxed_mode = relaxed_mode or os.getenv("PUMPFUN_RELAXED_MODE", "false").lower() == "true"
        
        # Creation filter mode from environment
        self.creation_filter_mode = os.getenv("PUMPFUN_CREATION_FILTER", creation_filter_mode).lower()
        
        self.enable_pump_fun = enable_pump_fun
        self.enable_launchlab = enable_launchlab
        
        # Apply relaxed thresholds if enabled
        if self.relaxed_mode:
            # Relaxed mode: Lower thresholds to catch more opportunities
            self.min_trades_to_alert = int(os.getenv("PUMPFUN_MIN_TRADES_RELAXED", "1"))
            self.min_volume_sol_to_alert = float(os.getenv("PUMPFUN_MIN_VOLUME_RELAXED", "0.1"))
            logger.info("🔓 RELAXED MODE ENABLED - Lower thresholds for more opportunities!")
        else:
            self.min_trades_to_alert = min_trades_to_alert
            self.min_volume_sol_to_alert = min_volume_sol_to_alert
        
        self.alert_on_creation = alert_on_creation
        self.alert_on_graduation = alert_on_graduation
        self.launchlab_poll_interval = launchlab_poll_interval
        
        # Token tracking
        self.tokens: Dict[str, BondingToken] = {}  # mint -> BondingToken
        self.migrations: List[MigrationEvent] = []
        self.seen_mints: set = set()
        
        # WebSocket state
        self._ws = None
        self._ws_connected = False
        self._reconnect_delay = 5
        self._max_reconnect_delay = 60
        
        # Callbacks for external integration
        self._on_new_token: Optional[Callable[[BondingToken], None]] = None
        self._on_trade: Optional[Callable[[str, dict], None]] = None
        self._on_migration: Optional[Callable[[MigrationEvent], None]] = None
        
        # Discord webhook
        self.discord_webhook_url = self._get_discord_webhook()
        
        # Statistics
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.total_tokens_seen = 0
        self.total_migrations = 0
        self.total_alerts_sent = 0
        self.filtered_tokens = 0  # Tokens filtered out by creation_filter_mode
        
        # Ensure data directory exists
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load persisted data
        self._load_state()
        
        logger.info("=" * 60)
        logger.info("🎰 BONDING CURVE MONITOR INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"   🔓 RELAXED MODE: {'✅ ON (more opportunities)' if self.relaxed_mode else '❌ OFF (stricter filters)'}")
        logger.info(f"   🎯 CREATION FILTER: {self.creation_filter_mode.upper()}")
        logger.info(f"   pump.fun: {'✅ Enabled (WebSocket)' if enable_pump_fun else '❌ Disabled'}")
        logger.info(f"   LaunchLab: {'✅ Enabled (Polling)' if enable_launchlab else '❌ Disabled'}")
        logger.info(f"   Alert on creation: {'✅' if alert_on_creation else '❌'}")
        logger.info(f"   Alert on graduation: {'✅' if alert_on_graduation else '❌'}")
        logger.info(f"   Min trades to alert: {self.min_trades_to_alert}")
        logger.info(f"   Min volume to alert: {self.min_volume_sol_to_alert} SOL")
        logger.info(f"   Discord alerts: {'✅ Enabled' if self.discord_webhook_url else '❌ Disabled'}")
        logger.info(f"   Tracked tokens: {len(self.tokens)}")
        logger.info("=" * 60)
    
    def _get_discord_webhook(self) -> Optional[str]:
        """Get Discord webhook URL - routes to PUMPFUN_ALERTS channel for bonding curve tokens"""
        if DISCORD_AVAILABLE:
            # Primary: Use PUMPFUN_ALERTS for pump.fun bonding curve gambling
            webhook = get_discord_webhook(AlertCategory.PUMPFUN_ALERTS)
            if webhook:
                logger.debug("Using PUMPFUN_ALERTS webhook")
                return webhook
            # Fallback: DEX_PUMP_ALERTS for bonding curve launches
            webhook = get_discord_webhook(AlertCategory.DEX_PUMP_ALERTS)
            if webhook:
                logger.debug("Using DEX_PUMP_ALERTS webhook (fallback)")
                return webhook
            # Final fallback to fast monitor
            webhook = get_discord_webhook(AlertCategory.DEX_FAST_MONITOR)
            if webhook:
                return webhook
        # NO fallback to general DISCORD_WEBHOOK_URL - prevents duplicate alerts to general channel
        # If no pumpfun-specific webhook is configured, alerts are disabled
        logger.warning("No PUMPFUN_ALERTS webhook configured - bonding curve alerts disabled")
        return None
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def set_callbacks(
        self,
        on_new_token: Optional[Callable[[BondingToken], None]] = None,
        on_trade: Optional[Callable[[str, dict], None]] = None,
        on_migration: Optional[Callable[[MigrationEvent], None]] = None
    ):
        """Set callbacks for external integration (e.g., DEX Launch Hunter)"""
        self._on_new_token = on_new_token
        self._on_trade = on_trade
        self._on_migration = on_migration
    
    async def start(self):
        """Start monitoring all enabled platforms"""
        logger.info("🚀 Starting Bonding Curve Monitor...")
        self.is_running = True
        self.start_time = datetime.now()
        
        tasks = []
        
        if self.enable_pump_fun:
            tasks.append(asyncio.create_task(self._run_pumpfun_websocket()))
        
        if self.enable_launchlab:
            tasks.append(asyncio.create_task(self._run_launchlab_polling()))
        
        # Cleanup task
        tasks.append(asyncio.create_task(self._cleanup_loop()))
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Monitor tasks cancelled")
    
    def stop(self):
        """Stop monitoring"""
        logger.info("🛑 Stopping Bonding Curve Monitor...")
        self.is_running = False
        self._save_state()
    
    def get_token(self, mint: str) -> Optional[BondingToken]:
        """Get a specific token by mint address"""
        return self.tokens.get(mint)
    
    def get_recent_tokens(self, minutes: int = 30, platform: Optional[BondingPlatform] = None) -> List[BondingToken]:
        """Get tokens created in the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        tokens = [
            t for t in self.tokens.values()
            if t.created_at > cutoff
        ]
        if platform:
            tokens = [t for t in tokens if t.platform == platform]
        return sorted(tokens, key=lambda t: t.created_at, reverse=True)
    
    def get_hot_tokens(self, min_trades: int = 10, min_volume_sol: float = 5.0) -> List[BondingToken]:
        """Get tokens with significant activity"""
        hot = [
            t for t in self.tokens.values()
            if t.total_trades >= min_trades and t.total_volume_sol >= min_volume_sol
        ]
        return sorted(hot, key=lambda t: t.total_volume_sol, reverse=True)
    
    def get_graduating_tokens(self, min_progress: float = 80.0) -> List[BondingToken]:
        """Get tokens close to graduating (80%+ progress)"""
        graduating = [
            t for t in self.tokens.values()
            if t.progress_pct >= min_progress and t.stage != TokenStage.GRADUATED
        ]
        return sorted(graduating, key=lambda t: t.progress_pct, reverse=True)
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            "is_running": self.is_running,
            "uptime_minutes": (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0,
            "websocket_connected": self._ws_connected,
            "total_tokens_seen": self.total_tokens_seen,
            "active_tokens": len(self.tokens),
            "total_migrations": self.total_migrations,
            "total_alerts": self.total_alerts_sent,
            "pump_fun_enabled": self.enable_pump_fun,
            "launchlab_enabled": self.enable_launchlab,
        }
    
    # ========================================================================
    # PUMP.FUN WEBSOCKET (REAL-TIME)
    # ========================================================================
    
    async def _run_pumpfun_websocket(self):
        """Connect to PumpPortal WebSocket for real-time pump.fun data"""
        import websockets
        
        while self.is_running:
            try:
                logger.info(f"🔌 Connecting to PumpPortal WebSocket: {self.PUMPPORTAL_WS}")
                print(f"[BONDING] Connecting to PumpPortal WebSocket...", flush=True)
                
                async with websockets.connect(
                    self.PUMPPORTAL_WS,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self._ws = ws
                    self._ws_connected = True
                    self._reconnect_delay = 5  # Reset reconnect delay on successful connect
                    
                    logger.info("✅ Connected to PumpPortal WebSocket")
                    print("[BONDING] ✅ Connected to PumpPortal WebSocket!", flush=True)
                    
                    # Subscribe to new token events
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    logger.info("📡 Subscribed to new token events")
                    
                    # Subscribe to migration events
                    await ws.send(json.dumps({"method": "subscribeMigration"}))
                    logger.info("📡 Subscribed to migration events")
                    
                    # Process incoming messages
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            await self._handle_pumpfun_message(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from PumpPortal: {message[:100]}")
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"PumpPortal WebSocket closed: {e}")
                self._ws_connected = False
            except Exception as e:
                logger.error(f"PumpPortal WebSocket error: {e}")
                self._ws_connected = False
            
            # Reconnect with backoff
            if self.is_running:
                logger.info(f"Reconnecting in {self._reconnect_delay}s...")
                print(f"[BONDING] Reconnecting in {self._reconnect_delay}s...", flush=True)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
    
    async def _handle_pumpfun_message(self, data: dict):
        """Handle message from PumpPortal WebSocket"""
        
        # New token creation
        if "mint" in data and "traderPublicKey" not in data:
            # This is a token creation event
            await self._handle_new_token_pumpfun(data)
            
        # Trade event
        elif "mint" in data and "traderPublicKey" in data:
            await self._handle_trade_pumpfun(data)
            
        # Migration event
        elif "signature" in data and "slot" in data and "mint" in data:
            # Check if this looks like a migration (has specific structure)
            if data.get("txType") == "migration" or "pool" in str(data).lower():
                await self._handle_migration_pumpfun(data)
    
    async def _handle_new_token_pumpfun(self, data: dict):
        """Handle new token creation on pump.fun"""
        mint = data.get("mint", "")
        if not mint or mint in self.seen_mints:
            return
        
        self.seen_mints.add(mint)
        self.total_tokens_seen += 1
        
        # Parse token data
        token = BondingToken(
            mint=mint,
            symbol=data.get("symbol", "???"),
            name=data.get("name", "Unknown"),
            platform=BondingPlatform.PUMP_FUN,
            created_at=datetime.now(),
            creator=data.get("traderPublicKey", ""),
            uri=data.get("uri", ""),
            image_uri=data.get("imageUri", ""),
            twitter=data.get("twitter", ""),
            telegram=data.get("telegram", ""),
            website=data.get("website", ""),
        )
        
        # Parse initial market cap if available
        if "marketCapSol" in data:
            token.market_cap_sol = float(data.get("marketCapSol", 0))
        if "vSolInBondingCurve" in data:
            # Estimate progress from virtual SOL
            vsol = float(data.get("vSolInBondingCurve", 0))
            # pump.fun graduates at ~85 SOL
            token.progress_pct = min((vsol / 85.0) * 100, 100)
        
        self.tokens[mint] = token
        self._save_state()
        
        # Log and alert
        logger.info(f"🆕 NEW TOKEN: {token.symbol} ({token.name}) on pump.fun")
        logger.info(f"   └─ Mint: {mint[:20]}...")
        print(f"[BONDING] 🆕 NEW: {token.symbol} on pump.fun | {mint[:20]}...", flush=True)
        
        # Callback
        if self._on_new_token:
            try:
                self._on_new_token(token)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        # Send Discord alert if configured (with filtering)
        if self.alert_on_creation:
            should_alert = True
            
            # Apply creation filter
            if self.creation_filter_mode == "with_socials":
                # Only alert if has at least 1 social link
                has_socials = bool(token.twitter or token.telegram or token.website)
                if not has_socials:
                    should_alert = False
                    self.filtered_tokens += 1
                    logger.debug(f"   └─ Filtered (no socials): {token.symbol}")
                    
            elif self.creation_filter_mode == "promising_only":
                # Only alert if instant assessment is promising
                trust_emoji, trust_label, _ = self._get_instant_trust_assessment(token)
                if trust_label not in ["PROMISING", "WORTH WATCHING"]:
                    should_alert = False
                    self.filtered_tokens += 1
                    logger.debug(f"   └─ Filtered ({trust_label}): {token.symbol}")
            
            # "all" mode alerts on everything
            if should_alert:
                await self._send_creation_alert(token)
    
    async def _handle_trade_pumpfun(self, data: dict):
        """Handle trade event on pump.fun"""
        mint = data.get("mint", "")
        if not mint:
            return
        
        # Get or create token
        if mint not in self.tokens:
            # Token we haven't seen creation for - create placeholder
            self.tokens[mint] = BondingToken(
                mint=mint,
                symbol=data.get("symbol", "???"),
                name=data.get("name", "Unknown"),
                platform=BondingPlatform.PUMP_FUN,
            )
            self.seen_mints.add(mint)
        
        token = self.tokens[mint]
        token.last_update = datetime.now()
        
        # Update trade metrics
        token.total_trades += 1
        is_buy = data.get("txType") == "buy"
        
        if is_buy:
            token.buy_count += 1
        else:
            token.sell_count += 1
        
        # Volume
        sol_amount = float(data.get("solAmount", 0)) / 1e9  # lamports to SOL
        token.total_volume_sol += sol_amount
        
        # Market cap update
        if "marketCapSol" in data:
            token.market_cap_sol = float(data.get("marketCapSol", 0))
        if "newTokenBalance" in data and "bondingCurveKey" in data:
            # Can estimate progress from bonding curve balance
            pass
        
        # Buy pressure (rolling average)
        if token.total_trades > 0:
            token.buy_pressure = token.buy_count / token.total_trades
        
        # Callback
        if self._on_trade:
            try:
                self._on_trade(mint, data)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
        
        # Check if we should alert (first time meeting criteria)
        if not token.alerted:
            if token.total_trades >= self.min_trades_to_alert and token.total_volume_sol >= self.min_volume_sol_to_alert:
                token.alerted = True
                await self._send_momentum_alert(token)
    
    async def _handle_migration_pumpfun(self, data: dict):
        """Handle token graduation/migration on pump.fun"""
        mint = data.get("mint", "")
        if not mint:
            return
        
        # Get token info
        symbol = "???"
        if mint in self.tokens:
            self.tokens[mint].stage = TokenStage.GRADUATED
            symbol = self.tokens[mint].symbol
        
        migration = MigrationEvent(
            mint=mint,
            symbol=symbol,
            platform=BondingPlatform.PUMP_FUN,
            migrated_at=datetime.now(),
            pool_address=data.get("pool", ""),
        )
        
        self.migrations.append(migration)
        self.total_migrations += 1
        
        logger.info(f"🎓 GRADUATED: {symbol} ({mint[:20]}...) - Now on Raydium!")
        print(f"[BONDING] 🎓 GRADUATED: {symbol} - Now tradeable on Raydium!", flush=True)
        
        # Callback
        if self._on_migration:
            try:
                self._on_migration(migration)
            except Exception as e:
                logger.error(f"Migration callback error: {e}")
        
        # Alert
        if self.alert_on_graduation:
            await self._send_graduation_alert(migration)
    
    # ========================================================================
    # LAUNCHLAB POLLING
    # ========================================================================
    
    async def _run_launchlab_polling(self):
        """Poll Raydium LaunchLab API for new tokens"""
        logger.info(f"📊 Starting LaunchLab polling (every {self.launchlab_poll_interval}s)")
        
        while self.is_running:
            try:
                await self._poll_launchlab()
            except Exception as e:
                logger.error(f"LaunchLab polling error: {e}")
            
            await asyncio.sleep(self.launchlab_poll_interval)
    
    async def _poll_launchlab(self):
        """Poll LaunchLab for new tokens"""
        try:
            # Try to get recent LaunchLab tokens from Raydium API
            # Note: The exact endpoint may need adjustment based on Raydium's current API structure
            async with httpx.AsyncClient(timeout=30) as client:
                # Try the pools endpoint filtered for LaunchLab
                url = f"{self.RAYDIUM_API}/pools/info/list"
                params = {
                    "poolType": "all",
                    "poolSortField": "default",
                    "sortType": "desc",
                    "pageSize": 50,
                    "page": 1,
                }
                
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    pools = data.get("data", {}).get("data", [])
                    
                    # Filter for LaunchLab pools (usually have specific characteristics)
                    for pool in pools:
                        # Check if this looks like a LaunchLab pool
                        pool_type = pool.get("type", "").lower()
                        program_id = pool.get("programId", "")
                        
                        # LaunchLab has specific markers
                        if "launchlab" in pool_type or "launch" in pool_type:
                            mint = pool.get("mintA", {}).get("address", "")
                            if mint and mint not in self.seen_mints:
                                await self._process_launchlab_token(pool)
                
        except httpx.TimeoutException:
            logger.warning("LaunchLab API timeout")
        except Exception as e:
            logger.debug(f"LaunchLab API error: {e}")
    
    async def _process_launchlab_token(self, pool_data: dict):
        """Process a LaunchLab token from API response"""
        mint_info = pool_data.get("mintA", {})
        mint = mint_info.get("address", "")
        
        if not mint or mint in self.seen_mints:
            return
        
        self.seen_mints.add(mint)
        self.total_tokens_seen += 1
        
        token = BondingToken(
            mint=mint,
            symbol=mint_info.get("symbol", "???"),
            name=mint_info.get("name", "Unknown"),
            platform=BondingPlatform.LAUNCHLAB,
            created_at=datetime.now(),  # Approximation
            market_cap_usd=float(pool_data.get("tvl", 0)),
        )
        
        self.tokens[mint] = token
        
        logger.info(f"🆕 NEW TOKEN: {token.symbol} on LaunchLab")
        print(f"[BONDING] 🆕 NEW: {token.symbol} on LaunchLab | {mint[:20]}...", flush=True)
        
        if self._on_new_token:
            try:
                self._on_new_token(token)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        if self.alert_on_creation:
            await self._send_creation_alert(token)
    
    # ========================================================================
    # ALERTS
    # ========================================================================
    
    def _get_instant_trust_assessment(self, token: BondingToken) -> tuple:
        """
        Instant trust assessment at token CREATION - before any trades.
        This uses only what's available at mint time: name, symbol, socials, creator.
        
        Returns:
            (trust_emoji, trust_label, quick_flags)
        """
        score = 0
        flags = []
        
        # Check social presence (biggest indicator at creation)
        has_twitter = bool(token.twitter)
        has_telegram = bool(token.telegram)
        has_website = bool(token.website)
        social_count = sum([has_twitter, has_telegram, has_website])
        
        if social_count >= 3:
            score += 25
            flags.append("✅ Full socials")
        elif social_count == 2:
            score += 15
            flags.append("📱 2 socials")
        elif social_count == 1:
            score += 5
            flags.append("📱 1 social")
        else:
            score -= 15
            flags.append("❌ No socials")
        
        # Check name/symbol for rug patterns
        name_lower = token.name.lower()
        symbol_lower = token.symbol.lower()
        
        # Positive patterns (known meme formats that can pump)
        positive_patterns = ["pepe", "doge", "shib", "cat", "dog", "elon", "trump", "ai", "gpt", "moon", "rocket"]
        if any(p in name_lower or p in symbol_lower for p in positive_patterns):
            score += 10
            flags.append("🔥 Trending theme")
        
        # Negative patterns (common rug indicators)
        negative_patterns = ["test", "scam", "rug", "honeypot", "fake", "copy", "clone"]
        if any(p in name_lower or p in symbol_lower for p in negative_patterns):
            score -= 30
            flags.append("⚠️ Suspicious name")
        
        # Very short or generic symbols
        if len(token.symbol) < 2:
            score -= 10
            flags.append("⚠️ Short symbol")
        
        # Check if has image URI (projects with art are more likely legit)
        if token.image_uri:
            score += 5
            flags.append("🖼️ Has image")
        
        # Determine tier
        if score >= 30:
            return "🟢", "PROMISING", flags
        elif score >= 10:
            return "🔵", "WORTH WATCHING", flags
        elif score >= -5:
            return "🟡", "NEUTRAL", flags
        elif score >= -15:
            return "🟠", "RISKY", flags
        else:
            return "🔴", "LIKELY RUG", flags
    
    async def _send_creation_alert(self, token: BondingToken):
        """Send Discord alert for new token creation with interactive embed"""
        if not self.discord_webhook_url:
            return
        
        try:
            platform_emoji = "🎰" if token.platform == BondingPlatform.PUMP_FUN else "🚀"
            platform_name = token.platform.value
            
            # Get instant trust assessment
            trust_emoji, trust_label, trust_flags = self._get_instant_trust_assessment(token)
            
            # Build links
            if token.platform == BondingPlatform.PUMP_FUN:
                platform_link = f"https://pump.fun/{token.mint}"
            else:
                platform_link = f"https://raydium.io/launchpad/?mint={token.mint}"
            dex_link = f"https://dexscreener.com/solana/{token.mint}"
            
            # Build Discord embed for richer formatting
            embed = {
                "title": f"{platform_emoji} NEW TOKEN: {token.symbol}",
                "description": (
                    f"**{token.name}**\n\n"
                    f"⚡ **INSTANT ALERT - Caught at CREATION!**\n"
                    f"_No trades yet - you're first!_"
                ),
                "color": 0x00FF00 if "PROMISING" in trust_label else (0xFFAA00 if "WATCH" in trust_label else 0xFF6600),
                "fields": [
                    {
                        "name": f"{trust_emoji} INSTANT ASSESSMENT",
                        "value": f"**{trust_label}**\n{' | '.join(trust_flags[:3])}",
                        "inline": False
                    },
                    {
                        "name": "📈 Progress",
                        "value": f"{token.progress_pct:.1f}%",
                        "inline": True
                    },
                    {
                        "name": "💰 Mcap",
                        "value": f"{token.market_cap_sol:.2f} SOL",
                        "inline": True
                    },
                    {
                        "name": "🔗 Platform",
                        "value": f"[{platform_name}]({platform_link})",
                        "inline": True
                    },
                    {
                        "name": "📋 Mint Address",
                        "value": f"`{token.mint}`",
                        "inline": False
                    },
                ],
                "footer": {
                    "text": "⚡ CREATION ALERT | Reply: BUY $XX | PASS | MONITOR"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add social links if available
            social_links = []
            if token.twitter:
                social_links.append(f"[Twitter]({token.twitter})")
            if token.telegram:
                social_links.append(f"[Telegram]({token.telegram})")
            if token.website:
                social_links.append(f"[Website]({token.website})")
            
            if social_links:
                embed["fields"].append({
                    "name": "🔗 Social",
                    "value": " | ".join(social_links),
                    "inline": False
                })
            else:
                embed["fields"].append({
                    "name": "⚠️ Warning",
                    "value": "No social links found - higher risk!",
                    "inline": False
                })
            
            # Send embed message
            await self._send_discord_embed(embed)
            self.total_alerts_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending creation alert: {e}")
    
    
    async def _send_momentum_alert(self, token: BondingToken):
        """Send Discord alert when token gains momentum"""
        if not self.discord_webhook_url:
            return
        
        try:
            buy_pressure_emoji = "🟢" if token.buy_pressure > 0.6 else "🟡" if token.buy_pressure > 0.4 else "🔴"
            
            message = (
                f"🔥 **TOKEN GAINING MOMENTUM!**\n\n"
                f"**Token:** {token.symbol}\n"
                f"**Platform:** {token.platform.value}\n"
                f"**Trades:** {token.total_trades}\n"
                f"**Volume:** {token.total_volume_sol:.2f} SOL\n"
                f"**Buy/Sell:** {token.buy_count}/{token.sell_count} {buy_pressure_emoji}\n"
                f"**Progress:** {token.progress_pct:.1f}%\n\n"
                f"🔗 https://pump.fun/{token.mint}"
            )
            
            await self._send_discord_message(message)
            self.total_alerts_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending momentum alert: {e}")
    
    async def _send_graduation_alert(self, migration: MigrationEvent):
        """Send Discord alert when token graduates"""
        if not self.discord_webhook_url:
            return
        
        try:
            message = (
                f"🎓 **TOKEN GRADUATED!**\n\n"
                f"**Token:** {migration.symbol}\n"
                f"**Platform:** {migration.platform.value}\n"
                f"**Status:** Now tradeable on Raydium!\n"
                f"**Time:** Just now\n\n"
                f"📊 [DexScreener](https://dexscreener.com/solana/{migration.mint})\n"
                f"💱 [Raydium](https://raydium.io/swap/?inputMint={migration.mint})\n\n"
                f"⚠️ _Token has completed bonding curve - now has DEX liquidity_"
            )
            
            await self._send_discord_message(message)
            self.total_alerts_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending graduation alert: {e}")
    
    async def _send_discord_message(self, content: str):
        """Send message to Discord webhook"""
        if not self.discord_webhook_url:
            return
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    self.discord_webhook_url,
                    json={"content": content}
                )
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
    
    async def _send_discord_embed(self, embed: dict):
        """Send embed message to Discord webhook"""
        if not self.discord_webhook_url:
            return
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    self.discord_webhook_url,
                    json={"embeds": [embed]}
                )
                if response.status_code not in [200, 204]:
                    logger.warning(f"Discord webhook returned {response.status_code}")
        except Exception as e:
            logger.error(f"Discord embed error: {e}")
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def _save_state(self):
        """Save current state to disk"""
        try:
            state = {
                "tokens": {
                    mint: {
                        "mint": t.mint,
                        "symbol": t.symbol,
                        "name": t.name,
                        "platform": t.platform.value,
                        "created_at": t.created_at.isoformat(),
                        "stage": t.stage.value,
                        "progress_pct": t.progress_pct,
                        "total_trades": t.total_trades,
                        "total_volume_sol": t.total_volume_sol,
                        "buy_count": t.buy_count,
                        "sell_count": t.sell_count,
                        "alerted": t.alerted,
                    }
                    for mint, t in self.tokens.items()
                },
                "seen_mints": list(self.seen_mints),
                "total_tokens_seen": self.total_tokens_seen,
                "total_migrations": self.total_migrations,
                "total_alerts_sent": self.total_alerts_sent,
                "last_save": datetime.now().isoformat(),
            }
            
            with open(self.TOKENS_FILE, "w") as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _load_state(self):
        """Load state from disk"""
        try:
            if not self.TOKENS_FILE.exists():
                return
            
            with open(self.TOKENS_FILE, "r") as f:
                state = json.load(f)
            
            # Load seen mints
            self.seen_mints = set(state.get("seen_mints", []))
            self.total_tokens_seen = state.get("total_tokens_seen", 0)
            self.total_migrations = state.get("total_migrations", 0)
            self.total_alerts_sent = state.get("total_alerts_sent", 0)
            
            # Load tokens (only recent ones - last 24h)
            cutoff = datetime.now() - timedelta(hours=24)
            tokens_data = state.get("tokens", {})
            
            for mint, t_data in tokens_data.items():
                try:
                    created_at = datetime.fromisoformat(t_data.get("created_at", datetime.now().isoformat()))
                    if created_at < cutoff:
                        continue  # Skip old tokens
                    
                    platform = BondingPlatform(t_data.get("platform", "pump.fun"))
                    stage = TokenStage(t_data.get("stage", "created"))
                    
                    token = BondingToken(
                        mint=mint,
                        symbol=t_data.get("symbol", "???"),
                        name=t_data.get("name", "Unknown"),
                        platform=platform,
                        created_at=created_at,
                        stage=stage,
                        progress_pct=t_data.get("progress_pct", 0),
                        total_trades=t_data.get("total_trades", 0),
                        total_volume_sol=t_data.get("total_volume_sol", 0),
                        buy_count=t_data.get("buy_count", 0),
                        sell_count=t_data.get("sell_count", 0),
                        alerted=t_data.get("alerted", False),
                    )
                    self.tokens[mint] = token
                    
                except Exception as e:
                    logger.debug(f"Error loading token {mint}: {e}")
                    continue
            
            logger.info(f"Loaded {len(self.tokens)} tokens from disk")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Remove tokens older than 24 hours
                cutoff = datetime.now() - timedelta(hours=24)
                old_mints = [
                    mint for mint, t in self.tokens.items()
                    if t.created_at < cutoff
                ]
                
                for mint in old_mints:
                    del self.tokens[mint]
                
                if old_mints:
                    logger.info(f"Cleaned up {len(old_mints)} old tokens")
                
                self._save_state()
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


# Singleton instance
_monitor_instance: Optional[BondingCurveMonitor] = None


def get_bonding_curve_monitor(**kwargs) -> BondingCurveMonitor:
    """Get or create Bonding Curve Monitor singleton"""
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = BondingCurveMonitor(**kwargs)
    
    return _monitor_instance


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🎰 BONDING CURVE MONITOR - TEST RUN")
    print("=" * 60)
    
    async def test_run():
        monitor = get_bonding_curve_monitor(
            enable_pump_fun=True,
            enable_launchlab=True,
            alert_on_creation=True,
            alert_on_graduation=True,
            min_trades_to_alert=3,
            min_volume_sol_to_alert=0.5,
        )
        
        # Run for a bit
        print("\nMonitoring for new tokens... Press Ctrl+C to stop\n")
        
        try:
            await monitor.start()
        except KeyboardInterrupt:
            print("\n⛔ Stopping...")
            monitor.stop()
        
        # Print stats
        stats = monitor.get_stats()
        print(f"\n📊 Final Stats:")
        print(f"   Tokens seen: {stats['total_tokens_seen']}")
        print(f"   Migrations: {stats['total_migrations']}")
        print(f"   Alerts sent: {stats['total_alerts']}")
    
    asyncio.run(test_run())

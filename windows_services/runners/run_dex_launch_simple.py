"""
DEX Launch Monitor - SIMPLE VERSION
No threading, no timeouts - straightforward execution with async support

Now includes optional Bonding Curve Monitor integration for catching
pump.fun and LaunchLab tokens at launch (before DexScreener).

Environment Variables:
    DEX_ENABLE_BONDING_MONITOR: Enable bonding curve monitoring (default: true)
    DEX_SCAN_INTERVAL: Seconds between DexScreener scans (default: 30)
    DEX_LENIENT_MODE: Allow tokens with mint/freeze authority (default: true)
    DEX_DISCOVERY_MODE: aggressive, balanced, conservative (default: aggressive)
"""

import sys
import os
import time
import asyncio
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file (CRITICAL for Discord alerts, API keys, etc.)
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Add venv site-packages (handle both Windows and Linux)
if sys.platform == "win32":
    venv_site_packages = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
else:
    # Linux/Unix
    import glob
    venv_pattern = PROJECT_ROOT / "venv" / "lib" / "python3*" / "site-packages"
    venv_matches = glob.glob(str(venv_pattern))
    venv_site_packages = Path(venv_matches[0]) if venv_matches else None
    
if venv_site_packages and venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

# Setup logging with PST timezone
from loguru import logger
import pytz

# Configure PST timezone for logs
pst_tz = pytz.timezone('America/Los_Angeles')

def pst_time(record):
    """Convert log time to PST"""
    from datetime import datetime
    record["extra"]["pst_time"] = datetime.now(pst_tz).strftime("%Y-%m-%d %H:%M:%S")
    return record

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

logger.remove()
# Add stderr handler (so we see logs in terminal) - PST time
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{extra[pst_time]}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
    filter=pst_time
)
# Add file handler with immediate flush - PST time
logger.add(
    str(log_dir / "dex_launch_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{extra[pst_time]} PST | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False,
    filter=pst_time
)

print("=" * 70, flush=True)
print("🚀 DEX LAUNCH - SIMPLE RUNNER", flush=True)
print("=" * 70, flush=True)
logger.info("=" * 70)
logger.info("🚀 DEX LAUNCH - SIMPLE RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")
logger.info(f"✓ User: {os.getenv('USERNAME', 'UNKNOWN')}")

os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress verbose HTTP error output
import logging
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger.info("")
logger.info("Starting imports...")
print("Starting imports...", flush=True)

try:
    import_start = time.time()
    
    print("  → Importing launch_announcement_monitor...", flush=True)
    logger.info("  → Importing launch_announcement_monitor...")
    from services.launch_announcement_monitor import get_announcement_monitor
    print("  ✓ launch_announcement_monitor imported", flush=True)
    
    print("  → Importing dex_launch_hunter...", flush=True)
    logger.info("  → Importing dex_launch_hunter...")
    from services.dex_launch_hunter import get_dex_launch_hunter
    print("  ✓ dex_launch_hunter imported", flush=True)
    
    print("  → Importing alert_system...", flush=True)
    logger.info("  → Importing alert_system...")
    from services.alert_system import get_alert_system
    print("  ✓ alert_system imported", flush=True)
    
    # Optional: Bonding curve monitor for early detection
    enable_bonding = os.getenv("DEX_ENABLE_BONDING_MONITOR", "true").lower() == "true"
    bonding_monitor = None
    if enable_bonding:
        try:
            print("  → Importing bonding_curve_monitor...", flush=True)
            logger.info("  → Importing bonding_curve_monitor...")
            from services.bonding_curve_monitor import get_bonding_curve_monitor
            print("  ✓ bonding_curve_monitor imported", flush=True)
        except ImportError as e:
            print(f"  ⚠️ bonding_curve_monitor not available: {e}", flush=True)
            logger.warning(f"Bonding curve monitor not available: {e}")
            enable_bonding = False
    
    logger.info(f"✓ All imports complete in {time.time() - import_start:.1f}s")
    print(f"✓ All imports complete in {time.time() - import_start:.1f}s", flush=True)
    
    print("Initializing services...", flush=True)
    monitor = get_announcement_monitor(scan_interval=300)
    print("  ✓ Announcement monitor created", flush=True)
    
    # Import config to customize settings
    from models.dex_models import HunterConfig
    
    # Read settings from environment variables (can be set in .env file)
    # DEX_LENIENT_MODE: "true" or "false" - Allow tokens with mint/freeze authority
    # DEX_DISCOVERY_MODE: "aggressive", "balanced", or "conservative"
    # DEX_SCAN_INTERVAL: Seconds between scans (default 300 = 5 minutes)
    lenient_mode = os.getenv("DEX_LENIENT_MODE", "true").lower() == "true"
    discovery_mode = os.getenv("DEX_DISCOVERY_MODE", "aggressive").lower()
    min_liquidity = float(os.getenv("DEX_MIN_LIQUIDITY", "500"))
    scan_interval = int(os.getenv("DEX_SCAN_INTERVAL", "30"))  # Default 30 sec
    
    print(f"  📊 Scan interval: {scan_interval}s ({scan_interval/60:.1f} min)", flush=True)
    logger.info(f"  📊 Scan interval: {scan_interval}s ({scan_interval/60:.1f} min)")
    
    config = HunterConfig(
        lenient_solana_mode=lenient_mode,
        discovery_mode=discovery_mode,
        min_liquidity_usd=min_liquidity,
        min_composite_score=20.0,
    )
    dex_hunter = get_dex_launch_hunter(config=config)
    
    mode_str = "LENIENT" if lenient_mode else "STRICT"
    print(f"  ✓ DEX hunter created (mode={mode_str}, discovery={discovery_mode})", flush=True)
    alert_system = get_alert_system()
    print("  ✓ Alert system created", flush=True)
    
    # Initialize bonding curve monitor if enabled
    if enable_bonding:
        bonding_monitor = get_bonding_curve_monitor(
            enable_pump_fun=True,
            enable_launchlab=True,
            alert_on_creation=True,
            alert_on_graduation=True,
            min_trades_to_alert=5,
            min_volume_sol_to_alert=1.0,
        )
        print("  ✓ Bonding curve monitor created (pump.fun + LaunchLab)", flush=True)
        logger.info("✓ Bonding curve monitor initialized")
    else:
        bonding_monitor = None
        print("  ⚠️ Bonding curve monitor disabled", flush=True)
    
    logger.info("✓ Services initialized")
    print("✓ Services initialized", flush=True)
    
    # Print SERVICE READY
    service_ready_msg = f"🚀 SERVICE READY (total startup: {time.time() - import_start:.1f}s)"
    print("", flush=True)
    print("=" * 70, flush=True)
    print(service_ready_msg, flush=True)
    print("=" * 70, flush=True)
    print("", flush=True)
    logger.info("")
    logger.info("=" * 70)
    logger.info(service_ready_msg)
    logger.info("=" * 70)
    logger.info("")
    
    # Write status file for batch script verification
    try:
        status_file = PROJECT_ROOT / "logs" / ".dex_launch_ready"
        status_file.write_text(f"SERVICE READY at {time.time()}")
    except Exception as e:
        logger.warning(f"Could not write status file: {e}")
    
    async def monitor_loop():
        print("Starting monitor loop...", flush=True)
        logger.info("=" * 70)
        logger.info("Starting DEX launch monitoring (announcements + active scanning)...")
        logger.info("=" * 70)
        
        # Start announcement monitoring in background
        print("Creating background monitor task...", flush=True)
        async def start_monitor_with_timeout():
            await monitor.start_monitoring()
        
        monitor_task = asyncio.create_task(start_monitor_with_timeout())
        print("Background monitor task created", flush=True)
        
        # Start bonding curve monitor in background (if enabled)
        bonding_task = None
        if bonding_monitor:
            print("Creating bonding curve monitor task...", flush=True)
            logger.info("Starting bonding curve monitor for real-time pump.fun/LaunchLab detection...")
            
            # Set up callback to analyze bonding curve tokens with DEX hunter
            async def on_bonding_token_graduation(migration):
                """When a token graduates from bonding curve, analyze it immediately"""
                try:
                    logger.info(f"🎓 Bonding token graduated: {migration.symbol} - Analyzing...")
                    print(f"[BONDING→DEX] 🎓 {migration.symbol} graduated, analyzing...", flush=True)
                    
                    success, token = await dex_hunter._analyze_token(
                        migration.mint,
                        "solana"
                    )
                    
                    if success and token and token.composite_score >= 50:
                        # Use pair address for DexScreener URL (more reliable)
                        dex_address = token.pairs[0].pair_address if token.pairs else token.contract_address
                        alert_msg = (
                            f"🎓 **BONDING CURVE GRADUATE!**\n\n"
                            f"**Token:** {token.symbol}\n"
                            f"**Platform:** {migration.platform.value}\n"
                            f"**Score:** {token.composite_score:.1f}/100\n"
                            f"**Liquidity:** ${token.liquidity_usd:,.0f}\n"
                            f"**Risk:** {token.risk_level.value}\n\n"
                            f"⚡ _Caught at graduation - first seconds on DEX!_\n\n"
                            f"🔗 https://dexscreener.com/solana/{dex_address}\n\n"
                            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"**💬 Quick Reply Commands:**\n"
                            f"• `monitor` - Detailed analysis\n"
                            f"• `ADD $25` - Add position ($25)\n"
                            f"• `ADD $50` - Add position ($50)"
                        )
                        
                        alert_system.send_alert(
                            "BONDING_GRADUATE",
                            alert_msg,
                            priority="HIGH",
                            metadata={'symbol': token.symbol, 'score': token.composite_score, 'source': 'bonding_curve'}
                        )
                        logger.info(f"✅ Alert sent for graduated token {token.symbol}")
                        
                except Exception as e:
                    logger.error(f"Error analyzing graduated token: {e}")
            
            def on_migration_sync(migration):
                """Sync wrapper that schedules async analysis"""
                asyncio.create_task(on_bonding_token_graduation(migration))
            
            bonding_monitor.set_callbacks(
                on_migration=on_migration_sync
            )
            
            bonding_task = asyncio.create_task(bonding_monitor.start())
            print("Bonding curve monitor task created", flush=True)
            logger.info("✓ Bonding curve monitor started (pump.fun WebSocket + LaunchLab polling)")
        
        scan_counter = 0
        
        try:
            print("Entering main scan loop...", flush=True)
            while True:
                scan_counter += 1
                print(f"🔄 Scan cycle #{scan_counter} starting...", flush=True)
                logger.info(f"🔄 Scan cycle #{scan_counter} starting...")
                
                # ===== PART 1: Active DEX Scanning =====
                # Timeout configurable via DEX_SCAN_TIMEOUT (default 300s = 5 min)
                scan_timeout = int(os.getenv("DEX_SCAN_TIMEOUT", "300"))
                print(f"🔍 Running active DEX scan (max {scan_timeout}s)...", flush=True)
                logger.info(f"🔍 Running active DEX scan (timeout: {scan_timeout}s)...")
                try:
                    # Add timeout to prevent hanging
                    await asyncio.wait_for(
                        dex_hunter._scan_for_launches(),
                        timeout=scan_timeout
                    )
                    print("✓ Active DEX scan completed!", flush=True)
                    logger.info("✓ Active DEX scan completed")
                    
                    # Check for high-score tokens from active scan
                    for addr, token in dex_hunter.discovered_tokens.items():
                        if hasattr(token, '_alerted'):
                            continue  # Already alerted
                        
                        if token.composite_score >= 60:
                            # Use pair address for DexScreener URL (more reliable than token address)
                            dex_address = token.pairs[0].pair_address if token.pairs else token.contract_address
                            alert_msg = (
                                f"🚀 **NEW TOKEN DISCOVERED!**\n\n"
                                f"**Token:** {token.symbol}\n"
                                f"**Chain:** {token.chain.value}\n"
                                f"**Score:** {token.composite_score:.1f}/100\n"
                                f"**Liquidity:** ${token.liquidity_usd:,.0f}\n"
                                f"**Volume 24h:** ${token.volume_24h:,.0f}\n"
                                f"**Age:** {token.age_hours:.1f}h\n"
                                f"**Risk:** {token.risk_level.value}\n\n"
                                f"🔗 https://dexscreener.com/{token.chain.value}/{dex_address}\n\n"
                                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                f"**💬 Quick Reply Commands:**\n"
                                f"• `monitor` - Detailed analysis\n"
                                f"• `ADD $25` - Add position ($25)\n"
                                f"• `ADD $50` - Add position ($50)\n"
                                f"• `ADD $100` - Add position ($100)"
                            )
                            
                            alert_system.send_alert(
                                "DEX_DISCOVERY",
                                alert_msg,
                                priority="HIGH" if token.composite_score >= 70 else "MEDIUM",
                                metadata={'symbol': token.symbol, 'score': token.composite_score}
                            )
                            token._alerted = True
                            logger.info(f"✅ Alert sent for {token.symbol} (Score: {token.composite_score:.1f})")
                
                except asyncio.TimeoutError:
                    print(f"⚠️ DEX scan TIMEOUT after {scan_timeout}s!", flush=True)
                    logger.warning(f"⚠️ Active DEX scan timed out after {scan_timeout}s, continuing...")
                except Exception as e:
                    print(f"❌ Scan error: {e}", flush=True)
                    logger.error(f"Active scan error: {e}")
                
                # ===== PART 2: Process Announcements =====
                recent = monitor.get_recent_announcements(minutes=10)
                
                if recent:
                    logger.info(f"📢 Processing {len(recent)} announcements...")
                    
                    for announcement in recent:
                        if announcement.token_address.lower() in dex_hunter.discovered_tokens:
                            continue
                        
                        logger.info(f"🔍 Analyzing: {announcement.token_symbol} from {announcement.source}")
                        
                        try:
                            success, token = await dex_hunter.analyze_token(
                                announcement.token_address,
                                announcement.chain
                            )
                            
                            if success and token and token.composite_score >= 60:
                                # Use pair address for DexScreener URL (more reliable)
                                dex_address = token.pairs[0].pair_address if token.pairs else token.contract_address
                                alert_msg = (
                                    f"🚨 **HIGH SCORE LAUNCH!**\n\n"
                                    f"**Source:** {announcement.source}\n"
                                    f"**Token:** {token.symbol}\n"
                                    f"**Chain:** {token.chain.value}\n"
                                    f"**Score:** {token.composite_score:.1f}/100\n"
                                    f"**Liquidity:** ${token.liquidity_usd:,.0f}\n"
                                    f"**Risk:** {token.risk_level.value}\n\n"
                                    f"🔗 https://dexscreener.com/{token.chain.value}/{dex_address}\n\n"
                                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                                    f"**💬 Quick Reply Commands:**\n"
                                    f"• `monitor` - Detailed analysis\n"
                                    f"• `ADD $25` - Add position ($25)\n"
                                    f"• `ADD $50` - Add position ($50)\n"
                                    f"• `ADD $100` - Add position ($100)"
                                )
                                
                                alert_system.send_alert(
                                    "LAUNCH_DETECTED",
                                    alert_msg,
                                    priority="HIGH" if token.composite_score >= 70 else "MEDIUM",
                                    metadata={'symbol': token.symbol, 'score': token.composite_score}
                                )
                                logger.info(f"✅ Alert sent for {token.symbol} (Score: {token.composite_score:.1f})")
                        
                        except Exception as e:
                            logger.error(f"Error analyzing {announcement.token_symbol}: {e}")
                
                # ===== Show Stats =====
                stats = monitor.get_stats()
                discovered_count = len(dex_hunter.discovered_tokens)
                high_score_count = sum(1 for t in dex_hunter.discovered_tokens.values() if t.composite_score >= 60)
                
                stats_msg = f"📊 Stats: {discovered_count} discovered, {high_score_count} high-score"
                print(stats_msg, flush=True)
                logger.info(
                    f"📊 Stats: {discovered_count} tokens discovered, "
                    f"{high_score_count} high-score, "
                    f"{stats.get('total_announcements', 0)} announcements"
                )
                
                print(f"💤 Sleeping {scan_interval}s until next scan...", flush=True)
                logger.info(f"💤 Sleeping {scan_interval}s until next scan...")
                await asyncio.sleep(scan_interval)
                print("⏰ Waking up for next scan cycle...", flush=True)
                
        except Exception as e:
            logger.error(f"Monitor loop error: {e}", exc_info=True)
        finally:
            monitor.stop_monitoring()
            monitor_task.cancel()
            if bonding_task and bonding_monitor:
                bonding_monitor.stop()
                bonding_task.cancel()
    
    # Run the async loop
    print("Starting async monitor loop...", flush=True)
    asyncio.run(monitor_loop())

except KeyboardInterrupt:
    logger.info("Service stopped by user")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    sys.exit(1)

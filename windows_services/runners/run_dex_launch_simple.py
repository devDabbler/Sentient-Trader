"""
DEX Launch Monitor - SIMPLE VERSION
No threading, no timeouts - straightforward execution with async support
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

# Setup logging
from loguru import logger

log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

logger.remove()
# Add stderr handler (so we see logs in terminal)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True
)
# Add file handler with immediate flush
logger.add(
    str(log_dir / "dex_launch_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False
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
    
    logger.info(f"✓ All imports complete in {time.time() - import_start:.1f}s")
    print(f"✓ All imports complete in {time.time() - import_start:.1f}s", flush=True)
    
    print("Initializing services...", flush=True)
    monitor = get_announcement_monitor(scan_interval=300)
    print("  ✓ Announcement monitor created", flush=True)
    dex_hunter = get_dex_launch_hunter()
    print("  ✓ DEX hunter created", flush=True)
    alert_system = get_alert_system()
    print("  ✓ Alert system created", flush=True)
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
        
        scan_counter = 0
        
        try:
            print("Entering main scan loop...", flush=True)
            while True:
                scan_counter += 1
                print(f"🔄 Scan cycle #{scan_counter} starting...", flush=True)
                logger.info(f"🔄 Scan cycle #{scan_counter} starting...")
                
                # ===== PART 1: Active DEX Scanning =====
                print("🔍 Running active DEX scan (max 120s)...", flush=True)
                logger.info("🔍 Running active DEX scan...")
                try:
                    # Add timeout to prevent hanging
                    await asyncio.wait_for(
                        dex_hunter._scan_for_launches(),
                        timeout=120  # 2 minute timeout
                    )
                    print("✓ Active DEX scan completed!", flush=True)
                    logger.info("✓ Active DEX scan completed")
                    
                    # Check for high-score tokens from active scan
                    for addr, token in dex_hunter.discovered_tokens.items():
                        if hasattr(token, '_alerted'):
                            continue  # Already alerted
                        
                        if token.composite_score >= 60:
                            alert_msg = (
                                f"🚀 NEW TOKEN DISCOVERED!\n\n"
                                f"Token: {token.symbol}\n"
                                f"Chain: {token.chain.value}\n"
                                f"Score: {token.composite_score:.1f}/100\n"
                                f"Liquidity: ${token.liquidity_usd:,.0f}\n"
                                f"Volume 24h: ${token.volume_24h:,.0f}\n"
                                f"Age: {token.age_hours:.1f}h\n"
                                f"Risk: {token.risk_level.value}\n\n"
                                f"🔗 https://dexscreener.com/{token.chain.value}/{token.contract_address}"
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
                    print("⚠️ DEX scan TIMEOUT after 120s!", flush=True)
                    logger.warning("⚠️ Active DEX scan timed out after 120s, continuing...")
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
                                alert_msg = (
                                    f"🚨 HIGH SCORE LAUNCH!\n\n"
                                    f"Source: {announcement.source}\n"
                                    f"Token: {token.symbol}\n"
                                    f"Chain: {token.chain.value}\n"
                                    f"Score: {token.composite_score:.1f}/100\n"
                                    f"Liquidity: ${token.liquidity_usd:,.0f}\n"
                                    f"Risk: {token.risk_level.value}\n\n"
                                    f"🔗 https://dexscreener.com/{token.chain.value}/{token.contract_address}"
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
                
                print("💤 Sleeping 5 minutes until next scan...", flush=True)
                logger.info(f"💤 Sleeping 5 minutes until next scan...")
                await asyncio.sleep(300)  # 5 minutes
                print("⏰ Waking up for next scan cycle...", flush=True)
                
        except Exception as e:
            logger.error(f"Monitor loop error: {e}", exc_info=True)
        finally:
            monitor.stop_monitoring()
            monitor_task.cancel()
    
    # Run the async loop
    print("Starting async monitor loop...", flush=True)
    asyncio.run(monitor_loop())

except KeyboardInterrupt:
    logger.info("Service stopped by user")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    sys.exit(1)

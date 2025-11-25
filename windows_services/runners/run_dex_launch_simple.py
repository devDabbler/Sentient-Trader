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
# Add file handler
logger.add(
    str(log_dir / "dex_launch_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False,
    buffering=1
)

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
logger.info("Starting imports (be patient)...")

try:
    import_start = time.time()
    
    sys.stdout.write("DEBUG: Importing DEX launch services...\n")
    sys.stdout.flush()
    from services.launch_announcement_monitor import get_announcement_monitor
    from services.dex_launch_hunter import get_dex_launch_hunter
    from services.alert_system import get_alert_system
    
    logger.info(f"✓ Imported in {time.time() - import_start:.1f}s")
    sys.stdout.write("DEBUG: Import complete\n")
    sys.stdout.flush()
    
    logger.info("Initializing services...")
    sys.stdout.write("DEBUG: Initializing services...\n")
    sys.stdout.flush()
    monitor = get_announcement_monitor(scan_interval=300)
    sys.stdout.write("DEBUG: Announcement monitor created\n")
    sys.stdout.flush()
    dex_hunter = get_dex_launch_hunter()
    sys.stdout.write("DEBUG: DEX hunter created\n")
    sys.stdout.flush()
    alert_system = get_alert_system()
    logger.info("✓ Services initialized")
    sys.stdout.write("DEBUG: All services initialized\n")
    sys.stdout.flush()
    
    # Print SERVICE READY to both stdout AND logger
    logger.info("")
    logger.info("=" * 70)
    service_ready_msg = f"🚀 SERVICE READY (total startup: {time.time() - import_start:.1f}s)"
    logger.info(service_ready_msg)
    print(f"\n{'='*70}")
    print(service_ready_msg)
    print(f"{'='*70}\n")
    sys.stdout.flush()
    logger.info("=" * 70)
    logger.info("")
    
    # Write status file for batch script verification
    status_file = PROJECT_ROOT / "logs" / ".dex_launch_ready"
    status_file.write_text(f"SERVICE READY at {time.time()}")
    
    # Async monitoring loop
    async def monitor_loop():
        logger.info("Starting announcement monitoring...")
        
        # Start announcement monitoring in background
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        try:
            while True:
                # Get recent announcements
                recent = monitor.get_recent_announcements(minutes=10)
                
                if recent:
                    logger.info(f"📢 Processing {len(recent)} announcements...")
                    
                    for announcement in recent:
                        if announcement.token_address in dex_hunter.discovered_tokens:
                            continue
                        
                        logger.info(f"🔍 {announcement.token_symbol} from {announcement.source}")
                        
                        try:
                            success, token = await dex_hunter._analyze_token(
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
                                    f"DexScreener: https://dexscreener.com/{token.chain.value}/{token.contract_address}"
                                )
                                
                                alert_system.send_alert(
                                    "LAUNCH_DETECTED",
                                    alert_msg,
                                    priority="HIGH" if token.composite_score >= 70 else "MEDIUM"
                                )
                        
                        except Exception as e:
                            logger.error(f"Error analyzing {announcement.token_symbol}: {e}")
                
                # Show stats
                stats = monitor.get_stats()
                logger.info(
                    f"📊 {stats.get('total_announcements', 0)} total, "
                    f"{stats.get('last_30_min', 0)} in last 30min"
                )
                
                await asyncio.sleep(300)  # 5 minutes
                
        except Exception as e:
            logger.error(f"Monitor loop error: {e}", exc_info=True)
        finally:
            monitor.stop_monitoring()
            monitor_task.cancel()
    
    # Run the async loop
    asyncio.run(monitor_loop())

except KeyboardInterrupt:
    logger.info("Service stopped by user")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    sys.exit(1)

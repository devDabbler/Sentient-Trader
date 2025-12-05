"""
Crypto Breakout Monitor - SIMPLE VERSION
No threading, no timeouts - just straightforward execution
"""

import sys
import os
import time
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

PST = pytz.timezone('America/Los_Angeles')
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

logger.remove()
# Add stderr handler (so we see logs in terminal)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> PST | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True
)
# Add file handler
logger.add(
    str(log_dir / "crypto_breakout_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} PST | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False,
    buffering=1
)

logger.info("=" * 70)
logger.info("📈 CRYPTO BREAKOUT - SIMPLE RUNNER")
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
    
    sys.stdout.write("DEBUG: Importing CryptoBreakoutMonitor...\n")
    sys.stdout.flush()
    from services.crypto_breakout_monitor import CryptoBreakoutMonitor
    logger.info(f"✓ Imported in {time.time() - import_start:.1f}s")
    sys.stdout.write("DEBUG: Import complete\n")
    sys.stdout.flush()
    
    logger.info("Creating monitor instance...")
    sys.stdout.write("DEBUG: Creating monitor instance...\n")
    sys.stdout.flush()
    monitor = CryptoBreakoutMonitor()
    logger.info("✓ Monitor created")
    sys.stdout.write("DEBUG: Monitor created\n")
    sys.stdout.flush()
    
    # ============================================================
    # Load watchlist from Control Panel config (if available)
    # ============================================================
    save_analysis_results = None
    try:
        from service_config_loader import load_service_watchlist, load_discord_settings, save_analysis_results
        
        custom_watchlist = load_service_watchlist('sentient-crypto-breakout')
        if custom_watchlist:
            logger.info(f"📋 Control Panel watchlist: {len(custom_watchlist)} pairs")
            # Override monitor's watchlist
            if hasattr(monitor, 'watchlist'):
                monitor.watchlist = custom_watchlist
            elif hasattr(monitor, 'pairs'):
                monitor.pairs = custom_watchlist
            elif hasattr(monitor, 'crypto_pairs'):
                monitor.crypto_pairs = custom_watchlist
        
        # Load Discord settings
        discord_settings = load_discord_settings('sentient-crypto-breakout')
        if discord_settings.get('enabled') is False:
            logger.info("🔕 Discord alerts DISABLED via Control Panel")
            os.environ['DISCORD_ALERTS_DISABLED'] = 'true'
        else:
            min_conf = discord_settings.get('min_confidence', 70)
            logger.info(f"🔔 Discord alerts enabled (min confidence: {min_conf}%)")
            os.environ['DISCORD_MIN_CONFIDENCE'] = str(min_conf)
            
    except ImportError:
        logger.debug("service_config_loader not available - using defaults")
        save_analysis_results = None
    except Exception as e:
        logger.warning(f"Could not load Control Panel config: {e}")
        save_analysis_results = None
    
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
    status_file = PROJECT_ROOT / "logs" / ".crypto_breakout_ready"
    status_file.write_text(f"SERVICE READY at {time.time()}")
    
    # Get settings - scan_interval is in SECONDS (default 300 = 5 min)
    # Environment variable overrides: CRYPTO_SCAN_INTERVAL=60 for 1 minute
    env_interval = os.getenv('CRYPTO_SCAN_INTERVAL')
    if env_interval:
        scan_interval_seconds = int(env_interval)
        logger.info(f"Scan interval: {scan_interval_seconds}s ({scan_interval_seconds/60:.1f} min) (from env CRYPTO_SCAN_INTERVAL)")
    else:
        scan_interval_seconds = getattr(monitor, 'scan_interval', 300)  # seconds
        logger.info(f"Scan interval: {scan_interval_seconds}s ({scan_interval_seconds/60:.1f} min) (default)")
    scan_interval_minutes = scan_interval_seconds / 60
    
    # Run monitoring loop - CryptoBreakoutMonitor has start() method
    if hasattr(monitor, 'start'):
        logger.info("Using monitor.start() method - this runs the full monitoring loop")
        # Note: monitor.start() runs its own loop, so we can't intercept results easily
        # The monitor itself should save results if needed
        monitor.start()
    else:
        # Fallback: manual loop calling internal scan method
        logger.info("Using manual scan loop")
        scan_count = 0
        all_results = []
        
        while True:
            try:
                scan_count += 1
                logger.info(f"Starting scan #{scan_count}...")
                
                # CryptoBreakoutMonitor has _scan_and_alert() method
                if hasattr(monitor, '_scan_and_alert'):
                    results = monitor._scan_and_alert()
                    if results:
                        all_results.extend(results)
                        # Keep last 50 results
                        all_results = all_results[-50:]
                        
                        # Save to control panel
                        if save_analysis_results:
                            try:
                                save_analysis_results('sentient-crypto-breakout', all_results)
                                logger.debug(f"Saved {len(results)} results to control panel")
                            except Exception as e:
                                logger.debug(f"Could not save results: {e}")
                    
                    logger.info(f"Scan #{scan_count} complete - alerts sent if breakouts found")
                elif hasattr(monitor, 'scan_opportunities'):
                    opportunities = monitor.scan_opportunities()
                    logger.info(f"Found {len(opportunities) if opportunities else 0} opportunities")
                elif hasattr(monitor, 'scan'):
                    monitor.scan()
                else:
                    logger.warning("No scanning method found!")
                
                logger.info(f"Waiting {scan_interval_minutes:.1f} minutes until next scan...")
                logger.info("")
                time.sleep(scan_interval_seconds)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Scan error: {e}", exc_info=True)
                time.sleep(60)

except KeyboardInterrupt:
    logger.info("Service stopped by user")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    sys.exit(1)

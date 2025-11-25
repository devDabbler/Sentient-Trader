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
    str(log_dir / "crypto_breakout_service.log"),
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
    
    # Get settings
    scan_interval = getattr(monitor, 'scan_interval_minutes', 15)
    logger.info(f"Scan interval: {scan_interval} minutes")
    
    # Run monitoring loop
    if hasattr(monitor, 'run_continuous'):
        logger.info("Using run_continuous() method")
        monitor.run_continuous()
    else:
        logger.info("Using manual scan loop")
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                logger.info(f"Starting scan #{scan_count}...")
                
                if hasattr(monitor, 'scan_opportunities'):
                    opportunities = monitor.scan_opportunities()
                    logger.info(f"Found {len(opportunities) if opportunities else 0} opportunities")
                elif hasattr(monitor, 'scan'):
                    monitor.scan()
                
                logger.info(f"Scan complete. Waiting {scan_interval} minutes...")
                logger.info("")
                time.sleep(scan_interval * 60)
                
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

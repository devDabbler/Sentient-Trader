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
logger.add(
    str(log_dir / "crypto_breakout_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)

logger.info("=" * 70)
logger.info("📈 CRYPTO BREAKOUT - SIMPLE RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")

os.environ['PYTHONUNBUFFERED'] = '1'

logger.info("")
logger.info("Starting imports (be patient)...")

try:
    import_start = time.time()
    
    from services.crypto_breakout_monitor import CryptoBreakoutMonitor
    logger.info(f"✓ Imported in {time.time() - import_start:.1f}s")
    
    logger.info("Creating monitor instance...")
    monitor = CryptoBreakoutMonitor()
    logger.info("✓ Monitor created")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 SERVICE READY")
    logger.info("=" * 70)
    logger.info("")
    
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

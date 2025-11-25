"""
Stock Monitor Service Runner - ULTRA MINIMAL VERSION
Uses standard import with disabled optimizations
"""
import sys
import os
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
from loguru import logger
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", colorize=True)
logger.add(str(log_dir / "stock_monitor_ultra_minimal.log"), rotation="50 MB", retention="30 days", level="INFO")

logger.info("=" * 70)
logger.info("📊 STOCK MONITOR - ULTRA MINIMAL RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Project root: {PROJECT_ROOT}")
logger.info("")

# Set environment
os.environ['PYTHONUNBUFFERED'] = '1'

logger.info("Starting service...")
start_time = time.time()

try:
    # Standard import (fastest, most reliable)
    logger.info("Importing stock_informational_monitor...")
    import_start = time.time()
    
    from services.stock_informational_monitor import get_stock_informational_monitor
    
    logger.info(f"✓ Import completed in {time.time() - import_start:.1f}s")
    
    # Create monitor instance
    logger.info("Creating monitor instance...")
    instance_start = time.time()
    monitor = get_stock_informational_monitor()
    logger.info(f"✓ Instance created in {time.time() - instance_start:.1f}s")
    logger.info(f"✓ Watchlist: {len(monitor.watchlist)} symbols")
    
    scan_interval = getattr(monitor, 'scan_interval_minutes', 30) or getattr(monitor, 'scan_interval', 30)
    logger.info(f"✓ Scan interval: {scan_interval} minutes")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"🚀 SERVICE READY (startup: {time.time() - start_time:.1f}s)")
    logger.info("=" * 70)
    logger.info("")
    
    # Start monitoring
    logger.info("Starting monitoring...")
    if hasattr(monitor, 'run_continuous'):
        monitor.run_continuous()
    else:
        logger.warning("No run_continuous method - entering manual loop")
        while True:
            try:
                logger.info("Running scan...")
                if hasattr(monitor, 'scan_and_alert'):
                    monitor.scan_and_alert()
                time.sleep(scan_interval * 60)
            except KeyboardInterrupt:
                break

except KeyboardInterrupt:
    logger.info("Service stopped by user")
    sys.exit(0)

except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    logger.error("Service failed to start")
    sys.exit(1)

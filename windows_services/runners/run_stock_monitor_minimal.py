"""
Stock Monitor Service Runner - MINIMAL VERSION
Bypasses problematic import chain by importing only what's needed
"""
import sys
import os
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Add venv
if sys.platform == "win32":
    venv_site_packages = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
else:
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
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", colorize=True)
logger.add(str(log_dir / "stock_monitor_minimal.log"), rotation="50 MB", retention="30 days", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}", backtrace=True, diagnose=True)

logger.info("=" * 70)
logger.info("📊 STOCK MONITOR - MINIMAL RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Project root: {PROJECT_ROOT}")
logger.info("")

# Set environment
os.environ['PYTHONUNBUFFERED'] = '1'

logger.info("Starting service...")
start_time = time.time()

try:
    # Import with timeout protection
    logger.info("Importing stock_informational_monitor...")
    import_start = time.time()
    
    # Import directly - avoid services.__init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stock_informational_monitor",
        PROJECT_ROOT / "services" / "stock_informational_monitor.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load stock_informational_monitor module")
    
    stock_monitor_module = importlib.util.module_from_spec(spec)
    sys.modules["stock_informational_monitor"] = stock_monitor_module
    spec.loader.exec_module(stock_monitor_module)
    
    get_stock_informational_monitor = stock_monitor_module.get_stock_informational_monitor
    
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
    
    # Start monitoring loop
    if hasattr(monitor, 'run_continuous'):
        logger.info("Using monitor.run_continuous() method")
        monitor.run_continuous()
    elif hasattr(monitor, 'run_continuous_async'):
        logger.info("Using monitor.run_continuous_async() method")
        import asyncio
        asyncio.run(monitor.run_continuous_async())
    else:
        # Manual scan loop
        logger.info("Using manual scan loop")
        scan_count = 0
        
        while True:
            try:
                scan_count += 1
                logger.info(f"Starting scan #{scan_count}...")
                scan_start = time.time()
                
                if hasattr(monitor, 'scan'):
                    monitor.scan()
                elif hasattr(monitor, 'scan_and_alert'):
                    monitor.scan_and_alert()
                else:
                    logger.warning("No scan method found")
                
                scan_duration = time.time() - scan_start
                logger.info(f"Scan #{scan_count} complete ({scan_duration:.1f}s)")
                logger.info(f"Next scan in {scan_interval} minutes...")
                logger.info("")
                
                time.sleep(scan_interval * 60)
                
            except KeyboardInterrupt:
                logger.info("Scan interrupted by user")
                raise
            except Exception as e:
                logger.error(f"Scan error: {e}", exc_info=True)
                logger.info("Waiting 60s before retry...")
                time.sleep(60)

except KeyboardInterrupt:
    logger.info("Service stopped by user")
    sys.exit(0)

except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    logger.error("Service failed to start")
    sys.exit(1)

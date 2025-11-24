"""
Runner script for Stock Monitor
Used by NSSM to run the service
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add venv site-packages to path (critical for service execution)
venv_site_packages = project_root / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

# Setup logging first
from loguru import logger

log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

logger.remove()
logger.add(
    str(log_dir / "stock_monitor_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)

logger.info("=" * 70)
logger.info("🔍 STOCK INFORMATIONAL MONITOR SERVICE STARTED")
logger.info("=" * 70)

# Log environment for debugging
import os
logger.info(f"Current user: {os.getenv('USERNAME', 'UNKNOWN')}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH', 'NOT SET')}")
logger.info(f"PATH (first 200 chars): {os.getenv('PATH', '')[:200]}")
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Sys.path: {sys.path[:3]}")

try:
    logger.info("Starting stock monitor service...")
    logger.info("Note: Imports will happen lazily to avoid service startup hangs")
    
    # Import in main thread, but do it here after logging is set up
    logger.info("Importing stock_informational_monitor...")
    
    # Use delayed import - this avoids blocking during Windows Service initialization
    import importlib
    import sys
    
    # Ensure paths are set
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Import with explicit error handling
    try:
        stock_monitor_module = importlib.import_module('services.stock_informational_monitor')
        get_stock_informational_monitor = stock_monitor_module.get_stock_informational_monitor
        logger.info("✓ Import successful")
    except Exception as e:
        logger.error(f"❌ Failed to import: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Initializing stock monitor...")
    
    # Get monitor instance from config
    monitor = get_stock_informational_monitor()
    
    logger.info(f"Monitor initialized with {len(monitor.watchlist)} symbols")
    
    # Get scan interval from monitor attributes
    scan_interval = getattr(monitor, 'scan_interval_minutes', 
                           getattr(monitor, 'scan_interval', 30))
    logger.info(f"Scan interval: {scan_interval} minutes")
    
    # Run continuous monitoring
    if hasattr(monitor, 'run_continuous_async'):
        logger.info("Starting async continuous monitoring...")
        asyncio.run(monitor.run_continuous_async())
    elif hasattr(monitor, 'run_continuous'):
        logger.info("Starting continuous monitoring...")
        monitor.run_continuous()
    else:
        logger.error("Monitor does not have run_continuous method!")
        sys.exit(1)

except Exception as e:
    logger.error(f"Fatal error in service: {e}", exc_info=True)
    sys.exit(1)

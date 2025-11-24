"""
Stock Monitor Service Runner - V5 with Lazy Import Fix
Fixes Task Scheduler hang by ensuring llm_helper uses lazy imports
"""

import sys
import os
import time
from pathlib import Path

# CRITICAL: Set working directory FIRST
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)
os.environ['PWD'] = str(PROJECT_ROOT)

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))

# Add venv site-packages
venv_site_packages = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

# Setup logging
from loguru import logger

log_dir = PROJECT_ROOT / "logs"
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
logger.info("📊 STOCK MONITOR SERVICE V5 - LAZY IMPORT FIX")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Project root: {PROJECT_ROOT}")
logger.info(f"✓ User: {os.getenv('USERNAME', 'UNKNOWN')}")
logger.info(f"✓ Python: {sys.executable}")

# Set environment for stability
os.environ['PYTHONUNBUFFERED'] = '1'


def main():
    """Main service with error recovery"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            logger.info(f"Initialization attempt {retry_count + 1}/{max_retries}")
            
            # Import with explicit path verification
            logger.info("Importing stock_informational_monitor...")
            logger.info(f"Current sys.path: {sys.path[:3]}")
            
            from services.stock_informational_monitor import get_stock_informational_monitor
            
            logger.info("✓ Import successful!")
            
            # Initialize monitor
            logger.info("Creating monitor instance...")
            monitor = get_stock_informational_monitor()
            logger.info(f"✓ Monitor initialized with {len(monitor.watchlist)} symbols")
            
            # Get scan interval
            scan_interval = getattr(monitor, 'scan_interval_minutes', 
                                   getattr(monitor, 'scan_interval', 30))
            logger.info(f"Scan interval: {scan_interval} minutes")
            
            # Start monitoring
            logger.info("=" * 70)
            logger.info("🚀 SERVICE READY - STARTING MONITOR")
            logger.info("=" * 70)
            
            if hasattr(monitor, 'run_continuous'):
                logger.info("Using run_continuous method")
                monitor.run_continuous()
            elif hasattr(monitor, 'run_continuous_async'):
                logger.info("Using run_continuous_async method")
                import asyncio
                asyncio.run(monitor.run_continuous_async())
            else:
                # Manual loop
                logger.info("Using manual scan loop")
                while True:
                    try:
                        # Ensure working directory stays correct
                        os.chdir(PROJECT_ROOT)
                        
                        logger.info("Starting scan...")
                        if hasattr(monitor, 'scan'):
                            monitor.scan()
                        elif hasattr(monitor, 'scan_and_alert'):
                            monitor.scan_and_alert()
                        else:
                            logger.warning("No scan method found!")
                            
                        logger.info(f"Scan complete. Waiting {scan_interval} minutes...")
                        time.sleep(scan_interval * 60)
                        
                    except Exception as e:
                        logger.error(f"Scan error: {e}", exc_info=True)
                        time.sleep(60)
            
            # If we get here, service stopped normally
            break
            
        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
            retry_count += 1
            if retry_count < max_retries:
                wait_time = retry_count * 10
                logger.info(f"Restarting in {wait_time}s...")
                time.sleep(wait_time)
    
    if retry_count >= max_retries:
        logger.error("Max retries exceeded - service stopping")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

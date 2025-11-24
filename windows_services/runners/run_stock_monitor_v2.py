"""
Stock Monitor Service Runner - V2 with Timeout Protection
Handles Task Scheduler environment issues with timeout-protected imports
"""

import sys
import os
import time
import signal
import threading
from pathlib import Path
from queue import Queue, Empty

# Add project root to path FIRST
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add venv site-packages
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
logger.info("🔍 STOCK MONITOR SERVICE V2 - TIMEOUT PROTECTED")
logger.info("=" * 70)

# Set aggressive timeouts to prevent hangs
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HTTP_TIMEOUT'] = '5'
os.environ['REQUESTS_TIMEOUT'] = '5'
os.environ['OPENROUTER_TIMEOUT'] = '5'

# Patch socket with timeout BEFORE any imports that might use network
import socket
original_socket = socket.socket
def socket_with_timeout(*args, **kwargs):
    sock = original_socket(*args, **kwargs)
    sock.settimeout(5.0)  # 5 second timeout on all sockets
    return sock
socket.socket = socket_with_timeout

logger.info(f"User: {os.getenv('USERNAME', 'UNKNOWN')}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python: {sys.executable}")
logger.info("Socket timeout patch applied")


def import_with_timeout(module_name, timeout=30):
    """
    Import a module with timeout protection
    Returns: (success: bool, result: module or error)
    """
    result_queue = Queue()
    
    def do_import():
        try:
            logger.info(f"  Starting import: {module_name}...")
            if module_name == "stock_informational_monitor":
                from services.stock_informational_monitor import get_stock_informational_monitor
                result_queue.put(("success", get_stock_informational_monitor))
            else:
                raise ValueError(f"Unknown module: {module_name}")
        except Exception as e:
            logger.error(f"  Import failed: {e}", exc_info=True)
            result_queue.put(("error", e))
    
    # Run import in separate thread
    thread = threading.Thread(target=do_import, daemon=True)
    thread.start()
    
    # Wait for completion or timeout
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.error(f"⏱️ TIMEOUT: {module_name} import exceeded {timeout}s")
        return False, f"Import timeout after {timeout}s"
    
    try:
        status, result = result_queue.get_nowait()
        if status == "success":
            logger.info(f"  ✓ {module_name} imported successfully")
            return True, result
        else:
            logger.error(f"  ✗ {module_name} import error: {result}")
            return False, result
    except Empty:
        logger.error(f"  ✗ {module_name} import - no result")
        return False, "No result from import thread"


def main():
    """Main service loop with error recovery"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            logger.info(f"Initialization attempt {retry_count + 1}/{max_retries}")
            
            # Import with timeout
            success, result = import_with_timeout("stock_informational_monitor", timeout=45)
            
            if not success:
                logger.error(f"Failed to import: {result}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 10
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            
            # Get the function
            get_stock_informational_monitor = result
            
            # Verify it's callable before using
            if not callable(get_stock_informational_monitor):
                logger.error(f"Import failed - result is not callable: {get_stock_informational_monitor}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 10
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                continue
            
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
                monitor.run_continuous()
            elif hasattr(monitor, 'run_continuous_async'):
                import asyncio
                asyncio.run(monitor.run_continuous_async())
            else:
                # Manual loop
                logger.info("Using manual scan loop")
                while True:
                    try:
                        logger.info("Scanning...")
                        if hasattr(monitor, 'scan'):
                            monitor.scan()
                        elif hasattr(monitor, 'scan_and_alert'):
                            monitor.scan_and_alert()
                        logger.info(f"Waiting {scan_interval} minutes...")
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
                logger.info(f"Restarting in 30s...")
                time.sleep(30)
    
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

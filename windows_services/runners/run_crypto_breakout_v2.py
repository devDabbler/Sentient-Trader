"""
Crypto Breakout Monitor Service Runner - V2 with Timeout Protection
Handles Task Scheduler environment issues with timeout-protected imports
"""

import sys
import os
import time
import threading
from pathlib import Path
from queue import Queue, Empty

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add venv site-packages
venv_site_packages = project_root / "venv" / "Lib" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

# Setup logging
from loguru import logger

log_dir = project_root / "logs"
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
logger.info("📈 CRYPTO BREAKOUT MONITOR V2 - TIMEOUT PROTECTED")
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
    sock.settimeout(5.0)
    return sock
socket.socket = socket_with_timeout

logger.info(f"User: {os.getenv('USERNAME', 'UNKNOWN')}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info("Socket timeout patch applied")


def import_with_timeout(module_name, timeout=30):
    """Import with timeout protection"""
    result_queue = Queue()
    
    def do_import():
        try:
            logger.info(f"  Importing {module_name}...")
            if module_name == "crypto_breakout_monitor":
                from services.crypto_breakout_monitor import CryptoBreakoutMonitor
                result_queue.put(("success", CryptoBreakoutMonitor))
            else:
                raise ValueError(f"Unknown module: {module_name}")
        except Exception as e:
            logger.error(f"  Import error: {e}", exc_info=True)
            result_queue.put(("error", e))
    
    thread = threading.Thread(target=do_import, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.error(f"⏱️ TIMEOUT: {module_name} import exceeded {timeout}s")
        return False, None
    
    try:
        status, result = result_queue.get_nowait()
        if status == "success":
            logger.info(f"  ✓ {module_name}")
            return True, result
        else:
            return False, None
    except Empty:
        return False, None


def main():
    """Main service with error recovery"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            logger.info(f"Initialization attempt {retry_count + 1}/{max_retries}")
            
            # Import with timeout
            success, CryptoBreakoutMonitor = import_with_timeout("crypto_breakout_monitor", 45)
            
            if not success or not callable(CryptoBreakoutMonitor):
                logger.error("Failed to import CryptoBreakoutMonitor")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying in {retry_count * 10}s...")
                    time.sleep(retry_count * 10)
                continue
            
            # Initialize monitor
            logger.info("Creating monitor instance...")
            monitor = CryptoBreakoutMonitor(
                scan_interval_seconds=300,  # 5 minutes
                min_score=70.0,
                min_confidence='HIGH',
                use_ai=True,
                use_watchlist=True,
                alert_cooldown_minutes=60,
                auto_add_to_watchlist=True
            )
            
            logger.info("✓ Monitor initialized")
            logger.info("Scan interval: 5 minutes")
            logger.info("Min score: 70.0")
            logger.info("AI enabled: True")
            logger.info("=" * 70)
            logger.info("🚀 SERVICE READY")
            logger.info("=" * 70)
            
            # Start monitoring
            if hasattr(monitor, 'run_continuous'):
                monitor.run_continuous()
            else:
                # Manual loop
                logger.info("Using manual scan loop")
                while True:
                    try:
                        logger.info("Starting scan cycle...")
                        
                        if hasattr(monitor, 'scan_and_alert'):
                            monitor.scan_and_alert()
                        elif hasattr(monitor, 'scan'):
                            opportunities = monitor.scan()
                            logger.info(f"Found {len(opportunities)} opportunities")
                        
                        logger.info("Waiting 5 minutes...")
                        time.sleep(300)
                        
                    except Exception as e:
                        logger.error(f"Scan error: {e}", exc_info=True)
                        time.sleep(60)
            
            break
            
        except Exception as e:
            logger.error(f"Service error: {e}", exc_info=True)
            retry_count += 1
            if retry_count < max_retries:
                logger.info("Restarting in 30s...")
                time.sleep(30)
    
    logger.error("Max retries exceeded - stopping")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

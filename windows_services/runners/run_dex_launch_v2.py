"""
DEX Launch Monitor Service Runner - V2 with Timeout Protection
Handles Task Scheduler environment issues with timeout-protected imports
"""

import sys
import os
import time
import asyncio
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
    str(log_dir / "dex_launch_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)

logger.info("=" * 70)
logger.info("🚀 DEX LAUNCH MONITOR V2 - TIMEOUT PROTECTED")
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


def import_with_timeout(module_path, timeout=30):
    """Import with timeout protection"""
    result_queue = Queue()
    
    def do_import():
        try:
            logger.info(f"  Importing {module_path}...")
            if module_path == "launch_announcement_monitor":
                from services.launch_announcement_monitor import get_announcement_monitor
                result_queue.put(("success", get_announcement_monitor))
            elif module_path == "dex_launch_hunter":
                from services.dex_launch_hunter import get_dex_launch_hunter
                result_queue.put(("success", get_dex_launch_hunter))
            elif module_path == "alert_system":
                from services.alert_system import get_alert_system
                result_queue.put(("success", get_alert_system))
            else:
                raise ValueError(f"Unknown module: {module_path}")
        except Exception as e:
            logger.error(f"  Import error: {e}", exc_info=True)
            result_queue.put(("error", e))
    
    thread = threading.Thread(target=do_import, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.error(f"⏱️ TIMEOUT: {module_path} import exceeded {timeout}s")
        return False, None
    
    try:
        status, result = result_queue.get_nowait()
        if status == "success":
            logger.info(f"  ✓ {module_path}")
            return True, result
        else:
            return False, None
    except Empty:
        return False, None


async def monitor_loop(monitor, dex_hunter, alert_system):
    """Main monitoring loop"""
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
                    
                    logger.info(
                        f"🔍 {announcement.token_symbol} from {announcement.source}"
                    )
                    
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


def main():
    """Main service with error recovery"""
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            logger.info(f"Initialization attempt {retry_count + 1}/{max_retries}")
            
            # Import all services with timeout
            success1, get_announcement_monitor = import_with_timeout("launch_announcement_monitor", 45)
            success2, get_dex_launch_hunter = import_with_timeout("dex_launch_hunter", 45)
            success3, get_alert_system = import_with_timeout("alert_system", 30)
            
            if not (success1 and success2 and success3):
                logger.error("Failed to import one or more services")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying in {retry_count * 10}s...")
                    time.sleep(retry_count * 10)
                continue
            
            # Initialize services
            logger.info("Initializing services...")
            
            # Verify all imports succeeded and are callable before using
            if not (success1 and success2 and success3 and 
                   callable(get_announcement_monitor) and 
                   callable(get_dex_launch_hunter) and 
                   callable(get_alert_system)):
                logger.error("One or more imports failed - functions not callable")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying in {retry_count * 10}s...")
                    time.sleep(retry_count * 10)
                continue
            
            monitor = get_announcement_monitor(scan_interval=300)
            dex_hunter = get_dex_launch_hunter()
            alert_system = get_alert_system()
            
            logger.info("✓ All services initialized")
            logger.info("=" * 70)
            logger.info("🚀 SERVICE READY")
            logger.info("=" * 70)
            
            # Run monitoring loop
            asyncio.run(monitor_loop(monitor, dex_hunter, alert_system))
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

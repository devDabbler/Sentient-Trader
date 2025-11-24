"""
Runner script for DEX Launch Monitor
Used by NSSM to run the service
"""

import sys
import os
import time
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
    str(log_dir / "dex_launch_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)

logger.info("=" * 70)
logger.info("🚀 DEX LAUNCH MONITOR SERVICE STARTING")
logger.info("=" * 70)

# Log Python path for debugging
import os
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path[:3]}")

try:
    logger.info("Importing services...")
    import threading
    import queue
    
    def import_with_timeout(module_path, func_name, timeout=30):
        """Import with timeout to detect hangs"""
        result_queue = queue.Queue()
        
        def do_import():
            try:
                logger.info(f"Starting import of {module_path}...")
                if module_path == "launch_announcement_monitor":
                    from services.launch_announcement_monitor import get_announcement_monitor
                    result_queue.put(("success", get_announcement_monitor))
                elif module_path == "dex_launch_hunter":
                    from services.dex_launch_hunter import get_dex_launch_hunter
                    result_queue.put(("success", get_dex_launch_hunter))
                elif module_path == "alert_system":
                    from services.alert_system import get_alert_system
                    result_queue.put(("success", get_alert_system))
            except Exception as e:
                result_queue.put(("error", e))
        
        thread = threading.Thread(target=do_import, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            logger.error(f"⏱️ Import of {module_path} timed out after {timeout}s!")
            return None
        
        try:
            status, result = result_queue.get_nowait()
            if status == "success":
                logger.info(f"✓ {module_path} imported successfully")
                return result
            else:
                logger.error(f"❌ {module_path} import failed: {result}")
                return None
        except queue.Empty:
            logger.error(f"❌ {module_path} import failed - no result")
            return None
    
    get_announcement_monitor_func = import_with_timeout("launch_announcement_monitor", "get_announcement_monitor")
    get_dex_launch_hunter_func = import_with_timeout("dex_launch_hunter", "get_dex_launch_hunter")
    get_alert_system_func = import_with_timeout("alert_system", "get_alert_system")
    
    if not get_announcement_monitor_func or not get_dex_launch_hunter_func or not get_alert_system_func:
        logger.error("Failed to import one or more services")
        sys.exit(1)
    
    logger.info("All imports successful")
    
    async def monitor_loop():
        """Main monitoring loop"""
        logger.info("Initializing services...")
        
        # Safe to call - we verified not None above
        monitor = get_announcement_monitor_func(scan_interval=300)  # type: ignore # 5 min
        dex_hunter = get_dex_launch_hunter_func()  # type: ignore
        alert_system = get_alert_system_func()  # type: ignore
        
        logger.info("Services initialized successfully")
        logger.info("Starting announcement monitoring...")
        
        # Start announcement monitoring in background
        monitor_task = asyncio.create_task(monitor.start_monitoring())
        
        try:
            while True:
                # Get announcements from last 10 minutes
                recent = monitor.get_recent_announcements(minutes=10)
                
                if recent:
                    logger.info(f"📢 Processing {len(recent)} recent announcements...")
                    
                    for announcement in recent:
                        if announcement.token_address in dex_hunter.discovered_tokens:
                            continue
                        
                        logger.info(
                            f"🔍 Analyzing: {announcement.token_symbol} "
                            f"from {announcement.source}"
                        )
                        
                        try:
                            success, token = await dex_hunter._analyze_token(
                                announcement.token_address,
                                announcement.chain
                            )
                            
                            if success and token:
                                logger.info(
                                    f"✅ {token.symbol} - Score: {token.composite_score:.1f}, "
                                    f"Risk: {token.risk_level.value}"
                                )
                                
                                if token.composite_score >= 60:
                                    alert_msg = (
                                        f"🚨 HIGH SCORE LAUNCH DETECTED!\n\n"
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
                    f"📊 Stats: {stats['total_announcements']} total, "
                    f"{stats['last_30_min']} in last 30min"
                )
                
                await asyncio.sleep(300)  # 5 minutes
                
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}", exc_info=True)
        finally:
            monitor.stop_monitoring()
            monitor_task.cancel()
    
    # Run the async loop
    asyncio.run(monitor_loop())

except Exception as e:
    logger.error(f"Fatal error: {e}", exc_info=True)
    sys.exit(1)

"""
DEX Launch Monitor - SIMPLE VERSION
No threading, no timeouts - straightforward execution with async support
"""

import sys
import os
import time
import asyncio
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
    str(log_dir / "dex_launch_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)

logger.info("=" * 70)
logger.info("🚀 DEX LAUNCH - SIMPLE RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")

os.environ['PYTHONUNBUFFERED'] = '1'

logger.info("")
logger.info("Starting imports (be patient)...")

try:
    import_start = time.time()
    
    from services.launch_announcement_monitor import get_announcement_monitor
    from services.dex_launch_hunter import get_dex_launch_hunter
    from services.alert_system import get_alert_system
    
    logger.info(f"✓ Imported in {time.time() - import_start:.1f}s")
    
    logger.info("Initializing services...")
    monitor = get_announcement_monitor(scan_interval=300)
    dex_hunter = get_dex_launch_hunter()
    alert_system = get_alert_system()
    logger.info("✓ Services initialized")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 SERVICE READY")
    logger.info("=" * 70)
    logger.info("")
    
    # Async monitoring loop
    async def monitor_loop():
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
                        
                        logger.info(f"🔍 {announcement.token_symbol} from {announcement.source}")
                        
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
    
    # Run the async loop
    asyncio.run(monitor_loop())

except KeyboardInterrupt:
    logger.info("Service stopped by user")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    sys.exit(1)

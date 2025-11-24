"""
Runner script for Crypto Breakout Monitor
Used by NSSM to run the service
"""

import sys
import time
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
    str(log_dir / "crypto_breakout_service.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True
)

logger.info("=" * 70)
logger.info("📈 CRYPTO BREAKOUT MONITOR SERVICE STARTED")
logger.info("=" * 70)

try:
    from services.crypto_breakout_monitor import CryptoBreakoutMonitor
    
    # Initialize monitor with sensible defaults
    monitor = CryptoBreakoutMonitor(
        scan_interval_seconds=300,  # 5 minutes
        min_score=70.0,
        min_confidence='HIGH',
        use_ai=True,
        use_watchlist=True,
        alert_cooldown_minutes=60,
        auto_add_to_watchlist=True
    )
    
    logger.info("Crypto Breakout Monitor initialized")
    logger.info("Scan interval: 5 minutes")
    logger.info("Min score: 70.0")
    logger.info("AI enabled: True")
    
    # Check if monitor has run_continuous method
    if hasattr(monitor, 'run_continuous'):
        logger.info("Starting continuous monitoring...")
        monitor.run_continuous()
    else:
        # No run_continuous, implement our own loop
        logger.info("Implementing manual scan loop")
        
        while True:
            try:
                logger.info("Starting scan cycle...")
                
                # Scan for opportunities
                if hasattr(monitor, 'scan_and_alert'):
                    monitor.scan_and_alert()
                elif hasattr(monitor, 'scan'):
                    opportunities = monitor.scan()
                    logger.info(f"Found {len(opportunities)} opportunities")
                
                # Wait for scan interval
                logger.info("Scan complete, waiting for next cycle...")
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in scan cycle: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 min on error

except Exception as e:
    logger.error(f"Fatal error in service: {e}", exc_info=True)
    sys.exit(1)

"""
Crypto AI Position Manager - SIMPLE VERSION
Monitors open positions and uses AI to make intelligent exit decisions
Handles 24/7 crypto position management with multi-agent AI analysis
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

# Open log file in unbuffered/line-buffered mode
log_file_path = str(log_dir / "crypto_ai_position_manager_service.log")

logger.remove()
# Add stderr handler (so we see logs in terminal)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True
)
# Add file handler with immediate flush
# Using a custom sink function to ensure immediate writes
def file_sink(message):
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message)
        f.flush()

logger.add(
    file_sink,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False
)

logger.info("=" * 70)
logger.info("🤖 CRYPTO AI POSITION MANAGER - SIMPLE RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")
logger.info(f"✓ User: {os.getenv('USERNAME', 'UNKNOWN')}")

os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress verbose HTTP error output
import logging
logging.getLogger('yfinance').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger.info("")
logger.info("Starting imports (be patient)...")

try:
    import_start = time.time()
    
    logger.info("DEBUG: Step 1 - Importing AICryptoPositionManager...")
    sys.stdout.write("DEBUG: Importing AICryptoPositionManager...\n")
    sys.stdout.flush()
    from services.ai_crypto_position_manager import AICryptoPositionManager
    logger.info("DEBUG: Step 2 - AICryptoPositionManager imported")
    
    logger.info("DEBUG: Step 3 - Importing KrakenClient...")
    from clients.kraken_client import KrakenClient
    logger.info("DEBUG: Step 4 - KrakenClient imported")
    logger.info(f"✓ Imported in {time.time() - import_start:.1f}s")
    sys.stdout.write("DEBUG: Import complete\n")
    sys.stdout.flush()
    
    logger.info("DEBUG: Step 5 - Loading environment...")
    sys.stdout.write("DEBUG: Initializing Kraken client...\n")
    sys.stdout.flush()
    
    # Load API credentials from environment
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')
    
    logger.info(f"DEBUG: Step 6 - API key present: {bool(api_key)}, API secret present: {bool(api_secret)}")
    
    if not api_key or not api_secret:
        logger.error("❌ KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in .env")
        sys.exit(1)
    
    logger.info(f"DEBUG: Step 6 - API key present: {bool(api_key)}, API secret present: {bool(api_secret)}")
    
    if not api_key or not api_secret:
        logger.error("❌ KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in .env")
        sys.exit(1)
    
    logger.info("DEBUG: Step 7 - Creating Kraken client...")
    kraken_client = KrakenClient(api_key=api_key, api_secret=api_secret)
    logger.info("DEBUG: Step 8 - Kraken client created")
    logger.info("✓ Kraken client initialized")
    sys.stdout.write("DEBUG: Kraken client initialized\n")
    sys.stdout.flush()
    
    logger.info("DEBUG: Step 9 - Creating AI Position Manager instance...")
    sys.stdout.write("DEBUG: Creating AI Position Manager...\n")
    sys.stdout.flush()
    manager = AICryptoPositionManager(
        kraken_client=kraken_client,
        check_interval_seconds=60,         # Check positions every minute
        enable_ai_decisions=True,          # Enable AI-powered exit decisions
        enable_trailing_stops=True,        # Enable trailing stops
        enable_breakeven_moves=True,       # Enable break-even protection
        enable_partial_exits=True,         # Enable partial profit taking
        min_ai_confidence=70.0,            # Only act on high-confidence decisions (0-100)
        require_manual_approval=False      # PRODUCTION: Set to False for auto-execution (use with caution!)
    )
    logger.info("DEBUG: Step 10 - AI Position Manager created successfully")
    logger.info("✓ AI Position Manager created")
    sys.stdout.write("DEBUG: AI Position Manager created\n")
    sys.stdout.flush()
    
    # Print SERVICE READY to both stdout AND logger
    logger.info("")
    logger.info("=" * 70)
    service_ready_msg = f"🚀 SERVICE READY - AI ACTIVELY MONITORING POSITIONS (startup: {time.time() - import_start:.1f}s)"
    logger.info(service_ready_msg)
    print(f"\n{'='*70}")
    print(service_ready_msg)
    print(f"{'='*70}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    logger.info("=" * 70)
    logger.info("")
    
    # Write status file for batch script verification
    status_file = PROJECT_ROOT / "logs" / ".crypto_ai_position_manager_ready"
    status_file.write_text(f"SERVICE READY at {time.time()}")
    
    logger.info(f"✓ Check interval: {manager.check_interval_seconds}s")
    logger.info(f"✓ AI Decisions: {manager.enable_ai_decisions}")
    logger.info(f"✓ Trailing Stops: {manager.enable_trailing_stops}")
    logger.info(f"✓ Manual Approval Required: {manager.require_manual_approval}")
    logger.info(f"✓ Min confidence: {manager.min_ai_confidence}%")
    logger.info("")
    logger.warning("⚠️  AUTO-EXECUTION MODE - AI will execute trades without approval!")
    logger.info("")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Run monitoring loop - AICryptoPositionManager has start_monitoring_loop() and monitor_positions()
    if hasattr(manager, 'start_monitoring_loop'):
        logger.info("Using manager.start_monitoring_loop() - this runs continuous monitoring")
        sys.stdout.flush()
        manager.start_monitoring_loop()
        
        # Wait a moment and verify thread started
        time.sleep(2)
        if manager.is_running and manager.thread and manager.thread.is_alive():
            logger.info("✅ Background monitoring thread is ALIVE and running")
        else:
            logger.error("❌ Background thread failed to start!")
            logger.error(f"   is_running: {manager.is_running}")
            logger.error(f"   thread: {manager.thread}")
            if manager.thread:
                logger.error(f"   thread.is_alive(): {manager.thread.is_alive()}")
        
        sys.stdout.flush()
        
        # Keep main thread alive while background monitoring runs
        heartbeat_count = 0
        try:
            while True:
                time.sleep(60)
                heartbeat_count += 1
                # Main thread heartbeat every 5 minutes
                if heartbeat_count % 5 == 0:
                    logger.info(f"💓 Main thread heartbeat #{heartbeat_count} - Service uptime: {heartbeat_count} min")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            raise
    else:
        # Fallback: manual monitoring loop
        logger.info("Using manual monitoring loop")
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logger.info(f"Starting position check #{cycle_count}...")
                
                if hasattr(manager, 'monitor_positions'):
                    decisions = manager.monitor_positions()
                    if decisions:
                        logger.info(f"Check #{cycle_count} complete - {len(decisions)} AI decisions made")
                    else:
                        logger.info(f"Check #{cycle_count} complete - no positions to manage")
                elif hasattr(manager, 'check_and_manage_positions'):
                    manager.check_and_manage_positions()
                else:
                    logger.warning("No monitoring method found - service may not be functional")
                
                logger.info(f"Waiting {manager.check_interval_seconds}s until next check...")
                logger.info("")
                time.sleep(manager.check_interval_seconds)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Monitoring cycle error: {e}", exc_info=True)
                logger.info("Waiting 60s before retry...")
                time.sleep(60)

except KeyboardInterrupt:
    logger.info("Service stopped by user")
except Exception as e:
    logger.error(f"FATAL ERROR: {e}", exc_info=True)
    sys.exit(1)

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
    
    logger.info("Importing AICryptoPositionManager...")
    from services.ai_crypto_position_manager import AICryptoPositionManager
    
    logger.info("Importing KrakenClient...")
    from clients.kraken_client import KrakenClient
    logger.info(f"✓ Imports complete in {time.time() - import_start:.1f}s")
    
    # Load API credentials from environment
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('KRAKEN_API_KEY')
    api_secret = os.getenv('KRAKEN_API_SECRET')
    
    if not api_key or not api_secret:
        logger.error("❌ KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in .env")
        sys.exit(1)
    
    logger.info("✓ API credentials loaded")
    
    logger.info("Initializing Kraken client...")
    kraken_client = KrakenClient(api_key=api_key, api_secret=api_secret)
    logger.info("✓ Kraken client initialized")
    
    logger.info("Creating AI Position Manager...")
    sys.stdout.flush()
    manager = AICryptoPositionManager(
        kraken_client=kraken_client,
        check_interval_seconds=60,         # Check positions every minute
        enable_ai_decisions=True,          # Enable AI-powered exit decisions
        enable_trailing_stops=True,        # Enable trailing stops
        enable_breakeven_moves=True,       # Enable break-even protection
        enable_partial_exits=True,         # Enable partial profit taking
        min_ai_confidence=70.0,            # Only act on high-confidence decisions (0-100)
        require_manual_approval=True       # SAFETY: Require Discord approval before executing trades
    )
    logger.info("✓ AI Position Manager created")
    sys.stdout.flush()
    
    # ============================================================
    # SYNC WITH KRAKEN - Check actual positions on startup
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 SYNCING WITH KRAKEN - Checking your current positions...")
    logger.info("=" * 70)
    sys.stdout.flush()
    
    try:
        # First, show all Kraken balances/positions
        kraken_positions = kraken_client.get_open_positions(calculate_real_cost=True, min_value=1.0)
        
        if kraken_positions:
            logger.info(f"")
            logger.info(f"📈 Found {len(kraken_positions)} position(s) on Kraken:")
            logger.info("-" * 50)
            
            total_value = 0
            total_pnl = 0
            for pos in kraken_positions:
                value = pos.volume * pos.current_price
                pnl = pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else 0
                pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price * 100) if pos.entry_price > 0 else 0
                
                total_value += value
                total_pnl += pnl
                
                pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                logger.info(f"  {pnl_emoji} {pos.pair}: {pos.volume:.6f} @ ${pos.current_price:,.2f}")
                logger.info(f"      Entry: ${pos.entry_price:,.2f} | Value: ${value:,.2f} | P&L: {pnl_pct:+.2f}%")
            
            logger.info("-" * 50)
            logger.info(f"💰 Total Portfolio Value: ${total_value:,.2f}")
            logger.info("")
        else:
            logger.info("")
            logger.info("📭 No positions found on Kraken (or all positions < $1)")
            logger.info("   The AI will monitor once you open trades via the UI")
            logger.info("")
        
        # Now sync positions to the AI manager
        sync_result = manager.sync_with_kraken()
        logger.info(f"🔄 Sync Result: {sync_result['added']} added, {sync_result['removed']} removed, {sync_result['kept']} kept")
        
    except Exception as e:
        logger.error(f"❌ Error syncing with Kraken: {e}")
        logger.info("   Will continue with empty position list")
    
    sys.stdout.flush()
    
    # ============================================================
    # SERVICE READY
    # ============================================================
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
    logger.info(f"✓ Discord Approval: {'✅ ENABLED' if manager.discord_approval_manager else '❌ NOT CONFIGURED'}")
    logger.info(f"✓ Min confidence: {manager.min_ai_confidence}%")
    logger.info(f"✓ Positions being monitored: {len(manager.positions)}")
    logger.info("")
    
    if manager.require_manual_approval:
        if manager.discord_approval_manager:
            logger.info("🔐 DISCORD APPROVAL MODE - AI recommendations require your approval via Discord")
            logger.info("   Reply APPROVE or REJECT in Discord when prompted")
        else:
            logger.warning("⚠️ MANUAL APPROVAL MODE - AI recommendations require approval in the app")
    else:
        logger.warning("⚠️  AUTO-EXECUTION MODE - AI will execute trades without approval!")
    
    logger.info("")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Run monitoring loop - AICryptoPositionManager has start_monitoring_loop() and monitor_positions()
    if hasattr(manager, 'start_monitoring_loop'):
        print("🔄 Starting continuous monitoring loop...", flush=True)
        logger.info("🔄 Starting continuous monitoring loop...")
        sys.stdout.flush()
        manager.start_monitoring_loop()
        
        # Wait a moment and verify thread started
        time.sleep(2)
        if manager.is_running and manager.thread and manager.thread.is_alive():
            print("✅ Background monitoring thread is ALIVE and running", flush=True)
            logger.info("✅ Background monitoring thread is ALIVE and running")
        else:
            print("❌ Background thread failed to start!", flush=True)
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
                # Main thread heartbeat every minute
                print(f"💓 Heartbeat #{heartbeat_count} - Service running for {heartbeat_count} min", flush=True)
                logger.info(f"💓 Main thread heartbeat #{heartbeat_count} - Service uptime: {heartbeat_count} min")
                
                # Also print position count
                active_count = len([p for p in manager.positions.values() if hasattr(p, 'status') and p.status == 'ACTIVE'])
                print(f"   📊 Active positions: {active_count}/{len(manager.positions)}", flush=True)
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

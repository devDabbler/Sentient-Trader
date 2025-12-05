"""
Stock AI Position Manager - SIMPLE VERSION
Monitors open stock positions and uses AI to make intelligent exit decisions
Syncs with Tradier/IBKR broker for both paper and live trading
"""

import sys
import os
import time
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

# Setup logging with PST timezone
from loguru import logger
import pytz

PST = pytz.timezone('America/Los_Angeles')
log_dir = PROJECT_ROOT / "logs"
log_dir.mkdir(exist_ok=True)

# Open log file in unbuffered/line-buffered mode
log_file_path = str(log_dir / "stock_ai_trader_service.log")

logger.remove()
# Add stderr handler (so we see logs in terminal)
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> PST | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True
)
# Add file handler with immediate flush
def file_sink(message):
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(message)
        f.flush()

logger.add(
    file_sink,
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} PST | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False
)

logger.info("=" * 70)
logger.info("🤖 STOCK AI POSITION MANAGER - SIMPLE RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")
logger.info(f"✓ User: {os.getenv('USERNAME', os.getenv('USER', 'UNKNOWN'))}")

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
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("Importing AIStockPositionManager...")
    from services.ai_stock_position_manager import AIStockPositionManager, _create_broker_adapter
    logger.info(f"✓ Imports complete in {time.time() - import_start:.1f}s")
    
    # ============================================================
    # DETERMINE TRADING MODE (Paper vs Live)
    # ============================================================
    paper_mode = os.getenv('STOCK_PAPER_MODE', 'true').lower() == 'true'
    broker_type = os.getenv('BROKER_TYPE', 'TRADIER').upper()
    
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"📊 TRADING MODE: {'🔵 PAPER TRADING' if paper_mode else '🔴 LIVE TRADING'}")
    logger.info(f"📊 BROKER: {broker_type}")
    logger.info("=" * 70)
    
    if not paper_mode:
        logger.warning("⚠️  LIVE TRADING MODE - Real money at risk!")
        logger.warning("    Press Ctrl+C within 5 seconds to abort...")
        time.sleep(5)
        logger.info("    Continuing with live trading...")
    
    # ============================================================
    # CREATE BROKER ADAPTER
    # ============================================================
    logger.info("")
    logger.info("Creating broker adapter...")
    broker_adapter = _create_broker_adapter()
    
    if broker_adapter:
        logger.info(f"✓ Broker adapter created: {type(broker_adapter).__name__}")
    else:
        logger.warning("⚠️ No broker adapter available - will use yfinance for prices only")
        logger.warning("   Set BROKER_TYPE and credentials in .env to enable broker integration")
    
    # ============================================================
    # CREATE AI POSITION MANAGER
    # ============================================================
    logger.info("")
    
    # Check interval from environment (seconds, default 60)
    check_interval = int(os.getenv('STOCK_POSITION_CHECK_INTERVAL', '60'))
    logger.info(f"Position check interval: {check_interval}s (env: STOCK_POSITION_CHECK_INTERVAL)")
    
    logger.info("Creating AI Stock Position Manager...")
    sys.stdout.flush()
    
    manager = AIStockPositionManager(
        broker_adapter=broker_adapter,
        check_interval_seconds=check_interval,  # Check positions (env configurable)
        enable_trailing_stops=True,         # Enable trailing stops
        enable_breakeven_moves=True,        # Enable break-even protection
        min_confidence=70.0,                # Only act on high-confidence decisions (0-100)
        require_manual_approval=True,       # SAFETY: Require Discord approval before executing trades
        paper_mode=paper_mode,
        max_positions=10
    )
    logger.info("✓ AI Stock Position Manager created")
    sys.stdout.flush()
    
    # ============================================================
    # SYNC WITH BROKER - Check actual positions on startup
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 SYNCING WITH BROKER - Checking your current positions...")
    logger.info("=" * 70)
    sys.stdout.flush()
    
    if broker_adapter:
        try:
            # First, show all broker positions
            success, broker_positions = broker_adapter.get_positions()
            
            if success and broker_positions:
                logger.info(f"")
                logger.info(f"📈 Found {len(broker_positions)} position(s) in broker:")
                logger.info("-" * 50)
                
                total_value = 0
                total_pnl = 0
                for pos in broker_positions:
                    symbol = pos.get('symbol', 'UNKNOWN')
                    quantity = int(pos.get('quantity', 0))
                    cost_basis = float(pos.get('cost_basis', 0))
                    current_price = float(pos.get('current_price', 0))
                    
                    value = abs(quantity) * current_price if current_price > 0 else 0
                    entry_price = cost_basis / abs(quantity) if quantity != 0 else current_price
                    pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    total_value += value
                    total_pnl += (pnl_pct / 100) * cost_basis
                    
                    pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
                    logger.info(f"  {pnl_emoji} {symbol}: {quantity} shares @ ${current_price:.2f}")
                    logger.info(f"      Entry: ${entry_price:.2f} | Value: ${value:,.2f} | P&L: {pnl_pct:+.2f}%")
                
                logger.info("-" * 50)
                logger.info(f"💰 Total Portfolio Value: ${total_value:,.2f}")
                logger.info("")
            else:
                logger.info("")
                logger.info("📭 No positions found in broker account")
                logger.info("   The AI will monitor once you open trades via the UI or manually")
                logger.info("")
            
            # Now sync positions to the AI manager
            sync_result = manager.sync_with_broker()
            logger.info(f"🔄 Sync Result: {sync_result['added']} added, {sync_result['removed']} removed, {sync_result['kept']} kept")
            
        except Exception as e:
            logger.error(f"❌ Error syncing with broker: {e}")
            logger.info("   Will continue with saved position list")
    else:
        logger.info("📭 No broker connection - will monitor saved positions only")
    
    sys.stdout.flush()
    
    # ============================================================
    # SERVICE READY
    # ============================================================
    logger.info("")
    logger.info("=" * 70)
    service_ready_msg = f"🚀 SERVICE READY - AI ACTIVELY MONITORING STOCK POSITIONS (startup: {time.time() - import_start:.1f}s)"
    logger.info(service_ready_msg)
    print(f"\n{'='*70}")
    print(service_ready_msg)
    print(f"{'='*70}\n")
    sys.stdout.flush()
    sys.stderr.flush()
    logger.info("=" * 70)
    logger.info("")
    
    # Write status file for batch script verification
    status_file = PROJECT_ROOT / "logs" / ".stock_ai_trader_ready"
    status_file.write_text(f"SERVICE READY at {time.time()}")
    
    logger.info(f"✓ Check interval: {manager.check_interval_seconds}s")
    logger.info(f"✓ Paper Mode: {'✅ ENABLED' if manager.paper_mode else '⚠️ LIVE TRADING'}")
    logger.info(f"✓ Trailing Stops: {manager.enable_trailing_stops}")
    logger.info(f"✓ Manual Approval Required: {manager.require_manual_approval}")
    logger.info(f"✓ Discord Approval: {'✅ ENABLED' if manager.discord_approval_manager else '❌ NOT CONFIGURED'}")
    logger.info(f"✓ Min confidence: {manager.min_confidence}%")
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
    
    # Run monitoring loop
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
                
                # Sync with broker periodically
                if cycle_count % 10 == 1:  # Every 10 cycles, sync with broker
                    logger.info("🔄 Syncing with broker...")
                    manager.sync_with_broker()
                
                # Check positions
                active_positions = manager.get_active_positions()
                if active_positions:
                    logger.info(f"Check #{cycle_count} - monitoring {len(active_positions)} positions")
                    manager._check_positions(active_positions)
                else:
                    logger.info(f"Check #{cycle_count} complete - no positions to manage")
                
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


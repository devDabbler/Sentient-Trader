"""
Bonding Curve Monitor - Service Runner

Real-time monitoring for pump.fun and LaunchLab token launches.
Catches tokens at creation, not hours later from DexScreener.

Usage:
    python windows_services/runners/run_bonding_curve_monitor.py

Environment Variables:
    BONDING_ENABLE_PUMPFUN: Enable pump.fun WebSocket (default: true)
    BONDING_ENABLE_LAUNCHLAB: Enable LaunchLab polling (default: true)
    BONDING_ALERT_ON_CREATION: Alert on new token creation (default: true)
    BONDING_ALERT_ON_GRADUATION: Alert on token graduation (default: true)
    BONDING_MIN_TRADES: Minimum trades for momentum alert (default: 5)
    BONDING_MIN_VOLUME_SOL: Minimum volume SOL for momentum alert (default: 1.0)
    BONDING_LAUNCHLAB_INTERVAL: LaunchLab poll interval seconds (default: 30)

Author: Sentient Trader
Created: December 2025
"""

import sys
import os
import asyncio
import time
from pathlib import Path

# Track startup time
import_start = time.time()

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Add venv site-packages
if sys.platform == "win32":
    venv_site_packages = PROJECT_ROOT / "venv" / "Lib" / "site-packages"
else:
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
# Add stderr handler
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    colorize=True
)
# Add file handler
logger.add(
    str(log_dir / "bonding_curve_monitor.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False
)

print("=" * 70, flush=True)
print("🎰 BONDING CURVE MONITOR - SERVICE RUNNER", flush=True)
print("=" * 70, flush=True)
logger.info("=" * 70)
logger.info("🎰 BONDING CURVE MONITOR - SERVICE RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")

os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress verbose HTTP output
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('websockets').setLevel(logging.WARNING)

logger.info("")
logger.info("Starting imports...")


def parse_bool_env(key: str, default: bool = True) -> bool:
    """Parse boolean from environment variable"""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


async def run_monitor():
    """Run the bonding curve monitor"""
    from services.bonding_curve_monitor import get_bonding_curve_monitor
    from services.alert_system import get_alert_system
    
    # Parse configuration from environment
    enable_pumpfun = parse_bool_env("BONDING_ENABLE_PUMPFUN", True)
    enable_launchlab = parse_bool_env("BONDING_ENABLE_LAUNCHLAB", True)
    alert_on_creation = parse_bool_env("BONDING_ALERT_ON_CREATION", True)
    alert_on_graduation = parse_bool_env("BONDING_ALERT_ON_GRADUATION", True)
    min_trades = int(os.getenv("BONDING_MIN_TRADES", "5"))
    min_volume_sol = float(os.getenv("BONDING_MIN_VOLUME_SOL", "1.0"))
    launchlab_interval = int(os.getenv("BONDING_LAUNCHLAB_INTERVAL", "30"))
    
    print(f"", flush=True)
    print(f"📋 Configuration:", flush=True)
    print(f"   pump.fun WebSocket: {'✅ Enabled' if enable_pumpfun else '❌ Disabled'}", flush=True)
    print(f"   LaunchLab polling: {'✅ Enabled' if enable_launchlab else '❌ Disabled'}", flush=True)
    print(f"   Alert on creation: {'✅' if alert_on_creation else '❌'}", flush=True)
    print(f"   Alert on graduation: {'✅' if alert_on_graduation else '❌'}", flush=True)
    print(f"   Min trades to alert: {min_trades}", flush=True)
    print(f"   Min volume to alert: {min_volume_sol} SOL", flush=True)
    print(f"   LaunchLab interval: {launchlab_interval}s", flush=True)
    print(f"", flush=True)
    
    logger.info(f"Configuration loaded:")
    logger.info(f"  pump.fun: {enable_pumpfun}, LaunchLab: {enable_launchlab}")
    logger.info(f"  Alert creation: {alert_on_creation}, Alert graduation: {alert_on_graduation}")
    logger.info(f"  Min trades: {min_trades}, Min volume: {min_volume_sol} SOL")
    
    # Create monitor instance
    monitor = get_bonding_curve_monitor(
        enable_pump_fun=enable_pumpfun,
        enable_launchlab=enable_launchlab,
        alert_on_creation=alert_on_creation,
        alert_on_graduation=alert_on_graduation,
        min_trades_to_alert=min_trades,
        min_volume_sol_to_alert=min_volume_sol,
        launchlab_poll_interval=launchlab_interval,
    )
    
    # Get alert system for additional notifications
    alert_system = get_alert_system()
    
    # Set up callback to also notify via alert system
    def on_new_token(token):
        """Called when new token is detected"""
        msg = f"🎰 NEW: {token.symbol} on {token.platform.value}"
        print(f"[BONDING] {msg}", flush=True)
    
    def on_migration(migration):
        """Called when token graduates"""
        msg = f"🎓 GRADUATED: {migration.symbol} - Now on DEX!"
        print(f"[BONDING] {msg}", flush=True)
        # Send high priority alert for graduations
        alert_system.send_alert(
            "BONDING_GRADUATION",
            f"🎓 {migration.symbol} graduated from {migration.platform.value}!\n"
            f"Mint: {migration.mint[:30]}...\n"
            f"Now tradeable on Raydium/DEX",
            priority="HIGH"
        )
    
    monitor.set_callbacks(
        on_new_token=on_new_token,
        on_migration=on_migration,
    )
    
    # Print startup message
    startup_time = time.time() - import_start
    print(f"", flush=True)
    print(f"🚀 SERVICE READY (startup: {startup_time:.1f}s)", flush=True)
    print(f"", flush=True)
    print(f"Monitoring for new token launches...", flush=True)
    print(f"Press Ctrl+C to stop", flush=True)
    print("=" * 70, flush=True)
    
    logger.info(f"🚀 SERVICE READY (startup: {startup_time:.1f}s)")
    
    # Send startup notification
    alert_system.send_alert(
        "SERVICE_START",
        f"🎰 Bonding Curve Monitor started\n"
        f"pump.fun: {'✅' if enable_pumpfun else '❌'}\n"
        f"LaunchLab: {'✅' if enable_launchlab else '❌'}",
        priority="LOW"
    )
    
    # Status update loop (runs alongside monitor)
    async def status_loop():
        """Print periodic status updates"""
        while monitor.is_running:
            await asyncio.sleep(300)  # Every 5 minutes
            stats = monitor.get_stats()
            status_msg = (
                f"[BONDING] 📊 Status: "
                f"WS={'✅' if stats['websocket_connected'] else '❌'} | "
                f"Tokens={stats['total_tokens_seen']} | "
                f"Migrations={stats['total_migrations']} | "
                f"Alerts={stats['total_alerts']}"
            )
            print(status_msg, flush=True)
            logger.info(status_msg)
    
    # Run monitor and status loop concurrently
    try:
        await asyncio.gather(
            monitor.start(),
            status_loop()
        )
    except asyncio.CancelledError:
        logger.info("Tasks cancelled")
        monitor.stop()


async def main():
    """Main entry point"""
    try:
        logger.info("✓ All imports successful")
        logger.info("")
        logger.info("=" * 70)
        logger.info("Starting Bonding Curve Monitor...")
        logger.info("=" * 70)
        
        await run_monitor()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal (Ctrl+C)")
        print("\n⛔ Shutting down...", flush=True)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}", flush=True)
        raise


if __name__ == "__main__":
    # Use asyncio.run for clean async execution
    asyncio.run(main())

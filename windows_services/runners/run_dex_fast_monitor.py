"""
DEX Fast Position Monitor - SIMPLE RUNNER
Real-time 2-second monitoring for held DEX positions (meme coins / pump tokens)

This runs separately from DexLaunchHunter to provide fast price monitoring
for positions you've already bought on Phantom/DEX.

Usage:
    python windows_services/runners/run_dex_fast_monitor.py

Features:
- 2-second price monitoring loop
- Trailing stop detection (12% default)
- Hard stop loss detection (30% default)
- Pump spike detection (5%+ in 2s)
- Discord alerts for exit signals
- Profitability calculator with realistic slippage/fees
"""

import sys
import os
import asyncio
from pathlib import Path

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
    str(log_dir / "dex_fast_monitor.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False
)

print("=" * 70, flush=True)
print("🎰 DEX FAST POSITION MONITOR - RUNNER", flush=True)
print("=" * 70, flush=True)
logger.info("=" * 70)
logger.info("🎰 DEX FAST POSITION MONITOR - RUNNER")
logger.info("=" * 70)
logger.info(f"✓ Working directory: {os.getcwd()}")
logger.info(f"✓ Python: {sys.executable}")

os.environ['PYTHONUNBUFFERED'] = '1'

# Suppress verbose HTTP output
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

logger.info("")
logger.info("Starting imports...")


async def run_fast_monitor():
    """Run the fast position monitor"""
    from services.dex_fast_position_monitor import get_fast_position_monitor
    
    # Configuration
    check_interval = float(os.getenv('DEX_FAST_MONITOR_INTERVAL', '2.0'))
    # Let the monitor handle routing (Fast Monitor -> Crypto Positions -> Pump -> General)
    discord_webhook = None
    
    logger.info(f"Check interval: {check_interval}s")
    logger.info("Discord alerts: routing handled by monitor priority (Fast -> Crypto Positions -> Pump -> General)")
    
    # Get or create monitor instance
    monitor = get_fast_position_monitor(
        check_interval=check_interval,
        discord_webhook_url=discord_webhook
    )
    
    positions = monitor.get_all_positions()
    logger.info(f"Loaded {len(positions)} active positions")
    
    print("", flush=True)
    print("🎰 Fast Position Monitor starting...", flush=True)
    print(f"   Check interval: {check_interval}s", flush=True)
    print(f"   Active positions: {len(positions)}", flush=True)
    print("", flush=True)
    
    if positions:
        print("📊 Current Positions:", flush=True)
        for pos in positions:
            print(f"   • {pos.symbol}: Entry ${pos.entry_price:.8f}", flush=True)
    else:
        print("ℹ️ No positions currently being monitored", flush=True)
        print("   Add positions via CLI or service control panel", flush=True)
    
    print("", flush=True)
    print("Press Ctrl+C to stop", flush=True)
    print("=" * 70, flush=True)
    
    # Run the fast loop
    await monitor.run_fast_loop()


async def main():
    """Main entry point"""
    try:
        logger.info("✓ All imports successful")
        logger.info("")
        logger.info("=" * 70)
        logger.info("Starting Fast Position Monitor...")
        logger.info("=" * 70)
        
        await run_fast_monitor()
        
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

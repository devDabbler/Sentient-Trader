"""
Pump.fun Gambler Monitor - Service Runner

Real-time monitoring for pump.fun bonding curve tokens for small-money gambling.
Catches tokens at creation and graduation, with pump.fun-specific analysis.

Features:
- Real-time WebSocket connection to pump.fun via PumpPortal
- Bonding curve progress tracking
- Graduation alerts when tokens hit 100%
- pump.fun-specific AI analysis (not DexScreener)
- Discord alerts to dedicated #pumpfun-alerts channel
- Reply commands: ANALYZE, BUY $XX, PASS, MONITOR

Usage:
    python windows_services/runners/run_pumpfun_gambler.py

Environment Variables:
    DISCORD_WEBHOOK_PUMPFUN_ALERTS: Webhook for pump.fun gambling alerts
    DISCORD_CHANNEL_ID_PUMPFUN_ALERTS: Channel ID for reply commands
    PUMPFUN_MAX_BET: Default max bet (default: 25)
    PUMPFUN_ALERT_ON_CREATION: Alert on new token creation (default: true)
    PUMPFUN_ALERT_ON_GRADUATION: Alert on token graduation (default: true)

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
    str(log_dir / "pumpfun_gambler.log"),
    rotation="50 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    enqueue=False
)

print("=" * 70, flush=True)
print("🎰 PUMP.FUN GAMBLER MONITOR - SERVICE RUNNER", flush=True)
print("=" * 70, flush=True)
logger.info("=" * 70)
logger.info("🎰 PUMP.FUN GAMBLER MONITOR - SERVICE RUNNER")
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


async def run_pumpfun_gambler():
    """Run the pump.fun gambler monitor"""
    from services.bonding_curve_monitor import get_bonding_curve_monitor
    from services.pumpfun_analyzer import get_pumpfun_analyzer
    
    # Parse configuration from environment
    max_bet = float(os.getenv("PUMPFUN_MAX_BET", "25"))
    alert_on_creation = parse_bool_env("PUMPFUN_ALERT_ON_CREATION", True)
    alert_on_graduation = parse_bool_env("PUMPFUN_ALERT_ON_GRADUATION", True)
    
    # Gambling-specific settings: catch everything early
    min_trades = 3  # Lower threshold for gambling
    min_volume_sol = 0.5  # Lower threshold for gambling
    
    # Check Discord configuration
    webhook_url = os.getenv("DISCORD_WEBHOOK_PUMPFUN_ALERTS")
    channel_id = os.getenv("DISCORD_CHANNEL_ID_PUMPFUN_ALERTS")
    
    print(f"", flush=True)
    print(f"📋 Configuration:", flush=True)
    print(f"   Mode: 🎰 GAMBLING (small bets)", flush=True)
    print(f"   Max bet: ${max_bet}", flush=True)
    print(f"   Alert on creation: {'✅' if alert_on_creation else '❌'}", flush=True)
    print(f"   Alert on graduation: {'✅' if alert_on_graduation else '❌'}", flush=True)
    print(f"   Min trades to alert: {min_trades}", flush=True)
    print(f"   Min volume to alert: {min_volume_sol} SOL", flush=True)
    print(f"", flush=True)
    print(f"🔗 Discord:", flush=True)
    print(f"   Webhook: {'✅ Configured' if webhook_url else '❌ Not configured'}", flush=True)
    print(f"   Channel ID: {'✅ ' + channel_id if channel_id else '❌ Not configured'}", flush=True)
    
    if not webhook_url:
        print(f"", flush=True)
        print(f"⚠️ WARNING: DISCORD_WEBHOOK_PUMPFUN_ALERTS not set!", flush=True)
        print(f"   Create a channel called #pumpfun-alerts and set the webhook.", flush=True)
        print(f"   Alerts will fall back to other configured channels.", flush=True)
    
    if not channel_id:
        print(f"", flush=True)
        print(f"⚠️ WARNING: DISCORD_CHANNEL_ID_PUMPFUN_ALERTS not set!", flush=True)
        print(f"   Reply commands (ANALYZE, BUY, etc.) won't work in this channel.", flush=True)
    
    print(f"", flush=True)
    
    logger.info(f"Configuration loaded:")
    logger.info(f"  Mode: GAMBLING, Max bet: ${max_bet}")
    logger.info(f"  Alert creation: {alert_on_creation}, Alert graduation: {alert_on_graduation}")
    logger.info(f"  Discord: webhook={'✅' if webhook_url else '❌'}, channel={'✅' if channel_id else '❌'}")
    
    # Create monitor instance (pump.fun only, no LaunchLab for gambling)
    monitor = get_bonding_curve_monitor(
        enable_pump_fun=True,
        enable_launchlab=False,  # Focus on pump.fun for gambling
        alert_on_creation=alert_on_creation,
        alert_on_graduation=alert_on_graduation,
        min_trades_to_alert=min_trades,
        min_volume_sol_to_alert=min_volume_sol,
    )
    
    # Initialize pump.fun analyzer
    analyzer = get_pumpfun_analyzer(max_bet_default=max_bet)
    
    # Track tokens for auto-analysis
    analyzed_tokens = set()
    
    # Set up callback to auto-analyze promising tokens
    async def on_new_token_async(token):
        """Called when new token is detected - auto-analyze if promising"""
        msg = f"🎰 NEW: {token.symbol} on pump.fun"
        print(f"[PUMPFUN] {msg}", flush=True)
        
        # Auto-analyze after some initial activity (5+ trades)
        if token.total_trades >= 5 and token.mint not in analyzed_tokens:
            analyzed_tokens.add(token.mint)
            try:
                analysis = await analyzer.analyze_token(token.mint)
                if analysis and analysis.score >= 50:
                    # Send analysis alert for promising tokens
                    await analyzer.send_analysis_alert(analysis)
                    logger.info(f"📊 Auto-analyzed {token.symbol}: Score={analysis.score:.0f}")
            except Exception as e:
                logger.debug(f"Auto-analysis error: {e}")
    
    def on_new_token(token):
        """Sync wrapper for async callback"""
        asyncio.create_task(on_new_token_async(token))
    
    async def on_migration_async(migration):
        """Called when token graduates - this is the key gambling moment"""
        msg = f"🎓 GRADUATED: {migration.symbol} - Now on DEX!"
        print(f"[PUMPFUN] {msg}", flush=True)
        
        # Graduation is a key moment - run fresh analysis
        try:
            analysis = await analyzer.analyze_token(migration.mint)
            if analysis:
                await analyzer.send_analysis_alert(analysis)
                logger.info(f"🎓 Graduation analysis for {migration.symbol}: Score={analysis.score:.0f}")
        except Exception as e:
            logger.debug(f"Graduation analysis error: {e}")
        
        # Graduation alert is sent by analyzer.send_analysis_alert() above
        # No need for duplicate alert_system call
    
    def on_migration(migration):
        """Sync wrapper for async callback"""
        asyncio.create_task(on_migration_async(migration))
    
    monitor.set_callbacks(
        on_new_token=on_new_token,
        on_migration=on_migration,
    )
    
    # Print startup message
    startup_time = time.time() - import_start
    print(f"", flush=True)
    print(f"🚀 SERVICE READY (startup: {startup_time:.1f}s)", flush=True)
    print(f"", flush=True)
    print(f"🎰 Monitoring pump.fun for gambling opportunities...", flush=True)
    print(f"", flush=True)
    print(f"📋 Reply Commands (in #pumpfun-alerts):", flush=True)
    print(f"   ANALYZE - Full pump.fun analysis", flush=True)
    print(f"   BUY $25 - Gamble $25 on this token", flush=True)
    print(f"   PASS    - Skip this token", flush=True)
    print(f"   MONITOR - Track bonding curve progress", flush=True)
    print(f"", flush=True)
    print(f"Press Ctrl+C to stop", flush=True)
    print("=" * 70, flush=True)
    
    logger.info(f"🚀 SERVICE READY (startup: {startup_time:.1f}s)")
    
    # Send startup notification to PUMPFUN channel (not generic alert_system)
    import httpx
    pumpfun_webhook = os.getenv("DISCORD_WEBHOOK_PUMPFUN_ALERTS")
    if pumpfun_webhook:
        try:
            async def send_startup_alert():
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(pumpfun_webhook, json={
                        "content": (
                            f"🎰 **Pump.fun Gambler Monitor Started**\n\n"
                            f"Max bet: ${max_bet}\n"
                            f"Creation alerts: {'✅' if alert_on_creation else '❌'}\n"
                            f"Graduation alerts: {'✅' if alert_on_graduation else '❌'}\n\n"
                            f"_Reply to token alerts with: ANALYZE | BUY $XX | PASS_"
                        )
                    })
            asyncio.create_task(send_startup_alert())
        except Exception as e:
            logger.debug(f"Startup alert error: {e}")
    
    # Status update loop (runs alongside monitor)
    async def status_loop():
        """Print periodic status updates"""
        while monitor.is_running:
            await asyncio.sleep(300)  # Every 5 minutes
            stats = monitor.get_stats()
            status_msg = (
                f"[PUMPFUN] 📊 Status: "
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
        logger.info("Starting Pump.fun Gambler Monitor...")
        logger.info("=" * 70)
        
        await run_pumpfun_gambler()
        
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

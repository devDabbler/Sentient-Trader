"""
Start Discord Approval Bot
Standalone script to start the Discord trade approval bot
"""

import sys
from loguru import logger
from dotenv import load_dotenv
import os
import time

# ============================================================
# SINGLETON CHECK - Prevent multiple instances
# ============================================================
from utils.process_lock import ensure_single_instance

force_restart = '--force' in sys.argv or '-f' in sys.argv
process_lock = ensure_single_instance("discord_approval_bot", force=force_restart)
# ============================================================

# Load .env file FIRST
load_dotenv()
logger.info("🤖 Starting Discord Trade Approval Bot...")

# Verify token is loaded
token = os.getenv('DISCORD_BOT_TOKEN')
if not token:
    logger.error("❌ DISCORD_BOT_TOKEN not found in environment!")
    logger.error("   Make sure .env file exists and contains DISCORD_BOT_TOKEN")
    exit(1)

logger.info(f"✅ Token loaded (length: {len(token)})")

from services.discord_trade_approval import get_discord_approval_manager

# Get approval manager (this will auto-start the bot)
approval_manager = get_discord_approval_manager()

# ALSO Start the Alert Pipeline (for incoming alerts)
try:
    from services.discord_alert_pipeline import get_discord_pipeline
    alert_pipeline = get_discord_pipeline()
    if alert_pipeline:
        logger.info("🚀 Starting Discord Alert Pipeline...")
        alert_pipeline.start()
    else:
        logger.warning("⚠️ Discord Alert Pipeline could not be created")
except Exception as e:
    logger.error(f"❌ Failed to start Alert Pipeline: {e}")

if not approval_manager or not approval_manager.enabled:
    logger.error("❌ Discord approval manager not configured!")
    logger.error("   Check .env for DISCORD_BOT_TOKEN and DISCORD_CHANNEL_IDS")
    exit(1)

# Start the bot
logger.info("🚀 Starting bot...")
success = approval_manager.start()

if not success:
    logger.error("❌ Failed to start Discord bot!")
    exit(1)

# Wait for bot to be ready
logger.info("⏳ Waiting for bot to connect...")
for i in range(30):
    if approval_manager.is_running():
        logger.info("✅ Discord approval bot is RUNNING!")
        logger.info(f"   Channel ID: {approval_manager.channel_id}")
        logger.info(f"   Bot enabled: {approval_manager.enabled}")
        break
    time.sleep(1)
else:
    logger.warning("⚠️ Bot started but not ready yet (may still be connecting)")

logger.info("🔄 Bot is running in background. Keep this script running.")
logger.info("   Press Ctrl+C to stop")

try:
    # Keep alive
    while True:
        time.sleep(10)
        pending = len(approval_manager.get_pending_approvals())
        if pending > 0:
            logger.info(f"⏳ {pending} pending approval(s)")
except KeyboardInterrupt:
    logger.info("👋 Shutting down Discord bot...")
finally:
    process_lock.release()

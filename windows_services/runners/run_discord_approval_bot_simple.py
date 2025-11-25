#!/usr/bin/env python3
"""
Discord Approval Bot Service Runner
Simple, robust runner for systemd deployment

This bot listens for trade approval commands in Discord:
- APPROVE / REJECT for single trades
- APPROVE 1 / REJECT 2 for specific trades
- APPROVE ALL / REJECT ALL for bulk actions
- LIST to see pending trades
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("DEBUG: Starting Discord Approval Bot...", flush=True)

# Load environment FIRST
from dotenv import load_dotenv
load_dotenv()

# Setup logging
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level: <8} | {message}")
logger.add("logs/discord_approval_bot.log", level="DEBUG", rotation="10 MB", retention="7 days")

logger.info("🤖 Discord Approval Bot Service Starting...")

# Verify required environment variables
token = os.getenv('DISCORD_BOT_TOKEN')
channel_ids = os.getenv('DISCORD_CHANNEL_IDS') or os.getenv('DISCORD_APPROVAL_CHANNEL_ID')

if not token:
    logger.error("❌ DISCORD_BOT_TOKEN not found in environment!")
    logger.error("   Add DISCORD_BOT_TOKEN to your .env file")
    sys.exit(1)

if not channel_ids:
    logger.error("❌ DISCORD_CHANNEL_IDS not found in environment!")
    logger.error("   Add DISCORD_CHANNEL_IDS to your .env file")
    sys.exit(1)

logger.info(f"✅ Bot token loaded (length: {len(token)})")
logger.info(f"✅ Channel IDs: {channel_ids}")

# Import and start the bot
print("DEBUG: Importing DiscordTradeApprovalBot...", flush=True)

try:
    from services.discord_trade_approval import get_discord_approval_manager
    print("DEBUG: Import complete", flush=True)
except ImportError as e:
    logger.error(f"❌ Failed to import discord_trade_approval: {e}")
    sys.exit(1)

print("DEBUG: Getting approval manager...", flush=True)

# Get approval manager
approval_manager = get_discord_approval_manager()

if not approval_manager:
    logger.error("❌ Failed to create Discord approval manager!")
    sys.exit(1)

if not approval_manager.enabled:
    logger.error("❌ Discord approval manager is not enabled!")
    logger.error("   Check DISCORD_BOT_TOKEN and DISCORD_CHANNEL_IDS in .env")
    sys.exit(1)

print("DEBUG: Starting bot...", flush=True)

# Start the bot
success = approval_manager.start()

if not success:
    logger.error("❌ Failed to start Discord approval bot!")
    sys.exit(1)

# Wait for bot to connect
logger.info("⏳ Waiting for bot to connect to Discord...")
connected = False
for i in range(30):
    if approval_manager.is_running():
        connected = True
        break
    time.sleep(1)
    if i % 5 == 0:
        logger.info(f"   Still connecting... ({i}s)")

if connected:
    logger.info("=" * 60)
    logger.info("🚀 DISCORD APPROVAL BOT READY!")
    logger.info("=" * 60)
    logger.info(f"   Channel ID: {approval_manager.channel_id}")
    logger.info(f"   Commands: APPROVE, REJECT, LIST, APPROVE ALL")
    logger.info("")
else:
    logger.warning("⚠️ Bot started but connection not confirmed yet")

# Keep running and log periodic status
scan_count = 0
try:
    while True:
        time.sleep(60)  # Check every minute
        scan_count += 1
        
        pending = approval_manager.get_pending_approvals()
        pending_count = len(pending) if pending else 0
        
        if pending_count > 0:
            logger.info(f"⏳ {pending_count} pending approval(s) waiting")
            for p in pending[:3]:  # Show first 3
                logger.info(f"   • {p.get('pair', 'Unknown')} {p.get('side', '')} - {p.get('confidence', 0):.0f}% conf")
        
        # Hourly status
        if scan_count % 60 == 0:
            logger.info(f"📊 Hourly status: Bot running, {pending_count} pending approvals")

except KeyboardInterrupt:
    logger.info("👋 Received shutdown signal")
except Exception as e:
    logger.error(f"❌ Unexpected error: {e}", exc_info=True)
finally:
    logger.info("🛑 Discord Approval Bot shutting down")
    if approval_manager:
        approval_manager.stop()

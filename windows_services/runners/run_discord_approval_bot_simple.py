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

# ============================================================
# SINGLETON CHECK - Prevent multiple instances
# ============================================================
from utils.process_lock import ensure_single_instance

# Check for --force flag to kill existing instance
force_restart = '--force' in sys.argv or '-f' in sys.argv

# This will exit if another instance is running (unless force=True)
process_lock = ensure_single_instance("discord_approval_bot", force=force_restart)
# ============================================================

print("DEBUG: Starting Discord Approval Bot...", flush=True)

# Load environment FIRST - use explicit path for systemd compatibility
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging with PST timezone
from loguru import logger
import pytz

PST = pytz.timezone('America/Los_Angeles')
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} PST | {level: <8} | {message}")
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

# Debug: Check if Birdeye API key is loaded (for order flow feature)
birdeye_key = os.getenv('BIRDEYE_API_KEY')
if birdeye_key:
    logger.info(f"✅ BIRDEYE_API_KEY loaded (length: {len(birdeye_key)})")
else:
    logger.warning("⚠️ BIRDEYE_API_KEY not found - order flow features disabled")

# Import and start the bot
print("DEBUG: Importing DiscordTradeApprovalBot...", flush=True)

try:
    from services.discord_trade_approval import get_discord_approval_manager
    print("DEBUG: Import complete", flush=True)
except ImportError as e:
    logger.error(f"❌ Failed to import discord_trade_approval: {e}")
    sys.exit(1)

print("DEBUG: Getting approval manager...", flush=True)

# Define callback to execute approved trades
def execute_approved_trade(approval_id: str, approved: bool):
    """Execute or cancel trade when approved/rejected via Discord"""
    import asyncio
    
    logger.info(f"{'✅' if approved else '❌'} Trade {approval_id} {'approved' if approved else 'rejected'}")
    
    if not approved:
        logger.info(f"   Trade cancelled - no action taken")
        # Send cancellation confirmation to Discord
        _send_discord_message(f"❌ **Trade Cancelled:** {approval_id.split('_')[1] if '_' in approval_id else approval_id}")
        return
    
    try:
        # Get the pending approval details from the bot
        if approval_manager and approval_manager.bot:
            pending = approval_manager.bot.pending_approvals.get(approval_id)
            if pending:
                # Prevent duplicate execution
                if pending.executed:
                    logger.warning(f"   ⚠️ Trade {approval_id} already executed - skipping")
                    return
                
                shares = int(pending.position_size / pending.entry_price) if pending.entry_price > 0 else 0
                logger.info(f"   Executing: {pending.pair} {pending.side}")
                logger.info(f"   Shares: {shares}")
                logger.info(f"   Value: ${pending.position_size:,.2f}")
                
                if shares <= 0:
                    logger.error(f"   ❌ Invalid shares quantity: {shares}")
                    _send_discord_message(f"❌ **Order Failed:** Invalid share quantity for {pending.pair}")
                    return
                
                # Get stock position manager and use broker adapter
                from services.ai_stock_position_manager import get_ai_stock_position_manager
                stock_manager = get_ai_stock_position_manager()
                
                if stock_manager and stock_manager.broker_adapter:
                    # Place order via broker adapter
                    success, result = stock_manager.broker_adapter.place_equity_order(
                        symbol=pending.pair,
                        side=pending.side.lower(),  # 'buy' or 'sell'
                        quantity=shares,
                        order_type='market'
                    )
                    
                    if success:
                        order_id = result.get('order_id', 'N/A') if isinstance(result, dict) else str(result)
                        logger.info(f"   ✅ ORDER PLACED: {order_id}")
                        logger.info(f"   📊 {pending.pair} {pending.side} {shares} shares @ market")
                        
                        # Mark as executed and remove from pending to prevent duplicates
                        pending.executed = True
                        if approval_id in approval_manager.bot.pending_approvals:
                            del approval_manager.bot.pending_approvals[approval_id]
                            logger.info(f"   🗑️ Removed {approval_id} from pending approvals")
                        
                        # Send success confirmation to Discord
                        paper_mode = stock_manager.paper_mode
                        mode_emoji = "📝" if paper_mode else "💰"
                        _send_discord_message(
                            f"✅ **{mode_emoji} ORDER EXECUTED**\n\n"
                            f"**{pending.pair}** {pending.side}\n"
                            f"   Shares: **{shares:,}**\n"
                            f"   Value: **${pending.position_size:,.2f}**\n"
                            f"   Order ID: `{order_id}`\n"
                            f"   Type: {'Paper' if paper_mode else 'LIVE'} Trade"
                        )
                    else:
                        error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                        logger.error(f"   ❌ ORDER FAILED: {result}")
                        _send_discord_message(f"❌ **Order Failed:** {pending.pair} {pending.side}\nError: {error_msg[:100]}")
                else:
                    logger.error(f"   ❌ Stock position manager or broker adapter not available")
                    _send_discord_message(f"❌ **Order Failed:** Broker not configured")
            else:
                logger.warning(f"   ⚠️ Approval {approval_id} not found in pending")
    except Exception as e:
        logger.error(f"   ❌ Error executing trade: {e}", exc_info=True)
        _send_discord_message(f"❌ **Order Error:** {str(e)[:100]}")

def _send_discord_message(message: str):
    """Helper to send message to Discord from sync context"""
    import asyncio
    try:
        if approval_manager and approval_manager.bot and approval_manager.loop:
            async def _send():
                channel = approval_manager.bot.get_channel(approval_manager.bot.channel_id)
                if channel:
                    await channel.send(message)
            
            # Schedule the coroutine on the bot's event loop
            asyncio.run_coroutine_threadsafe(_send(), approval_manager.loop)
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")

# Get approval manager with trade execution callback
approval_manager = get_discord_approval_manager(approval_callback=execute_approved_trade)

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
            # Iterate over first 3 pending approvals (pending is a dict of {id: approval})
            for approval in list(pending.values())[:3]:
                logger.info(f"   • {approval.pair} {approval.side} - {approval.confidence:.0f}% conf")
        
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
    # Release process lock
    process_lock.release()

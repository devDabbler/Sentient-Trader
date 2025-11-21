"""
Discord Trade Approval System
Handles two-way Discord communication for trade approvals:
1. Sends entry alerts to Discord with trade details
2. Listens for user approval/rejection via Discord messages
3. Executes approved trades automatically
"""

from loguru import logger
import discord
from discord.ext import commands
import asyncio
import os
from datetime import datetime
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict
import json
import threading


@dataclass
class PendingTradeApproval:
    """Trade awaiting user approval"""
    approval_id: str
    pair: str
    side: str
    entry_price: float
    position_size: float
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float
    reasoning: str
    created_time: datetime
    expires_minutes: int = 60
    discord_message_id: Optional[str] = None
    approved: bool = False
    rejected: bool = False
    executed: bool = False
    execution_result: Optional[Dict] = None
    
    def to_dict(self):
        """Convert to dict for JSON serialization"""
        data = asdict(self)
        data['created_time'] = self.created_time.isoformat()
        return data
    
    def is_expired(self) -> bool:
        """Check if approval request has expired"""
        elapsed = (datetime.now() - self.created_time).total_seconds() / 60
        return elapsed > self.expires_minutes


class DiscordTradeApprovalBot(commands.Bot):
    """Discord bot for trade approvals with interactive buttons"""
    
    def __init__(self, token: str, channel_id: str, approval_callback: Callable):
        """
        Initialize Discord approval bot
        
        Args:
            token: Discord bot token
            channel_id: Channel ID to send approval requests
            approval_callback: Function to call when trade approved (trade_id, approved: bool)
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        
        super().__init__(command_prefix='!', intents=intents)
        
        self.token = token
        self.channel_id = int(channel_id)
        self.approval_callback = approval_callback
        self.pending_approvals: Dict[str, PendingTradeApproval] = {}
        self.is_ready = False
        
        logger.info(f"🤖 Discord Trade Approval Bot initialized")
        logger.info(f"   Channel ID: {channel_id}")
    
    async def on_ready(self):
        """Called when bot is ready"""
        self.is_ready = True
        logger.info(f'✅ Discord bot logged in as {self.user}')
        
        channel = self.get_channel(self.channel_id)
        if channel:
            logger.info(f"✅ Monitoring channel: {channel.name} (ID: {self.channel_id})")
        else:
            logger.error(f"❌ Could not find channel ID: {self.channel_id}")
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages for approvals"""
        # Ignore bot's own messages
        if message.author == self.user:
            return
        
        # Check if it's in our channel
        if message.channel.id != self.channel_id:
            return
        
        content = message.content.upper().strip()
        
        # Check for approval/rejection keywords
        if content in ['APPROVE', 'YES', 'GO', 'EXECUTE', 'BUY', 'Y']:
            await self._handle_approval(message, approve=True)
        elif content in ['REJECT', 'NO', 'CANCEL', 'SKIP', 'N']:
            await self._handle_approval(message, approve=False)
        
        # Process commands
        await self.process_commands(message)
    
    async def _handle_approval(self, message: discord.Message, approve: bool):
        """Handle user approval/rejection"""
        # Check if there's a pending approval (most recent)
        if not self.pending_approvals:
            await message.channel.send("❌ No pending trade approvals.")
            return
        
        # Get most recent pending approval
        latest_approval = None
        latest_time = None
        
        for approval_id, approval in self.pending_approvals.items():
            if not approval.approved and not approval.rejected:
                if latest_time is None or approval.created_time > latest_time:
                    latest_approval = approval
                    latest_time = approval.created_time
        
        if not latest_approval:
            await message.channel.send("❌ No pending approvals found.")
            return
        
        # Mark as approved/rejected
        if approve:
            latest_approval.approved = True
            await message.channel.send(
                f"✅ **APPROVED:** {latest_approval.pair} {latest_approval.side} trade\n"
                f"Executing trade now..."
            )
            logger.info(f"✅ User approved trade: {latest_approval.approval_id}")
        else:
            latest_approval.rejected = True
            await message.channel.send(
                f"❌ **REJECTED:** {latest_approval.pair} {latest_approval.side} trade\n"
                f"Trade cancelled."
            )
            logger.info(f"❌ User rejected trade: {latest_approval.approval_id}")
        
        # Call approval callback
        if self.approval_callback:
            try:
                self.approval_callback(latest_approval.approval_id, approve)
            except Exception as e:
                logger.error(f"Error in approval callback: {e}", exc_info=True)
    
    async def send_approval_request(
        self,
        approval_id: str,
        pair: str,
        side: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        confidence: float,
        reasoning: str,
        additional_info: Optional[str] = None
    ) -> bool:
        """
        Send trade approval request to Discord
        
        Returns:
            True if sent successfully
        """
        try:
            if not self.is_ready:
                logger.warning("Discord bot not ready, cannot send approval request")
                return False
            
            channel = self.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Cannot find Discord channel: {self.channel_id}")
                return False
            
            # Calculate trade metrics
            risk_pct = abs((entry_price - stop_loss) / entry_price) * 100
            reward_pct = abs((take_profit - entry_price) / entry_price) * 100
            rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
            
            # Build rich embed
            embed = discord.Embed(
                title=f"🚨 TRADE APPROVAL REQUIRED: {pair}",
                description=f"**{side}** position opportunity detected",
                color=discord.Color.gold(),
                timestamp=datetime.now()
            )
            
            # Add fields
            embed.add_field(
                name="💰 Entry Price",
                value=f"${entry_price:,.6f}",
                inline=True
            )
            embed.add_field(
                name="📦 Position Size",
                value=f"${position_size:,.2f}",
                inline=True
            )
            embed.add_field(
                name="📊 Strategy",
                value=strategy,
                inline=True
            )
            
            embed.add_field(
                name="🛑 Stop Loss",
                value=f"${stop_loss:,.6f} ({-risk_pct:.2f}%)",
                inline=True
            )
            embed.add_field(
                name="🎯 Take Profit",
                value=f"${take_profit:,.6f} (+{reward_pct:.2f}%)",
                inline=True
            )
            embed.add_field(
                name="⚖️ Risk:Reward",
                value=f"{rr_ratio:.2f}:1",
                inline=True
            )
            
            # Confidence with emoji
            conf_emoji = "🟢" if confidence >= 80 else "🟡" if confidence >= 60 else "🔴"
            embed.add_field(
                name=f"{conf_emoji} AI Confidence",
                value=f"{confidence:.1f}%",
                inline=True
            )
            
            embed.add_field(
                name="💡 AI Reasoning",
                value=reasoning[:1024],  # Discord limit
                inline=False
            )
            
            if additional_info:
                embed.add_field(
                    name="📋 Additional Info",
                    value=additional_info[:1024],
                    inline=False
                )
            
            # Instructions
            embed.add_field(
                name="⏰ How to Approve/Reject",
                value=(
                    "**Reply in this channel with:**\n"
                    "• `APPROVE` or `YES` → Execute trade ✅\n"
                    "• `REJECT` or `NO` → Cancel trade ❌\n\n"
                    "⏱️ *Expires in 60 minutes*"
                ),
                inline=False
            )
            
            embed.set_footer(text=f"Approval ID: {approval_id}")
            
            # Send message
            message = await channel.send(embed=embed)
            
            # Create pending approval
            approval = PendingTradeApproval(
                approval_id=approval_id,
                pair=pair,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy=strategy,
                confidence=confidence,
                reasoning=reasoning,
                created_time=datetime.now(),
                discord_message_id=str(message.id)
            )
            
            self.pending_approvals[approval_id] = approval
            
            logger.info(f"📨 Sent approval request to Discord: {approval_id}")
            logger.info(f"   {pair} {side} @ ${entry_price:,.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending approval request to Discord: {e}", exc_info=True)
            return False
    
    def get_approval_status(self, approval_id: str) -> Optional[PendingTradeApproval]:
        """Get status of pending approval"""
        return self.pending_approvals.get(approval_id)
    
    def cleanup_expired(self):
        """Remove expired approval requests"""
        expired = [
            aid for aid, approval in self.pending_approvals.items()
            if approval.is_expired() and not approval.approved and not approval.rejected
        ]
        
        for aid in expired:
            logger.info(f"⏰ Approval request expired: {aid}")
            del self.pending_approvals[aid]
        
        return len(expired)


class DiscordApprovalManager:
    """Manager for Discord trade approvals (thread-safe wrapper)"""
    
    def __init__(
        self,
        token: Optional[str] = None,
        channel_id: Optional[str] = None,
        approval_callback: Optional[Callable] = None
    ):
        """
        Initialize approval manager
        
        Args:
            token: Discord bot token (or from env DISCORD_BOT_TOKEN)
            channel_id: Channel ID for approvals (or from env DISCORD_CHANNEL_IDS)
            approval_callback: Function to call when trade approved
        """
        # Get credentials from env if not provided
        self.token = token or os.getenv('DISCORD_BOT_TOKEN')
        self.channel_id = channel_id or os.getenv('DISCORD_CHANNEL_IDS', '').split(',')[0].strip()
        
        if not self.token or not self.channel_id:
            logger.warning("Discord approval manager not configured (missing token or channel)")
            self.bot = None
            self.enabled = False
            return
        
        # Provide default no-op callback if none provided
        if approval_callback is None:
            approval_callback = lambda trade_id, approved: logger.info(f"Trade {trade_id} {'approved' if approved else 'rejected'}")
        
        self.bot = DiscordTradeApprovalBot(self.token, self.channel_id, approval_callback)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.task = None
        self.thread = None
        self.enabled = True
        
        logger.info("✅ Discord Approval Manager initialized")
    
    def start(self):
        """Start Discord bot in background thread"""
        if not self.enabled or not self.token or not self.bot:
            logger.warning("Discord approval manager not enabled or not properly configured")
            return False
        
        if self.thread and self.thread.is_alive():
            logger.info("Discord bot already running")
            return True
        
        try:
            logger.info("Starting Discord bot thread...")
            
            # Create event loop in background thread
            def run_bot():
                try:
                    logger.info("Bot thread started, setting up event loop...")
                    
                    # Fix for Windows + Python 3.11+ asyncio/aiohttp issue
                    import sys
                    if sys.platform == 'win32':
                        try:
                            # Use SelectorEventLoop instead of ProactorEventLoop on Windows
                            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                            logger.info("Set WindowsSelectorEventLoopPolicy for Windows compatibility")
                        except AttributeError:
                            logger.warning("WindowsSelectorEventLoopPolicy not available, using default")
                    
                    # Create and set new event loop for this thread
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
                    logger.info("Event loop created and set")
                    
                    # Start the Discord bot
                    logger.info("Connecting to Discord...")
                    # At this point, self.token is guaranteed to be a string since we checked enabled status
                    assert self.token is not None, "Token should not be None when manager is enabled"
                    if self.bot:
                        self.loop.run_until_complete(self.bot.start(self.token))
                    
                except KeyboardInterrupt:
                    logger.info("Bot thread received keyboard interrupt")
                except Exception as e:
                    logger.error(f"❌ Error in Discord bot thread: {e}", exc_info=True)
                finally:
                    if self.loop:
                        try:
                            self.loop.close()
                            logger.info("Event loop closed")
                        except Exception as e:
                            logger.error(f"Error closing event loop: {e}")
            
            self.thread = threading.Thread(target=run_bot, daemon=True, name="DiscordBot")
            self.thread.start()
            logger.info("Discord bot thread started, waiting for connection...")
            
            # Wait for bot to be ready
            import time as time_module
            for i in range(30):
                if self.bot.is_ready():
                    logger.info("✅ Discord approval bot connected and ready!")
                    return True
                time_module.sleep(0.5)
            
            logger.warning("⚠️ Discord bot thread started but not ready yet (may still be connecting)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting Discord approval manager: {e}", exc_info=True)
            return False
    
    def send_approval_request(
        self,
        approval_id: str,
        pair: str,
        side: str,
        entry_price: float,
        position_size: float,
        stop_loss: float,
        take_profit: float,
        strategy: str,
        confidence: float,
        reasoning: str,
        additional_info: Optional[str] = None
    ) -> bool:
        """Send approval request to Discord"""
        if not self.enabled or not self.bot:
            logger.debug("Discord approval manager not enabled")
            return False
        
        if not self.bot.is_ready:
            logger.warning("Discord bot not ready")
            return False
        
        try:
            # Schedule coroutine in bot's event loop
            future = asyncio.run_coroutine_threadsafe(
                self.bot.send_approval_request(
                    approval_id=approval_id,
                    pair=pair,
                    side=side,
                    entry_price=entry_price,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strategy=strategy,
                    confidence=confidence,
                    reasoning=reasoning,
                    additional_info=additional_info
                ),
                self.loop or asyncio.new_event_loop()
            )
            
            # Wait for result with timeout
            return future.result(timeout=10)
            
        except Exception as e:
            logger.error(f"Error sending approval request: {e}", exc_info=True)
            return False
    
    def get_approval_status(self, approval_id: str) -> Optional[PendingTradeApproval]:
        """Get status of pending approval"""
        if not self.enabled or not self.bot:
            return None
        return self.bot.get_approval_status(approval_id)
    
    def get_pending_approvals(self) -> Dict[str, PendingTradeApproval]:
        """Get all pending approvals"""
        if not self.enabled or not self.bot:
            return {}
        return self.bot.pending_approvals
    
    def cleanup_expired(self) -> int:
        """Remove expired approvals"""
        if not self.enabled or not self.bot:
            return 0
        return self.bot.cleanup_expired()
    
    def is_running(self) -> bool:
        """Check if bot is running"""
        return bool(self.enabled and self.bot and getattr(self.bot, 'is_ready', False))


# Global singleton instance
_approval_manager_instance: Optional[DiscordApprovalManager] = None


def get_discord_approval_manager(approval_callback: Optional[Callable] = None) -> Optional[DiscordApprovalManager]:
    """
    Get or create Discord approval manager singleton
    
    Args:
        approval_callback: Function to call when trade approved (trade_id, approved: bool)
    
    Returns:
        DiscordApprovalManager instance or None if not configured
    """
    global _approval_manager_instance
    
    if _approval_manager_instance is None:
        _approval_manager_instance = DiscordApprovalManager(approval_callback=approval_callback)
        
        # Auto-start if configured
        if _approval_manager_instance.enabled:
            _approval_manager_instance.start()
    
    return _approval_manager_instance if _approval_manager_instance.enabled else None

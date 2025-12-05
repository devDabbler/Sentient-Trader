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
from discord.ui import View, Button
import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, Set
from dataclasses import dataclass, asdict
import json
import threading

# Import channel utilities for multi-channel support
from src.integrations.discord_channels import get_all_channel_ids

from windows_services.runners.service_config_loader import queue_analysis_request

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
    
    @staticmethod
    def price_fmt(price: float) -> str:
        """Format price for display"""
        if price >= 1000:
            return f"{price:,.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        elif price >= 0.01:
            return f"{price:.6f}"
        else:
            return f"{price:.8f}"


class TradeActionView(View):
    """Interactive buttons for Trade Approval"""
    def __init__(self, bot, approval_id: str):
        super().__init__(timeout=3600)  # 1 hour timeout
        self.bot = bot
        self.approval_id = approval_id

    @discord.ui.button(label="✅ Approve", style=discord.ButtonStyle.green, custom_id="btn_approve")
    async def approve_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await self.bot._handle_button_click(interaction, self.approval_id, True)

    @discord.ui.button(label="❌ Reject", style=discord.ButtonStyle.red, custom_id="btn_reject")
    async def reject_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await self.bot._handle_button_click(interaction, self.approval_id, False)


class AlertActionView(View):
    """Interactive buttons for General Alerts (Watch/Analyze/Trade/Dismiss)"""
    def __init__(self, bot, symbol: str, asset_type: str = "crypto", alert_data: dict = None):
        super().__init__(timeout=7200)  # 2 hour timeout
        self.bot = bot
        self.symbol = symbol
        self.asset_type = asset_type
        self.alert_data = alert_data or {}  # Store alert data for trade execution
        
        # Create buttons dynamically with unique custom_ids per symbol
        # This fixes "interaction failed" when multiple alerts have buttons
        clean_symbol = symbol.replace("/", "_").replace(" ", "_")[:20]  # Discord limits custom_id to 100 chars
        timestamp = int(time.time())
        
        watch_btn = Button(
            label="👀 Watch", 
            style=discord.ButtonStyle.primary, 
            custom_id=f"watch_{clean_symbol}_{timestamp}"
        )
        watch_btn.callback = self._watch_callback
        self.add_item(watch_btn)
        
        analyze_btn = Button(
            label="🔍 Analyze", 
            style=discord.ButtonStyle.secondary, 
            custom_id=f"analyze_{clean_symbol}_{timestamp}"
        )
        analyze_btn.callback = self._analyze_callback
        self.add_item(analyze_btn)
        
        # Add TRADE button for crypto assets - executes via AI Position Manager
        if asset_type == "crypto":
            trade_btn = Button(
                label="🚀 Trade", 
                style=discord.ButtonStyle.success,  # Green button
                custom_id=f"trade_{clean_symbol}_{timestamp}"
            )
            trade_btn.callback = self._trade_callback
            self.add_item(trade_btn)
        
        dismiss_btn = Button(
            label="🗑️ Dismiss", 
            style=discord.ButtonStyle.gray, 
            custom_id=f"dismiss_{clean_symbol}_{timestamp}"
        )
        dismiss_btn.callback = self._dismiss_callback
        self.add_item(dismiss_btn)
    
    async def _watch_callback(self, interaction: discord.Interaction):
        """Handle Watch button click"""
        try:
            await interaction.response.send_message(f"👀 Adding **{self.symbol}** to watchlist...", ephemeral=True)
            await self.bot._handle_watch_command(interaction.message, self.symbol)
        except Exception as e:
            logger.error(f"Error in watch callback: {e}")
            try:
                await interaction.followup.send(f"❌ Error: {str(e)[:100]}", ephemeral=True)
            except:
                pass
    
    async def _analyze_callback(self, interaction: discord.Interaction):
        """Handle Analyze button click"""
        try:
            await interaction.response.send_message(f"🔍 Queuing analysis for **{self.symbol}**...", ephemeral=True)
            await self.bot._handle_analyze_command(interaction.message, self.symbol)
        except Exception as e:
            logger.error(f"Error in analyze callback: {e}")
            try:
                await interaction.followup.send(f"❌ Error: {str(e)[:100]}", ephemeral=True)
            except:
                pass
    
    async def _dismiss_callback(self, interaction: discord.Interaction):
        """Handle Dismiss button click"""
        try:
            await interaction.response.send_message(f"🗑️ Dismissing alert for **{self.symbol}**...", ephemeral=True)
            await self.bot._handle_dismiss_command(interaction.message, self.symbol)
        except Exception as e:
            logger.error(f"Error in dismiss callback: {e}")
            try:
                await interaction.followup.send(f"❌ Error: {str(e)[:100]}", ephemeral=True)
            except:
                pass
    
    async def _trade_callback(self, interaction: discord.Interaction):
        """Handle Trade button click - queues trade for execution via AI Position Manager"""
        try:
            await interaction.response.send_message(
                f"🚀 Preparing trade for **{self.symbol}**...\n"
                f"_Calculating position size and risk parameters..._", 
                ephemeral=True
            )
            
            # Execute trade via AI Crypto Position Manager
            result = await self.bot._handle_trade_command(interaction.message, self.symbol, self.alert_data)
            
            if result.get('success'):
                await interaction.followup.send(
                    f"✅ **Trade Queued:** {self.symbol}\n"
                    f"**Side:** {result.get('side', 'BUY')}\n"
                    f"**Size:** ${result.get('position_size', 0):,.2f}\n"
                    f"**Stop:** ${result.get('stop_loss', 0):,.4f}\n"
                    f"**Target:** ${result.get('take_profit', 0):,.4f}\n\n"
                    f"_Awaiting approval or auto-execution..._",
                    ephemeral=True
                )
            else:
                await interaction.followup.send(
                    f"❌ **Trade Failed:** {result.get('error', 'Unknown error')}\n"
                    f"_Check logs for details._",
                    ephemeral=True
                )
                
        except Exception as e:
            logger.error(f"Error in trade callback: {e}")
            try:
                await interaction.followup.send(f"❌ Trade error: {str(e)[:100]}", ephemeral=True)
            except:
                pass


class DiscordTradeApprovalBot(commands.Bot):
    """Discord bot for trade approvals with interactive buttons"""
    
    def __init__(self, token: str, channel_id: str, approval_callback: Callable):
        """
        Initialize Discord approval bot
        
        Args:
            token: Discord bot token
            channel_id: Primary channel ID to send approval requests (also loads all configured channels)
            approval_callback: Function to call when trade approved (trade_id, approved: bool)
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        
        super().__init__(command_prefix='!', intents=intents)
        
        self.token = token
        self.channel_id = int(channel_id)  # Primary channel for approval messages
        self.approval_callback = approval_callback
        self.pending_approvals: Dict[str, PendingTradeApproval] = {}
        self.bot_ready = False  # Renamed from is_ready to avoid shadowing inherited method
        
        # 🎯 MULTI-CHANNEL SUPPORT: Load all configured channel IDs
        # This allows the bot to listen for replies in ANY channel it monitors
        self.monitored_channel_ids: Set[int] = get_all_channel_ids()
        # Always include the primary channel
        self.monitored_channel_ids.add(self.channel_id)
        
        # Message deduplication - prevent processing same message twice
        self._processed_messages: set = set()
        self._max_processed_cache = 100  # Keep last 100 message IDs
        
        # 🛡️ ANALYSIS DEDUPLICATION - prevent double analysis when multiple bots or reconnects
        self._recent_analysis_requests: Dict[str, datetime] = {}  # {symbol_mode: last_request_time}
        self._analysis_cooldown_seconds = 10  # Minimum seconds between analyses for same symbol+mode
        
        # 🛡️ TRADE DEDUPLICATION - prevent double trades when multiple bots or reconnects
        self._recent_trade_requests: Dict[str, datetime] = {}  # {symbol: last_trade_time}
        self._trade_cooldown_seconds = 30  # Minimum seconds between trades for same symbol
        
        # 🛡️ PREVENT DUPLICATE PROCESSING - track trades currently being processed
        self._processing_trades: set = set()  # {trade_key} to prevent duplicate processing
        
        logger.info(f"🤖 Discord Trade Approval Bot initialized")
        logger.info(f"   Primary Channel ID: {channel_id}")
        logger.info(f"   Monitoring {len(self.monitored_channel_ids)} channels: {self.monitored_channel_ids}")
    
    async def on_ready(self):
        """Called when bot is ready"""
        self.bot_ready = True
        logger.info(f'✅ Discord bot logged in as {self.user}')
        
        # Log all monitored channels
        logger.info(f"📡 Monitoring {len(self.monitored_channel_ids)} channels for commands:")
        for cid in self.monitored_channel_ids:
            channel = self.get_channel(cid)
            if channel:
                logger.info(f"   ✅ #{channel.name} (ID: {cid})")
            else:
                logger.warning(f"   ⚠️ Channel ID {cid} not accessible")
        
        channel = self.get_channel(self.channel_id)
        if channel:
            logger.info(f"✅ Primary channel: {channel.name} (ID: {self.channel_id})")
        else:
            logger.error(f"❌ Could not find primary channel ID: {self.channel_id}")
    
    async def on_message(self, message: discord.Message):
        """Handle incoming messages for approvals"""
        # DEBUG: Log ALL incoming messages
        logger.debug(f"📨 RAW MESSAGE: channel={message.channel.id}, author={message.author}, content='{message.content[:50] if message.content else ''}'")
        
        # Ignore bot's own messages
        if message.author == self.user:
            logger.debug(f"   ↳ Ignoring own message")
            return
        
        # 🎯 MULTI-CHANNEL SUPPORT: Check if message is in ANY monitored channel
        if message.channel.id not in self.monitored_channel_ids:
            logger.debug(f"   ↳ Channel {message.channel.id} not in monitored channels {self.monitored_channel_ids}")
            return
        
        # Log that we're processing this message
        channel_name = getattr(message.channel, 'name', 'unknown')
        logger.info(f"📩 Processing message in #{channel_name} from {message.author.name}: '{message.content}'")
        
        # MESSAGE DEDUPLICATION: Prevent processing same message twice
        # (can happen due to Discord reconnects or network issues)
        msg_id = str(message.id)
        if msg_id in self._processed_messages:
            logger.debug(f"Skipping duplicate message: {msg_id}")
            return
        
        # Add to processed set
        self._processed_messages.add(msg_id)
        
        # Trim cache if too large
        if len(self._processed_messages) > self._max_processed_cache:
            # Remove oldest entries (convert to list, remove first half)
            to_remove = list(self._processed_messages)[:self._max_processed_cache // 2]
            for old_id in to_remove:
                self._processed_messages.discard(old_id)
        
        content = message.content.upper().strip()
        logger.info(f"   Content (upper): '{content}' | Pending approvals: {len(self.pending_approvals)}")
        logger.info(f"   Message reference: {message.reference}")

        # 🔔 CHECK IF THIS IS A REPLY TO A SPECIFIC APPROVAL MESSAGE
        if message.reference and message.reference.message_id:
            logger.info(f"   📎 This is a REPLY to message: {message.reference.message_id}")
            replied_to_id = str(message.reference.message_id)
            
            # 1. Check pending trade approvals (internal memory)
            for approval_id, approval in self.pending_approvals.items():
                if approval.discord_message_id == replied_to_id:
                    # Found the approval - check if it's approve or reject
                    if content in ['APPROVE', 'YES', 'GO', 'EXECUTE', 'Y', '✅', 'OK']:
                        await self._handle_reply_approval(message, approval_id, approval, approve=True)
                        return
                    elif content in ['REJECT', 'NO', 'CANCEL', 'SKIP', 'N', '❌']:
                        await self._handle_reply_approval(message, approval_id, approval, approve=False)
                        return
                    elif content == 'RISK':
                        # Show risk profile
                        await self._handle_risk_command(message)
                        return
                    elif content in ['SIZE', 'SIZING']:
                        # Show position sizing for this approval
                        await self._handle_sizing_command(message, approval.pair)
                        return
                    elif content in ['MODIFY', 'EDIT', 'CHANGE', 'M']:
                        # Show modification options
                        await self._show_position_modify_help(message, approval)
                        return
                    elif content in ['?', 'HELP', 'H']:
                        # Show help for trade approvals
                        help_text = f"""📋 **Trade Approval Commands for {approval.pair}:**

**Approve/Reject:**
`YES` - ✅ Execute the trade as shown
`NO` - ❌ Cancel the trade

**Modify Position (before approving):**
`50` - Change to 50 shares (just type the number)
`SHARES 100` - Change to 100 shares
`$2000` - Change to $2,000 position value
`HALF` - Reduce to 50% of suggested
`DOUBLE` - Increase to 200%
`1.5X` or `3X` - Custom multiplier
`MODIFY` - Show all modification options

**Info:**
`SIZE` - 📊 Recalculate position size
`RISK` - 📊 Show current risk profile
`?` - Show this help
"""
                        await message.channel.send(help_text)
                        return
                    else:
                        # Check for position modification commands
                        import re
                        original_content = message.content.strip()
                        
                        # Pattern 1: Just a number (interpreted as shares) - e.g., "50" or "100"
                        if content.isdigit():
                            new_shares = int(content)
                            await self._modify_position_shares(message, approval_id, approval, new_shares)
                            return
                        
                        # Pattern 2: SHARES 50 or 50 SHARES
                        shares_match = re.match(r'(?:SHARES?\s+)?(\d+)(?:\s+SHARES?)?$', content)
                        if shares_match and ('SHARE' in content or content.replace(' ', '').isdigit()):
                            new_shares = int(shares_match.group(1))
                            await self._modify_position_shares(message, approval_id, approval, new_shares)
                            return
                        
                        # Pattern 3: $2000 or $2,000 or $2000.00 (dollar sign with amount)
                        dollar_match = re.match(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', original_content)
                        if dollar_match:
                            new_value = float(dollar_match.group(1).replace(',', ''))
                            await self._modify_position_value(message, approval_id, approval, new_value)
                            return
                        
                        # Pattern 4: VALUE 2000 or VAL 2000
                        value_match = re.match(r'(?:VALUE?|VAL)\s+(\d+(?:,\d{3})*(?:\.\d{2})?)', content)
                        if value_match:
                            new_value = float(value_match.group(1).replace(',', ''))
                            await self._modify_position_value(message, approval_id, approval, new_value)
                            return
                        
                        # Pattern 5: HALF, DOUBLE, 2X, 0.5X, 1.5X, 3X, etc.
                        multiplier_match = re.match(r'(HALF|DOUBLE|(\d+(?:\.\d+)?)\s*X)', content)
                        if multiplier_match:
                            if multiplier_match.group(1) == 'HALF':
                                multiplier = 0.5
                            elif multiplier_match.group(1) == 'DOUBLE':
                                multiplier = 2.0
                            else:
                                multiplier = float(multiplier_match.group(2))
                            await self._modify_position_multiplier(message, approval_id, approval, multiplier)
                            return
                        
                        # Reply to the approval message but unclear intent
                        await message.channel.send(
                            f"❓ Reply with **YES** to confirm or **NO** to cancel.\n\n"
                            f"**To modify position, reply with:**\n"
                            f"• `50` or `SHARES 50` → Set to 50 shares\n"
                            f"• `$2000` → Set to $2,000 value\n"
                            f"• `HALF` / `DOUBLE` / `1.5X` → Adjust size"
                        )
                        return
            
            # 2. Check Orchestrator Alert Queue (Generic Alerts)
            await self._handle_alert_reply(message, content)
            return
            
        # Check for LIST command to show pending trades
        if content in ['LIST', 'PENDING', 'TRADES']:
            await self._list_pending_trades(message)
            return
        
        logger.info(f"   🔍 Checking standalone commands for: '{content}'")
        
        # ============================================================
        # STANDALONE YES/NO - Apply to most recent pending approval
        # ============================================================
        if content in ['APPROVE', 'YES', 'GO', 'EXECUTE', 'Y', '✅', 'OK']:
            logger.info(f"   🔍 Matched standalone APPROVE command")
            try:
                if await self._handle_standalone_approval(message, approve=True):
                    return
            except Exception as e:
                logger.error(f"   ❌ Error in standalone approval: {e}", exc_info=True)
                await message.channel.send(f"❌ Error: {str(e)[:100]}")
                return
        
        if content in ['REJECT', 'NO', 'CANCEL', 'SKIP', 'N', '❌']:
            logger.info(f"   🔍 Matched standalone REJECT command")
            try:
                if await self._handle_standalone_approval(message, approve=False):
                    return
            except Exception as e:
                logger.error(f"   ❌ Error in standalone rejection: {e}", exc_info=True)
                await message.channel.send(f"❌ Error: {str(e)[:100]}")
                return
        
        # ============================================================
        # STANDALONE COMMANDS (no reply needed)
        # ============================================================
        
        # Account & Risk Commands
        if content in ['BALANCE', 'BAL', 'ACCOUNT', 'STATUS']:
            await self._handle_balance_command(message)
            return
        
        if content == 'RISK':
            await self._handle_risk_command(message)
            return
        
        if content == 'SYNC':
            await self._handle_sync_command(message)
            return
        
        if content in ['BROKER', 'BROKERS']:
            await self._handle_broker_command(message)
            return
        
        # Broker switching: BROKER TRADIER or BROKER IBKR
        import re
        broker_match = re.match(r'BROKER\s+(TRADIER|IBKR)', content)
        if broker_match:
            broker = broker_match.group(1)
            await self._handle_broker_switch_command(message, broker)
            return
        
        # SIZE SYMBOL command (e.g., "SIZE NVDA")
        size_match = re.match(r'SIZE\s+([A-Z]+)', content)
        if size_match:
            symbol = size_match.group(1)
            await self._handle_sizing_command(message, symbol)
            return
        
        # Help command
        if content in ['HELP', '?', 'H', 'COMMANDS']:
            await self._show_global_help(message)
            return
        
        # AI validation toggle
        if content == 'AICHECK':
            await self._handle_aicheck_command(message)
            return
        
        # ============================================================
        # STANDALONE POSITION MODIFICATION COMMANDS (no reply needed)
        # These apply to the most recent pending approval
        # ============================================================
        original_content = message.content.strip()
        logger.debug(f"   🔍 Checking standalone position modification: '{content}'")
        if await self._handle_standalone_position_modification(message, content, original_content):
            logger.debug(f"   ✅ Handled by standalone position modification")
            return
        logger.debug(f"   ❌ Not a standalone position modification command")
        
        # Check for specific trade approval: "APPROVE 1" or "APPROVE BTC" or "1 APPROVE"
        
        # Generic Alert Commands: WATCH, ANALYZE, DISMISS, MULTI, ULTIMATE
        # Pattern: CMD SYMBOL (e.g., "WATCH BTC", "ANALYZE AAPL")
        alert_match = re.match(r'(WATCH|ANALYZE|DISMISS|REMOVE|MULTI|ULTIMATE)\s+([A-Z0-9/]+)', content)
        if alert_match:
            action = alert_match.group(1)
            symbol = alert_match.group(2).upper()  # Keep / for crypto pairs
            
            if action == 'WATCH':
                await self._handle_watch_command(message, symbol.replace('/', ''))
            elif action == 'ANALYZE':
                await self._handle_analyze_command(message, symbol, mode="standard")
            elif action == 'MULTI':
                await self._handle_analyze_command(message, symbol, mode="multi")
            elif action == 'ULTIMATE':
                await self._handle_analyze_command(message, symbol, mode="ultimate")
            elif action in ['DISMISS', 'REMOVE']:
                await self._handle_dismiss_command(message, symbol.replace('/', ''))
            return
        
        # Standalone TRADE commands: "TRADE AAPL", "T NVDA", "BUY TSLA", "PAPER GOOG"
        # Also supports crypto: "TRADE BTC/USD", "T ETH", "BUY SOL/USD"
        trade_match = re.match(r'(TRADE|T|BUY|ENTER|PAPER|P)\s+([A-Z0-9/]+)', content)
        if trade_match:
            action = trade_match.group(1)
            symbol = trade_match.group(2).upper()
            paper_mode = action in ['PAPER', 'P']
            logger.info(f"   🔍 Standalone TRADE command: {action} {symbol} (paper={paper_mode})")
            
            # Check if crypto symbol (contains / or is known crypto)
            from src.integrations.discord_channels import is_crypto_symbol
            if "/" in symbol or is_crypto_symbol(symbol):
                # Crypto trade - ensure proper format
                if "/" not in symbol:
                    symbol = f"{symbol}/USD"
                await self._handle_trade_command(message, symbol)
            else:
                # Stock trade
                await self._handle_stock_trade_execution(message, symbol, side="BUY", paper_mode=paper_mode)
            return

        # Pattern: APPROVE/REJECT followed by number or symbol
        specific_match = re.match(r'(APPROVE|YES|REJECT|NO)\s+(\d+|[A-Z0-9/]+)', content)
        if not specific_match:
            # Also try number first: "1 APPROVE" or "BTC APPROVE"
            specific_match = re.match(r'(\d+|[A-Z0-9/]+)\s+(APPROVE|YES|REJECT|NO)', content)
            if specific_match:
                # Swap order for consistent handling
                identifier = specific_match.group(1)
                action = specific_match.group(2)
            else:
                identifier = None
                action = None
        else:
            action = specific_match.group(1)
            identifier = specific_match.group(2)
        
        if identifier and action:
            approve = action in ['APPROVE', 'YES']
            await self._handle_specific_approval(message, identifier, approve)
            return
        
        # Check for simple approval/rejection keywords (applies to most recent)
        if content in ['APPROVE', 'YES', 'GO', 'EXECUTE', 'BUY', 'Y']:
            await self._handle_approval(message, approve=True)
        elif content in ['REJECT', 'NO', 'CANCEL', 'SKIP', 'N']:
            await self._handle_approval(message, approve=False)
        elif content in ['APPROVE ALL', 'YES ALL']:
            await self._handle_approve_all(message, approve=True)
        elif content in ['REJECT ALL', 'NO ALL', 'CANCEL ALL']:
            await self._handle_approve_all(message, approve=False)
        
        # Process commands
        await self.process_commands(message)

    async def _handle_button_click(self, interaction: discord.Interaction, approval_id: str, approve: bool):
        """Handle button click on trade approval"""
        approval = self.pending_approvals.get(approval_id)
        
        if not approval:
            await interaction.followup.send("❌ This approval request has expired or was already processed.", ephemeral=True)
            return

        if approval.approved or approval.rejected:
            await interaction.followup.send(f"⚠️ This trade ({approval.pair}) has already been processed.", ephemeral=True)
            return
            
        # Mark as approved/rejected
        if approve:
            approval.approved = True
            await interaction.followup.send(
                f"✅ **APPROVED:** {approval.pair} {approval.side} trade\n🚀 Executing trade now..."
            )
            logger.info(f"✅ User approved trade via button: {approval_id} ({approval.pair})")
        else:
            approval.rejected = True
            await interaction.followup.send(
                f"❌ **REJECTED:** {approval.pair} {approval.side} trade\nTrade cancelled."
            )
            logger.info(f"❌ User rejected trade via button: {approval_id} ({approval.pair})")
            
        # Disable buttons on the message
        try:
            if interaction.message:
                await interaction.message.edit(view=None)
        except:
            pass

        # Call approval callback in a thread to prevent blocking Discord event loop
        if self.approval_callback:
            try:
                import asyncio
                # Run callback in thread executor to prevent blocking
                await asyncio.to_thread(self.approval_callback, approval_id, approve)
                logger.info(f"✅ Approval callback completed for {approval_id}")
            except Exception as e:
                logger.error(f"Error in approval callback: {e}", exc_info=True)
                try:
                    await interaction.followup.send(f"⚠️ Callback error: {str(e)[:100]}", ephemeral=True)
                except:
                    pass

    async def _handle_alert_reply(self, message: discord.Message, content: str):
        """Handle replies to generic alerts (from Orchestrator queue)"""
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.refresh_state()  # Sync with other processes
        
        # Get recent pending alerts
        pending = orch.get_pending_alerts()
        
        if not message.reference:
            await message.channel.send("⚠️ This command requires a reply to a message.")
            return

        ref_msg = message.reference.cached_message
        if not ref_msg and message.reference.message_id:
             # Fetch message if not cached
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
            except:
                pass
        
        if not ref_msg:
            await message.channel.send("⚠️ Could not retrieve original message context.")
            return

        # Extract symbol from the referenced message embed or content
        target_symbol = None
        
        # Pattern to match crypto (XXX/USD, BTC/USDT) and stock symbols (AAPL, NVDA)
        import re
        symbol_pattern = re.compile(r'\b([A-Z0-9]{2,10}/[A-Z]{2,4}|[A-Z]{1,5})\b')
        
        if ref_msg.embeds:
            for embed in ref_msg.embeds:
                if embed.title:
                    # Method 1: Title with colon - "🔔 BREAKOUT: BLUR/USD"
                    if ":" in embed.title:
                        parts = embed.title.split(":")
                        if len(parts) > 1:
                            possible_symbol = parts[-1].strip().split(' ')[0]
                            if possible_symbol and len(possible_symbol) >= 2:
                                target_symbol = possible_symbol
                                break
                    
                    # Method 2: Title without colon - "📈 BTC/USD" or "🔥 BLUR/USD"
                    if not target_symbol:
                        # Find symbol pattern in title
                        matches = symbol_pattern.findall(embed.title)
                        # Prefer crypto pairs (contain /) over stock symbols
                        for match in matches:
                            if '/' in match:
                                target_symbol = match
                                break
                        # If no crypto pair, use first match that's not a common word
                        if not target_symbol and matches:
                            for match in matches:
                                if match not in ['RSI', 'EMA', 'SMA', 'MACD', 'BUY', 'SELL', 'HIGH', 'LOW']:
                                    target_symbol = match
                                    break
                    
                    if target_symbol:
                        break
        
        # Method 3: Try message content if no embed match
        if not target_symbol and ref_msg.content:
            content_to_check = ref_msg.content
            
            # Check for colon format first
            if ":" in content_to_check:
                parts = content_to_check.split(":")
                if len(parts) > 1:
                    possible_symbol = parts[-1].strip().split(' ')[0].split('\n')[0]
                    if possible_symbol and len(possible_symbol) >= 2:
                        target_symbol = possible_symbol
            
            # Otherwise search for symbol patterns in content
            if not target_symbol:
                matches = symbol_pattern.findall(content_to_check)
                for match in matches:
                    if '/' in match:  # Prefer crypto pairs
                        target_symbol = match
                        break
                if not target_symbol and matches:
                    for match in matches:
                        if match not in ['RSI', 'EMA', 'SMA', 'MACD', 'BUY', 'SELL', 'HIGH', 'LOW', 'USD', 'USDT']:
                            target_symbol = match
                            break

        if not target_symbol:
            await message.channel.send("⚠️ Could not determine symbol from the message you replied to.")
            return
            
        target_symbol = target_symbol.upper()
        
        if content in ['WATCH', 'ADD', 'TRACK', 'W']:
            await self._handle_watch_command(message, target_symbol)
            
            # Also auto-approve in orchestrator if it exists
            for alert in pending:
                if alert.symbol == target_symbol:
                    orch.approve_alert(alert.id, add_to_watchlist=True)
        
        # Analysis commands with shortcuts (works for both crypto and stocks)
        elif content in ['1', 'S', 'STD', 'STANDARD', 'ANALYZE', 'SCAN', 'CHECK']:
            await self._handle_analyze_command(message, target_symbol, mode="standard")
            
        elif content in ['2', 'M', 'MULTI']:
            await self._handle_analyze_command(message, target_symbol, mode="multi")

        elif content in ['3', 'U', 'ULT', 'ULTIMATE']:
            await self._handle_analyze_command(message, target_symbol, mode="ultimate")
        
        # TRADE EXECUTION COMMANDS - Route to appropriate handler based on asset type
        elif content in ['T', 'TRADE', 'EXECUTE', 'BUY', 'ENTER']:
            # Check if crypto (contains /) or stock
            if "/" in target_symbol:
                # Crypto trade - use crypto position manager
                # _handle_trade_command sends its own success/failure messages
                await self._handle_trade_command(message, target_symbol)
            else:
                # Stock trade
                await self._handle_stock_trade_execution(message, target_symbol, side="BUY")
        
        elif content in ['SHORT', 'SELL', 'S-TRADE']:
            if "/" in target_symbol:
                # Crypto short/sell - not yet implemented
                await message.channel.send(f"⚠️ Short selling not yet available for crypto. Use the exchange directly.")
            else:
                await self._handle_stock_trade_execution(message, target_symbol, side="SELL")
        
        elif content in ['P', 'PAPER', 'PAPER-TRADE', 'TEST']:
            if "/" in target_symbol:
                await message.channel.send(f"⚠️ Paper trading for crypto not yet implemented. Use `T` or `TRADE` for live trading.")
            else:
                await self._handle_stock_trade_execution(message, target_symbol, side="BUY", paper_mode=True)
            
        elif content in ['DISMISS', 'REMOVE', 'DELETE', 'X', 'D']:
            # Reject in orchestrator
            found = False
            for alert in pending:
                if alert.symbol == target_symbol:
                    orch.reject_alert(alert.id)
                    found = True
            
            if found:
                await message.channel.send(f"🗑️ Dismissed alert for {target_symbol}")
            else:
                await message.channel.send(f"⚠️ No pending alert found for {target_symbol} to dismiss, but noted.")
        
        elif content in ['?', 'HELP', 'H']:
            # Show available commands - different for crypto vs stocks
            is_crypto = "/" in target_symbol
            
            if is_crypto:
                help_text = f"""📋 **Commands for {target_symbol}:** (Crypto)

**Watchlist:**
`W` or `WATCH` - Add to watchlist

**Analysis (reply with):**
`1` or `S` - 🔬 Standard (single strategy)
`2` or `M` - 🎯 Multi (Long/Short + timeframes)
`3` or `U` - 🚀 Ultimate (ALL combinations)

**Trade (after analysis):**
`T` or `TRADE` - Queue BUY trade for approval

**After Trade Queued:**
`YES` - ✅ Confirm and execute trade
`NO` - ❌ Cancel trade
`100` or `TOKENS 100` - Change to 100 tokens
`$500` - Change to $500 position
`HALF` / `DOUBLE` - Adjust size

**Position Sizing & Account:**
`SIZE` or `SIZING` - Show recommended position size
`RISK` - Show current risk profile
`BALANCE` or `BAL` - Show broker account balance

**Other:**
`X` or `D` - Dismiss alert
"""
            else:
                help_text = f"""📋 **Commands for {target_symbol}:** (Stock)

**Watchlist:**
`W` or `WATCH` - Add to watchlist

**Analysis (reply with):**
`1` or `S` - 🔬 Standard (single strategy)
`2` or `M` - 🎯 Multi (Long/Short + timeframes)
`3` or `U` - 🚀 Ultimate (ALL combinations)

**Trade Execution (after analysis):**
`T` or `TRADE` - Queue BUY trade for approval
`SHORT` - Execute SHORT/SELL trade
`P` or `PAPER` - Paper trade (test mode)

**After Trade Queued:**
`YES` - ✅ Confirm and execute trade
`NO` - ❌ Cancel trade
`50` or `SHARES 50` - Change to 50 shares
`$2000` - Change to $2,000 position
`HALF` / `DOUBLE` - Adjust size

**Position Sizing & Account:**
`SIZE` or `SIZING` - Show recommended position size
`RISK` - Show current risk profile
`BALANCE` or `BAL` - Show broker account balance
`SYNC` - Sync broker balance to risk profile

**Other:**
`X` or `D` - Dismiss alert
"""
            await message.channel.send(help_text)
        
        elif content in ['SIZE', 'SIZING']:
            # Show position sizing recommendation
            await self._handle_sizing_command(message, target_symbol)
        
        elif content == 'RISK':
            # Show risk profile
            await self._handle_risk_command(message)
        
        elif content in ['BALANCE', 'BAL', 'ACCOUNT']:
            # Show broker account balance
            await self._handle_balance_command(message)
        
        elif content == 'SYNC':
            # Sync broker balance to risk profile
            await self._handle_sync_command(message)
        
        elif content in ['BALANCE', 'BAL', 'ACCOUNT', 'STATUS']:
            # Show broker account status
            await self._handle_balance_command(message)
        
        elif content == 'SYNC':
            # Sync broker balance to risk profile
            await self._handle_sync_command(message)

    async def _handle_watch_command(self, message: discord.Message, symbol: str):
        """Handle WATCH command - adds to BOTH database and service watchlist"""
        try:
            if "/" in symbol:
                # Crypto - add to crypto watchlist manager
                from services.crypto_watchlist_manager import CryptoWatchlistManager
                wm = CryptoWatchlistManager()
                if wm.add_crypto(symbol):
                    # Also add to service watchlist for crypto monitor
                    try:
                        from windows_services.runners.service_config_loader import load_service_watchlist
                        from service_control_panel import set_service_watchlist
                        current_watchlist = load_service_watchlist('sentient-crypto-breakout') or []
                        if symbol not in current_watchlist:
                            current_watchlist.append(symbol)
                            set_service_watchlist('sentient-crypto-breakout', current_watchlist)
                            logger.info(f"✅ {symbol} also added to crypto service watchlist")
                    except Exception as e:
                        logger.warning(f"Could not add to service watchlist: {e}")
                    
                    await message.channel.send(f"✅ **{symbol}** added to Crypto Watchlist + Service Monitor")
                else:
                    await message.channel.send(f"⚠️ Failed to add {symbol} (duplicate?)")
            else:
                # Stock - add to BOTH TickerManager (Supabase) AND Stock Monitor service watchlist
                from services.ticker_manager import TickerManager
                tm = TickerManager()
                success = tm.add_ticker(symbol)
                
                if success:
                    # Also add to Stock Monitor service watchlist so it gets scanned
                    try:
                        from windows_services.runners.service_config_loader import load_service_watchlist
                        from service_control_panel import set_service_watchlist
                        current_watchlist = load_service_watchlist('sentient-stock-monitor') or []
                        if symbol not in current_watchlist:
                            current_watchlist.append(symbol)
                            set_service_watchlist('sentient-stock-monitor', current_watchlist)
                            logger.info(f"✅ {symbol} also added to stock service watchlist")
                    except Exception as e:
                        logger.warning(f"Could not add to service watchlist: {e}")
                    
                    await message.channel.send(f"✅ **{symbol}** added to Stock Watchlist + Service Monitor")
                else:
                    await message.channel.send(f"⚠️ Failed to add {symbol} to watchlist")
                
        except Exception as e:
            logger.error(f"Error processing WATCH {symbol}: {e}")
            await message.channel.send(f"❌ Error adding {symbol}: {str(e)}")
    
    async def _handle_risk_command(self, message: discord.Message):
        """Handle RISK command - show current risk profile"""
        try:
            from services.risk_profile_config import get_risk_profile_manager, RISK_PRESETS
            
            manager = get_risk_profile_manager()
            profile = manager.get_profile()
            
            # Emoji based on tolerance
            emoji = {"Conservative": "🛡️", "Moderate": "⚖️", "Aggressive": "🚀"}.get(
                profile.risk_tolerance, "📊"
            )
            
            risk_text = f"""📊 **Current Risk Profile**

{emoji} **Tolerance:** {profile.risk_tolerance}

💰 **Capital:**
   Total: ${profile.total_capital:,.2f}
   Available: ${profile.available_capital:,.2f}
   Usable: ${profile.get_usable_capital():,.2f}
   Reserve: {profile.reserved_pct}%

📈 **Position Limits:**
   Max Position: {profile.max_position_pct}% (${profile.get_max_position_value():,.2f})
   Min Position: {profile.min_position_pct}%
   Max Positions: {profile.max_positions}

⚠️ **Risk Management:**
   Risk/Trade: {profile.risk_per_trade_pct}%
   Max Daily Loss: {profile.max_loss_per_day_pct}%
   Min Confidence: {profile.min_confidence_to_trade}%

🤖 **AI Sizing:** {'Enabled' if profile.use_ai_sizing else 'Disabled'}

💡 *Configure via Service Control Panel → Risk Profile tab*
"""
            await message.channel.send(risk_text)
            
        except Exception as e:
            logger.error(f"Error showing risk profile: {e}")
            await message.channel.send(f"❌ Error loading risk profile: {str(e)}")
    
    async def _handle_balance_command(self, message: discord.Message):
        """Handle BALANCE command - show broker account status and cash"""
        try:
            from ui.risk_profile_ui import get_broker_status
            
            status = get_broker_status()
            
            if status['connected']:
                mode_emoji = "📝" if status['paper_mode'] else "💰"
                mode_str = "PAPER" if status['paper_mode'] else "LIVE"
                
                balance_text = f"""🏦 **Broker Account Status**

✅ **Connected:** {status['broker_type']} ({mode_emoji} {mode_str})

💵 **Account Balance:**
   Total Equity: **${status['total_equity']:,.2f}**
   Cash: **${status['cash']:,.2f}**
   Buying Power: **${status['buying_power']:,.2f}**

📊 **Quick Stats:**
   Available for Trading: ${status['buying_power']:,.2f}
   
💡 *Reply `RISK` to see risk profile, `SYNC` to update risk profile with these balances*
"""
            else:
                balance_text = f"""🏦 **Broker Account Status**

❌ **Not Connected**
   Broker: {status['broker_type']}
   Mode: {'PAPER' if status['paper_mode'] else 'LIVE'}
   Error: {status.get('error', 'Unknown')}

💡 *Check your .env configuration and broker connection*
"""
            
            await message.channel.send(balance_text)
            
        except Exception as e:
            logger.error(f"Error showing broker balance: {e}")
            await message.channel.send(f"❌ Error loading broker status: {str(e)}")
    
    async def _handle_sync_command(self, message: discord.Message):
        """Handle SYNC command - sync broker balance to risk profile"""
        try:
            from ui.risk_profile_ui import get_broker_status
            from services.risk_profile_config import get_risk_profile_manager
            
            # Get broker status
            status = get_broker_status()
            
            if not status['connected']:
                await message.channel.send(
                    f"❌ Cannot sync - broker not connected.\n"
                    f"Error: {status.get('error', 'Unknown')}"
                )
                return
            
            # Update risk profile
            manager = get_risk_profile_manager()
            manager.update_capital(
                total=status['total_equity'],
                available=status['buying_power']
            )
            
            await message.channel.send(
                f"✅ **Risk Profile Synced!**\n\n"
                f"💵 Total Capital: ${status['total_equity']:,.2f}\n"
                f"💰 Available Capital: ${status['buying_power']:,.2f}\n\n"
                f"Your risk profile now uses live broker balances."
            )
            
        except Exception as e:
            logger.error(f"Error syncing broker balance: {e}")
            await message.channel.send(f"❌ Error syncing: {str(e)}")
    
    async def _handle_broker_command(self, message: discord.Message):
        """Handle BROKER command - show current broker config and options"""
        try:
            import os
            import socket
            from ui.risk_profile_ui import get_broker_status
            
            status = get_broker_status()
            
            # Check which brokers are configured
            tradier_configured = bool(
                os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or 
                os.getenv('TRADIER_ACCESS_TOKEN') or
                os.getenv('TRADIER_PROD_ACCESS_TOKEN')
            )
            ibkr_configured = bool(os.getenv('IBKR_PAPER_PORT'))
            
            # Check if we're on a remote server
            hostname = socket.gethostname()
            is_remote = 'sentient' in hostname.lower() or not os.path.exists('C:\\')
            
            # IBKR availability note
            ibkr_note = " ⚠️ (requires TWS locally)" if is_remote and ibkr_configured else ""
            
            broker_text = f"""🏦 **Broker Configuration**

**Current Broker:** {status['broker_type']} ({'📝 PAPER' if status['paper_mode'] else '💰 LIVE'})
**Status:** {'✅ Connected' if status['connected'] else '❌ Not Connected'}
**Host:** `{hostname}` {'🌐 (remote)' if is_remote else '💻 (local)'}

**Available Brokers:**
{'✅' if tradier_configured else '❌'} **TRADIER** - {'Configured ✨ (works anywhere!)' if tradier_configured else 'Not configured'}
{'✅' if ibkr_configured else '❌'} **IBKR** - {'Configured' + ibkr_note if ibkr_configured else 'Not configured'}

**Switch Broker:**
`BROKER TRADIER` or `BROKER IBKR`

**Commands:**
• `BALANCE` - Show account balance
• `SYNC` - Sync balance to risk profile
• `RISK` - Show risk profile
"""
            await message.channel.send(broker_text)
            
        except Exception as e:
            logger.error(f"Error showing broker config: {e}")
            await message.channel.send(f"❌ Error: {str(e)}")
    
    async def _handle_broker_switch_command(self, message: discord.Message, broker: str):
        """Handle BROKER TRADIER or BROKER IBKR command"""
        try:
            import os
            
            broker = broker.upper()
            
            if broker == 'TRADIER':
                # Check if Tradier is configured
                has_token = bool(
                    os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or 
                    os.getenv('TRADIER_ACCESS_TOKEN')
                )
                if not has_token:
                    await message.channel.send(
                        "❌ **Tradier not configured**\n\n"
                        "Add to your `.env` file:\n"
                        "```\n"
                        "TRADIER_PAPER_ACCESS_TOKEN=your_token\n"
                        "TRADIER_PAPER_ACCOUNT_ID=your_account_id\n"
                        "```"
                    )
                    return
                
                # Set broker type (note: this is runtime only, doesn't persist)
                os.environ['BROKER_TYPE'] = 'TRADIER'
                
                # Test connection
                from ui.risk_profile_ui import get_broker_status
                status = get_broker_status()
                
                if status['connected']:
                    await message.channel.send(
                        f"✅ **Switched to TRADIER**\n\n"
                        f"Mode: {'📝 PAPER' if status['paper_mode'] else '💰 LIVE'}\n"
                        f"Balance: ${status['total_equity']:,.2f}\n"
                        f"Buying Power: ${status['buying_power']:,.2f}"
                    )
                else:
                    await message.channel.send(
                        f"⚠️ Switched to TRADIER but connection failed:\n{status.get('error', 'Unknown error')}"
                    )
                    
            elif broker == 'IBKR':
                # Check if IBKR is configured
                has_port = bool(os.getenv('IBKR_PAPER_PORT'))
                if not has_port:
                    await message.channel.send(
                        "❌ **IBKR not configured**\n\n"
                        "Add to your `.env` file:\n"
                        "```\n"
                        "IBKR_PAPER_PORT=7497\n"
                        "IBKR_PAPER_CLIENT_ID=1\n"
                        "```\n"
                        "And make sure TWS/Gateway is running!"
                    )
                    return
                
                # Check if we're on a remote server (IBKR won't work)
                import socket
                hostname = socket.gethostname()
                is_remote = 'sentient' in hostname.lower() or not os.path.exists('C:\\')
                
                if is_remote:
                    await message.channel.send(
                        f"⚠️ **IBKR requires TWS/Gateway running locally**\n\n"
                        f"You're on: `{hostname}` (remote server)\n"
                        f"IBKR TWS is a desktop GUI app that must run on your **local Windows machine**.\n\n"
                        f"**Options:**\n"
                        f"1. Run this bot on your local machine with TWS running\n"
                        f"2. Use **Tradier** instead (`BROKER TRADIER`) - works anywhere!\n"
                        f"3. Set up VPN/port forwarding to your local TWS (advanced)"
                    )
                    return
                
                os.environ['BROKER_TYPE'] = 'IBKR'
                
                from ui.risk_profile_ui import get_broker_status
                status = get_broker_status()
                
                if status['connected']:
                    await message.channel.send(
                        f"✅ **Switched to IBKR**\n\n"
                        f"Mode: {'📝 PAPER' if status['paper_mode'] else '💰 LIVE'}\n"
                        f"Balance: ${status['total_equity']:,.2f}\n"
                        f"Buying Power: ${status['buying_power']:,.2f}"
                    )
                else:
                    await message.channel.send(
                        f"⚠️ **IBKR connection failed**\n\n"
                        f"Error: {status.get('error', 'Connection refused')}\n\n"
                        f"**Checklist:**\n"
                        f"☐ TWS or IB Gateway is running\n"
                        f"☐ API connections enabled in TWS (File → Global Config → API)\n"
                        f"☐ Port matches: `{os.getenv('IBKR_PAPER_PORT', '7497')}`\n"
                        f"☐ 'Read-Only API' is **disabled**\n\n"
                        f"💡 Or use `BROKER TRADIER` - no desktop app needed!"
                    )
            else:
                await message.channel.send(f"❌ Unknown broker: {broker}. Use `TRADIER` or `IBKR`")
                
        except Exception as e:
            logger.error(f"Error switching broker: {e}")
            await message.channel.send(f"❌ Error switching broker: {str(e)}")
    
    async def _show_global_help(self, message: discord.Message):
        """Show global help for all available commands"""
        help_text = """📋 **Sentient Trader Discord Commands**

**📡 Multi-Channel Support:**
_Commands work in all alert channels: stocks, crypto, options, and executions_

**🏦 Account & Broker:**
`BALANCE` or `BAL` - Show broker account balance
`BROKER` - Show broker configuration
`BROKER TRADIER` - Switch to Tradier
`BROKER IBKR` - Switch to IBKR
`SYNC` - Sync broker balance to risk profile

**📊 Risk & Position Sizing:**
`RISK` - Show your risk profile
`SIZE NVDA` - Calculate position size for a symbol
`AICHECK` - Toggle AI validation before trades

**📋 Trades & Alerts:**
`LIST` or `PENDING` - Show pending trade approvals
`WATCH AAPL` or `WATCH BTC/USD` - Add symbol to watchlist
`ANALYZE NVDA` or `ANALYZE ETH/USD` - Run standard analysis
`MULTI BTC/USD` - Run multi-strategy analysis
`ULTIMATE SOL/USD` - Run ultimate analysis
`TRADE BTC/USD` or `T ETH` - Queue crypto trade
`TRADE AAPL` - Queue stock trade

**Reply to Alerts with:**
`1` / `2` / `3` - Standard / Multi / Ultimate analysis
`T` or `TRADE` - Queue trade for approval
`SHORT` - Execute short trade (stocks only)
`?` - Show alert-specific help

**After Trade is Queued:**
`YES` - ✅ Confirm and execute trade
`NO` - ❌ Cancel trade
`100` or `TOKENS 100` - Change to 100 tokens/shares
`$500` - Change to $500 position
`HALF` / `DOUBLE` - Adjust size

💡 *Crypto trades require confirmation before execution!*
💡 *Type any command directly - no need to reply!*
"""
        await message.channel.send(help_text)
    
    async def _handle_aicheck_command(self, message: discord.Message):
        """Toggle AI pre-trade validation on/off"""
        try:
            import os
            
            # Get current setting (default True)
            current = os.environ.get('AI_TRADE_VALIDATION', 'true').lower() == 'true'
            
            # Toggle
            new_value = not current
            os.environ['AI_TRADE_VALIDATION'] = str(new_value).lower()
            
            status = "✅ ENABLED" if new_value else "❌ DISABLED"
            
            await message.channel.send(
                f"🥊 **AI Pre-Trade Validation: {status}**\n\n"
                f"{'Trades will be validated by AI before execution.' if new_value else 'Trades will execute immediately after approval (no AI check).'}\n\n"
                f"💡 Type `AICHECK` again to toggle."
            )
            
        except Exception as e:
            logger.error(f"Error toggling AI check: {e}")
            await message.channel.send(f"❌ Error: {str(e)}")
    
    async def _handle_sizing_command(self, message: discord.Message, symbol: str):
        """Handle SIZING command - calculate position size for symbol"""
        try:
            from services.risk_profile_config import get_risk_profile_manager
            from services.service_orchestrator import get_orchestrator
            
            # Get current price from alert or market data
            orch = get_orchestrator()
            orch.refresh_state()
            pending = orch.get_pending_alerts()
            
            # Find alert for this symbol
            alert = None
            for a in pending:
                if a.symbol.upper() == symbol.upper():
                    alert = a
                    break
            
            if alert and alert.price:
                price = alert.price
                stop_loss = alert.stop_loss or (price * 0.95)
                confidence = float(alert.metadata.get('score', 75)) if alert.metadata else 75.0
            else:
                # Try to get current price from yfinance
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1d')
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                        stop_loss = price * 0.95  # Default 5% stop
                        confidence = 70.0  # Default confidence
                    else:
                        await message.channel.send(f"⚠️ Could not get price for {symbol}")
                        return
                except Exception as e:
                    await message.channel.send(f"⚠️ Could not get price for {symbol}: {str(e)[:50]}")
                    return
            
            # Calculate sizing
            manager = get_risk_profile_manager()
            sizing = manager.calculate_position_size(
                price=price,
                stop_loss=stop_loss,
                confidence=confidence
            )
            
            # Calculate potential targets
            risk_per_share = price - stop_loss
            target_1r = price + risk_per_share
            target_2r = price + (risk_per_share * 2)
            target_3r = price + (risk_per_share * 3)
            
            sizing_text = f"""📊 **Position Sizing for {symbol}**

💰 **Entry:** ${price:.2f}
🛑 **Stop Loss:** ${stop_loss:.2f} ({((price - stop_loss) / price * 100):.1f}% risk)

📈 **Recommended Position:**
   Shares: **{sizing['recommended_shares']:,}**
   Value: **${sizing['recommended_value']:,.2f}**
   % of Portfolio: {sizing['position_pct']:.1f}%

⚠️ **Risk:**
   Amount at Risk: ${sizing['risk_amount']:,.2f}
   % of Capital: {sizing['risk_pct']:.1f}%

🎯 **Targets (R-multiples):**
   1R: ${target_1r:.2f} (+${sizing['recommended_shares'] * risk_per_share:,.2f})
   2R: ${target_2r:.2f} (+${sizing['recommended_shares'] * risk_per_share * 2:,.2f})
   3R: ${target_3r:.2f} (+${sizing['recommended_shares'] * risk_per_share * 3:,.2f})

📊 Confidence: {confidence:.0f}% | Adjustment: {sizing['confidence_adjustment']:.2f}x
🛡️ Profile: {sizing['profile_tolerance']}
"""
            await message.channel.send(sizing_text)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            await message.channel.send(f"❌ Error calculating size: {str(e)[:100]}")

    async def _handle_analyze_command(self, message: discord.Message, symbol: str, mode: str = "standard"):
        """Handle ANALYZE command - supports both crypto and stock analysis modes"""
        from windows_services.runners.service_config_loader import queue_analysis_request
        
        try:
            # 🛡️ ANALYSIS DEDUPLICATION: Use file-based lock to prevent double analysis across processes
            # This catches cases where multiple bot instances or Discord reconnects cause duplicate processing
            cache_key = f"{symbol.upper()}_{mode}"
            lock_file = os.path.join(os.path.dirname(__file__), "..", "logs", f".analysis_lock_{cache_key.replace('/', '_')}.lock")
            
            try:
                # Check if lock file exists and is recent (within cooldown period)
                if os.path.exists(lock_file):
                    lock_age = time.time() - os.path.getmtime(lock_file)
                    if lock_age < self._analysis_cooldown_seconds:
                        logger.warning(f"⏳ Duplicate analysis blocked (file lock): {symbol} {mode} ({lock_age:.1f}s ago)")
                        return
                
                # Create/update lock file
                os.makedirs(os.path.dirname(lock_file), exist_ok=True)
                with open(lock_file, 'w') as f:
                    f.write(f"{datetime.now().isoformat()}|{symbol}|{mode}")
            except Exception as lock_err:
                logger.debug(f"Lock file error (non-critical): {lock_err}")
            
            # Also check in-memory cache for same-process duplicates
            if cache_key in self._recent_analysis_requests:
                last_request_time = self._recent_analysis_requests[cache_key]
                elapsed = (datetime.now() - last_request_time).total_seconds()
                if elapsed < self._analysis_cooldown_seconds:
                    logger.warning(f"⏳ Duplicate analysis blocked (memory): {symbol} {mode} (requested {elapsed:.1f}s ago)")
                    return
            
            # Record this analysis request
            self._recent_analysis_requests[cache_key] = datetime.now()
            
            # Clean up old entries (older than 2 minutes)
            cutoff = datetime.now() - timedelta(minutes=2)
            self._recent_analysis_requests = {
                k: t for k, t in self._recent_analysis_requests.items() 
                if t > cutoff
            }
            
            # Determine asset type
            asset_type = "crypto" if "/" in symbol or symbol in ["BTC", "ETH", "SOL", "AVAX", "LINK", "DOGE", "XRP"] else "stock"
            
            # Updated preset map with stock-specific analysis modes
            if asset_type == "crypto":
                preset_map = {
                    "standard": "crypto_standard",
                    "multi": "crypto_multi",
                    "ultimate": "crypto_ultimate"
                }
            else:
                # Stock-specific analysis modes
                preset_map = {
                    "standard": "stock_standard",
                    "multi": "stock_multi",
                    "ultimate": "stock_ultimate"
                }
            
            preset = preset_map.get(mode, preset_map["standard"])
            
            # Mode descriptions for user feedback
            mode_descriptions = {
                "standard": "🔬 Single strategy analysis",
                "multi": "🎯 Multi-strategy (Long/Short + leverage)",
                "ultimate": "🚀 Complete analysis (ALL strategies + timeframes)"
            }
            mode_desc = mode_descriptions.get(mode, "📊 Standard analysis")
            
            if queue_analysis_request(preset, [symbol], asset_type=asset_type, analysis_mode=mode):
                await message.channel.send(
                    f"🔍 **{mode.upper()}** Analysis queued for **{symbol}** ({asset_type})\n"
                    f"   {mode_desc}\n"
                    f"   Check Control Panel for results or wait for Discord notification."
                )
            else:
                await message.channel.send(f"❌ Failed to queue analysis for {symbol}")
                 
        except Exception as e:
            logger.error(f"Error processing ANALYZE {symbol}: {e}")
            await message.channel.send(f"❌ Error analyzing {symbol}: {str(e)}")

    async def _handle_dismiss_command(self, message: discord.Message, symbol: str):
        """Handle DISMISS command"""
        from services.service_orchestrator import get_orchestrator
        orch = get_orchestrator()
        orch.refresh_state()
        
        pending = orch.get_pending_alerts()
        rejected_count = 0
        
        for alert in pending:
            if alert.symbol == symbol:
                orch.reject_alert(alert.id)
                rejected_count += 1
        
        if rejected_count > 0:
            await message.channel.send(f"🗑️ Dismissed {rejected_count} alert(s) for **{symbol}**")
        else:
             await message.channel.send(f"ℹ️ No pending alerts found for **{symbol}**")
    
    async def _handle_trade_command(self, message: discord.Message, symbol: str, alert_data: dict = None) -> dict:
        """
        Handle TRADE command for CRYPTO - Queue trade for approval (mirrors stock workflow)
        Does NOT execute immediately - waits for user confirmation with YES
        
        Args:
            message: Discord message
            symbol: Crypto pair (e.g., BTC/USD, ETH/USD)
            alert_data: Optional data from the alert (price, score, etc.)
            
        Returns:
            dict with success status
        """
        import asyncio
        
        # Ensure we have a valid crypto pair format
        if "/" not in symbol:
            symbol = f"{symbol}/USD"
        
        # 🛡️ TRADE DEDUPLICATION: Check cooldown FIRST, before any processing or messages
        trade_lock_file = os.path.join(os.path.dirname(__file__), "..", "logs", f".trade_lock_{symbol.replace('/', '_')}.lock")
        
        cooldown_active = False
        lock_age = 0
        elapsed = 0
        cooldown_source = None
        
        # Check file-based lock (cross-process)
        try:
            if os.path.exists(trade_lock_file):
                lock_age = time.time() - os.path.getmtime(trade_lock_file)
                if lock_age < self._trade_cooldown_seconds:
                    cooldown_active = True
                    cooldown_source = "file lock"
        except Exception as lock_err:
            logger.debug(f"Trade lock file error (non-critical): {lock_err}")
        
        # Check in-memory cache (same-process)
        if not cooldown_active and symbol in self._recent_trade_requests:
            last_trade_time = self._recent_trade_requests[symbol]
            elapsed = (datetime.now() - last_trade_time).total_seconds()
            if elapsed < self._trade_cooldown_seconds:
                cooldown_active = True
                cooldown_source = "memory cache"
        
        # If cooldown is active, send ONE message and return IMMEDIATELY (no processing set entry)
        if cooldown_active:
            # Use the most recent cooldown time
            cooldown_time = max(lock_age, elapsed) if lock_age > 0 else elapsed
            remaining = int(self._trade_cooldown_seconds - cooldown_time)
            logger.warning(f"⏳ Duplicate crypto trade blocked ({cooldown_source}): {symbol} ({cooldown_time:.1f}s ago)")
            await message.channel.send(
                f"⏳ **Trade Cooldown:** {symbol} was just traded {cooldown_time:.0f}s ago.\n"
                f"Wait {remaining}s or check your positions."
            )
            return {'success': False, 'error': 'Trade cooldown active'}
        
        # 🛡️ PREVENT DUPLICATE PROCESSING: Use symbol-only key to catch duplicates from any code path
        trade_key = f"crypto_trade_{symbol}"
        if trade_key in self._processing_trades:
            logger.warning(f"⏳ Duplicate trade command ignored: {symbol} (already processing)")
            return {'success': False, 'error': 'Already processing this trade'}
        
        # No cooldown and not already processing - add to processing set and create locks IMMEDIATELY
        self._processing_trades.add(trade_key)
        
        try:
            # Create/update lock file IMMEDIATELY to prevent race conditions
            try:
                os.makedirs(os.path.dirname(trade_lock_file), exist_ok=True)
                with open(trade_lock_file, 'w') as f:
                    f.write(f"{datetime.now().isoformat()}|{symbol}")
            except Exception as lock_err:
                logger.debug(f"Trade lock file creation error (non-critical): {lock_err}")
            
            # Record this trade request in memory IMMEDIATELY
            self._recent_trade_requests[symbol] = datetime.now()
            
            # Clean up old entries (older than 5 minutes)
            cutoff = datetime.now() - timedelta(minutes=5)
            self._recent_trade_requests = {
                sym: t for sym, t in self._recent_trade_requests.items() 
                if t > cutoff
            }
            
            # NOW send the preparing message (after all checks and locks are in place)
            await message.channel.send(f"🚀 **Preparing Crypto Trade:** {symbol}...")
            
            # Get current price and calculate trade parameters
            def _get_trade_params():
                try:
                    from clients.kraken_client import KrakenClient
                    from services.risk_profile_config import get_risk_profile_manager
                    
                    # Get Kraken credentials from environment
                    api_key = os.getenv('KRAKEN_API_KEY')
                    api_secret = os.getenv('KRAKEN_API_SECRET')
                    
                    if not api_key or not api_secret:
                        return {'success': False, 'error': 'KRAKEN_API_KEY and KRAKEN_API_SECRET must be set in .env'}
                    
                    # Get current price
                    kraken = KrakenClient(api_key=api_key, api_secret=api_secret)
                    ticker_info = kraken.get_ticker_info(symbol)
                    if not ticker_info:
                        return {'success': False, 'error': f"Could not get price for {symbol}"}
                    
                    current_price = float(ticker_info.get('c', [0])[0])
                    if current_price <= 0:
                        return {'success': False, 'error': f"Invalid price for {symbol}"}
                    
                    # Use alert data if available, otherwise use defaults
                    score = alert_data.get('score', 75) if alert_data else 75
                    
                    # Calculate stop loss and take profit (2% stop, 4% target default)
                    stop_loss_pct = 0.02
                    take_profit_pct = 0.04
                    
                    stop_loss = current_price * (1 - stop_loss_pct)
                    take_profit = current_price * (1 + take_profit_pct)
                    
                    # Get position sizing from risk profile
                    try:
                        risk_manager = get_risk_profile_manager()
                        sizing = risk_manager.calculate_position_size(
                            price=current_price,
                            stop_loss=stop_loss,
                            confidence=float(score)
                        )
                        position_value = sizing.get('recommended_value', 100.0)
                        position_pct = sizing.get('position_pct', 5.0)
                        risk_pct = sizing.get('risk_pct', 1.0)
                    except Exception as e:
                        logger.warning(f"Risk profile sizing failed: {e}, using default")
                        position_value = 100.0
                        position_pct = 5.0
                        risk_pct = 1.0
                    
                    # Calculate volume (number of tokens)
                    volume = position_value / current_price if current_price > 0 else 0
                    
                    return {
                        'success': True,
                        'symbol': symbol,
                        'side': 'BUY',
                        'price': current_price,
                        'volume': volume,
                        'position_size': position_value,
                        'position_pct': position_pct,
                        'risk_pct': risk_pct,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'score': score
                    }
                    
                except Exception as e:
                    logger.error(f"Error getting trade params: {e}")
                    return {'success': False, 'error': str(e)}
            
            # Run in thread to avoid blocking
            params = await asyncio.to_thread(_get_trade_params)
            
            if not params.get('success'):
                await message.channel.send(f"❌ **Trade Failed:** {params.get('error', 'Unknown error')}")
                return params
            
            # Format price based on magnitude
            def fmt_price(p):
                if p >= 1000: return f"${p:,.2f}"
                elif p >= 1: return f"${p:.4f}"
                elif p >= 0.01: return f"${p:.6f}"
                else: return f"${p:.8f}"
            
            # Send trade details and wait for approval (mirrors stock workflow)
            sent_msg = await message.channel.send(
                f"🚀 **CRYPTO TRADE QUEUED: {symbol}**\n\n"
                f"📊 **Position Sizing:**\n"
                f"   Tokens: **{params['volume']:,.6f}**\n"
                f"   Value: **${params['position_size']:,.2f}** ({params['position_pct']:.1f}% of portfolio)\n"
                f"   Risk: **{params['risk_pct']:.1f}%**\n\n"
                f"📈 **Trade Details:**\n"
                f"   Side: **{params['side']}**\n"
                f"   Entry: {fmt_price(params['price'])}\n"
                f"   Stop: {fmt_price(params['stop_loss'])}\n"
                f"   Target: {fmt_price(params['take_profit'])}\n"
                f"   Confidence: {params['score']}%\n\n"
                f"**Commands (reply or just type):**\n"
                f"• `YES` - ✅ Confirm trade as shown\n"
                f"• `NO` - ❌ Cancel trade\n"
                f"• `100` or `TOKENS 100` - Change to 100 tokens\n"
                f"• `$500` - Change to $500 position\n"
                f"• `HALF` / `DOUBLE` - Adjust size"
            )
            
            # Create pending trade approval (same as stock workflow)
            approval_id = f"crypto_{symbol.replace('/', '_')}_{int(datetime.now().timestamp())}"
            
            self.pending_approvals[approval_id] = PendingTradeApproval(
                approval_id=approval_id,
                pair=symbol,
                side=params['side'],
                entry_price=params['price'],
                position_size=params['position_size'],
                stop_loss=params['stop_loss'],
                take_profit=params['take_profit'],
                strategy="DISCORD_TRADE",
                confidence=params['score'],
                reasoning=f"Crypto trade: {params['volume']:.6f} tokens @ ${params['position_size']:,.2f}",
                created_time=datetime.now(),
                discord_message_id=str(sent_msg.id)
            )
            
            logger.info(f"📊 Crypto trade queued for approval: {symbol} (approval_id={approval_id})")
            
            return {'success': True, 'queued': True, 'approval_id': approval_id}
            
        except Exception as e:
            logger.error(f"Error in _handle_trade_command: {e}", exc_info=True)
            error_result = {'success': False, 'error': str(e)}
            await message.channel.send(f"❌ **Trade Error:** {str(e)[:100]}")
            return error_result
        finally:
            # Remove from processing set after completion (with delay to prevent rapid re-processing)
            if trade_key in self._processing_trades:
                # Remove after a short delay to prevent immediate re-processing
                async def cleanup():
                    await asyncio.sleep(5)  # Increased delay to prevent rapid re-processing
                    if hasattr(self, '_processing_trades'):
                        self._processing_trades.discard(trade_key)
                        logger.debug(f"🧹 Cleaned up processing key: {trade_key}")
                asyncio.create_task(cleanup())
    
    async def _handle_stock_trade_execution(self, message: discord.Message, symbol: str, side: str = "BUY", paper_mode: bool = False):
        """
        Handle TRADE command - Execute stock trade via broker adapter
        This is the critical link between analysis approval and trade execution
        
        Args:
            message: Discord message
            symbol: Stock symbol to trade
            side: "BUY" or "SELL" 
            paper_mode: If True, use paper trading regardless of config
        """
        import asyncio
        
        # 🛡️ TRADE DEDUPLICATION: Check cooldown to prevent double trades
        if symbol.upper() in self._recent_trade_requests:
            last_trade_time = self._recent_trade_requests[symbol.upper()]
            elapsed = (datetime.now() - last_trade_time).total_seconds()
            if elapsed < self._trade_cooldown_seconds:
                remaining = int(self._trade_cooldown_seconds - elapsed)
                logger.warning(f"⏳ Duplicate stock trade blocked: {symbol} (traded {elapsed:.1f}s ago)")
                await message.channel.send(
                    f"⏳ **Trade Cooldown:** {symbol} was just traded {elapsed:.0f}s ago.\n"
                    f"Wait {remaining}s or check your positions."
                )
                return
        
        # Record this trade request
        self._recent_trade_requests[symbol.upper()] = datetime.now()
        
        # Clean up old entries (older than 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        self._recent_trade_requests = {
            sym: t for sym, t in self._recent_trade_requests.items() 
            if t > cutoff
        }
        
        try:
            from services.ai_stock_position_manager import get_ai_stock_position_manager
            from services.service_orchestrator import get_orchestrator
            
            # Run blocking initialization in thread pool to not block event loop
            def _get_orchestrator_data():
                orch = get_orchestrator()
                orch.refresh_state()
                return orch, orch.get_pending_alerts(asset_type="stock")
            
            orch, pending = await asyncio.to_thread(_get_orchestrator_data)
            
            # Find the alert for this symbol
            alert_info = None
            for alert in pending:
                if alert.symbol.upper() == symbol.upper():
                    alert_info = alert
                    break
            
            if not alert_info:
                await message.channel.send(
                    f"⚠️ No pending analysis found for **{symbol}**.\n"
                    f"Run analysis first: Reply with `1` (Standard), `2` (Multi), or `3` (Ultimate)"
                )
                return
            
            # Get stock position manager (run in thread to avoid blocking)
            stock_manager = await asyncio.to_thread(get_ai_stock_position_manager)
            
            if not stock_manager:
                await message.channel.send(
                    f"❌ Stock Position Manager not available.\n"
                    f"Ensure broker (Tradier/IBKR) is configured in your .env file."
                )
                return
            
            # Determine if paper trading
            use_paper = paper_mode or stock_manager.paper_mode
            mode_str = "📝 PAPER" if use_paper else "💰 LIVE"
            
            # Get position sizing from risk profile (run in thread to avoid blocking)
            try:
                from services.risk_profile_config import get_risk_profile_manager
                
                def _calculate_sizing():
                    risk_manager = get_risk_profile_manager()
                    conf_score = float(alert_info.metadata.get('score', 75)) if alert_info.metadata else 75.0
                    e_price = alert_info.price or 0.0
                    s_loss = alert_info.stop_loss or (e_price * 0.95)  # Default 5% stop
                    
                    sizing = risk_manager.calculate_position_size(
                        price=e_price,
                        stop_loss=s_loss,
                        confidence=conf_score
                    )
                    return sizing, e_price, s_loss, conf_score
                
                sizing, entry_price, stop_loss, confidence_score = await asyncio.to_thread(_calculate_sizing)
                
                recommended_shares = sizing['recommended_shares']
                recommended_value = sizing['recommended_value']
                position_pct = sizing['position_pct']
                risk_pct = sizing['risk_pct']
                
            except Exception as e:
                logger.warning(f"Could not get position sizing: {e}")
                # Fallback to default sizing
                entry_price = alert_info.price or 0.0
                stop_loss = alert_info.stop_loss or (entry_price * 0.95)
                confidence_score = float(alert_info.metadata.get('score', 75)) if alert_info.metadata else 75.0
                recommended_shares = int(stock_manager.default_position_size / alert_info.price) if alert_info.price else 10
                recommended_value = stock_manager.default_position_size
                position_pct = 10.0
                risk_pct = 2.0
            
            # Queue trade for approval with sizing info
            entry_str = f"${entry_price:.2f}" if entry_price else "Market"
            sent_msg = await message.channel.send(
                f"🚀 **{mode_str} TRADE QUEUED: {symbol}**\n\n"
                f"📊 **Position Sizing:**\n"
                f"   Shares: **{recommended_shares:,}**\n"
                f"   Value: **${recommended_value:,.2f}** ({position_pct:.1f}% of portfolio)\n"
                f"   Risk: **{risk_pct:.1f}%**\n\n"
                f"📈 **Trade Details:**\n"
                f"   Side: **{side}**\n"
                f"   Entry: {entry_str}\n"
                f"   Stop: ${stop_loss:.2f}\n"
                f"   Confidence: {alert_info.confidence}\n\n"
                f"**Commands (reply or just type):**\n"
                f"• `YES` - ✅ Confirm trade as shown\n"
                f"• `NO` - ❌ Cancel trade\n"
                f"• `SHARES 50` or `50` - Change to 50 shares\n"
                f"• `$2000` - Change to $2,000 position\n"
                f"• `HALF` / `DOUBLE` - Adjust size"
            )
            
            # Create pending trade approval
            approval_id = f"stock_{symbol}_{int(datetime.now().timestamp())}"
            
            # Store in pending approvals with stock-specific data including sizing
            # Use sent_msg.id so users can reply to the trade queue message
            self.pending_approvals[approval_id] = PendingTradeApproval(
                approval_id=approval_id,
                pair=symbol,
                side=side,
                entry_price=entry_price,
                position_size=recommended_value,
                stop_loss=stop_loss,
                take_profit=alert_info.target or (entry_price * 1.10),  # Default 10% target
                strategy=alert_info.alert_type,
                confidence=confidence_score,
                reasoning=f"AI-sized: {recommended_shares} shares @ ${recommended_value:,.2f} | {alert_info.reasoning or 'Stock opportunity detected'}",
                created_time=datetime.now(),
                discord_message_id=str(sent_msg.id)  # Use the bot's message ID for replies
            )
            
            logger.info(f"📊 Stock trade queued for approval: {symbol} {side} (approval_id={approval_id})")
            
            # Mark the orchestrator alert as in progress (run in thread to avoid blocking)
            await asyncio.to_thread(orch.approve_alert, alert_info.id, False)
            
        except ImportError as e:
            logger.error(f"Import error handling stock trade: {e}")
            await message.channel.send(
                f"❌ Stock trading module not available.\n"
                f"Run: `pip install ib_insync` for IBKR or configure Tradier API."
            )
        except Exception as e:
            logger.error(f"Error handling stock trade execution: {e}", exc_info=True)
            await message.channel.send(f"❌ Error processing trade: {str(e)[:200]}")
    
    async def _list_pending_trades(self, message: discord.Message):
        """List all pending trade approvals with numbers"""
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.is_expired()]
        
        if not pending:
            await message.channel.send("📋 No pending trade approvals.")
            return
        
        # Sort by creation time
        pending.sort(key=lambda x: x[1].created_time)
        
        lines = ["📋 **PENDING TRADE APPROVALS:**\n"]
        for i, (approval_id, approval) in enumerate(pending, 1):
            elapsed = (datetime.now() - approval.created_time).total_seconds() / 60
            lines.append(
                f"**{i}.** {approval.pair} {approval.side} @ ${approval.price_fmt(approval.entry_price)} "
                f"(Conf: {approval.confidence:.0f}%) - ⏱️ {elapsed:.0f}m ago"
            )
        
        lines.append("\n**Commands:**")
        lines.append("• `APPROVE 1` or `1 YES` - Approve specific trade")
        lines.append("• `REJECT 2` or `2 NO` - Reject specific trade")
        lines.append("• `APPROVE BTC/USD` - Approve by symbol")
        lines.append("• `APPROVE ALL` - Approve all pending")
        lines.append("• `ANALYZE BTC` - Run analysis")
        lines.append("• `MULTI BTC` - Run Multi-Config Analysis")
        
        await message.channel.send("\n".join(lines))
    
    async def _show_position_modify_help(self, message: discord.Message, approval: 'PendingTradeApproval'):
        """Show position modification options for a pending trade"""
        current_shares = int(approval.position_size / approval.entry_price) if approval.entry_price else 0
        
        help_text = f"""📊 **Modify Position for {approval.pair}:**

**Current Position:**
   Shares: **{current_shares:,}**
   Value: **${approval.position_size:,.2f}**

**Modification Commands (reply with):**
`SHARES 50` - Set to exactly 50 shares
`$2000` or `VALUE 2000` - Set to $2,000 value
`HALF` - Reduce to 50% ({current_shares // 2:,} shares)
`DOUBLE` - Increase to 200% ({current_shares * 2:,} shares)
`1.5X` - Increase to 150%

**After modifying, reply `YES` to confirm or `NO` to cancel.**
"""
        await message.channel.send(help_text)
    
    async def _handle_standalone_approval(self, message: discord.Message, approve: bool) -> bool:
        """
        Handle standalone YES/NO commands for the most recent pending approval.
        
        Returns True if there was a pending approval to act on, False otherwise.
        """
        logger.debug(f"   📋 _handle_standalone_approval called (approve={approve})")
        logger.debug(f"   📋 All pending_approvals: {list(self.pending_approvals.keys())}")
        
        # Get the most recent pending approval
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.is_expired()]
        
        logger.debug(f"   📋 Valid pending approvals: {[aid for aid, _ in pending]}")
        
        if not pending:
            logger.debug(f"   ⚠️ No pending approvals found!")
            return False
        
        # Sort by created_time to get most recent
        pending.sort(key=lambda x: x[1].created_time, reverse=True)
        approval_id, approval = pending[0]
        
        action_str = "APPROVED" if approve else "REJECTED"
        logger.info(f"📝 Standalone {action_str} command applied to {approval.pair}")
        
        # Use existing reply approval handler
        await self._handle_reply_approval(message, approval_id, approval, approve)
        return True
    
    async def _handle_standalone_position_modification(self, message: discord.Message, content: str, original_content: str) -> bool:
        """
        Handle position modification commands sent as standalone messages (not replies).
        Applies the modification to the most recent pending approval.
        
        Returns True if the message was handled, False otherwise.
        """
        import re
        
        logger.debug(f"   📋 _handle_standalone_position_modification called: content='{content}'")
        logger.debug(f"   📋 All pending_approvals: {list(self.pending_approvals.keys())}")
        
        # Get the most recent pending approval
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.is_expired()]
        
        logger.debug(f"   📋 Valid pending approvals: {[aid for aid, _ in pending]}")
        
        if not pending:
            # No pending approvals - don't handle these commands
            logger.debug(f"   ⚠️ No pending approvals - not handling position modification")
            return False
        
        # Sort by created_time to get most recent
        pending.sort(key=lambda x: x[1].created_time, reverse=True)
        approval_id, approval = pending[0]
        
        # Pattern 1: Just a number (interpreted as shares) - e.g., "50" or "100"
        if content.isdigit():
            new_shares = int(content)
            await self._modify_position_shares(message, approval_id, approval, new_shares)
            logger.info(f"📝 Standalone SHARES command applied to {approval.pair}: {new_shares}")
            return True
        
        # Pattern 2: SHARES 50 or 50 SHARES or SHARE 50
        shares_match = re.match(r'(?:SHARES?\s+)?(\d+)(?:\s+SHARES?)?$', content)
        if shares_match and 'SHARE' in content:
            new_shares = int(shares_match.group(1))
            await self._modify_position_shares(message, approval_id, approval, new_shares)
            logger.info(f"📝 Standalone SHARES command applied to {approval.pair}: {new_shares}")
            return True
        
        # Pattern 3: $2000 or $2,000 or $2000.00 (dollar sign with amount)
        dollar_match = re.match(r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', original_content)
        if dollar_match:
            new_value = float(dollar_match.group(1).replace(',', ''))
            await self._modify_position_value(message, approval_id, approval, new_value)
            logger.info(f"📝 Standalone $ command applied to {approval.pair}: ${new_value:,.2f}")
            return True
        
        # Pattern 4: VALUE 2000 or VAL 2000
        value_match = re.match(r'(?:VALUE?|VAL)\s+(\d+(?:,\d{3})*(?:\.\d{2})?)', content)
        if value_match:
            new_value = float(value_match.group(1).replace(',', ''))
            await self._modify_position_value(message, approval_id, approval, new_value)
            logger.info(f"📝 Standalone VALUE command applied to {approval.pair}: ${new_value:,.2f}")
            return True
        
        # Pattern 5: HALF, DOUBLE, 2X, 0.5X, 1.5X, 3X, etc.
        multiplier_match = re.match(r'(HALF|DOUBLE|(\d+(?:\.\d+)?)\s*X)', content)
        if multiplier_match:
            if multiplier_match.group(1) == 'HALF':
                multiplier = 0.5
            elif multiplier_match.group(1) == 'DOUBLE':
                multiplier = 2.0
            else:
                multiplier = float(multiplier_match.group(2))
            await self._modify_position_multiplier(message, approval_id, approval, multiplier)
            logger.info(f"📝 Standalone multiplier command applied to {approval.pair}: {multiplier}x")
            return True
        
        # Not a position modification command
        return False
    
    async def _modify_position_shares(self, message: discord.Message, approval_id: str, approval: 'PendingTradeApproval', new_shares: int):
        """Modify position to specific number of shares"""
        if new_shares <= 0:
            await message.channel.send("❌ Shares must be greater than 0")
            return
        
        old_shares = int(approval.position_size / approval.entry_price) if approval.entry_price else 0
        new_value = new_shares * approval.entry_price if approval.entry_price else approval.position_size
        
        # Update the approval
        approval.position_size = new_value
        approval.reasoning = f"Modified: {new_shares} shares @ ${new_value:,.2f} | {approval.reasoning.split('|')[-1].strip() if '|' in approval.reasoning else approval.reasoning}"
        
        # Send confirmation and update discord_message_id so replies to THIS message work
        confirm_msg = await message.channel.send(
            f"✅ **Position Modified for {approval.pair}:**\n"
            f"   Shares: {old_shares:,} → **{new_shares:,}**\n"
            f"   Value: ${new_value:,.2f}\n\n"
            f"Reply **YES** to confirm this trade or **NO** to cancel."
        )
        # Update the stored message ID so user can reply to this message
        approval.discord_message_id = str(confirm_msg.id)
        logger.info(f"📊 Position modified: {approval.pair} → {new_shares} shares (${new_value:,.2f})")
    
    async def _modify_position_value(self, message: discord.Message, approval_id: str, approval: 'PendingTradeApproval', new_value: float):
        """Modify position to specific dollar value"""
        if new_value <= 0:
            await message.channel.send("❌ Value must be greater than 0")
            return
        
        old_value = approval.position_size
        new_shares = int(new_value / approval.entry_price) if approval.entry_price else 0
        
        # Update the approval
        approval.position_size = new_value
        approval.reasoning = f"Modified: {new_shares} shares @ ${new_value:,.2f} | {approval.reasoning.split('|')[-1].strip() if '|' in approval.reasoning else approval.reasoning}"
        
        # Send confirmation and update discord_message_id so replies to THIS message work
        confirm_msg = await message.channel.send(
            f"✅ **Position Modified for {approval.pair}:**\n"
            f"   Value: ${old_value:,.2f} → **${new_value:,.2f}**\n"
            f"   Shares: **{new_shares:,}**\n\n"
            f"Reply **YES** to confirm this trade or **NO** to cancel."
        )
        # Update the stored message ID so user can reply to this message
        approval.discord_message_id = str(confirm_msg.id)
        logger.info(f"📊 Position modified: {approval.pair} → ${new_value:,.2f} ({new_shares} shares)")
    
    async def _modify_position_multiplier(self, message: discord.Message, approval_id: str, approval: 'PendingTradeApproval', multiplier: float):
        """Modify position by multiplier (HALF, DOUBLE, etc.)"""
        old_value = approval.position_size
        new_value = old_value * multiplier
        new_shares = int(new_value / approval.entry_price) if approval.entry_price else 0
        
        multiplier_str = {0.5: "HALF", 2.0: "DOUBLE", 1.5: "1.5X"}.get(multiplier, f"{multiplier}X")
        
        # Update the approval
        approval.position_size = new_value
        approval.reasoning = f"Modified ({multiplier_str}): {new_shares} shares @ ${new_value:,.2f} | {approval.reasoning.split('|')[-1].strip() if '|' in approval.reasoning else approval.reasoning}"
        
        # Send confirmation and update discord_message_id so replies to THIS message work
        confirm_msg = await message.channel.send(
            f"✅ **Position Modified ({multiplier_str}) for {approval.pair}:**\n"
            f"   Value: ${old_value:,.2f} → **${new_value:,.2f}**\n"
            f"   Shares: **{new_shares:,}**\n\n"
            f"Reply **YES** to confirm this trade or **NO** to cancel."
        )
        # Update the stored message ID so user can reply to this message
        approval.discord_message_id = str(confirm_msg.id)
        logger.info(f"📊 Position modified ({multiplier_str}): {approval.pair} → ${new_value:,.2f} ({new_shares} shares)")
    
    async def _handle_specific_approval(self, message: discord.Message, identifier: str, approve: bool):
        """Handle approval for a specific trade by number or symbol"""
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.is_expired()]
        
        if not pending:
            await message.channel.send("❌ No pending trade approvals.")
            return
        
        # Sort by creation time for consistent numbering
        pending.sort(key=lambda x: x[1].created_time)
        
        target_approval = None
        
        # Check if identifier is a number
        if identifier.isdigit():
            idx = int(identifier) - 1  # Convert to 0-based
            if 0 <= idx < len(pending):
                target_approval = pending[idx][1]
            else:
                await message.channel.send(f"❌ Invalid trade number: {identifier}. Use `LIST` to see pending trades.")
                return
        else:
            # Search by symbol (partial match)
            identifier_clean = identifier.replace('/', '')
            for _, approval in pending:
                if identifier_clean in approval.pair.replace('/', '').upper():
                    target_approval = approval
                    break
            
            if not target_approval:
                await message.channel.send(f"❌ No pending trade found for: {identifier}. Use `LIST` to see pending trades.")
                return
        
        # Process the approval
        if approve:
            target_approval.approved = True
            await message.channel.send(
                f"✅ **APPROVED:** {target_approval.pair} {target_approval.side} trade\n"
                f"Executing trade now..."
            )
            logger.info(f"✅ User approved specific trade: {target_approval.approval_id}")
        else:
            target_approval.rejected = True
            await message.channel.send(
                f"❌ **REJECTED:** {target_approval.pair} {target_approval.side} trade\n"
                f"Trade cancelled."
            )
            logger.info(f"❌ User rejected specific trade: {target_approval.approval_id}")
        
        # Call approval callback in thread to prevent blocking
        if self.approval_callback:
            try:
                import asyncio
                await asyncio.to_thread(self.approval_callback, target_approval.approval_id, approve)
                logger.info(f"✅ Approval callback completed for {target_approval.approval_id}")
            except Exception as e:
                logger.error(f"Error in approval callback: {e}", exc_info=True)
    
    async def _handle_reply_approval(
        self, 
        message: discord.Message, 
        approval_id: str, 
        approval: 'PendingTradeApproval', 
        approve: bool
    ):
        """
        Handle approval/rejection when user replies directly to an approval message
        This is the most intuitive way to approve a specific trade
        """
        # Check if already processed
        if approval.approved or approval.rejected:
            await message.channel.send(
                f"⚠️ This trade ({approval.pair}) has already been {'approved' if approval.approved else 'rejected'}."
            )
            return
        
        # Check if expired
        if approval.is_expired():
            await message.channel.send(
                f"⏰ This approval request for {approval.pair} has **expired**. "
                f"The trade opportunity may no longer be valid."
            )
            return
        
        # Process the approval
        if approve:
            approval.approved = True
            # Mark message ID to prevent duplicate processing
            self._processed_messages.add(message.id)
            
            # Reply to the original message for clear context
            await message.reply(
                f"✅ **APPROVED:** {approval.pair} {approval.side}\n"
                f"🚀 Executing trade now..."
            )
            logger.info(f"✅ User approved trade via reply: {approval_id} ({approval.pair})")
            
            # Check if this is a position management action (CLOSE_NOW, TIGHTEN_STOP, etc.)
            # These should use the callback to go through the AI position manager
            is_position_action = approval.side.upper() in ['CLOSE_NOW', 'TIGHTEN_STOP', 'EXTEND_TARGET', 'TAKE_PARTIAL', 'MOVE_TO_BREAKEVEN']
            
            if is_position_action:
                # Position management actions must go through the approval callback
                # to properly close/modify existing positions via AI Position Manager
                logger.info(f"🔄 Position action detected ({approval.side}) - routing through AI Position Manager callback")
                if self.approval_callback:
                    try:
                        import asyncio
                        await asyncio.to_thread(self.approval_callback, approval_id, approve)
                        # Remove from pending after callback
                        if approval_id in self.pending_approvals:
                            del self.pending_approvals[approval_id]
                            logger.info(f"🗑️ Removed {approval_id} from pending approvals after position action")
                    except Exception as e:
                        logger.error(f"Error in position action callback: {e}", exc_info=True)
                        await message.channel.send(f"⚠️ Error processing position action: {str(e)[:100]}")
                else:
                    logger.error("❌ No approval callback configured for position action!")
                    await message.channel.send(f"❌ **Error:** Position manager callback not configured. Please use buttons or restart the crypto AI trader service.")
            elif "/" in approval.pair:
                # New crypto trade (BUY/SELL entry) - execute directly via Kraken
                await self._execute_crypto_trade(message, approval)
            else:
                # Stock trade - use callback
                if self.approval_callback:
                    try:
                        import asyncio
                        await asyncio.to_thread(self.approval_callback, approval_id, approve)
                        # Remove from pending after callback
                        if approval_id in self.pending_approvals:
                            del self.pending_approvals[approval_id]
                            logger.info(f"🗑️ Removed {approval_id} from pending approvals after stock execution")
                    except Exception as e:
                        logger.error(f"Error in approval callback: {e}", exc_info=True)
                        await message.channel.send(f"⚠️ Error processing trade: {str(e)[:100]}")
        else:
            approval.rejected = True
            await message.reply(
                f"❌ **REJECTED:** {approval.pair} {approval.side}\n"
                f"Trade cancelled."
            )
            logger.info(f"❌ User rejected trade via reply: {approval_id} ({approval.pair})")
    
    async def _execute_crypto_trade(self, message: discord.Message, approval: 'PendingTradeApproval'):
        """
        Execute an approved NEW crypto trade via Kraken and add to AI Position Manager.
        
        NOTE: This function is for NEW trade entries (BUY/SELL) only.
        Position management actions (CLOSE_NOW, TIGHTEN_STOP, etc.) should be routed
        through the approval_callback to use the AI Position Manager's execute_decision.
        """
        import asyncio
        import uuid
        
        # Safety check: Reject position management actions
        position_actions = ['CLOSE_NOW', 'TIGHTEN_STOP', 'EXTEND_TARGET', 'TAKE_PARTIAL', 'MOVE_TO_BREAKEVEN', 'HOLD']
        if approval.side.upper() in position_actions:
            error_msg = (
                f"❌ **Error:** Cannot execute position action '{approval.side}' as a new trade. "
                f"This appears to be a position management request that was incorrectly routed. "
                f"Please restart the Crypto AI Trader service and try again using the Approve button."
            )
            logger.error(f"_execute_crypto_trade called with position action: {approval.side}")
            await message.channel.send(error_msg)
            return
        
        def _do_execute():
            try:
                from services.ai_crypto_position_manager import get_ai_crypto_position_manager
                from clients.kraken_client import OrderSide, OrderType
                
                manager = get_ai_crypto_position_manager()
                if not manager:
                    return {'success': False, 'error': 'Crypto Position Manager not available'}
                
                if not manager.kraken_client:
                    return {'success': False, 'error': 'Kraken client not available'}
                
                # Generate unique trade ID
                trade_id = f"discord_{approval.pair.replace('/', '_')}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                
                # Calculate volume from position size and price
                volume = approval.position_size / approval.entry_price if approval.entry_price > 0 else 0
                
                if volume <= 0:
                    return {'success': False, 'error': 'Invalid volume calculated'}
                
                # Convert side string to OrderSide enum - only BUY/SELL should reach here
                if approval.side.upper() not in ['BUY', 'SELL']:
                    return {'success': False, 'error': f"Invalid order side: {approval.side}. Expected BUY or SELL."}
                
                side_enum = OrderSide.BUY if approval.side.upper() == 'BUY' else OrderSide.SELL
                
                # Place order with Kraken first (like stock execution does)
                logger.info(f"🚀 Placing {approval.side} order for {approval.pair}: {volume:.6f} @ ${approval.entry_price:,.4f}")
                executed_order = manager.kraken_client.place_order(
                    pair=approval.pair,
                    side=side_enum,
                    order_type=OrderType.MARKET,
                    volume=volume,
                    price=None,  # Market order
                    stop_loss=approval.stop_loss,
                    take_profit=approval.take_profit
                )
                
                if not executed_order:
                    return {'success': False, 'error': 'Failed to place order with Kraken'}
                
                order_id = executed_order.order_id if hasattr(executed_order, 'order_id') else None
                if not order_id:
                    return {'success': False, 'error': 'Order placed but no order ID returned'}
                
                logger.info(f"✅ Order placed successfully: {order_id} for {approval.pair}")
                
                # Get actual execution price from order if available
                actual_price = executed_order.avg_price if hasattr(executed_order, 'avg_price') and executed_order.avg_price > 0 else approval.entry_price
                
                # Now add position to monitoring with the order ID
                success = manager.add_position(
                    trade_id=trade_id,
                    pair=approval.pair,
                    side=approval.side,
                    volume=volume,
                    entry_price=actual_price,
                    stop_loss=approval.stop_loss,
                    take_profit=approval.take_profit,
                    strategy="DISCORD_TRADE",
                    trailing_stop_pct=2.0,
                    breakeven_trigger_pct=3.0,
                    entry_order_id=order_id
                )
                
                if success:
                    return {
                        'success': True,
                        'trade_id': trade_id,
                        'order_id': order_id,
                        'volume': volume,
                        'price': actual_price,
                        'position_size': approval.position_size
                    }
                else:
                    return {'success': False, 'error': 'Order placed but failed to add position to manager'}
                    
            except Exception as e:
                logger.error(f"Error executing crypto trade: {e}", exc_info=True)
                return {'success': False, 'error': str(e)}
        
        result = await asyncio.to_thread(_do_execute)
        
        # Mark approval as executed to prevent duplicate processing
        approval.executed = True
        
        # Remove from pending approvals to prevent duplicate messages
        if approval.approval_id in self.pending_approvals:
            del self.pending_approvals[approval.approval_id]
            logger.info(f"🗑️ Removed {approval.approval_id} from pending approvals after execution")
        
        if result.get('success'):
            execution_msg = (
                f"✅ **Crypto Trade Executed:** {approval.pair}\n"
                f"**Side:** {approval.side} | **Price:** ${result['price']:,.4f}\n"
                f"**Volume:** {result['volume']:.6f} | **Value:** ${result['position_size']:,.2f}\n"
                f"**Stop:** ${approval.stop_loss:,.4f} | **Target:** ${approval.take_profit:,.4f}\n"
                f"**Order ID:** `{result.get('order_id', 'N/A')}`\n\n"
                f"🤖 _AI Position Manager is now monitoring this trade_"
            )
            # Send to the channel where the command was issued
            await message.channel.send(execution_msg)
            
            # Also send to dedicated crypto executions channel if different
            try:
                from src.integrations.discord_channels import get_channel_id_for_category, AlertCategory
                exec_channel_id = get_channel_id_for_category(AlertCategory.CRYPTO_EXECUTIONS)
                if exec_channel_id and exec_channel_id != message.channel.id:
                    exec_channel = self.get_channel(exec_channel_id)
                    if exec_channel:
                        await exec_channel.send(f"📊 **Trade Confirmation**\n{execution_msg}")
                        logger.info(f"✅ Sent execution confirmation to #crypto-executions")
            except Exception as e:
                logger.warning(f"Could not send to executions channel: {e}")
        else:
            await message.channel.send(f"❌ **Trade Execution Failed:** {result.get('error', 'Unknown error')}")

    async def _handle_approve_all(self, message: discord.Message, approve: bool):
        """Approve or reject all pending trades"""
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.is_expired()]
        
        if not pending:
            await message.channel.send("❌ No pending trade approvals.")
            return
        
        action = "APPROVED" if approve else "REJECTED"
        count = 0
        
        import asyncio
        
        for approval_id, approval in pending:
            if approve:
                approval.approved = True
            else:
                approval.rejected = True
            count += 1
            
            # Call approval callback for each in thread to prevent blocking
            if self.approval_callback:
                try:
                    await asyncio.to_thread(self.approval_callback, approval.approval_id, approve)
                    logger.info(f"✅ Approval callback completed for {approval.approval_id}")
                except Exception as e:
                    logger.error(f"Error in approval callback: {e}", exc_info=True)
        
        await message.channel.send(f"{'✅' if approve else '❌'} **{action} {count} trade(s)**")
        logger.info(f"{'✅' if approve else '❌'} User {action.lower()} all {count} pending trades")
    
    async def _handle_approval(self, message: discord.Message, approve: bool):
        """Handle user approval/rejection (most recent trade)"""
        # Check if there's a pending approval (most recent)
        if not self.pending_approvals:
            # Only send message if this isn't a duplicate check after execution
            if message.id not in self._processed_messages:
                await message.channel.send("❌ No pending trade approvals.")
            return
        
        # Get all pending (not approved/rejected/executed)
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.executed and not a.is_expired()]
        
        if not pending:
            # Only send message if this isn't a duplicate check after execution
            if message.id not in self._processed_messages:
                await message.channel.send("❌ No pending approvals found.")
            return
        
        # If multiple pending, show list and ask for specific selection
        if len(pending) > 1:
            action_word = "APPROVE" if approve else "REJECT"
            await message.channel.send(
                f"⚠️ **{len(pending)} trades pending!** Please specify which one:\n"
                f"• `{action_word} 1` - {action_word.capitalize()} first trade\n"
                f"• `{action_word} BTC/USD` - {action_word.capitalize()} by symbol\n"
                f"• `{action_word} ALL` - {action_word.capitalize()} all\n"
                f"• `LIST` - Show all pending trades"
            )
            return
        
        # Only one pending - approve/reject it
        latest_approval = pending[0][1]
        approval_id = latest_approval.approval_id
        
        # Check if already processed
        if latest_approval.approved or latest_approval.rejected or latest_approval.executed:
            await message.channel.send(
                f"⚠️ This trade ({latest_approval.pair}) has already been processed."
            )
            return
        
        # Mark as approved/rejected
        if approve:
            latest_approval.approved = True
            # Mark message ID to prevent duplicate processing
            self._processed_messages.add(message.id)
            
            await message.channel.send(
                f"✅ **APPROVED:** {latest_approval.pair} {latest_approval.side} trade\n"
                f"Executing trade now..."
            )
            logger.info(f"✅ User approved trade: {approval_id}")
            
            # Check if this is a crypto trade (contains /) and execute it directly
            if "/" in latest_approval.pair:
                await self._execute_crypto_trade(message, latest_approval)
            else:
                # Stock trade - use callback in thread to prevent blocking
                if self.approval_callback:
                    try:
                        import asyncio
                        await asyncio.to_thread(self.approval_callback, approval_id, approve)
                        logger.info(f"✅ Stock approval callback completed for {approval_id}")
                        # Remove from pending after callback
                        if approval_id in self.pending_approvals:
                            del self.pending_approvals[approval_id]
                            logger.info(f"🗑️ Removed {approval_id} from pending approvals after stock execution")
                    except Exception as e:
                        logger.error(f"Error in approval callback: {e}", exc_info=True)
                        await message.channel.send(f"⚠️ Callback error: {str(e)[:100]}")
        else:
            latest_approval.rejected = True
            await message.channel.send(
                f"❌ **REJECTED:** {latest_approval.pair} {latest_approval.side} trade\n"
                f"Trade cancelled."
            )
            logger.info(f"❌ User rejected trade: {approval_id}")
            # Remove rejected approval
            if approval_id in self.pending_approvals:
                del self.pending_approvals[approval_id]
    
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
            if not self.bot_ready:
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
            
            # Instructions - emphasize reply feature
            embed.add_field(
                name="⏰ How to Approve/Reject",
                value=(
                    "**Option 1: Use Buttons (Preferred)**\n"
                    "Click ✅ Approve or ❌ Reject below\n\n"
                    "**Option 2: Reply to THIS message**\n"
                    "↩️ Reply with `YES` or `NO`"
                ),
                inline=False
            )
            
            embed.set_footer(text=f"Approval ID: {approval_id}")
            
            # Add Interactive Buttons
            view = TradeActionView(self, approval_id)
            
            # Send message
            message = await channel.send(embed=embed, view=view)
            
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
    
    async def send_alert_notification(
        self,
        symbol: str,
        alert_type: str,
        message_text: str,
        confidence: str = "MEDIUM",
        color: int = 3447003,
        asset_type: str = "crypto",
        alert_data: dict = None,
        target_channel_id: int = None
    ) -> bool:
        """
        Send a generic alert notification with action buttons
        
        Args:
            symbol: Trading symbol (e.g., BTC/USD, GME)
            alert_type: Type of alert (BREAKOUT, BUZZING, etc.)
            message_text: Alert message content
            confidence: Confidence level (HIGH, MEDIUM, LOW)
            color: Embed color
            asset_type: 'crypto' or 'stock' - determines available buttons
            alert_data: Optional dict with price, score, etc. for trade execution
            target_channel_id: Optional specific channel ID to send to (for channel routing)
        """
        try:
            if not self.bot_ready:
                return False
            
            # Use target channel if specified, otherwise fall back to default
            channel_id = target_channel_id if target_channel_id else self.channel_id
            channel = self.get_channel(channel_id)
            if not channel:
                logger.warning(f"Could not find channel {channel_id}, falling back to default {self.channel_id}")
                channel = self.get_channel(self.channel_id)
                if not channel:
                    return False
            
            # Build alert data from parameters if not provided
            if alert_data is None:
                alert_data = {
                    'symbol': symbol,
                    'alert_type': alert_type,
                    'confidence': confidence
                }
                
            embed = discord.Embed(
                title=f"🔔 {alert_type}: {symbol}",
                description=message_text,
                color=color,
                timestamp=datetime.now()
            )
            
            # Pass asset_type and alert_data to view for Trade button functionality
            view = AlertActionView(self, symbol, asset_type=asset_type, alert_data=alert_data)
            await channel.send(embed=embed, view=view)
            logger.debug(f"Sent alert to channel {channel.name} (ID: {channel_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
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
                        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self.loop = loop
                    
                    # Run bot
                    logger.info("Running bot.start()...")
                    if self.token and self.bot:
                        loop.run_until_complete(self.bot.start(self.token))
                    
                except Exception as e:
                    logger.error(f"Discord bot thread crashed: {e}", exc_info=True)
            
            self.thread = threading.Thread(target=run_bot, daemon=True)
            self.thread.start()
            
            # Wait a bit for bot to be ready
            time.sleep(3)
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Discord bot: {e}", exc_info=True)
            return False
    
    def stop(self):
        """Stop Discord bot"""
        if self.loop and self.bot:
            asyncio.run_coroutine_threadsafe(self.bot.close(), self.loop)
    
    def send_approval_request(self, **kwargs) -> bool:
        """Send trade approval request (thread-safe)"""
        if not self.enabled or not self.loop or not self.bot:
            return False
            
        future = asyncio.run_coroutine_threadsafe(
            self.bot.send_approval_request(**kwargs), 
            self.loop
        )
        try:
            return future.result(timeout=5)
        except Exception as e:
            logger.error(f"Error sending approval request: {e}")
            return False
            
    def send_notification(self, message: str, color: int = 3447003) -> bool:
        """Send generic notification to approval channel"""
        if not self.enabled or not self.loop or not self.bot:
            return False
        
        async def _send():
            try:
                if self.bot and hasattr(self.bot, 'channel_id'):
                    channel = self.bot.get_channel(self.bot.channel_id)
                    if channel:
                        embed = discord.Embed(description=message, color=color)
                        await channel.send(embed=embed)
                        return True
                return False
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
                return False
                
        future = asyncio.run_coroutine_threadsafe(_send(), self.loop)
        try:
            return future.result(timeout=5)
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False

    def is_running(self) -> bool:
        """Check if bot is running and connected"""
        return bool(self.bot and self.bot.bot_ready)
    
    def get_pending_approvals(self) -> Dict[str, 'PendingTradeApproval']:
        """Get all pending trade approvals (thread-safe)"""
        try:
            if not self.enabled or not self.bot:
                return {}
            
            # Verify bot has the attribute and it's valid
            if not hasattr(self.bot, 'pending_approvals'):
                logger.warning("Discord bot missing pending_approvals attribute")
                return {}
            
            # Direct access to bot's pending_approvals dict (read-only access is safe)
            return self.bot.pending_approvals.copy() if self.bot.pending_approvals else {}
        except Exception as e:
            logger.error(f"Error getting pending approvals: {e}")
            return {}
    
    def cleanup_expired(self) -> int:
        """Remove expired approval requests (thread-safe)"""
        if not self.enabled or not self.bot:
            return 0
        
        try:
            # Direct call to bot's cleanup_expired (it's a regular method, not async)
            return self.bot.cleanup_expired()
        except Exception as e:
            logger.error(f"Error cleaning up expired approvals: {e}")
            return 0

# Global instance
_approval_manager: Optional[DiscordApprovalManager] = None

def get_discord_approval_manager(approval_callback: Optional[Callable] = None, force_new: bool = False) -> Optional[DiscordApprovalManager]:
    """
    Get or create global Discord approval manager
    
    Args:
        approval_callback: Callback function when trade is approved/rejected
        force_new: If True, create a new instance even if one exists (for reset scenarios)
    """
    global _approval_manager
    
    # Force reset if requested or if existing manager is in bad state
    if force_new or (_approval_manager is not None and not hasattr(_approval_manager, 'get_pending_approvals')):
        logger.warning("Resetting Discord approval manager (invalid state or force requested)")
        _approval_manager = None
    
    if _approval_manager is None:
        _approval_manager = DiscordApprovalManager(approval_callback=approval_callback)
        
        # Auto-start if created
        if _approval_manager.enabled:
            _approval_manager.start()
            
    return _approval_manager

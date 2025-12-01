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
from datetime import datetime
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict
import json
import threading


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
    """Interactive buttons for General Alerts (Watch/Analyze/Dismiss)"""
    def __init__(self, bot, symbol: str, asset_type: str = "crypto"):
        super().__init__(timeout=7200)  # 2 hour timeout
        self.bot = bot
        self.symbol = symbol
        self.asset_type = asset_type

    @discord.ui.button(label="👀 Watch", style=discord.ButtonStyle.primary, custom_id="btn_watch")
    async def watch_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.send_message(f"👀 Adding **{self.symbol}** to watchlist...", ephemeral=True)
        await self.bot._handle_watch_command(interaction.message, self.symbol)

    @discord.ui.button(label="🔍 Analyze", style=discord.ButtonStyle.secondary, custom_id="btn_analyze")
    async def analyze_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.send_message(f"🔍 Queuing analysis for **{self.symbol}**...", ephemeral=True)
        await self.bot._handle_analyze_command(interaction.message, self.symbol)

    @discord.ui.button(label="🗑️ Dismiss", style=discord.ButtonStyle.gray, custom_id="btn_dismiss")
    async def dismiss_button(self, interaction: discord.Interaction, button: Button):
        await interaction.response.send_message(f"🗑️ Dismissing alert for **{self.symbol}**...", ephemeral=True)
        await self.bot._handle_dismiss_command(interaction.message, self.symbol)


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
        self.bot_ready = False  # Renamed from is_ready to avoid shadowing inherited method
        
        logger.info(f"🤖 Discord Trade Approval Bot initialized")
        logger.info(f"   Channel ID: {channel_id}")
    
    async def on_ready(self):
        """Called when bot is ready"""
        self.bot_ready = True
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
        
        # 🔔 CHECK IF THIS IS A REPLY TO A SPECIFIC APPROVAL MESSAGE
        if message.reference and message.reference.message_id:
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
                    elif content in ['?', 'HELP', 'H']:
                        # Show help for trade approvals
                        help_text = f"""📋 **Trade Approval Commands for {approval.pair}:**

**Approve/Reject:**
`APPROVE` or `YES` - ✅ Execute the trade
`REJECT` or `NO` - ❌ Cancel the trade

**Position Sizing:**
`SIZE` or `SIZING` - 📊 Recalculate position size
`RISK` - 📊 Show current risk profile

**Other:**
`?` or `HELP` - Show this help
"""
                        await message.channel.send(help_text)
                        return
                    else:
                        # Reply to the approval message but unclear intent
                        await message.channel.send(
                            f"❓ Reply with **APPROVE** or **REJECT** to confirm.\n"
                            f"Or use `RISK` to see profile, `SIZE` for sizing, `?` for help."
                        )
                        return
            
            # 2. Check Orchestrator Alert Queue (Generic Alerts)
            await self._handle_alert_reply(message, content)
            return
            
        # Check for LIST command to show pending trades
        if content in ['LIST', 'PENDING', 'TRADES']:
            await self._list_pending_trades(message)
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
        
        # Check for specific trade approval: "APPROVE 1" or "APPROVE BTC" or "1 APPROVE"
        
        # Generic Alert Commands: WATCH, ANALYZE, DISMISS, MULTI, ULTIMATE
        # Pattern: CMD SYMBOL (e.g., "WATCH BTC", "ANALYZE AAPL")
        alert_match = re.match(r'(WATCH|ANALYZE|DISMISS|REMOVE|MULTI|ULTIMATE)\s+([A-Z0-9/]+)', content)
        if alert_match:
            action = alert_match.group(1)
            symbol = alert_match.group(2).replace('/', '').upper()
            
            if action == 'WATCH':
                await self._handle_watch_command(message, symbol)
            elif action == 'ANALYZE':
                await self._handle_analyze_command(message, symbol, mode="standard")
            elif action == 'MULTI':
                await self._handle_analyze_command(message, symbol, mode="multi")
            elif action == 'ULTIMATE':
                await self._handle_analyze_command(message, symbol, mode="ultimate")
            elif action in ['DISMISS', 'REMOVE']:
                await self._handle_dismiss_command(message, symbol)
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

        # Call approval callback
        if self.approval_callback:
            try:
                self.approval_callback(approval_id, approve)
            except Exception as e:
                logger.error(f"Error in approval callback: {e}", exc_info=True)

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
        if ref_msg.embeds:
            for embed in ref_msg.embeds:
                if embed.title and ":" in embed.title:
                    # Try extracting after colon
                    parts = embed.title.split(":")
                    if len(parts) > 1:
                        possible_symbol = parts[-1].strip().split(' ')[0]
                        target_symbol = possible_symbol
                        break
        
        if not target_symbol:
            # Try content
            if ":" in ref_msg.content:
                 parts = ref_msg.content.split(":")
                 if len(parts) > 1:
                     target_symbol = parts[-1].strip().split(' ')[0]

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
        
        # STOCK TRADE EXECUTION COMMANDS (after analysis approval)
        elif content in ['T', 'TRADE', 'EXECUTE', 'BUY', 'ENTER']:
            await self._handle_stock_trade_execution(message, target_symbol, side="BUY")
        
        elif content in ['SHORT', 'SELL', 'S-TRADE']:
            await self._handle_stock_trade_execution(message, target_symbol, side="SELL")
        
        elif content in ['P', 'PAPER', 'PAPER-TRADE', 'TEST']:
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
            # Show available commands
            help_text = f"""📋 **Commands for {target_symbol}:**

**Watchlist:**
`W` or `WATCH` - Add to watchlist

**Analysis (reply with):**
`1` or `S` - 🔬 Standard (single strategy)
`2` or `M` - 🎯 Multi (Long/Short + timeframes)
`3` or `U` - 🚀 Ultimate (ALL combinations)

**Trade Execution (after analysis):**
`T` or `TRADE` - Execute BUY trade
`SHORT` - Execute SHORT/SELL trade
`P` or `PAPER` - Paper trade (test mode)

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
        """Handle WATCH command"""
        # Add to watchlist via orchestrator helper (handles both crypto and stock)
        try:
            if "/" in symbol:
                # Crypto
                from services.crypto_watchlist_manager import CryptoWatchlistManager
                wm = CryptoWatchlistManager()
                if wm.add_crypto(symbol):
                    await message.channel.send(f"✅ **{symbol}** added to Crypto Watchlist")
                else:
                    await message.channel.send(f"⚠️ Failed to add {symbol} (duplicate?)")
            else:
                # Stock or Crypto without /
                from services.ticker_manager import TickerManager
                tm = TickerManager()
                tm.add_ticker(symbol)
                await message.channel.send(f"✅ **{symbol}** added to Watchlist (TickerManager)")
                
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
            from ui.risk_profile_ui import get_broker_status
            
            status = get_broker_status()
            
            # Check which brokers are configured
            tradier_configured = bool(
                os.getenv('TRADIER_PAPER_ACCESS_TOKEN') or 
                os.getenv('TRADIER_ACCESS_TOKEN') or
                os.getenv('TRADIER_PROD_ACCESS_TOKEN')
            )
            ibkr_configured = bool(os.getenv('IBKR_PAPER_PORT'))
            
            broker_text = f"""🏦 **Broker Configuration**

**Current Broker:** {status['broker_type']} ({'📝 PAPER' if status['paper_mode'] else '💰 LIVE'})
**Status:** {'✅ Connected' if status['connected'] else '❌ Not Connected'}

**Available Brokers:**
{'✅' if tradier_configured else '❌'} **TRADIER** - {'Configured' if tradier_configured else 'Not configured'}
{'✅' if ibkr_configured else '❌'} **IBKR** - {'Configured' if ibkr_configured else 'Not configured'}

**Switch Broker:**
Type `BROKER TRADIER` or `BROKER IBKR` to switch

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
                        f"⚠️ Switched to IBKR but connection failed:\n{status.get('error', 'Unknown error')}\n\n"
                        f"Make sure TWS/Gateway is running on port {os.getenv('IBKR_PAPER_PORT', '7497')}"
                    )
            else:
                await message.channel.send(f"❌ Unknown broker: {broker}. Use `TRADIER` or `IBKR`")
                
        except Exception as e:
            logger.error(f"Error switching broker: {e}")
            await message.channel.send(f"❌ Error switching broker: {str(e)}")
    
    async def _show_global_help(self, message: discord.Message):
        """Show global help for all available commands"""
        help_text = """📋 **Sentient Trader Discord Commands**

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
`WATCH AAPL` - Add symbol to watchlist
`ANALYZE NVDA` - Run standard analysis

**Reply to Alerts with:**
`1` / `2` / `3` - Standard / Multi / Ultimate analysis
`T` or `TRADE` - Execute long trade
`SHORT` - Execute short trade
`APPROVE` / `REJECT` - For trade approvals
`?` - Show alert-specific help

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
        try:
            from services.ai_stock_position_manager import get_ai_stock_position_manager
            from services.service_orchestrator import get_orchestrator
            
            # Get pending alert info for this symbol
            orch = get_orchestrator()
            orch.refresh_state()
            pending = orch.get_pending_alerts(asset_type="stock")
            
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
            
            # Get stock position manager
            stock_manager = get_ai_stock_position_manager()
            
            if not stock_manager:
                await message.channel.send(
                    f"❌ Stock Position Manager not available.\n"
                    f"Ensure broker (Tradier/IBKR) is configured in your .env file."
                )
                return
            
            # Determine if paper trading
            use_paper = paper_mode or stock_manager.paper_mode
            mode_str = "📝 PAPER" if use_paper else "💰 LIVE"
            
            # Get position sizing from risk profile
            try:
                from services.risk_profile_config import get_risk_profile_manager
                risk_manager = get_risk_profile_manager()
                
                # Calculate position size
                confidence_score = float(alert_info.metadata.get('score', 75)) if alert_info.metadata else 75.0
                entry_price = alert_info.price or 0.0
                stop_loss = alert_info.stop_loss or (entry_price * 0.95)  # Default 5% stop
                
                sizing = risk_manager.calculate_position_size(
                    price=entry_price,
                    stop_loss=stop_loss,
                    confidence=confidence_score
                )
                
                recommended_shares = sizing['recommended_shares']
                recommended_value = sizing['recommended_value']
                position_pct = sizing['position_pct']
                risk_pct = sizing['risk_pct']
                
            except Exception as e:
                logger.warning(f"Could not get position sizing: {e}")
                # Fallback to default sizing
                recommended_shares = int(stock_manager.default_position_size / alert_info.price) if alert_info.price else 10
                recommended_value = stock_manager.default_position_size
                position_pct = 10.0
                risk_pct = 2.0
            
            # Queue trade for approval with sizing info
            await message.channel.send(
                f"🚀 **{mode_str} TRADE QUEUED: {symbol}**\n\n"
                f"📊 **Position Sizing:**\n"
                f"   Shares: **{recommended_shares:,}**\n"
                f"   Value: **${recommended_value:,.2f}** ({position_pct:.1f}% of portfolio)\n"
                f"   Risk: **{risk_pct:.1f}%**\n\n"
                f"📈 **Trade Details:**\n"
                f"   Side: **{side}**\n"
                f"   Entry: ${entry_price:.2f if entry_price else 'Market'}\n"
                f"   Stop: ${stop_loss:.2f}\n"
                f"   Confidence: {alert_info.confidence}\n\n"
                f"Reply **YES** to confirm or **NO** to cancel."
            )
            
            # Create pending trade approval
            approval_id = f"stock_{symbol}_{int(datetime.now().timestamp())}"
            
            # Store in pending approvals with stock-specific data including sizing
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
                discord_message_id=str(message.id)
            )
            
            # Mark the orchestrator alert as in progress
            orch.approve_alert(alert_info.id, add_to_watchlist=False)
            
            logger.info(f"📊 Stock trade queued for approval: {symbol} {side} (approval_id={approval_id})")
            
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
        
        # Call approval callback
        if self.approval_callback:
            try:
                self.approval_callback(target_approval.approval_id, approve)
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
            # Reply to the original message for clear context
            await message.reply(
                f"✅ **APPROVED:** {approval.pair} {approval.side}\n"
                f"🚀 Executing trade now..."
            )
            logger.info(f"✅ User approved trade via reply: {approval_id} ({approval.pair})")
        else:
            approval.rejected = True
            await message.reply(
                f"❌ **REJECTED:** {approval.pair} {approval.side}\n"
                f"Trade cancelled."
            )
            logger.info(f"❌ User rejected trade via reply: {approval_id} ({approval.pair})")
        
        # Call approval callback to execute/cancel the trade
        if self.approval_callback:
            try:
                self.approval_callback(approval_id, approve)
            except Exception as e:
                logger.error(f"Error in approval callback: {e}", exc_info=True)
                await message.channel.send(f"⚠️ Error processing trade: {str(e)[:100]}")

    async def _handle_approve_all(self, message: discord.Message, approve: bool):
        """Approve or reject all pending trades"""
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.is_expired()]
        
        if not pending:
            await message.channel.send("❌ No pending trade approvals.")
            return
        
        action = "APPROVED" if approve else "REJECTED"
        count = 0
        
        for approval_id, approval in pending:
            if approve:
                approval.approved = True
            else:
                approval.rejected = True
            count += 1
            
            # Call approval callback for each
            if self.approval_callback:
                try:
                    self.approval_callback(approval.approval_id, approve)
                except Exception as e:
                    logger.error(f"Error in approval callback: {e}", exc_info=True)
        
        await message.channel.send(f"{'✅' if approve else '❌'} **{action} {count} trade(s)**")
        logger.info(f"{'✅' if approve else '❌'} User {action.lower()} all {count} pending trades")
    
    async def _handle_approval(self, message: discord.Message, approve: bool):
        """Handle user approval/rejection (most recent trade)"""
        # Check if there's a pending approval (most recent)
        if not self.pending_approvals:
            await message.channel.send("❌ No pending trade approvals.")
            return
        
        # Get all pending (not approved/rejected)
        pending = [(aid, a) for aid, a in self.pending_approvals.items() 
                   if not a.approved and not a.rejected and not a.is_expired()]
        
        if not pending:
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
        color: int = 3447003
    ) -> bool:
        """Send a generic alert notification with action buttons"""
        try:
            if not self.bot_ready:
                return False
                
            channel = self.get_channel(self.channel_id)
            if not channel:
                return False
                
            embed = discord.Embed(
                title=f"🔔 {alert_type}: {symbol}",
                description=message_text,
                color=color,
                timestamp=datetime.now()
            )
            
            view = AlertActionView(self, symbol)
            await channel.send(embed=embed, view=view)
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

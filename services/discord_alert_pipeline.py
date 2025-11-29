"""
Discord Alert Pipeline

Connects Discord alerts to the Service Orchestrator's alert queue.
Allows you to:
1. Receive alerts from Discord channels
2. Queue them for review in the Control Panel
3. Approve/reject with one click
4. Auto-add approved symbols to watchlist

This bridges the gap between "I got an alert on Discord" and "I'm trading it"
"""

import os
import re
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from loguru import logger

try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("discord.py not installed - Discord pipeline disabled")


@dataclass
class ParsedDiscordAlert:
    """Parsed alert from Discord message"""
    symbol: str
    alert_type: str  # ENTRY, EXIT, BREAKOUT, WATCH, etc.
    asset_type: str  # crypto, stock
    price: Optional[float]
    target: Optional[float]
    stop_loss: Optional[float]
    reasoning: str
    confidence: str
    source_channel: str
    source_author: str
    raw_message: str


class DiscordAlertParser:
    """Parse Discord messages for trading alerts"""
    
    # Crypto patterns
    CRYPTO_PATTERNS = [
        r'\$([A-Z]{2,10})',  # $BTC, $ETH
        r'\b(BTC|ETH|SOL|DOGE|SHIB|PEPE|WIF|BONK|AVAX|LINK|XRP|ADA|DOT|MATIC|UNI|AAVE)\b',
        r'([A-Z]{2,6})/USD',  # BTC/USD
        r'([A-Z]{2,6})/USDT',  # BTC/USDT
    ]
    
    # Stock patterns
    STOCK_PATTERNS = [
        r'\$([A-Z]{1,5})\b',  # $TSLA
        r'ticker[:\s]+([A-Z]{1,5})',
        r'\b([A-Z]{2,5})\s+(?:calls?|puts?|options?)',  # TSLA calls
    ]
    
    # Price patterns
    PRICE_PATTERNS = [
        r'@\s*\$?(\d+\.?\d*)',
        r'price[:\s]+\$?(\d+\.?\d*)',
        r'entry[:\s]+\$?(\d+\.?\d*)',
        r'current[:\s]+\$?(\d+\.?\d*)',
    ]
    
    TARGET_PATTERNS = [
        r'target[:\s]+\$?(\d+\.?\d*)',
        r'TP[:\s]+\$?(\d+\.?\d*)',
        r'take\s*profit[:\s]+\$?(\d+\.?\d*)',
        r'PT[:\s]+\$?(\d+\.?\d*)',
    ]
    
    STOP_PATTERNS = [
        r'stop[:\s]+\$?(\d+\.?\d*)',
        r'SL[:\s]+\$?(\d+\.?\d*)',
        r'stop\s*loss[:\s]+\$?(\d+\.?\d*)',
    ]
    
    # Alert type keywords
    ALERT_KEYWORDS = {
        'ENTRY': ['entry', 'buy', 'long', 'entering', 'bought', 'going long', 'bullish'],
        'EXIT': ['exit', 'sell', 'close', 'sold', 'taking profit', 'closed'],
        'BREAKOUT': ['breakout', 'breaking', 'broke out', 'runner', 'ripping'],
        'WATCH': ['watch', 'watching', 'alert', 'monitor', 'keep eye on'],
        'SHORT': ['short', 'shorting', 'puts', 'bearish'],
    }
    
    # Confidence keywords
    HIGH_CONFIDENCE = ['high confidence', 'strong', 'confident', 'conviction', '🔥', '💎']
    LOW_CONFIDENCE = ['risky', 'speculative', 'gamble', 'lotto', 'yolo']
    
    # Known crypto symbols (to distinguish from stocks)
    KNOWN_CRYPTO = {
        'BTC', 'ETH', 'SOL', 'DOGE', 'SHIB', 'PEPE', 'WIF', 'BONK', 'AVAX', 
        'LINK', 'XRP', 'ADA', 'DOT', 'MATIC', 'UNI', 'AAVE', 'LTC', 'BCH',
        'ATOM', 'NEAR', 'FTM', 'ALGO', 'XLM', 'VET', 'HBAR', 'ICP', 'FIL',
        'APE', 'SAND', 'MANA', 'AXS', 'GALA', 'ENJ', 'CHZ', 'FLOW', 'IMX',
        'OP', 'ARB', 'SUI', 'APT', 'SEI', 'TIA', 'INJ', 'PYTH', 'JUP', 'JTO',
        'RENDER', 'FET', 'AGIX', 'OCEAN', 'TAO', 'RNDR', 'GRT', 'LDO', 'RPL',
        'TURBO', 'FLOKI', 'BRETT', 'POPCAT', 'MEW', 'BILLY', 'NEIRO', 'GOAT'
    }
    
    @classmethod
    def parse(cls, content: str, channel_name: str, author: str) -> Optional[ParsedDiscordAlert]:
        """Parse a Discord message into a structured alert"""
        content_upper = content.upper()
        content_lower = content.lower()
        
        # Try to extract symbol
        symbol = None
        asset_type = "stock"  # Default
        
        # Check crypto patterns first
        for pattern in cls.CRYPTO_PATTERNS:
            match = re.search(pattern, content_upper)
            if match:
                symbol = match.group(1).upper()
                if symbol in cls.KNOWN_CRYPTO:
                    asset_type = "crypto"
                    break
        
        # If no crypto found, try stock patterns
        if not symbol or asset_type != "crypto":
            for pattern in cls.STOCK_PATTERNS:
                match = re.search(pattern, content_upper)
                if match:
                    potential = match.group(1).upper()
                    # Validate it's not a common word
                    if len(potential) >= 2 and potential.isalpha():
                        if potential in cls.KNOWN_CRYPTO:
                            symbol = potential
                            asset_type = "crypto"
                        else:
                            symbol = potential
                            asset_type = "stock"
                        break
        
        if not symbol:
            return None
        
        # Determine alert type
        alert_type = "WATCH"  # Default
        for atype, keywords in cls.ALERT_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                alert_type = atype
                break
        
        # Extract prices
        price = cls._extract_price(content, cls.PRICE_PATTERNS)
        target = cls._extract_price(content, cls.TARGET_PATTERNS)
        stop_loss = cls._extract_price(content, cls.STOP_PATTERNS)
        
        # Determine confidence
        confidence = "MEDIUM"
        if any(kw in content_lower for kw in cls.HIGH_CONFIDENCE):
            confidence = "HIGH"
        elif any(kw in content_lower for kw in cls.LOW_CONFIDENCE):
            confidence = "LOW"
        
        # Build reasoning (first 200 chars of message)
        reasoning = content[:200].strip()
        if len(content) > 200:
            reasoning += "..."
        
        return ParsedDiscordAlert(
            symbol=symbol,
            alert_type=alert_type,
            asset_type=asset_type,
            price=price,
            target=target,
            stop_loss=stop_loss,
            reasoning=reasoning,
            confidence=confidence,
            source_channel=channel_name,
            source_author=author,
            raw_message=content
        )
    
    @staticmethod
    def _extract_price(text: str, patterns: List[str]) -> Optional[float]:
        """Extract price from text using patterns"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    if 0.0000001 <= price <= 1000000:  # Reasonable range
                        return price
                except (ValueError, IndexError):
                    continue
        return None


class DiscordAlertPipeline:
    """
    Discord bot that listens for alerts and queues them to the orchestrator.
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        channel_ids: Optional[List[str]] = None,
        on_alert_callback: Optional[Callable[[ParsedDiscordAlert], None]] = None
    ):
        if not DISCORD_AVAILABLE:
            logger.error("discord.py not available - pipeline disabled")
            self.enabled = False
            return
        
        self.token = token or os.getenv('DISCORD_BOT_TOKEN')
        if not self.token:
            logger.warning("No Discord token - pipeline disabled")
            self.enabled = False
            return
        
        # Get channel IDs from env if not provided
        if not channel_ids:
            channel_str = os.getenv('DISCORD_ALERT_CHANNEL_IDS', '')
            channel_ids = [c.strip() for c in channel_str.split(',') if c.strip()]
        
        self.channel_ids = set(channel_ids) if channel_ids else set()
        self.on_alert_callback = on_alert_callback
        self.enabled = True
        
        # Bot setup
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        
        self.bot = commands.Bot(command_prefix='!st ', intents=intents)
        self.is_running = False
        self.loop = None
        self.thread = None
        
        # Stats
        self.alerts_parsed = 0
        self.alerts_queued = 0
        
        # Setup event handlers
        self._setup_handlers()
        
        logger.info(f"📡 Discord Alert Pipeline initialized")
        logger.info(f"   Monitoring {len(self.channel_ids)} channels")
    
    def _setup_handlers(self):
        """Setup Discord event handlers"""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"🤖 Discord Pipeline connected as {self.bot.user}")
            self.is_running = True
            
            # List monitored channels
            for cid in self.channel_ids:
                channel = self.bot.get_channel(int(cid))
                if channel:
                    logger.info(f"   📺 Monitoring: #{channel.name}")
        
        @self.bot.event
        async def on_message(message):
            # Ignore own messages
            if message.author == self.bot.user:
                return
            
            # Check if from monitored channel
            if str(message.channel.id) not in self.channel_ids:
                # Also check if it's a DM with specific format
                if not isinstance(message.channel, discord.DMChannel):
                    return
            
            # Parse the message
            channel_name = getattr(message.channel, 'name', 'DM')
            parsed = DiscordAlertParser.parse(
                message.content,
                channel_name,
                str(message.author)
            )
            
            if parsed:
                self.alerts_parsed += 1
                logger.info(f"📨 Parsed alert: {parsed.symbol} ({parsed.alert_type})")
                
                # Queue to orchestrator
                if self.on_alert_callback:
                    try:
                        self.on_alert_callback(parsed)
                        self.alerts_queued += 1
                        
                        # React to show we processed it
                        await message.add_reaction("👀")
                    except Exception as e:
                        logger.error(f"Failed to queue alert: {e}")
                        await message.add_reaction("⚠️")
            
            # Process commands
            await self.bot.process_commands(message)
        
        # Add commands for manual control
        @self.bot.command(name='add')
        async def add_to_watchlist(ctx, symbol: str):
            """Add a symbol to watchlist: !st add BTC"""
            symbol = symbol.upper()
            
            # Determine if crypto or stock
            asset_type = "crypto" if symbol in DiscordAlertParser.KNOWN_CRYPTO else "stock"
            
            alert = ParsedDiscordAlert(
                symbol=symbol,
                alert_type="WATCH",
                asset_type=asset_type,
                price=None,
                target=None,
                stop_loss=None,
                reasoning=f"Manually added via Discord by {ctx.author}",
                confidence="MEDIUM",
                source_channel=ctx.channel.name if hasattr(ctx.channel, 'name') else 'DM',
                source_author=str(ctx.author),
                raw_message=ctx.message.content
            )
            
            if self.on_alert_callback:
                self.on_alert_callback(alert)
                await ctx.send(f"✅ Added {symbol} to alert queue")
            else:
                await ctx.send(f"⚠️ Alert queue not connected")
        
        @self.bot.command(name='status')
        async def show_status(ctx):
            """Show pipeline status: !st status"""
            await ctx.send(
                f"📊 **Alert Pipeline Status**\n"
                f"• Alerts Parsed: {self.alerts_parsed}\n"
                f"• Alerts Queued: {self.alerts_queued}\n"
                f"• Channels Monitored: {len(self.channel_ids)}"
            )
    
    def start(self) -> bool:
        """Start the Discord bot in background thread"""
        if not self.enabled or not self.token:
            return False
        
        if self.is_running:
            logger.warning("Pipeline already running")
            return True
        
        try:
            self.loop = asyncio.new_event_loop()
            token = self.token  # Capture for closure
            
            def run_bot():
                asyncio.set_event_loop(self.loop)
                if self.loop is not None:
                    self.loop.run_until_complete(self.bot.start(token))
            
            self.thread = threading.Thread(target=run_bot, daemon=True)
            self.thread.start()
            
            logger.info("🚀 Discord Alert Pipeline started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def stop(self):
        """Stop the Discord bot"""
        if self.loop and self.bot:
            try:
                asyncio.run_coroutine_threadsafe(self.bot.close(), self.loop)
                self.is_running = False
                logger.info("Discord Alert Pipeline stopped")
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "enabled": self.enabled,
            "running": self.is_running,
            "channels_monitored": len(self.channel_ids),
            "alerts_parsed": self.alerts_parsed,
            "alerts_queued": self.alerts_queued
        }


# Integration with Service Orchestrator
def create_discord_pipeline_with_orchestrator() -> Optional[DiscordAlertPipeline]:
    """
    Create Discord pipeline that automatically queues alerts to the orchestrator.
    """
    try:
        from services.service_orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        def queue_alert(parsed: ParsedDiscordAlert):
            """Callback to queue parsed alerts to orchestrator"""
            orchestrator.add_alert(
                symbol=parsed.symbol,
                alert_type=parsed.alert_type,
                source="discord",
                asset_type=parsed.asset_type,
                price=parsed.price,
                target=parsed.target,
                stop_loss=parsed.stop_loss,
                reasoning=f"[{parsed.source_channel}] {parsed.reasoning}",
                confidence=parsed.confidence,
                expires_minutes=120,  # 2 hour expiry for Discord alerts
                metadata={
                    "channel": parsed.source_channel,
                    "author": parsed.source_author,
                    "raw_message": parsed.raw_message[:500]
                }
            )
        
        pipeline = DiscordAlertPipeline(on_alert_callback=queue_alert)
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to create pipeline with orchestrator: {e}")
        return None


# Global instance
_pipeline: Optional[DiscordAlertPipeline] = None


def get_discord_pipeline() -> Optional[DiscordAlertPipeline]:
    """Get or create global Discord pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = create_discord_pipeline_with_orchestrator()
    return _pipeline

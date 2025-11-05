"""
Discord Alert Listener for Trading Signals
Monitors Discord channels for trading alerts and parses them into standardized format
"""

import discord
from discord.ext import commands
import asyncio
from loguru import logger
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os
from collections import deque



@dataclass
class DiscordAlert:
    """Standardized Discord trading alert"""
    symbol: str
    alert_type: str  # ENTRY, EXIT, RUNNER, BREAKOUT, etc.
    price: Optional[float]
    target: Optional[float]
    stop_loss: Optional[float]
    entry_price: Optional[float]
    reasoning: str
    confidence: Optional[str]  # HIGH, MEDIUM, LOW
    timestamp: str
    channel_name: str
    author: str
    raw_message: str
    premium_channel: bool = False
    
    def to_dict(self):
        return asdict(self)


class DiscordAlertParser:
    """Parse Discord messages for trading alerts"""
    
    # Common patterns for trading alerts
    TICKER_PATTERNS = [
        r'\$([A-Z]{1,5})',  # $TSLA
        r'\b([A-Z]{2,5})\b(?=\s|$)',  # TSLA (standalone)
        r'ticker[:\s]+([A-Z]{1,5})',  # ticker: TSLA
    ]
    
    PRICE_PATTERNS = [
        r'@\s*\$?(\d+\.?\d*)',  # @ $150.50
        r'price[:\s]+\$?(\d+\.?\d*)',  # price: 150.50
        r'entry[:\s]+\$?(\d+\.?\d*)',  # entry: 150.50
        r'\$(\d+\.?\d*)',  # $150.50
    ]
    
    TARGET_PATTERNS = [
        r'target[:\s]+\$?(\d+\.?\d*)',  # target: 160
        r'TP[:\s]+\$?(\d+\.?\d*)',  # TP: 160
        r'price\s+target[:\s]+\$?(\d+\.?\d*)',  # price target: 160
    ]
    
    STOP_PATTERNS = [
        r'stop\s*loss[:\s]+\$?(\d+\.?\d*)',  # stop loss: 145
        r'SL[:\s]+\$?(\d+\.?\d*)',  # SL: 145
        r'stop[:\s]+\$?(\d+\.?\d*)',  # stop: 145
    ]
    
    ALERT_TYPE_KEYWORDS = {
        'ENTRY': ['entry', 'buy', 'long', 'entering', 'entered'],
        'EXIT': ['exit', 'sell', 'close', 'closed', 'took profit'],
        'RUNNER': ['runner', 'running', 'moving', 'breakout'],
        'ALERT': ['alert', 'watch', 'watching', 'monitoring'],
        'STOP': ['stopped', 'stop hit', 'stopped out'],
    }
    
    @staticmethod
    def extract_ticker(text: str) -> Optional[str]:
        """Extract ticker symbol from message"""
        for pattern in DiscordAlertParser.TICKER_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ticker = match.group(1).upper()
                # Validate ticker (2-5 chars, all caps)
                if 2 <= len(ticker) <= 5 and ticker.isalpha():
                    return ticker
        return None
    
    @staticmethod
    def extract_price(text: str, pattern_list: List[str]) -> Optional[float]:
        """Extract price from message using pattern list"""
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    price = float(match.group(1))
                    # Sanity check (reasonable stock price range)
                    if 0.01 <= price <= 100000:
                        return price
                except (ValueError, IndexError):
                    continue
        return None
    
    @staticmethod
    def determine_alert_type(text: str) -> str:
        """Determine alert type from message content"""
        text_lower = text.lower()
        
        for alert_type, keywords in DiscordAlertParser.ALERT_TYPE_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return alert_type
        
        return 'ALERT'
    
    @classmethod
    def parse_alert(
        cls,
        message: discord.Message,
        premium_channel: bool = False
    ) -> Optional[DiscordAlert]:
        """Parse Discord message into DiscordAlert"""
        try:
            content = message.content
            
            # Extract ticker
            ticker = cls.extract_ticker(content)
            if not ticker:
                return None
            
            # Determine alert type
            alert_type = cls.determine_alert_type(content)
            
            # Extract prices
            price = cls.extract_price(content, cls.PRICE_PATTERNS)
            target = cls.extract_price(content, cls.TARGET_PATTERNS)
            stop_loss = cls.extract_price(content, cls.STOP_PATTERNS)
            
            # Entry price (if not already extracted as price)
            entry_price = price
            
            # Determine confidence based on keywords
            confidence = None
            if any(word in content.lower() for word in ['high confidence', 'strong', 'confident']):
                confidence = 'HIGH'
            elif any(word in content.lower() for word in ['low confidence', 'risky', 'uncertain']):
                confidence = 'LOW'
            else:
                confidence = 'MEDIUM'
            
            # Create alert
            alert = DiscordAlert(
                symbol=ticker,
                alert_type=alert_type,
                price=price,
                target=target,
                stop_loss=stop_loss,
                entry_price=entry_price,
                reasoning=content[:200],  # First 200 chars
                confidence=confidence,
                timestamp=message.created_at.isoformat(),
                channel_name=message.channel.name if hasattr(message.channel, 'name') else 'DM',
                author=str(message.author),
                raw_message=content,
                premium_channel=premium_channel
            )
            
            logger.info(f"Parsed Discord alert: {ticker} {alert_type} @ {price}")
            return alert
        
        except Exception as e:
            logger.error(f"Error parsing Discord message: {e}")
            return None


class DiscordAlertListener(commands.Bot):
    """Discord bot that listens for trading alerts"""
    
    def __init__(
        self,
        token: str,
        monitored_channels: Dict[str, bool],  # {channel_id: is_premium}
        max_alerts: int = 100
    ):
        """
        Initialize Discord alert listener
        
        Args:
            token: Discord bot token
            monitored_channels: Dict of channel IDs to monitor {channel_id: is_premium}
            max_alerts: Maximum number of alerts to store in memory
        """
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True
        
        super().__init__(command_prefix='!', intents=intents)
        
        self.token = token
        self.monitored_channels = monitored_channels
        self.alerts = deque(maxlen=max_alerts)
        self.parser = DiscordAlertParser()
        self.is_ready = False
        
        logger.info(f"Discord bot initialized, monitoring {len(monitored_channels)} channels")
    
    async def on_ready(self):
        """Called when bot is ready"""
        self.is_ready = True
        logger.info(f'Discord bot logged in as {self.user}')
        logger.info(f'Monitoring {len(self.monitored_channels)} channels')
        
        # List monitored channels
        for channel_id in self.monitored_channels.keys():
            channel = self.get_channel(int(channel_id))
            if channel:
                logger.info(f"Monitoring: {channel.name} (ID: {channel_id})")
            else:
                logger.warning(f"Could not find channel ID: {channel_id}")
    
    async def on_message(self, message: discord.Message):
        """Called when a message is received"""
        # Ignore bot's own messages
        if message.author == self.user:
            return
        
        # Check if message is from monitored channel
        channel_id = str(message.channel.id)
        if channel_id not in self.monitored_channels:
            return
        
        is_premium = self.monitored_channels[channel_id]
        
        # Parse alert
        alert = self.parser.parse_alert(message, premium_channel=is_premium)
        
        if alert:
            self.alerts.append(alert)
            logger.info(f"New alert: {alert.symbol} {alert.alert_type} @ {alert.price}")
            
            # Optionally process commands
            await self.process_commands(message)
    
    def get_recent_alerts(self, limit: int = 50) -> List[DiscordAlert]:
        """Get recent alerts"""
        return list(self.alerts)[-limit:]
    
    def get_alerts_by_symbol(self, symbol: str, limit: int = 10) -> List[DiscordAlert]:
        """Get alerts for specific symbol"""
        symbol_alerts = [a for a in self.alerts if a.symbol.upper() == symbol.upper()]
        return symbol_alerts[-limit:]
    
    def get_alerts_by_type(self, alert_type: str, limit: int = 50) -> List[DiscordAlert]:
        """Get alerts by type"""
        type_alerts = [a for a in self.alerts if a.alert_type == alert_type.upper()]
        return type_alerts[-limit:]
    
    def export_alerts_json(self, filename: str = 'discord_alerts.json'):
        """Export alerts to JSON file"""
        try:
            alerts_data = [alert.to_dict() for alert in self.alerts]
            with open(filename, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            logger.info(f"Exported {len(alerts_data)} alerts to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")
            return False
    
    async def start_listening(self):
        """Start the bot"""
        try:
            await self.start(self.token)
        except Exception as e:
            logger.error(f"Error starting Discord bot: {e}")
            raise


class DiscordAlertManager:
    """Manager for Discord alerts (non-async interface for Streamlit)"""
    
    def __init__(self, token: str, monitored_channels: Dict[str, bool]):
        """
        Initialize alert manager
        
        Args:
            token: Discord bot token
            monitored_channels: Dict of channel IDs to monitor
        """
        self.bot = DiscordAlertListener(token, monitored_channels)
        self.loop = None
        self.task = None
    
    def start(self):
        """Start the Discord bot in background"""
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Start bot in background
            self.task = self.loop.create_task(self.bot.start_listening())
            
            # Run loop in background thread
            import threading
            thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            thread.start()
            
            logger.info("Discord alert manager started")
            return True
        
        except Exception as e:
            logger.error(f"Error starting Discord alert manager: {e}")
            return False
    
    def stop(self):
        """Stop the Discord bot"""
        try:
            if self.loop and self.task:
                self.loop.call_soon_threadsafe(self.task.cancel)
                self.loop.stop()
            logger.info("Discord alert manager stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord alert manager: {e}")
    
    def get_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts as dictionaries"""
        if not self.bot.is_ready:
            return []
        alerts = self.bot.get_recent_alerts(limit)
        return [alert.to_dict() for alert in alerts]
    
    def get_symbol_alerts(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get alerts for specific symbol"""
        if not self.bot.is_ready:
            return []
        alerts = self.bot.get_alerts_by_symbol(symbol, limit)
        return [alert.to_dict() for alert in alerts]
    
    def is_running(self) -> bool:
        """Check if bot is running"""
        return self.bot.is_ready


def create_discord_manager(
    token: Optional[str] = None,
    channel_ids: Optional[List[str]] = None,
    premium_channel_ids: Optional[List[str]] = None
) -> Optional[DiscordAlertManager]:
    """
    Create Discord alert manager from environment or parameters
    
    Args:
        token: Discord bot token (or from env DISCORD_BOT_TOKEN)
        channel_ids: List of channel IDs to monitor (free channels)
        premium_channel_ids: List of premium channel IDs
        
    Returns:
        DiscordAlertManager or None if configuration missing
    """
    # Get token
    token = token or os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        logger.warning("No Discord bot token found")
        return None
    
    # Get channels from env if not provided
    if not channel_ids and not premium_channel_ids:
        channel_str = os.getenv('DISCORD_CHANNEL_IDS', '')
        premium_str = os.getenv('DISCORD_PREMIUM_CHANNEL_IDS', '')
        
        channel_ids = [c.strip() for c in channel_str.split(',') if c.strip()]
        premium_channel_ids = [c.strip() for c in premium_str.split(',') if c.strip()]
    
    # Build monitored channels dict
    monitored = {}
    if channel_ids:
        for cid in channel_ids:
            monitored[cid] = False  # Free channel
    if premium_channel_ids:
        for cid in premium_channel_ids:
            monitored[cid] = True  # Premium channel
    
    if not monitored:
        logger.warning("No Discord channels configured")
        return None
    
    # Create manager
    try:
        manager = DiscordAlertManager(token, monitored)
        logger.info(f"Created Discord manager for {len(monitored)} channels")
        return manager
    except Exception as e:
        logger.error(f"Error creating Discord manager: {e}")
        return None

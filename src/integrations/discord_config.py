"""
Discord Configuration Management
Handles Discord bot settings and channel configurations
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class DiscordChannelConfig:
    """Configuration for a Discord channel"""
    channel_id: str
    channel_name: str
    is_premium: bool
    enabled: bool = True
    alert_types: List[str] = None  # None = all types
    
    def __post_init__(self):
        if self.alert_types is None:
            self.alert_types = ['ENTRY', 'EXIT', 'RUNNER', 'ALERT', 'STOP']


class DiscordConfig:
    """Discord bot configuration"""
    
    DEFAULT_CONFIG_FILE = 'discord_config.json'
    
    def __init__(self, config_file: str = None):
        """
        Initialize Discord configuration
        
        Args:
            config_file: Path to JSON config file
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.bot_token = None
        self.channels: Dict[str, DiscordChannelConfig] = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file and environment"""
        # Try to load from file first
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self._load_from_dict(data)
                logger.info(f"Loaded Discord config from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading Discord config: {e}")
        
        # Load from environment (overrides file)
        self._load_from_env()
    
    def _load_from_dict(self, data: dict):
        """Load configuration from dictionary"""
        self.bot_token = data.get('bot_token')
        
        channels_data = data.get('channels', [])
        for ch_data in channels_data:
            ch = DiscordChannelConfig(
                channel_id=ch_data['channel_id'],
                channel_name=ch_data.get('channel_name', 'Unknown'),
                is_premium=ch_data.get('is_premium', False),
                enabled=ch_data.get('enabled', True),
                alert_types=ch_data.get('alert_types')
            )
            self.channels[ch.channel_id] = ch
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Bot token
        token = os.getenv('DISCORD_BOT_TOKEN')
        if token:
            self.bot_token = token
        
        # Free channels
        channel_str = os.getenv('DISCORD_CHANNEL_IDS', '')
        if channel_str:
            for cid in channel_str.split(','):
                cid = cid.strip()
                if cid and cid not in self.channels:
                    self.channels[cid] = DiscordChannelConfig(
                        channel_id=cid,
                        channel_name='Free Channel',
                        is_premium=False
                    )
        
        # Premium channels
        premium_str = os.getenv('DISCORD_PREMIUM_CHANNEL_IDS', '')
        if premium_str:
            for cid in premium_str.split(','):
                cid = cid.strip()
                if cid:
                    if cid in self.channels:
                        self.channels[cid].is_premium = True
                    else:
                        self.channels[cid] = DiscordChannelConfig(
                            channel_id=cid,
                            channel_name='Premium Channel',
                            is_premium=True
                        )
    
    def save_config(self):
        """Save configuration to file"""
        try:
            data = {
                'bot_token': self.bot_token,
                'channels': [
                    {
                        'channel_id': ch.channel_id,
                        'channel_name': ch.channel_name,
                        'is_premium': ch.is_premium,
                        'enabled': ch.enabled,
                        'alert_types': ch.alert_types
                    }
                    for ch in self.channels.values()
                ]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved Discord config to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving Discord config: {e}")
            return False
    
    def add_channel(
        self,
        channel_id: str,
        channel_name: str = None,
        is_premium: bool = False
    ):
        """Add a channel to monitor"""
        if channel_id not in self.channels:
            self.channels[channel_id] = DiscordChannelConfig(
                channel_id=channel_id,
                channel_name=channel_name or f"Channel {channel_id}",
                is_premium=is_premium
            )
            logger.info(f"Added channel: {channel_id}")
    
    def remove_channel(self, channel_id: str):
        """Remove a channel"""
        if channel_id in self.channels:
            del self.channels[channel_id]
            logger.info(f"Removed channel: {channel_id}")
    
    def get_monitored_channels(self) -> Dict[str, bool]:
        """Get dict of monitored channels {channel_id: is_premium}"""
        return {
            ch.channel_id: ch.is_premium
            for ch in self.channels.values()
            if ch.enabled
        }
    
    def is_configured(self) -> bool:
        """Check if Discord is properly configured"""
        return bool(self.bot_token and self.channels)
    
    def get_channel_info(self) -> List[Dict]:
        """Get list of channel information"""
        return [
            {
                'channel_id': ch.channel_id,
                'channel_name': ch.channel_name,
                'is_premium': ch.is_premium,
                'enabled': ch.enabled
            }
            for ch in self.channels.values()
        ]


# Example configuration template
EXAMPLE_CONFIG = {
    "bot_token": "YOUR_DISCORD_BOT_TOKEN_HERE",
    "channels": [
        {
            "channel_id": "1427896857274617939",
            "channel_name": "Free Alerts Channel",
            "is_premium": False,
            "enabled": True,
            "alert_types": ["ENTRY", "EXIT", "RUNNER", "ALERT", "STOP"]
        },
        {
            "channel_id": "PREMIUM_CHANNEL_ID_HERE",
            "channel_name": "Premium Alerts",
            "is_premium": True,
            "enabled": True,
            "alert_types": None
        }
    ]
}


def create_example_config(filename: str = 'discord_config.json'):
    """Create example configuration file"""
    try:
        with open(filename, 'w') as f:
            json.dump(EXAMPLE_CONFIG, f, indent=2)
        logger.info(f"Created example config at {filename}")
        return True
    except Exception as e:
        logger.error(f"Error creating example config: {e}")
        return False

"""
Stock Discovery Configuration Manager

Manages stock discovery settings for the Service Control Panel:
- Enable/disable discovery
- Configure which discovery modes are active
- Load/save discovery preferences

Used by: service_control_panel.py
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from loguru import logger


# Get data directory for configs
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DISCOVERY_CONFIG_FILE = DATA_DIR / "discovery_config.json"


def load_discovery_config() -> Dict[str, Any]:
    """Load discovery configuration"""
    try:
        if DISCOVERY_CONFIG_FILE.exists():
            with open(DISCOVERY_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.debug(f"Loaded discovery config: enabled={config.get('enabled')}")
                return config
    except Exception as e:
        logger.warning(f"Could not load discovery config: {e}")
    
    # Return defaults
    return get_default_discovery_config()


def save_discovery_config(config: Dict[str, Any]):
    """Save discovery configuration"""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DISCOVERY_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved discovery config")
    except Exception as e:
        logger.error(f"Error saving discovery config: {e}")


def get_default_discovery_config() -> Dict[str, Any]:
    """Get default discovery configuration - uses TieredStockScanner categories"""
    return {
        'enabled': False,  # Disabled by default
        'last_updated': datetime.now().isoformat(),
        'modes': {
            'mega_cap': {
                'enabled': False,
                'description': 'Mega Caps - Options-friendly large caps (AAPL, MSFT, etc.)',
                'max_universe_size': 30,
                'priority': 1
            },
            'high_beta_tech': {
                'enabled': False,
                'description': 'High Beta Tech - Volatile tech stocks (PLTR, SOFI, etc.)',
                'max_universe_size': 30,
                'priority': 1
            },
            'momentum': {
                'enabled': False,
                'description': 'Momentum/Meme - High momentum and meme stocks',
                'max_universe_size': 20,
                'priority': 1
            },
            'ev_energy': {
                'enabled': False,
                'description': 'EV/Clean Energy - Electric vehicle and clean energy stocks',
                'max_universe_size': 20,
                'priority': 2
            },
            'crypto_related': {
                'enabled': False,
                'description': 'Crypto-Related - Stocks tied to crypto (MARA, RIOT, COIN)',
                'max_universe_size': 20,
                'priority': 2
            },
            'ai_stocks': {
                'enabled': False,
                'description': 'AI Stocks - Artificial intelligence related stocks',
                'max_universe_size': 20,
                'priority': 1
            },
            'biotech': {
                'enabled': False,
                'description': 'Biotech - Biotechnology and pharma stocks',
                'max_universe_size': 20,
                'priority': 2
            },
            'financial': {
                'enabled': False,
                'description': 'Financial - Banks and financial services',
                'max_universe_size': 20,
                'priority': 3
            },
            'energy': {
                'enabled': False,
                'description': 'Energy - Oil and gas stocks',
                'max_universe_size': 20,
                'priority': 3
            },
            'high_iv': {
                'enabled': False,
                'description': 'High IV Options - High implied volatility for options trading',
                'max_universe_size': 20,
                'priority': 2
            },
            'penny_stocks': {
                'enabled': False,
                'description': 'Penny Stocks - Low-priced stocks under $5',
                'max_universe_size': 20,
                'priority': 3
            },
        }
    }


def toggle_discovery(enabled: bool):
    """Toggle discovery on/off"""
    config = load_discovery_config()
    config['enabled'] = enabled
    config['last_updated'] = datetime.now().isoformat()
    save_discovery_config(config)
    status = "✅ ENABLED" if enabled else "❌ DISABLED"
    logger.info(f"Stock discovery {status}")


def toggle_discovery_mode(mode_name: str, enabled: bool):
    """Toggle specific discovery mode"""
    config = load_discovery_config()
    if mode_name in config['modes']:
        config['modes'][mode_name]['enabled'] = enabled
        config['last_updated'] = datetime.now().isoformat()
        save_discovery_config(config)
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        logger.info(f"Discovery mode '{mode_name}' {status}")


def set_mode_universe_size(mode_name: str, size: int):
    """Set maximum universe size for a discovery mode"""
    config = load_discovery_config()
    if mode_name in config['modes']:
        config['modes'][mode_name]['max_universe_size'] = size
        config['last_updated'] = datetime.now().isoformat()
        save_discovery_config(config)
        logger.info(f"Mode '{mode_name}' universe size set to {size}")


def get_active_modes() -> Dict[str, bool]:
    """Get currently active discovery modes"""
    config = load_discovery_config()
    return {
        name: settings['enabled']
        for name, settings in config['modes'].items()
    }


def get_mode_descriptions() -> Dict[str, str]:
    """Get descriptions for all modes"""
    config = load_discovery_config()
    return {
        name: settings['description']
        for name, settings in config['modes'].items()
    }


def apply_config_to_monitor(monitor):
    """
    Apply discovery configuration to stock monitor
    
    Args:
        monitor: StockInformationalMonitor instance
    """
    try:
        config = load_discovery_config()
        
        # Set discovery enabled/disabled
        monitor.set_discovery_enabled(config['enabled'])
        
        # Configure modes
        modes_config = {
            name: settings['enabled']
            for name, settings in config['modes'].items()
        }
        monitor.configure_discovery_modes(modes_config)
        
        logger.info(f"Applied discovery config to monitor")
    except Exception as e:
        logger.error(f"Error applying config to monitor: {e}")


if __name__ == "__main__":
    # Example usage
    print("Discovery Config Manager")
    print("=" * 50)
    
    # Load current config
    config = load_discovery_config()
    print(f"Discovery enabled: {config['enabled']}")
    print(f"Active modes: {get_active_modes()}")
    
    # Example: Enable discovery
    # toggle_discovery(True)
    # toggle_discovery_mode('top_gainers', True)


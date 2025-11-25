"""
Configuration Loader with Broker-Specific Support
Automatically selects broker-specific config files based on DEFAULT_BROKER environment variable
"""

import os
import importlib
from pathlib import Path
from loguru import logger
from typing import Optional


def get_broker_specific_config_filename(config_filename: str) -> str:
    """
    Get broker-specific config filename based on DEFAULT_BROKER environment variable
    
    Args:
        config_filename: Original config filename (e.g., 'config_paper_trading.py')
        
    Returns:
        Broker-specific config filename if DEFAULT_BROKER is set and file exists,
        otherwise returns original filename
        
    Examples:
        >>> get_broker_specific_config_filename('config_paper_trading.py')
        'config_paper_trading_tradier.py'  # if DEFAULT_BROKER=TRADIER
        'config_paper_trading_ibkr.py'     # if DEFAULT_BROKER=IBKR
        'config_paper_trading.py'          # if DEFAULT_BROKER not set or file doesn't exist
    """
    default_broker = os.getenv('DEFAULT_BROKER', '').upper()
    
    # If DEFAULT_BROKER is not set, return original filename
    if not default_broker or default_broker not in ['TRADIER', 'IBKR']:
        return config_filename
    
    # Extract base name and extension
    if config_filename.endswith('.py'):
        base_name = config_filename[:-3]  # Remove .py
        extension = '.py'
    else:
        base_name = config_filename
        extension = ''
    
    # Create broker-specific filename
    broker_suffix = default_broker.lower()
    broker_specific_filename = f"{base_name}_{broker_suffix}{extension}"
    
    # Check if broker-specific file exists
    config_path = Path(broker_specific_filename)
    if config_path.exists():
        logger.info(f"🔧 Using broker-specific config: {broker_specific_filename} (DEFAULT_BROKER={default_broker})")
        return broker_specific_filename
    else:
        logger.debug(f"📋 Broker-specific config not found: {broker_specific_filename}, using original: {config_filename}")
        return config_filename


def load_config_module(config_filename: str, force_reload: bool = False):
    """
    Load configuration module, automatically selecting broker-specific version if available
    
    Args:
        config_filename: Config filename (e.g., 'config_paper_trading.py')
        force_reload: If True, forces module reload even if already imported
        
    Returns:
        Config module object
        
    Raises:
        ImportError: If config file cannot be imported
    """
    # Get broker-specific filename if available
    actual_filename = get_broker_specific_config_filename(config_filename)
    
    # Remove .py extension for import
    module_name = actual_filename.replace('.py', '')
    
    # Import the module
    try:
        # Reload if already imported and forced
        if module_name in importlib.sys.modules and force_reload:
            cfg = importlib.reload(importlib.sys.modules[module_name])
            broker_type = getattr(cfg, 'BROKER_TYPE', 'NOT SET')
            logger.info(f"✅ Reloaded config: {actual_filename} (BROKER_TYPE={broker_type})")
        elif module_name in importlib.sys.modules:
            cfg = importlib.sys.modules[module_name]
            # Don't log on cached hit to reduce noise
        else:
            cfg = importlib.import_module(module_name)
            # Log which config was loaded
            broker_type = getattr(cfg, 'BROKER_TYPE', 'NOT SET')
            logger.info(f"✅ Loaded config: {actual_filename} (BROKER_TYPE={broker_type})")
        
        return cfg
    except ImportError as e:
        logger.error(f"❌ Failed to import config module '{module_name}': {str(e)}")
        raise


def get_config_broker_type(config_filename: str) -> Optional[str]:
    """
    Get BROKER_TYPE from config file (checks broker-specific version first)
    
    Args:
        config_filename: Config filename (e.g., 'config_paper_trading.py')
        
    Returns:
        BROKER_TYPE value from config, or None if not found
    """
    try:
        cfg = load_config_module(config_filename)
        return getattr(cfg, 'BROKER_TYPE', None)
    except Exception as e:
        logger.debug(f"Could not get BROKER_TYPE from {config_filename}: {e}")
        return None


def get_api_key(key_name: str, section: Optional[str] = None) -> Optional[str]:
    """
    Get API key from multiple sources with fallback priority:
    1. Environment variable
    2. Streamlit secrets (for cloud deployment)
    3. .env file (loaded by dotenv)
    
    Args:
        key_name: Name of the API key (e.g., 'OPENROUTER_API_KEY')
        section: Optional section in secrets.toml (e.g., 'openrouter')
        
    Returns:
        API key string or None if not found
        
    Examples:
        >>> get_api_key('OPENROUTER_API_KEY', 'openrouter')
        'sk-or-v1-...'
        >>> get_api_key('TRADIER_ACCESS_TOKEN')
        'ABC123...'
    """
    # Try environment variable first
    api_key = os.getenv(key_name)
    if api_key:
        return api_key
    
    # Try Streamlit secrets (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            # Try section.key format first (e.g., openrouter.api_key)
            if section:
                try:
                    api_key = st.secrets.get(section, {}).get(key_name.lower().replace('_', '_'))
                    if api_key:
                        return api_key
                    # Try with original key name
                    api_key = st.secrets.get(section, {}).get('api_key')
                    if api_key:
                        return api_key
                except Exception:
                    pass
            
            # Try direct key name (e.g., OPENROUTER_API_KEY)
            try:
                api_key = st.secrets.get(key_name)
                if api_key:
                    return api_key
            except Exception:
                pass
    except ImportError:
        pass  # Streamlit not installed or not in Streamlit context
    
    return None

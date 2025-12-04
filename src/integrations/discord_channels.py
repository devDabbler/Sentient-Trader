"""
Discord Channel Router
Routes different alert types to specific Discord channels/webhooks

Channel Categories:
- STOCK_ALERTS: Stock trading signals, scans, and opportunities
- CRYPTO_ALERTS: Crypto breakouts, signals, and opportunities
- OPTIONS_ALERTS: Options trading signals and opportunities
- DEX_PUMP_ALERTS: DEX Hunter pump/launch detection alerts
- STOCK_POSITIONS: Stock position monitor updates (AI decisions, stop adjustments)
- CRYPTO_POSITIONS: Crypto position monitor updates (AI decisions, stop adjustments)
- OPTIONS_POSITIONS: Options position monitor updates
- STOCK_EXECUTIONS: Stock trade executions (buys/sells)
- CRYPTO_EXECUTIONS: Crypto trade executions (buys/sells)
- OPTIONS_EXECUTIONS: Options trade executions (buys/sells)
- GENERAL: Fallback for any uncategorized alerts

Usage:
    from src.integrations.discord_channels import get_discord_webhook, AlertCategory
    
    webhook_url = get_discord_webhook(AlertCategory.STOCK_ALERTS)
    # or
    webhook_url = get_discord_webhook_for_asset("AAPL", is_execution=True)
"""

import os
from enum import Enum
from typing import Optional
from loguru import logger


class AlertCategory(Enum):
    """Discord channel categories for routing alerts"""
    # Trading signals and opportunities
    STOCK_ALERTS = "STOCK_ALERTS"           # Stock trading signals and opportunities
    CRYPTO_ALERTS = "CRYPTO_ALERTS"         # Crypto breakouts and opportunities
    OPTIONS_ALERTS = "OPTIONS_ALERTS"       # Options trading signals
    DEX_PUMP_ALERTS = "DEX_PUMP_ALERTS"     # DEX Hunter pump/launch alerts
    
    # Position monitor updates (AI decisions, stop adjustments, P&L updates)
    STOCK_POSITIONS = "STOCK_POSITIONS"     # Stock position monitor
    CRYPTO_POSITIONS = "CRYPTO_POSITIONS"   # Crypto position monitor
    OPTIONS_POSITIONS = "OPTIONS_POSITIONS" # Options position monitor
    
    # Trade executions (actual buys/sells)
    STOCK_EXECUTIONS = "STOCK_EXECUTIONS"   # Stock trade executions
    CRYPTO_EXECUTIONS = "CRYPTO_EXECUTIONS" # Crypto trade executions
    OPTIONS_EXECUTIONS = "OPTIONS_EXECUTIONS"  # Options trade executions
    
    GENERAL = "GENERAL"                     # Fallback channel


# Environment variable names for each webhook
WEBHOOK_ENV_VARS = {
    AlertCategory.STOCK_ALERTS: "DISCORD_WEBHOOK_STOCK_ALERTS",
    AlertCategory.CRYPTO_ALERTS: "DISCORD_WEBHOOK_CRYPTO_ALERTS",
    AlertCategory.OPTIONS_ALERTS: "DISCORD_WEBHOOK_OPTIONS_ALERTS",
    AlertCategory.DEX_PUMP_ALERTS: "DISCORD_WEBHOOK_DEX_PUMP_ALERTS",
    AlertCategory.STOCK_POSITIONS: "DISCORD_WEBHOOK_STOCK_POSITIONS",
    AlertCategory.CRYPTO_POSITIONS: "DISCORD_WEBHOOK_CRYPTO_POSITIONS",
    AlertCategory.OPTIONS_POSITIONS: "DISCORD_WEBHOOK_OPTIONS_POSITIONS",
    AlertCategory.STOCK_EXECUTIONS: "DISCORD_WEBHOOK_STOCK_EXECUTIONS",
    AlertCategory.CRYPTO_EXECUTIONS: "DISCORD_WEBHOOK_CRYPTO_EXECUTIONS",
    AlertCategory.OPTIONS_EXECUTIONS: "DISCORD_WEBHOOK_OPTIONS_EXECUTIONS",
    AlertCategory.GENERAL: "DISCORD_WEBHOOK_URL",  # Fallback to original
}


# Known crypto symbols (add more as needed)
KNOWN_CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "SHIB", "AVAX", "DOT", "LINK",
    "MATIC", "UNI", "ATOM", "LTC", "BCH", "XLM", "ALGO", "VET", "FIL", "THETA",
    "XMR", "ETC", "AAVE", "MKR", "COMP", "SNX", "CRV", "YFI", "SUSHI", "INCH",
    "APE", "SAND", "MANA", "AXS", "ENJ", "CHZ", "GALA", "IMX", "LRC", "ENS",
    "OP", "ARB", "PEPE", "FLOKI", "BONK", "WIF", "BOME", "MEME", "SLERF",
    "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD",  # Kraken-style pairs
}


def is_crypto_symbol(symbol: str) -> bool:
    """Check if a symbol is a cryptocurrency"""
    if not symbol:
        return False
    
    # Clean the symbol
    clean_symbol = symbol.upper().replace("/", "").replace("-", "")
    
    # Check direct match
    if clean_symbol in KNOWN_CRYPTO_SYMBOLS:
        return True
    
    # Check if it ends with USD (Kraken-style)
    if clean_symbol.endswith("USD") and clean_symbol[:-3] in KNOWN_CRYPTO_SYMBOLS:
        return True
    
    # Check if it contains known crypto symbols (e.g., "BTC/USD", "ETH-USD")
    for crypto in KNOWN_CRYPTO_SYMBOLS:
        if clean_symbol.startswith(crypto):
            return True
    
    return False


def is_options_symbol(symbol: str) -> bool:
    """Check if a symbol is an options contract"""
    if not symbol:
        return False
    
    # Options typically have format like: AAPL240119C00190000 or AAPL_240119C190
    # Check for common options patterns
    if len(symbol) > 10 and any(c in symbol for c in ['C', 'P']):
        # Check if contains date-like patterns
        if any(char.isdigit() for char in symbol):
            # Options usually have significant digit count
            digit_count = sum(1 for c in symbol if c.isdigit())
            if digit_count >= 6:
                return True
    
    return False


def get_discord_webhook(category: AlertCategory) -> Optional[str]:
    """
    Get the Discord webhook URL for a specific alert category
    
    Falls back to DISCORD_WEBHOOK_URL if specific channel not configured
    """
    # Get the specific webhook
    env_var = WEBHOOK_ENV_VARS.get(category, "DISCORD_WEBHOOK_URL")
    webhook_url = os.getenv(env_var)
    
    # Fallback to general webhook if specific not set
    if not webhook_url:
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if webhook_url:
            logger.debug(f"Using fallback DISCORD_WEBHOOK_URL for {category.value}")
    
    return webhook_url


def get_discord_webhook_for_asset(
    symbol: str,
    is_execution: bool = False,
    force_category: Optional[AlertCategory] = None
) -> Optional[str]:
    """
    Get the appropriate Discord webhook for an asset symbol
    
    Args:
        symbol: The trading symbol (e.g., "AAPL", "BTC/USD", "AAPL240119C00190000")
        is_execution: If True, routes to execution channel; otherwise alert channel
        force_category: Override auto-detection with specific category
    
    Returns:
        Discord webhook URL or None if not configured
    """
    if force_category:
        return get_discord_webhook(force_category)
    
    # Auto-detect asset type
    if is_options_symbol(symbol):
        category = AlertCategory.OPTIONS_EXECUTIONS if is_execution else AlertCategory.OPTIONS_ALERTS
    elif is_crypto_symbol(symbol):
        category = AlertCategory.CRYPTO_EXECUTIONS if is_execution else AlertCategory.CRYPTO_ALERTS
    else:
        # Default to stocks
        category = AlertCategory.STOCK_EXECUTIONS if is_execution else AlertCategory.STOCK_ALERTS
    
    return get_discord_webhook(category)


def get_all_configured_channels() -> dict:
    """Get status of all Discord channel configurations"""
    status = {}
    for category in AlertCategory:
        env_var = WEBHOOK_ENV_VARS.get(category)
        webhook_url = os.getenv(env_var) if env_var else None
        status[category.value] = {
            "env_var": env_var,
            "configured": bool(webhook_url),
            "url_preview": f"{webhook_url[:50]}..." if webhook_url and len(webhook_url) > 50 else webhook_url
        }
    return status


def log_channel_configuration():
    """Log the current Discord channel configuration"""
    logger.info("=" * 60)
    logger.info("Discord Channel Configuration")
    logger.info("=" * 60)
    
    for category in AlertCategory:
        env_var = WEBHOOK_ENV_VARS.get(category)
        webhook_url = os.getenv(env_var) if env_var else None
        fallback = os.getenv("DISCORD_WEBHOOK_URL")
        
        if webhook_url:
            logger.info(f"  ✅ {category.value}: Configured ({env_var})")
        elif fallback:
            logger.info(f"  ⚠️ {category.value}: Using fallback (DISCORD_WEBHOOK_URL)")
        else:
            logger.warning(f"  ❌ {category.value}: Not configured")
    
    logger.info("=" * 60)

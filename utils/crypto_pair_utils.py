"""
Global Crypto Pair Normalization Utility

Normalizes various crypto pair formats to a standard format.
Handles formats like:
- BTC/USD, BTCUSD, btcusd, btc/usd -> BTC/USD
- ETH/USDT, ETHUSDT, ethusdt, eth/usdt -> ETH/USDT

This utility should be used throughout the codebase for consistent pair handling.
"""

import re
from typing import Optional, Tuple
from loguru import logger


# Common quote currencies (in order of preference)
COMMON_QUOTES = ['USD', 'USDT', 'USDC', 'EUR', 'GBP', 'BTC', 'ETH']


def normalize_crypto_pair(pair: str, default_quote: str = 'USD') -> str:
    """
    Normalize a crypto pair to standard format (BASE/QUOTE).
    
    Handles various input formats:
    - BTC/USD, BTCUSD, btcusd, btc/usd -> BTC/USD
    - ETH/USDT, ETHUSDT, ethusdt, eth/usdt -> ETH/USDT
    - BTC -> BTC/USD (if default_quote provided)
    - btc -> BTC/USD (if default_quote provided)
    
    Args:
        pair: Crypto pair in any format (e.g., 'BTC/USD', 'BTCUSD', 'btcusd', 'btc/usd', 'BTC')
        default_quote: Default quote currency if not found in pair (default: 'USD')
        
    Returns:
        Normalized pair in format 'BASE/QUOTE' (e.g., 'BTC/USD')
        
    Examples:
        >>> normalize_crypto_pair('BTC/USD')
        'BTC/USD'
        >>> normalize_crypto_pair('BTCUSD')
        'BTC/USD'
        >>> normalize_crypto_pair('btcusd')
        'BTC/USD'
        >>> normalize_crypto_pair('btc/usd')
        'BTC/USD'
        >>> normalize_crypto_pair('BTC')
        'BTC/USD'
        >>> normalize_crypto_pair('ETHUSDT')
        'ETH/USDT'
    """
    if not pair or not isinstance(pair, str):
        return pair
    
    # Strip whitespace
    pair = pair.strip()
    
    if not pair:
        return pair
    
    # Handle pairs with slash separator (e.g., BTC/USD, btc/usd)
    if '/' in pair:
        parts = pair.split('/')
        if len(parts) == 2:
            base = parts[0].strip().upper()
            quote = parts[1].strip().upper()
            return f"{base}/{quote}"
        elif len(parts) == 1:
            # Just base asset (e.g., 'BTC/')
            base = parts[0].strip().upper()
            return f"{base}/{default_quote}"
    
    # Handle pairs without slash (e.g., BTCUSD, btcusd)
    pair_upper = pair.upper()
    
    # Try to find quote currency at the end
    for quote in COMMON_QUOTES:
        if pair_upper.endswith(quote):
            base = pair_upper[:-len(quote)]
            if base:  # Make sure we have a base
                return f"{base}/{quote}"
    
    # If no quote found, assume it's just the base asset
    # Return with default quote
    return f"{pair_upper}/{default_quote}"


def parse_crypto_pair(pair: str, default_quote: str = 'USD') -> Tuple[str, str]:
    """
    Parse a crypto pair into base and quote assets.
    
    Args:
        pair: Crypto pair in any format
        default_quote: Default quote currency if not found (default: 'USD')
        
    Returns:
        Tuple of (base_asset, quote_asset)
        
    Examples:
        >>> parse_crypto_pair('BTC/USD')
        ('BTC', 'USD')
        >>> parse_crypto_pair('BTCUSD')
        ('BTC', 'USD')
        >>> parse_crypto_pair('BTC')
        ('BTC', 'USD')
    """
    normalized = normalize_crypto_pair(pair, default_quote)
    if '/' in normalized:
        base, quote = normalized.split('/')
        return base, quote
    return normalized, default_quote


def extract_base_asset(pair: str) -> str:
    """
    Extract base asset from a crypto pair.
    
    Args:
        pair: Crypto pair in any format
        
    Returns:
        Base asset symbol (e.g., 'BTC')
        
    Examples:
        >>> extract_base_asset('BTC/USD')
        'BTC'
        >>> extract_base_asset('BTCUSD')
        'BTC'
        >>> extract_base_asset('btcusd')
        'BTC'
    """
    base, _ = parse_crypto_pair(pair)
    return base


def extract_quote_asset(pair: str, default_quote: str = 'USD') -> str:
    """
    Extract quote asset from a crypto pair.
    
    Args:
        pair: Crypto pair in any format
        default_quote: Default quote currency if not found (default: 'USD')
        
    Returns:
        Quote asset symbol (e.g., 'USD')
        
    Examples:
        >>> extract_quote_asset('BTC/USD')
        'USD'
        >>> extract_quote_asset('ETHUSDT')
        'USDT'
        >>> extract_quote_asset('BTC')
        'USD'
    """
    _, quote = parse_crypto_pair(pair, default_quote)
    return quote


def is_valid_pair_format(pair: str) -> bool:
    """
    Check if a pair string looks like a valid crypto pair format.
    
    Args:
        pair: Crypto pair string to validate
        
    Returns:
        True if format looks valid, False otherwise
        
    Examples:
        >>> is_valid_pair_format('BTC/USD')
        True
        >>> is_valid_pair_format('BTCUSD')
        True
        >>> is_valid_pair_format('')
        False
        >>> is_valid_pair_format('123')
        False
    """
    if not pair or not isinstance(pair, str):
        return False
    
    pair = pair.strip()
    if not pair:
        return False
    
    # Check if it contains at least one letter (crypto symbols are alphabetic)
    if not re.search(r'[A-Za-z]', pair):
        return False
    
    # Check if it's a reasonable length (2-20 characters)
    if len(pair) < 2 or len(pair) > 20:
        return False
    
    return True


def normalize_pair_list(pairs: list, default_quote: str = 'USD') -> list:
    """
    Normalize a list of crypto pairs.
    
    Args:
        pairs: List of crypto pairs in various formats
        default_quote: Default quote currency if not found (default: 'USD')
        
    Returns:
        List of normalized pairs
        
    Examples:
        >>> normalize_pair_list(['BTC/USD', 'BTCUSD', 'btcusd'])
        ['BTC/USD', 'BTC/USD', 'BTC/USD']
    """
    return [normalize_crypto_pair(pair, default_quote) for pair in pairs if pair]


def get_pair_variations(pair: str) -> list:
    """
    Get all common variations of a normalized pair.
    
    Useful for trying different formats when querying APIs.
    
    Args:
        pair: Normalized pair (e.g., 'BTC/USD')
        
    Returns:
        List of pair variations
        
    Examples:
        >>> get_pair_variations('BTC/USD')
        ['BTC/USD', 'BTCUSD', 'btc/usd', 'btcusd']
    """
    if not pair or '/' not in pair:
        return [pair]
    
    base, quote = pair.split('/')
    
    variations = [
        f"{base}/{quote}",      # BTC/USD
        f"{base}{quote}",       # BTCUSD
        f"{base.lower()}/{quote.lower()}",  # btc/usd
        f"{base.lower()}{quote.lower()}",  # btcusd
        f"{base}/{quote.lower()}",  # BTC/usd
        f"{base.lower()}/{quote}",  # btc/USD
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    
    return unique_variations


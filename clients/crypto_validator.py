"""
Crypto Validator Utility

Validates and converts crypto symbols to valid Kraken trading pairs.
Ensures all crypto implementations only work with tradable pairs.
"""

from typing import List, Dict, Optional, Tuple
from loguru import logger
from clients.kraken_client import KrakenClient


class CryptoValidator:
    """
    Validates crypto symbols and converts them to valid Kraken trading pairs.
    """
    
    def __init__(self, kraken_client: KrakenClient):
        """
        Initialize crypto validator
        
        Args:
            kraken_client: KrakenClient instance for validation
        """
        self.client = kraken_client
        self._valid_pairs_cache = {}
        self._invalid_symbols_cache = set()
    
    def validate_and_convert(
        self,
        symbol: str,
        try_formats: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Validate a crypto symbol and convert it to a valid Kraken pair format.
        
        Args:
            symbol: Crypto symbol (e.g., 'ACE', 'BTC', 'ETH')
            try_formats: Optional list of formats to try. If None, uses default formats.
            
        Returns:
            Valid Kraken pair format (e.g., 'ACE/USD', 'BTC/USD') or None if invalid
        """
        if not symbol:
            return None
        
        # Check cache first
        if symbol in self._valid_pairs_cache:
            return self._valid_pairs_cache[symbol]
        
        if symbol in self._invalid_symbols_cache:
            return None
        
        # Default formats to try
        if try_formats is None:
            symbol_upper = symbol.upper().strip()
            # Remove common suffixes/prefixes
            symbol_clean = symbol_upper.replace('/USD', '').replace('/USDT', '').replace('USD', '').replace('USDT', '')
            
            try_formats = [
                f"{symbol_clean}/USD",
                f"{symbol_clean}USD",
                f"{symbol_clean}/USDT",
                f"{symbol_clean}USDT",
                symbol_upper,  # Try as-is
            ]
        
        # Try each format
        for pair_format in try_formats:
            try:
                test_info = self.client.get_ticker_info(pair_format)
                if test_info and 'c' in test_info:
                    # Valid pair found - cache it
                    self._valid_pairs_cache[symbol] = pair_format
                    logger.debug(f"✅ Validated {symbol} -> {pair_format}")
                    return pair_format
            except Exception:
                continue
        
        # No valid format found - cache as invalid
        self._invalid_symbols_cache.add(symbol)
        logger.debug(f"❌ No valid Kraken pair found for {symbol}")
        return None
    
    def validate_batch(
        self,
        symbols: List[str],
        return_invalid: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Validate a batch of crypto symbols.
        
        Args:
            symbols: List of crypto symbols to validate
            return_invalid: If True, also returns list of invalid symbols
            
        Returns:
            Tuple of (valid_pairs, invalid_symbols) if return_invalid=True
            Otherwise, just returns valid_pairs
        """
        valid_pairs = []
        invalid_symbols = []
        
        for symbol in symbols:
            valid_pair = self.validate_and_convert(symbol)
            if valid_pair:
                valid_pairs.append(valid_pair)
            else:
                invalid_symbols.append(symbol)
        
        if return_invalid:
            return valid_pairs, invalid_symbols
        return valid_pairs
    
    def filter_valid_pairs(
        self,
        crypto_data: List[Dict],
        symbol_key: str = 'symbol'
    ) -> Tuple[List[Dict], List[str]]:
        """
        Filter a list of crypto data dictionaries, keeping only those with valid Kraken pairs.
        
        Args:
            crypto_data: List of dicts containing crypto data (must have symbol_key)
            symbol_key: Key name for the symbol in each dict
            
        Returns:
            Tuple of (filtered_data, invalid_symbols)
        """
        filtered_data = []
        invalid_symbols = []
        
        for item in crypto_data:
            symbol = item.get(symbol_key, '')
            if not symbol:
                continue
            
            valid_pair = self.validate_and_convert(symbol)
            if valid_pair:
                # Update the symbol in the data to the valid pair format
                item[symbol_key] = valid_pair
                filtered_data.append(item)
            else:
                invalid_symbols.append(symbol)
        
        return filtered_data, invalid_symbols
    
    def clear_cache(self):
        """Clear validation cache"""
        self._valid_pairs_cache.clear()
        self._invalid_symbols_cache.clear()
        logger.debug("Cleared crypto validator cache")


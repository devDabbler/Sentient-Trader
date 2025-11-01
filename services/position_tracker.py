"""
Position Tracker - Unified interface for getting active positions from Tradier or IBKR

Provides a simple way to check if you have active positions in a ticker,
used by event detectors to warn about earnings/news when you hold the stock.
"""

import logging
from typing import Set, Optional, Dict, List

logger = logging.getLogger(__name__)


class PositionTracker:
    """Tracks active positions from Tradier or IBKR"""
    
    def __init__(self, tradier_client=None, ibkr_client=None):
        """
        Initialize position tracker
        
        Args:
            tradier_client: Tradier client instance
            ibkr_client: IBKR client instance (ib_insync)
        """
        self.tradier_client = tradier_client
        self.ibkr_client = ibkr_client
        self._position_cache: Set[str] = set()
        self._cache_valid = False
    
    def set_tradier_client(self, client):
        """Set or update Tradier client"""
        self.tradier_client = client
        self._cache_valid = False
    
    def set_ibkr_client(self, client):
        """Set or update IBKR client"""
        self.ibkr_client = client
        self._cache_valid = False
    
    def refresh_positions(self) -> bool:
        """
        Refresh position cache from broker
        
        Returns:
            True if successful
        """
        positions = set()
        
        # Try Tradier first
        if self.tradier_client:
            try:
                success, positions_data = self.tradier_client.get_positions()
                
                if success and positions_data:
                    # Parse Tradier positions
                    pos_list = []
                    if isinstance(positions_data, dict):
                        if 'positions' in positions_data:
                            pos_data = positions_data['positions']
                            if isinstance(pos_data, dict) and 'position' in pos_data:
                                pos_list = pos_data['position']
                                if isinstance(pos_list, dict):
                                    pos_list = [pos_list]
                            elif isinstance(pos_data, list):
                                pos_list = pos_data
                    elif isinstance(positions_data, list):
                        pos_list = positions_data
                    
                    # Extract symbols
                    for pos in pos_list:
                        symbol = pos.get('symbol', '').upper()
                        if symbol:
                            positions.add(symbol)
                    
                    logger.info(f"Loaded {len(positions)} positions from Tradier")
            
            except Exception as e:
                logger.error(f"Error fetching Tradier positions: {e}")
        
        # Try IBKR if no Tradier positions
        if not positions and self.ibkr_client:
            try:
                # IBKR uses ib_insync
                portfolio = self.ibkr_client.portfolio()
                
                for item in portfolio:
                    symbol = item.contract.symbol.upper()
                    if symbol:
                        positions.add(symbol)
                
                logger.info(f"Loaded {len(positions)} positions from IBKR")
            
            except Exception as e:
                logger.error(f"Error fetching IBKR positions: {e}")
        
        # Update cache
        self._position_cache = positions
        self._cache_valid = True
        
        return len(positions) > 0
    
    def has_position(self, ticker: str) -> bool:
        """
        Check if you have an active position in this ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            True if position exists
        """
        # Refresh cache if not valid
        if not self._cache_valid:
            self.refresh_positions()
        
        return ticker.upper() in self._position_cache
    
    def get_all_positions(self) -> Set[str]:
        """
        Get all active position symbols
        
        Returns:
            Set of ticker symbols
        """
        # Refresh cache if not valid
        if not self._cache_valid:
            self.refresh_positions()
        
        return self._position_cache.copy()
    
    def get_position_details(self, ticker: str) -> Optional[Dict]:
        """
        Get detailed position information for a ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dict with position details or None
        """
        ticker = ticker.upper()
        
        # Try Tradier first
        if self.tradier_client:
            try:
                success, positions_data = self.tradier_client.get_positions()
                
                if success and positions_data:
                    pos_list = []
                    if isinstance(positions_data, dict):
                        if 'positions' in positions_data:
                            pos_data = positions_data['positions']
                            if isinstance(pos_data, dict) and 'position' in pos_data:
                                pos_list = pos_data['position']
                                if isinstance(pos_list, dict):
                                    pos_list = [pos_list]
                            elif isinstance(pos_data, list):
                                pos_list = pos_data
                    elif isinstance(positions_data, list):
                        pos_list = positions_data
                    
                    # Find matching position
                    for pos in pos_list:
                        if pos.get('symbol', '').upper() == ticker:
                            quantity = float(pos.get('quantity', 0))
                            cost_basis = float(pos.get('cost_basis', 0))
                            entry_price = cost_basis / quantity if quantity != 0 else 0
                            
                            return {
                                'symbol': ticker,
                                'quantity': quantity,
                                'entry_price': entry_price,
                                'cost_basis': cost_basis,
                                'current_price': float(pos.get('last', entry_price)),
                                'source': 'tradier'
                            }
            
            except Exception as e:
                logger.error(f"Error fetching position details from Tradier: {e}")
        
        # Try IBKR
        if self.ibkr_client:
            try:
                portfolio = self.ibkr_client.portfolio()
                
                for item in portfolio:
                    if item.contract.symbol.upper() == ticker:
                        return {
                            'symbol': ticker,
                            'quantity': item.position,
                            'entry_price': item.averageCost,
                            'cost_basis': item.averageCost * abs(item.position),
                            'current_price': item.marketPrice,
                            'unrealized_pnl': item.unrealizedPNL,
                            'source': 'ibkr'
                        }
            
            except Exception as e:
                logger.error(f"Error fetching position details from IBKR: {e}")
        
        return None


# Global position tracker instance
_global_position_tracker: Optional[PositionTracker] = None


def get_position_tracker(tradier_client=None, ibkr_client=None) -> PositionTracker:
    """
    Get or create global position tracker instance
    
    Args:
        tradier_client: Optional Tradier client
        ibkr_client: Optional IBKR client
        
    Returns:
        PositionTracker instance
    """
    global _global_position_tracker
    
    if _global_position_tracker is None:
        _global_position_tracker = PositionTracker(tradier_client, ibkr_client)
    else:
        # Update clients if provided
        if tradier_client:
            _global_position_tracker.set_tradier_client(tradier_client)
        if ibkr_client:
            _global_position_tracker.set_ibkr_client(ibkr_client)
    
    return _global_position_tracker

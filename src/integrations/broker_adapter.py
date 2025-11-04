"""
Unified Broker Adapter
Provides a consistent interface for both Tradier and IBKR clients
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BrokerAdapter(ABC):
    """Abstract base class for broker adapters"""
    
    @abstractmethod
    def get_positions(self) -> Tuple[bool, List[Dict]]:
        """
        Get current positions
        
        Returns:
            Tuple of (success: bool, positions: List[Dict])
            Each position dict should have: symbol, quantity, cost_basis, current_price
        """
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Tuple[bool, Dict]:
        """
        Get account balance information
        
        Returns:
            Tuple of (success: bool, balance_data: Dict)
            balance_data should have: total_equity, cash, buying_power
        """
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with quote data including: last_price, bid, ask, volume
        """
        pass
    
    @abstractmethod
    def place_order(self, order_data: Dict) -> Tuple[bool, Dict]:
        """
        Place a generic order
        
        Args:
            order_data: Order details
            
        Returns:
            Tuple of (success: bool, order_result: Dict)
        """
        pass
    
    @abstractmethod
    def place_equity_order(self, symbol: str, side: str, quantity: int, 
                          order_type: str = "market", duration: str = "day",
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          tag: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Place an equity order
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: 'market', 'limit', 'stop'
            duration: 'day', 'gtc'
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            tag: Order tag/identifier
            
        Returns:
            Tuple of (success: bool, order_result: Dict)
        """
        pass
    
    @abstractmethod
    def place_bracket_order(self, symbol: str, side: str, quantity: int,
                           stop_loss_price: float, take_profit_price: float,
                           duration: str = "day", tag: Optional[str] = None) -> Tuple[bool, Dict]:
        """
        Place a bracket order (entry with stop loss and take profit)
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            duration: 'day', 'gtc'
            tag: Order tag/identifier
            
        Returns:
            Tuple of (success: bool, order_result: Dict)
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker"""
        pass


class TradierAdapter(BrokerAdapter):
    """Adapter for Tradier client"""
    
    def __init__(self, tradier_client):
        """
        Initialize Tradier adapter
        
        Args:
            tradier_client: TradierClient instance
        """
        self.client = tradier_client
        logger.info("✅ TradierAdapter initialized")
    
    def get_positions(self) -> Tuple[bool, List[Dict]]:
        """Get positions from Tradier"""
        try:
            success, positions = self.client.get_positions()
            if not success:
                return False, []
            
            # Normalize position format
            normalized = []
            for pos in positions:
                normalized.append({
                    'symbol': pos.get('symbol'),
                    'quantity': int(pos.get('quantity', 0)),
                    'cost_basis': float(pos.get('cost_basis', 0)),
                    'current_price': float(pos.get('last', 0))
                })
            return True, normalized
        except Exception as e:
            logger.error(f"Error getting Tradier positions: {e}")
            return False, []
    
    def get_account_balance(self) -> Tuple[bool, Dict]:
        """Get account balance from Tradier"""
        try:
            success, balance = self.client.get_account_balance()
            if not success:
                return False, {}
            
            # Normalize balance format
            normalized = {
                'total_equity': float(balance.get('total_equity', 0)),
                'cash': float(balance.get('total_cash', 0)),
                'buying_power': float(balance.get('option_buying_power', 0) or balance.get('stock_buying_power', 0))
            }
            return True, normalized
        except Exception as e:
            logger.error(f"Error getting Tradier balance: {e}")
            return False, {}
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote from Tradier"""
        try:
            quote = self.client.get_quote(symbol)
            if not quote:
                return None
            return quote
        except Exception as e:
            logger.error(f"Error getting Tradier quote for {symbol}: {e}")
            return None
    
    def place_order(self, order_data: Dict) -> Tuple[bool, Dict]:
        """Place order via Tradier"""
        try:
            success, result = self.client.place_order(order_data)
            return success, result
        except Exception as e:
            logger.error(f"Error placing Tradier order: {e}")
            return False, {}
    
    def place_equity_order(self, symbol: str, side: str, quantity: int,
                          order_type: str = "market", duration: str = "day",
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          tag: Optional[str] = None) -> Tuple[bool, Dict]:
        """Place equity order via Tradier"""
        try:
            success, result = self.client.place_equity_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                duration=duration,
                limit_price=limit_price,
                stop_price=stop_price,
                tag=tag
            )
            return success, result
        except Exception as e:
            logger.error(f"Error placing Tradier equity order: {e}")
            return False, {}
    
    def place_bracket_order(self, symbol: str, side: str, quantity: int,
                           stop_loss_price: float, take_profit_price: float,
                           duration: str = "day", tag: Optional[str] = None) -> Tuple[bool, Dict]:
        """Place bracket order via Tradier"""
        try:
            success, result = self.client.place_bracket_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                duration=duration,
                tag=tag
            )
            return success, result
        except Exception as e:
            logger.error(f"Error placing Tradier bracket order: {e}")
            return False, {}
    
    def is_connected(self) -> bool:
        """Check if Tradier client is functional"""
        try:
            # Try to get account balance as a health check
            success, _ = self.client.get_account_balance()
            return success
        except:
            return False


class IBKRAdapter(BrokerAdapter):
    """Adapter for IBKR client"""
    
    def __init__(self, ibkr_client):
        """
        Initialize IBKR adapter
        
        Args:
            ibkr_client: IBKRClient instance
        """
        self.client = ibkr_client
        logger.info("✅ IBKRAdapter initialized")
    
    def get_positions(self) -> Tuple[bool, List[Dict]]:
        """Get positions from IBKR"""
        try:
            positions = self.client.get_positions()
            
            # Normalize position format
            normalized = []
            for pos in positions:
                normalized.append({
                    'symbol': pos.symbol,
                    'quantity': int(pos.quantity),
                    'cost_basis': float(pos.avg_cost * abs(pos.quantity)),
                    'current_price': float(pos.market_price)
                })
            return True, normalized
        except Exception as e:
            logger.error(f"Error getting IBKR positions: {e}")
            return False, []
    
    def get_account_balance(self) -> Tuple[bool, Dict]:
        """Get account balance from IBKR"""
        try:
            account_info = self.client.get_account_info()
            if not account_info:
                return False, {}
            
            # Normalize balance format
            normalized = {
                'total_equity': float(account_info.net_liquidation),
                'cash': float(account_info.total_cash_value),
                'buying_power': float(account_info.buying_power)
            }
            return True, normalized
        except Exception as e:
            logger.error(f"Error getting IBKR balance: {e}")
            return False, {}
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get quote from IBKR"""
        try:
            market_data = self.client.get_market_data(symbol)
            if not market_data:
                return None
            
            # Normalize to Tradier-like format
            return {
                'symbol': symbol,
                'last': market_data.get('last'),
                'bid': market_data.get('bid'),
                'ask': market_data.get('ask'),
                'volume': market_data.get('volume', 0)
            }
        except Exception as e:
            logger.error(f"Error getting IBKR quote for {symbol}: {e}")
            return None
    
    def place_order(self, order_data: Dict) -> Tuple[bool, Dict]:
        """Place order via IBKR (generic)"""
        try:
            symbol = order_data.get('symbol')
            side = order_data.get('side')  # 'buy' or 'sell'
            quantity = int(order_data.get('quantity'))
            order_type = order_data.get('type', 'market')
            
            # Map to IBKR action ('BUY' or 'SELL')
            action = side.upper()
            
            if order_type == 'market':
                order = self.client.place_market_order(symbol, action, quantity)
            elif order_type == 'limit':
                limit_price = float(order_data.get('price'))
                order = self.client.place_limit_order(symbol, action, quantity, limit_price)
            elif order_type == 'stop':
                stop_price = float(order_data.get('stop'))
                order = self.client.place_stop_order(symbol, action, quantity, stop_price)
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return False, {}
            
            if order:
                return True, {'order': {'id': order.order_id, 'status': order.status}}
            return False, {}
        except Exception as e:
            logger.error(f"Error placing IBKR order: {e}")
            return False, {}
    
    def place_equity_order(self, symbol: str, side: str, quantity: int,
                          order_type: str = "market", duration: str = "day",
                          limit_price: Optional[float] = None,
                          stop_price: Optional[float] = None,
                          tag: Optional[str] = None) -> Tuple[bool, Dict]:
        """Place equity order via IBKR"""
        try:
            action = side.upper()
            
            if order_type == 'market':
                order = self.client.place_market_order(symbol, action, quantity)
            elif order_type == 'limit' and limit_price:
                order = self.client.place_limit_order(symbol, action, quantity, limit_price)
            elif order_type == 'stop' and stop_price:
                order = self.client.place_stop_order(symbol, action, quantity, stop_price)
            else:
                logger.error(f"Invalid order type or missing price: {order_type}")
                return False, {}
            
            if order:
                return True, {'order': {'id': order.order_id, 'status': order.status}}
            return False, {}
        except Exception as e:
            logger.error(f"Error placing IBKR equity order: {e}")
            return False, {}
    
    def place_bracket_order(self, symbol: str, side: str, quantity: int,
                           stop_loss_price: float, take_profit_price: float,
                           duration: str = "day", tag: Optional[str] = None) -> Tuple[bool, Dict]:
        """Place bracket order via IBKR"""
        try:
            action = side.upper()
            
            # For IBKR, we need to place parent order + 2 child orders
            # 1. Place the entry market order
            parent_order = self.client.place_market_order(symbol, action, quantity)
            if not parent_order:
                return False, {}
            
            logger.info(f"✅ IBKR parent order placed: {parent_order.order_id}")
            
            # 2. Place stop loss order (opposite action)
            stop_action = 'SELL' if action == 'BUY' else 'BUY'
            stop_order = self.client.place_stop_order(symbol, stop_action, quantity, stop_loss_price)
            
            # 3. Place take profit order (opposite action)
            profit_order = self.client.place_limit_order(symbol, stop_action, quantity, take_profit_price)
            
            logger.info(f"✅ IBKR bracket orders placed - Parent: {parent_order.order_id}, "
                       f"Stop: {stop_order.order_id if stop_order else 'failed'}, "
                       f"Profit: {profit_order.order_id if profit_order else 'failed'}")
            
            return True, {
                'order': {
                    'id': parent_order.order_id,
                    'status': parent_order.status,
                    'stop_order_id': stop_order.order_id if stop_order else None,
                    'profit_order_id': profit_order.order_id if profit_order else None
                }
            }
        except Exception as e:
            logger.error(f"Error placing IBKR bracket order: {e}")
            return False, {}
    
    def is_connected(self) -> bool:
        """Check if IBKR client is connected"""
        return self.client.is_connected()


def create_broker_adapter(broker_type: str, broker_client) -> BrokerAdapter:
    """
    Factory function to create the appropriate broker adapter
    
    Args:
        broker_type: "TRADIER" or "IBKR"
        broker_client: The broker client instance
        
    Returns:
        BrokerAdapter instance
    """
    broker_type_upper = broker_type.upper()
    
    if broker_type_upper == "TRADIER":
        return TradierAdapter(broker_client)
    elif broker_type_upper == "IBKR":
        return IBKRAdapter(broker_client)
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}. Use 'TRADIER' or 'IBKR'")


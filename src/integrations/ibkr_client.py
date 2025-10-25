"""
Interactive Brokers Client for Day Trading/Scalping
Supports live trading, market data, and position management for IBKR accounts
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
import sys

# Configure logging first
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Fix asyncio event loop issue with Streamlit - CRITICAL FIX
def setup_event_loop():
    """Setup event loop for Streamlit compatibility"""
    try:
        # Apply nest_asyncio to allow nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
            logger.info("nest_asyncio applied successfully")
        except ImportError:
            logger.warning("nest_asyncio not available, attempting alternative fix")
        
        # Set up event loop policy for Windows/Streamlit compatibility
        if sys.platform == 'win32':
            try:
                # Use SelectorEventLoop on Windows for better compatibility
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                logger.info("Windows SelectorEventLoopPolicy set")
            except Exception as e:
                logger.warning(f"Could not set Windows event loop policy: {e}")
        
        # Ensure there's an event loop in the current thread
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
            logger.info("Existing event loop found and active")
        except RuntimeError as e:
            logger.info(f"No event loop found ({e}), creating new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.info("New event loop created and set")
        
        return True
    except Exception as e:
        logger.error(f"Failed to setup event loop: {e}", exc_info=True)
        return False

# Setup event loop before importing ib_insync
setup_event_loop()

# Now import ib_insync
try:
    from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, StopOrder, Contract
    from ib_insync import util, Trade, Position, PortfolioItem, AccountValue
    IB_INSYNC_AVAILABLE = True
    logger.info("ib_insync imported successfully")
except ImportError as e:
    IB_INSYNC_AVAILABLE = False
    logger.warning(f"ib_insync not available: {e}")
except Exception as e:
    IB_INSYNC_AVAILABLE = False
    logger.error(f"Error importing ib_insync: {e}", exc_info=True)


class OrderAction(Enum):
    """Order action types"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order types supported"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


@dataclass
class IBKRPosition:
    """Represents a position in IBKR account"""
    symbol: str
    position: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    account: str


@dataclass
class IBKROrder:
    """Represents an order in IBKR"""
    order_id: int
    symbol: str
    action: str
    order_type: str
    quantity: int
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled: int
    remaining: int
    avg_fill_price: float


@dataclass
class IBKRAccountInfo:
    """Account information"""
    account_id: str
    net_liquidation: float
    total_cash_value: float
    settled_cash: float
    buying_power: float
    day_trades_remaining: int
    is_pdt: bool  # Pattern Day Trader flag


class IBKRClient:
    """
    Interactive Brokers Client for day trading and scalping
    Requires IB Gateway or TWS to be running
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Initialize IBKR client
        
        Args:
            host: IB Gateway/TWS host (default: localhost)
            port: IB Gateway port (7497 for paper, 7496 for live) or TWS port (7497)
            client_id: Unique client ID (1-32)
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError(
                "ib_insync is not installed. Please install it with: pip install ib_insync"
            )
        
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        self._event_loop = None
        self._thread = None
        
        logger.info(f"IBKR Client initialized for {host}:{port} (client_id={client_id})")
    
    def connect(self, timeout: int = 10) -> bool:
        """
        Connect to IB Gateway or TWS
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        try:
            if self.connected:
                logger.warning("Already connected to IBKR")
                return True
            
            # Connect to IB
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
            self.connected = True
            
            # Get account information
            accounts = self.ib.managedAccounts()
            logger.info(f"Successfully connected to IBKR. Accounts: {accounts}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            logger.error("Make sure IB Gateway or TWS is running and API connections are enabled")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"Error disconnecting from IBKR: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connected and self.ib.isConnected()
    
    def get_account_info(self, account: Optional[str] = None) -> Optional[IBKRAccountInfo]:
        """
        Get account information including buying power and cash
        
        Args:
            account: Account ID (uses first account if None)
            
        Returns:
            IBKRAccountInfo object or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Get account if not specified
            if account is None:
                accounts = self.ib.managedAccounts()
                if not accounts:
                    logger.error("No accounts found")
                    return None
                account = accounts[0]
            
            # Request account summary
            account_values = self.ib.accountValues(account)
            
            # Parse account values
            values_dict = {}
            for av in account_values:
                values_dict[av.tag] = av.value
            
            # Get day trades remaining (if applicable)
            day_trades = 0
            is_pdt = False
            if 'DayTradesRemaining' in values_dict:
                day_trades = int(float(values_dict.get('DayTradesRemaining', 0)))
                is_pdt = day_trades == 0 or float(values_dict.get('NetLiquidation', 0)) >= 25000
            
            return IBKRAccountInfo(
                account_id=account,
                net_liquidation=float(values_dict.get('NetLiquidation', 0)),
                total_cash_value=float(values_dict.get('TotalCashValue', 0)),
                settled_cash=float(values_dict.get('SettledCash', 0)),
                buying_power=float(values_dict.get('BuyingPower', 0)),
                day_trades_remaining=day_trades,
                is_pdt=is_pdt
            )
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self, account: Optional[str] = None) -> List[IBKRPosition]:
        """
        Get current positions
        
        Args:
            account: Account ID (uses all accounts if None)
            
        Returns:
            List of IBKRPosition objects
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return []
            
            positions = self.ib.positions(account)
            
            result = []
            for pos in positions:
                # Get market price
                self.ib.reqMktData(pos.contract, '', False, False)
                self.ib.sleep(0.5)  # Wait for data
                
                ticker = self.ib.ticker(pos.contract)
                market_price = ticker.marketPrice() if ticker.marketPrice() else pos.avgCost
                
                result.append(IBKRPosition(
                    symbol=pos.contract.symbol,
                    position=pos.position,
                    avg_cost=pos.avgCost,
                    market_price=market_price,
                    market_value=pos.position * market_price,
                    unrealized_pnl=pos.unrealizedPNL,
                    realized_pnl=pos.realizedPNL,
                    account=pos.account
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_open_orders(self) -> List[IBKROrder]:
        """
        Get all open orders
        
        Returns:
            List of IBKROrder objects
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return []
            
            trades = self.ib.openTrades()
            
            result = []
            for trade in trades:
                order = trade.order
                contract = trade.contract
                
                result.append(IBKROrder(
                    order_id=order.orderId,
                    symbol=contract.symbol,
                    action=order.action,
                    order_type=order.orderType,
                    quantity=int(order.totalQuantity),
                    limit_price=order.lmtPrice if order.lmtPrice else None,
                    stop_price=order.auxPrice if order.auxPrice else None,
                    status=trade.orderStatus.status,
                    filled=int(trade.orderStatus.filled),
                    remaining=int(trade.orderStatus.remaining),
                    avg_fill_price=trade.orderStatus.avgFillPrice
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def place_market_order(self, symbol: str, action: str, quantity: int) -> Optional[IBKROrder]:
        """
        Place a market order (for immediate execution)
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            
        Returns:
            IBKROrder object or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Create stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Qualify contract
            self.ib.qualifyContracts(contract)
            
            # Create market order
            order = MarketOrder(action, quantity)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to be submitted
            self.ib.sleep(1)
            
            logger.info(f"Market order placed: {action} {quantity} {symbol}")
            
            return IBKROrder(
                order_id=order.orderId,
                symbol=symbol,
                action=action,
                order_type='MKT',
                quantity=quantity,
                limit_price=None,
                stop_price=None,
                status=trade.orderStatus.status,
                filled=int(trade.orderStatus.filled),
                remaining=int(trade.orderStatus.remaining),
                avg_fill_price=trade.orderStatus.avgFillPrice
            )
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, action: str, quantity: int, 
                         limit_price: float) -> Optional[IBKROrder]:
        """
        Place a limit order
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            limit_price: Limit price
            
        Returns:
            IBKROrder object or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Create stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Create limit order
            order = LimitOrder(action, quantity, limit_price)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            
            logger.info(f"Limit order placed: {action} {quantity} {symbol} @ ${limit_price}")
            
            return IBKROrder(
                order_id=order.orderId,
                symbol=symbol,
                action=action,
                order_type='LMT',
                quantity=quantity,
                limit_price=limit_price,
                stop_price=None,
                status=trade.orderStatus.status,
                filled=int(trade.orderStatus.filled),
                remaining=int(trade.orderStatus.remaining),
                avg_fill_price=trade.orderStatus.avgFillPrice
            )
            
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def place_stop_order(self, symbol: str, action: str, quantity: int, 
                        stop_price: float) -> Optional[IBKROrder]:
        """
        Place a stop order (stop-loss or stop-entry)
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            stop_price: Stop trigger price
            
        Returns:
            IBKROrder object or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Create stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Create stop order
            order = StopOrder(action, quantity, stop_price)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(1)
            
            logger.info(f"Stop order placed: {action} {quantity} {symbol} @ stop ${stop_price}")
            
            return IBKROrder(
                order_id=order.orderId,
                symbol=symbol,
                action=action,
                order_type='STP',
                quantity=quantity,
                limit_price=None,
                stop_price=stop_price,
                status=trade.orderStatus.status,
                filled=int(trade.orderStatus.filled),
                remaining=int(trade.orderStatus.remaining),
                avg_fill_price=trade.orderStatus.avgFillPrice
            )
            
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return False
            
            # Find the trade
            trades = self.ib.openTrades()
            for trade in trades:
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Order {order_id} cancelled")
                    return True
            
            logger.warning(f"Order {order_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders
        
        Returns:
            Number of orders cancelled
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return 0
            
            self.ib.reqGlobalCancel()
            logger.info("All orders cancelled")
            
            return len(self.ib.openTrades())
            
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time market data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with market data or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Request market data
            self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)  # Wait for data
            
            # Get ticker
            ticker = self.ib.ticker(contract)
            
            return {
                'symbol': symbol,
                'bid': ticker.bid if ticker.bid else 0,
                'ask': ticker.ask if ticker.ask else 0,
                'last': ticker.last if ticker.last else 0,
                'close': ticker.close if ticker.close else 0,
                'volume': ticker.volume if ticker.volume else 0,
                'bid_size': ticker.bidSize if ticker.bidSize else 0,
                'ask_size': ticker.askSize if ticker.askSize else 0,
            }
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def flatten_position(self, symbol: str) -> bool:
        """
        Close entire position for a symbol (day trader's panic button)
        
        Args:
            symbol: Stock symbol to close
            
        Returns:
            True if order placed successfully
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return False
            
            # Get current position
            positions = self.get_positions()
            position_qty = 0
            
            for pos in positions:
                if pos.symbol == symbol:
                    position_qty = pos.position
                    break
            
            if position_qty == 0:
                logger.warning(f"No position found for {symbol}")
                return False
            
            # Determine action (opposite of current position)
            action = 'SELL' if position_qty > 0 else 'BUY'
            quantity = abs(int(position_qty))
            
            # Place market order to close
            order = self.place_market_order(symbol, action, quantity)
            
            if order:
                logger.info(f"Flattened position for {symbol}: {action} {quantity}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error flattening position for {symbol}: {e}")
            return False
    
    def get_historical_data(self, symbol: str, duration: str = "1 D", 
                           bar_size: str = "1 min") -> Optional[List[Dict]]:
        """
        Get historical bar data for charting/analysis
        
        Args:
            symbol: Stock symbol
            duration: Duration string (e.g., "1 D", "1 W", "1 M")
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day")
            
        Returns:
            List of bar dictionaries or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Create contract
            contract = Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            # Convert to list of dicts
            result = []
            for bar in bars:
                result.append({
                    'date': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None


def create_ibkr_client_from_env() -> Optional[IBKRClient]:
    """
    Create IBKR client from environment variables
    
    Environment variables:
        IBKR_HOST: IB Gateway/TWS host (default: 127.0.0.1)
        IBKR_PORT: Port number (7497 for paper, 7496 for live)
        IBKR_CLIENT_ID: Client ID (default: 1)
    
    Returns:
        IBKRClient instance or None if error
    """
    import os
    
    host = os.getenv('IBKR_HOST', '127.0.0.1')
    port = int(os.getenv('IBKR_PORT', '7497'))  # Default to paper trading
    client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
    
    try:
        client = IBKRClient(host=host, port=port, client_id=client_id)
        return client
    except Exception as e:
        logger.error(f"Error creating IBKR client from environment: {e}")
        return None


def validate_ibkr_connection() -> Tuple[bool, str]:
    """
    Validate IBKR connection
    
    Returns:
        Tuple of (success, message)
    """
    try:
        client = create_ibkr_client_from_env()
        if client is None:
            return False, "Failed to create IBKR client (check ib_insync installation)"
        
        if not client.connect():
            return False, "Failed to connect to IBKR (make sure IB Gateway/TWS is running)"
        
        # Get account info to verify connection
        account_info = client.get_account_info()
        if account_info is None:
            client.disconnect()
            return False, "Connected but failed to retrieve account information"
        
        client.disconnect()
        return True, f"Successfully connected to IBKR account {account_info.account_id}"
        
    except Exception as e:
        return False, f"IBKR validation error: {str(e)}"

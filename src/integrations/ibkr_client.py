"""
Interactive Brokers Client for Day Trading/Scalping
Supports live trading, market data, and position management for IBKR accounts
Integrated with Trading Mode Manager for paper and live trading
"""

from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
import sys

# Loguru logging is configured in utils/logging_config.py
# No need to configure here

# Import trading configuration
try:
    from .trading_config import get_trading_mode_manager, TradingMode
    TRADING_CONFIG_AVAILABLE = True
except ImportError:
    TRADING_CONFIG_AVAILABLE = False
    logger.warning("Trading config not available, using legacy mode")

# Import hybrid data fetcher
try:
    from .hybrid_data_fetcher import HybridDataFetcher
    HYBRID_DATA_AVAILABLE = True
except ImportError:
    HYBRID_DATA_AVAILABLE = False
    logger.debug("Hybrid data fetcher not available")

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
        logger.error("Failed to setup event loop: {}", str(e), exc_info=True)
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
    logger.error("Error importing ib_insync: {}", str(e), exc_info=True)


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
    Supports paper and live trading modes via TradingModeManager
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1, trading_mode: Optional['TradingMode'] = None, market_data_source: str = "IBKR"):
        """
        Initialize IBKR client
        
        Args:
            host: IB Gateway/TWS host (default: localhost)
            port: IB Gateway port (7497 for paper TWS, 7496 for live TWS, 4002 for paper Gateway, 4001 for live Gateway)
            client_id: Unique client ID (1-32)
            trading_mode: Trading mode (PAPER or PRODUCTION). If None, uses provided connection details.
            market_data_source: Data source ("IBKR", "YFINANCE", or "HYBRID")
        """
        if not IB_INSYNC_AVAILABLE:
            raise ImportError(
                "ib_insync is not installed. Please install it with: pip install ib_insync"
            )
        
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.trading_mode = trading_mode
        self.connected = False
        self._event_loop = None
        self._thread = None
        
        # Market data configuration
        self.market_data_source = market_data_source.upper()
        self.hybrid_fetcher = None
        
        # Initialize hybrid data fetcher if needed
        if HYBRID_DATA_AVAILABLE and self.market_data_source in ["HYBRID", "YFINANCE"]:
            prefer_yfinance = (self.market_data_source == "YFINANCE")
            self.hybrid_fetcher = HybridDataFetcher(ibkr_client=self, prefer_yfinance=prefer_yfinance)
            logger.info(f"📊 Hybrid Data Fetcher initialized (mode: {self.market_data_source})")
        
        mode_str = f" ({trading_mode.value} mode)" if trading_mode else ""
        logger.info(f"IBKR Client initialized for {host}:{port} (client_id={client_id}){mode_str}")
    
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
            
            # Set request timeout to prevent hanging (default is 60s)
            self.ib.RequestTimeout = 5.0  # 5 second timeout for requests
            logger.info(f"Set IB.RequestTimeout to {self.ib.RequestTimeout}s")
            
            # Connect to IB
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
            self.connected = True
            
            # Get account information
            accounts = self.ib.managedAccounts()
            logger.info(f"Successfully connected to IBKR. Accounts: {accounts}")
            
            # Request delayed market data (free for paper trading)
            # Type 3 = delayed data, Type 4 = delayed frozen data
            try:
                self.ib.reqMarketDataType(3)
                logger.info("✅ Delayed market data enabled (free for paper trading)")
            except Exception as e:
                logger.warning(f"Could not enable delayed market data: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            logger.error("Make sure IB Gateway or TWS is running and API connections are enabled")
            self.connected = False
            return False
    
    def _qualify_contract_with_timeout(self, contract, timeout: float = 10.0):
        """
        Qualify a contract using ib_insync's built-in RequestTimeout
        
        The IB.RequestTimeout is set during connect() to prevent hanging.
        When a timeout occurs, ib_insync raises an asyncio.TimeoutError.
        
        Args:
            contract: The contract to qualify
            timeout: Timeout in seconds (handled by IB.RequestTimeout)
            
        Returns:
            Qualified contract or None if error/timeout
        """
        try:
            logger.debug(f"Qualifying contract (using IB.RequestTimeout={self.ib.RequestTimeout}s)...")
            start_time = datetime.now()
            
            # Use qualifyContracts - ib_insync will handle timeout via RequestTimeout
            qualified = self.ib.qualifyContracts(contract)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if not qualified or len(qualified) == 0:
                logger.error("❌ Failed to qualify contract - no results returned")
                logger.error("   💡 TIP: Check if symbol is valid and TWS is responsive")
                return None
                
            logger.debug(f"✅ Contract qualified in {elapsed:.2f}s")
            return qualified[0]
            
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Contract qualification timeout after {self.ib.RequestTimeout}s")
            logger.error("   💡 TIP: TWS may be frozen - try restarting it")
            return None
        except Exception as e:
            logger.error("❌ Error qualifying contract: {}", str(e), exc_info=True)
            logger.error("   💡 TIP: Verify symbol is valid and TWS is connected")
            return None
    
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
    
    def check_connection_health(self) -> bool:
        """
        Check if IBKR connection is healthy and responsive
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if not self.is_connected():
                logger.warning("🔴 IBKR connection check failed: Not connected")
                return False
            
            # Quick health check - try to get account info
            start_time = datetime.now()
            try:
                accounts = self.ib.managedAccounts()
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if elapsed > 5.0:
                    logger.warning(f"⚠️ IBKR connection slow: {}s response time {elapsed:.1f}")
                    return False
                
                if not accounts or len(accounts) == 0:
                    logger.warning("🔴 IBKR connection check failed: No accounts")
                    return False
                
                logger.debug(f"✅ IBKR connection healthy ({}s) {elapsed:.2f}")
                return True
                
            except Exception as e:
                logger.error(f"🔴 IBKR health check failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking IBKR connection health: {e}")
            return False
    
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
    
    def _ensure_event_loop(self):
        """
        Ensure there's an event loop in the current thread.
        Critical for methods called from non-main threads (like PositionExitMonitor).
        """
        try:
            loop = asyncio.get_event_loop()
            # Check if loop is closed or None
            if loop is None or loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                logger.debug("Created new event loop for current thread")
        except RuntimeError:
            # No event loop in current thread - create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            logger.debug("Created event loop for thread without loop")
    
    def get_positions(self, account: Optional[str] = None) -> List[IBKRPosition]:
        """
        Get current positions
        
        Args:
            account: Account ID (uses all accounts if None)
            
        Returns:
            List of IBKRPosition objects
        """
        try:
            # Ensure event loop exists in current thread (critical for multi-threading)
            self._ensure_event_loop()
            
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
                
                # Calculate unrealized PnL manually
                # Position object doesn't have unrealizedPNL attribute
                unrealized_pnl = (market_price - pos.avgCost) * pos.position
                
                result.append(IBKRPosition(
                    symbol=pos.contract.symbol,
                    position=pos.position,
                    avg_cost=pos.avgCost,
                    market_price=market_price,
                    market_value=pos.position * market_price,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=0.0,  # Position object doesn't track realized PnL
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
            # Ensure event loop exists in current thread (critical for multi-threading)
            self._ensure_event_loop()
            
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
    
    def place_market_order(self, symbol: str, action: str, quantity: float) -> Optional[IBKROrder]:
        """
        Place a market order (for immediate execution)
        Supports fractional shares for US stocks
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares (can be fractional, e.g., 0.5)
            
        Returns:
            IBKROrder object or None if error
        """
        try:
            # Ensure event loop exists in current thread (critical for multi-threading)
            self._ensure_event_loop()
            
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            logger.info(f"🔄 Creating contract for {symbol}...")
            # Create stock contract - SMART routing for US stocks
            # Note: For standard US stocks, we can skip qualification to avoid timeout issues
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Log if this is a fractional share order
            is_fractional = (quantity % 1 != 0)
            if is_fractional:
                logger.info(f"📊 Fractional share order: {quantity} shares")
            logger.info(f"✅ Contract created for {symbol} (SMART routing, USD)")
            
            # Create market order
            logger.info(f"📝 Creating market order: {action} {quantity} {symbol}")
            order = MarketOrder(action, quantity)
            
            # Place order
            logger.info(f"📤 Placing order with IBKR...")
            trade = self.ib.placeOrder(contract, order)
            
            # Wait briefly for order to be submitted
            self.ib.sleep(0.5)
            
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
    
    def place_limit_order(self, symbol: str, action: str, quantity: float, 
                         limit_price: float) -> Optional[IBKROrder]:
        """
        Place a limit order
        Supports fractional shares for US stocks
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares (can be fractional, e.g., 0.5)
            limit_price: Limit price
            
        Returns:
            IBKROrder object or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            logger.info(f"🔄 Creating limit order contract for {symbol}...")
            # Create stock contract - SMART routing for US stocks
            contract = Stock(symbol, 'SMART', 'USD')
            logger.info(f"✅ Contract created for {symbol} (SMART routing, USD)")
            
            # Create limit order
            logger.info(f"📝 Creating limit order: {action} {quantity} {symbol} @ ${limit_price}")
            order = LimitOrder(action, quantity, limit_price)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(0.5)
            
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
    
    def place_stop_order(self, symbol: str, action: str, quantity: float, 
                        stop_price: float) -> Optional[IBKROrder]:
        """
        Place a stop order (stop-loss or stop-entry)
        Supports fractional shares for US stocks
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares (can be fractional, e.g., 0.5)
            stop_price: Stop trigger price
            
        Returns:
            IBKROrder object or None if error
        """
        try:
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            logger.info(f"🔄 Creating stop order contract for {symbol}...")
            # Create stock contract - SMART routing for US stocks
            contract = Stock(symbol, 'SMART', 'USD')
            logger.info(f"✅ Contract created for {symbol} (SMART routing, USD)")
            
            # Create stop order
            logger.info(f"📝 Creating stop order: {action} {quantity} {symbol} @ ${stop_price}")
            order = StopOrder(action, quantity, stop_price)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            self.ib.sleep(0.5)
            
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
    
    def place_bracket_order(self, symbol: str, action: str, quantity: float,
                           take_profit_price: float, stop_loss_price: float,
                           limit_price: Optional[float] = None) -> Optional[Dict]:
        """
        Place a proper IBKR bracket order (parent + 2 child orders)
        Uses ib_insync's bracketOrder() for proper order linking
        Supports fractional shares for US stocks
        
        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares (can be fractional, e.g., 0.5)
            take_profit_price: Take profit limit price
            stop_loss_price: Stop loss price
            limit_price: Limit price for parent order (if None, uses midpoint)
            
        Returns:
            Dict with parent, takeProfit, and stopLoss order IDs, or None if error
        """
        try:
            # Ensure event loop exists in current thread (critical for multi-threading)
            self._ensure_event_loop()
            
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            logger.info(f"🎯 Creating IBKR bracket order for {symbol}: {action} {quantity} shares")
            logger.info(f"   Take Profit: ${}, Stop Loss: ${stop_loss_price:.2f} {take_profit_price:.2f}")
            
            # Create stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # If no limit price provided, get current ask price to ensure immediate fill
            if limit_price is None:
                # Get current market price
                self.ib.reqMktData(contract, '', False, False)
                self.ib.sleep(0.5)
                ticker = self.ib.ticker(contract)
                
                # For BUY: use ask price to ensure immediate fill
                # For SELL: use bid price to ensure immediate fill
                if action.upper() == 'BUY':
                    limit_price = ticker.ask if ticker.ask and ticker.ask > 0 else take_profit_price
                    logger.info(f"   Using current ASK as limit for immediate fill: ${} {limit_price:.2f}")
                else:
                    limit_price = ticker.bid if ticker.bid and ticker.bid > 0 else stop_loss_price
                    logger.info(f"   Using current BID as limit for immediate fill: ${} {limit_price:.2f}")
            
            # Create bracket order using ib_insync's built-in function
            # This properly links parent and child orders with OCA group
            parent, takeProfit, stopLoss = self.ib.bracketOrder(
                action=action.upper(),
                quantity=quantity,
                limitPrice=limit_price,
                takeProfitPrice=take_profit_price,
                stopLossPrice=stop_loss_price
            )
            
            # Place all three orders (parent + both children)
            # They are already linked by bracketOrder() via OCA group
            logger.info(f"📤 Placing bracket order with IBKR...")
            logger.info(f"   Submitting parent order...")
            parent_trade = self.ib.placeOrder(contract, parent)
            self.ib.sleep(0.2)
            
            logger.info(f"   Submitting take-profit order...")
            tp_trade = self.ib.placeOrder(contract, takeProfit)
            self.ib.sleep(0.2)
            
            logger.info(f"   Submitting stop-loss order...")
            sl_trade = self.ib.placeOrder(contract, stopLoss)
            self.ib.sleep(0.2)
            
            logger.info(f"✅ IBKR bracket order placed successfully")
            logger.info(f"   Parent Order ID: {parent.orderId}")
            logger.info(f"   Take Profit ID: {takeProfit.orderId}")
            logger.info(f"   Stop Loss ID: {stopLoss.orderId}")
            
            return {
                'parent': {
                    'order_id': parent.orderId,
                    'status': parent_trade.orderStatus.status
                },
                'take_profit': {
                    'order_id': takeProfit.orderId,
                    'price': take_profit_price
                },
                'stop_loss': {
                    'order_id': stopLoss.orderId,
                    'price': stop_loss_price
                }
            }
            
        except Exception as e:
            logger.error("Error placing bracket order: {}", str(e), exc_info=True)
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
            # Ensure event loop exists in current thread (critical for multi-threading)
            self._ensure_event_loop()
            
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
    
    def get_market_data_direct(self, symbol: str, timeout: float = 0.5) -> Optional[Dict]:
        """
        Get market data directly from IBKR (bypasses hybrid fetcher)
        Used internally by hybrid fetcher to prevent recursion
        
        Args:
            symbol: Stock symbol
            timeout: Seconds to wait for data (default 0.5 for fast bulk checks)
            
        Returns:
            Dictionary with market data or None if error
        """
        try:
            # Ensure event loop exists in current thread (critical for multi-threading)
            self._ensure_event_loop()
            
            if not self.is_connected():
                logger.error("Not connected to IBKR")
                return None
            
            # Create contract with timeout protection
            contract = Stock(symbol, 'SMART', 'USD')
            contract = self._qualify_contract_with_timeout(contract, timeout=5.0)
            if not contract:
                logger.debug(f"Failed to qualify contract for {symbol}")
                return None
            
            # Request market data (will use delayed if real-time not available)
            self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(timeout)  # Configurable wait time
            
            # Get ticker
            ticker = self.ib.ticker(contract)
            
            # Use delayed data if available
            last_price = ticker.last if ticker.last and not (ticker.last != ticker.last) else \
                        ticker.close if ticker.close and not (ticker.close != ticker.close) else 0
            
            # If no price data, return None quickly
            if last_price == 0:
                logger.debug(f"No price data for {symbol} after {timeout}s")
                return None
            
            return {
                'symbol': symbol,
                'bid': ticker.bid if ticker.bid and not (ticker.bid != ticker.bid) else 0,
                'ask': ticker.ask if ticker.ask and not (ticker.ask != ticker.ask) else 0,
                'last': last_price,
                'close': ticker.close if ticker.close else 0,
                'volume': ticker.volume if ticker.volume else 0,
                'bid_size': ticker.bidSize if ticker.bidSize else 0,
                'ask_size': ticker.askSize if ticker.askSize else 0,
                'source': 'IBKR'
            }
            
        except Exception as e:
            logger.debug(f"Error getting direct IBKR market data for {symbol}: {e}")
            return None
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get market data for a symbol
        Uses hybrid data fetcher if configured, otherwise uses IBKR directly
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with market data or None if error
        """
        try:
            # Use hybrid fetcher if available and configured
            if self.hybrid_fetcher and self.market_data_source in ["HYBRID", "YFINANCE"]:
                use_ibkr = (self.market_data_source == "HYBRID")  # Try IBKR first in hybrid mode
                quote = self.hybrid_fetcher.get_quote(symbol, use_ibkr=use_ibkr)
                if quote:
                    logger.debug("📊 {} quote from {quote.get('source', 'unknown')}", str(symbol))
                    return quote
                logger.warning(f"Hybrid fetcher failed for {symbol}, falling back to IBKR")
            
            # Fallback to direct IBKR data
            return self.get_market_data_direct(symbol)
            
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
            
            # Create contract with timeout protection
            contract = Stock(symbol, 'SMART', 'USD')
            contract = self._qualify_contract_with_timeout(contract, timeout=10.0)
            if not contract:
                logger.error(f"Failed to qualify contract for {symbol}")
                return None
            
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


def create_ibkr_client_from_env(trading_mode: Optional['TradingMode'] = None, market_data_source: str = "IBKR") -> Optional[IBKRClient]:
    """
    Create IBKR client from environment variables or trading mode manager
    
    Args:
        trading_mode: Trading mode (PAPER or PRODUCTION). If None, tries to load from trading mode manager.
        market_data_source: Data source ("IBKR", "YFINANCE", or "HYBRID")
    
    Environment variables (legacy support):
        IBKR_HOST: IB Gateway/TWS host (default: 127.0.0.1)
        IBKR_PORT: Port number (7497 for paper TWS, 7496 for live TWS, 4002 for paper Gateway, 4001 for live Gateway)
        IBKR_CLIENT_ID: Client ID (default: 1)
    
    Or use mode-specific variables:
        IBKR_PAPER_HOST, IBKR_PAPER_PORT, IBKR_PAPER_CLIENT_ID
        IBKR_LIVE_HOST, IBKR_LIVE_PORT, IBKR_LIVE_CLIENT_ID
    
    Returns:
        IBKRClient instance or None if error
    """
    import os
    
    try:
        # Try using TradingModeManager first if available
        if TRADING_CONFIG_AVAILABLE:
            mode_manager = get_trading_mode_manager()
            
            # Set mode if specified
            if trading_mode:
                if not mode_manager.set_mode(trading_mode):
                    logger.warning(f"No IBKR credentials available for {trading_mode.value} mode, falling back to env vars")
                else:
                    creds = mode_manager.get_ibkr_credentials()
                    if creds:
                        logger.info(f"🔧 Creating IBKR client from {trading_mode.value} mode credentials")
                        return IBKRClient(
                            host=creds.host,
                            port=creds.port,
                            client_id=creds.client_id,
                            trading_mode=trading_mode,
                            market_data_source=market_data_source
                        )
            else:
                # Use current mode
                creds = mode_manager.get_ibkr_credentials()
                if creds:
                    current_mode = mode_manager.get_mode()
                    logger.info(f"🔧 Creating IBKR client from current {current_mode.value} mode credentials")
                    return IBKRClient(
                        host=creds.host,
                        port=creds.port,
                        client_id=creds.client_id,
                        trading_mode=current_mode,
                        market_data_source=market_data_source
                    )
        
        # Fall back to legacy environment variables
        host = os.getenv('IBKR_HOST', '127.0.0.1')
        port = int(os.getenv('IBKR_PORT', '7497'))  # Default to paper trading TWS
        client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
        
        logger.info("🔧 Creating IBKR client from legacy environment variables")
        client = IBKRClient(host=host, port=port, client_id=client_id, trading_mode=trading_mode, market_data_source=market_data_source)
        return client
        
    except Exception as e:
        logger.error(f"Error creating IBKR client from environment: {e}")
        return None


def create_ibkr_client_for_mode(trading_mode: 'TradingMode') -> Optional[IBKRClient]:
    """
    Create IBKR client for a specific trading mode
    
    Args:
        trading_mode: Trading mode (PAPER or PRODUCTION)
    
    Returns:
        IBKRClient instance or None if error
    """
    try:
        return create_ibkr_client_from_env(trading_mode)
    except Exception as e:
        logger.error(f"Failed to create IBKR client for {trading_mode.value}: {e}")
        return None


def validate_ibkr_connection(trading_mode: Optional['TradingMode'] = None) -> Tuple[bool, str]:
    """
    Validate IBKR connection for current or specified mode
    
    Args:
        trading_mode: Trading mode to validate. If None, uses current mode.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        client = create_ibkr_client_from_env(trading_mode)
        if client is None:
            mode_name = trading_mode.value if trading_mode else "current"
            return False, f"Failed to create IBKR client for {mode_name} mode (check ib_insync installation and credentials)"
        
        if not client.connect():
            mode_name = client.trading_mode.value if client.trading_mode else "unknown"
            return False, f"Failed to connect to IBKR {mode_name} mode (make sure IB Gateway/TWS is running)"
        
        # Get account info to verify connection
        account_info = client.get_account_info()
        if account_info is None:
            client.disconnect()
            return False, "Connected but failed to retrieve account information"
        
        mode_name = client.trading_mode.value if client.trading_mode else "unknown"
        account_id = account_info.account_id
        client.disconnect()
        return True, f"Successfully connected to IBKR {mode_name} account {account_id}"
        
    except Exception as e:
        return False, f"IBKR validation error: {str(e)}"


def validate_all_ibkr_modes() -> Dict[str, Tuple[bool, str]]:
    """
    Validate IBKR connections for all available trading modes
    
    Returns:
        Dictionary mapping mode names to (success, message) tuples
    """
    if not TRADING_CONFIG_AVAILABLE:
        return {"error": (False, "Trading config not available")}
    
    mode_manager = get_trading_mode_manager()
    results = {}
    
    for mode in [TradingMode.PAPER, TradingMode.PRODUCTION]:
        if mode_manager.has_ibkr_credentials(mode):
            success, message = validate_ibkr_connection(mode)
            results[mode.value] = (success, message)
        else:
            results[mode.value] = (False, f"No IBKR credentials configured for {mode.value} mode")
    
    return results

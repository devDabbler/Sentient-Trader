"""
Kraken API Client for Cryptocurrency Trading Integration
Handles account management, market data, order placement, and position tracking for crypto assets

Kraken API Documentation: https://docs.kraken.com/rest/
"""

import requests
from loguru import logger
import time
import hmac
import hashlib
import base64
import urllib.parse
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from functools import wraps



class OrderType(Enum):
    """Kraken order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop-loss"
    TAKE_PROFIT = "take-profit"
    STOP_LOSS_LIMIT = "stop-loss-limit"
    TAKE_PROFIT_LIMIT = "take-profit-limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class KrakenBalance:
    """Account balance information"""
    currency: str
    balance: float
    available: float
    hold: float


@dataclass
class KrakenPosition:
    """Open position information"""
    pair: str
    side: str
    volume: float
    cost: float
    fee: float
    margin: float
    net_pnl: float
    current_price: float
    entry_price: float


@dataclass
class KrakenOrder:
    """Order information"""
    order_id: str
    pair: str
    side: str
    order_type: str
    price: Optional[float]
    volume: float
    status: str
    timestamp: datetime
    executed_volume: float = 0.0
    avg_price: float = 0.0


def retry_on_kraken_error(max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """
    Decorator to retry Kraken API calls on rate limits and server errors
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    # Check if it's a rate limit (429) or server error (5xx)
                    if e.response is not None and e.response.status_code in [429, 500, 502, 503, 504]:
                        last_exception = e
                        error_code = e.response.status_code
                        
                        if error_code == 429:
                            error_name = "Rate Limit Exceeded"
                        else:
                            error_name = f"{error_code} Server Error"
                        
                        if attempt < max_retries:
                            logger.warning(
                                f"⏳ {error_name} on {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                                f"Retrying in {delay:.1f}s..."
                            )
                            time.sleep(delay)
                            delay *= backoff_factor
                            continue
                        else:
                            logger.error(
                                f"❌ {error_name} on {func.__name__} after {max_retries + 1} attempts."
                            )
                    # Re-raise non-rate-limit/server errors immediately
                    raise
                except requests.exceptions.Timeout as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"⏳ Connection timeout on {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                        continue
                    else:
                        logger.error(f"❌ Connection timeout on {func.__name__} after {max_retries + 1} attempts.")
                    raise
                except Exception:
                    # Re-raise all other exceptions immediately
                    raise
            
            # If we exhausted all retries, raise the last exception
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class KrakenClient:
    """Client for Kraken cryptocurrency exchange API integration"""
    
    # Kraken REST API endpoints
    API_URL = "https://api.kraken.com"
    API_VERSION = "0"
    
    # Common crypto pairs on Kraken (in Kraken format)
    POPULAR_PAIRS = {
        # Major cryptocurrencies
        'BTC/USD': 'XXBTZUSD',
        'ETH/USD': 'XETHZUSD',
        'SOL/USD': 'SOLUSD',
        'XRP/USD': 'XXRPZUSD',
        'ADA/USD': 'ADAUSD',
        'DOGE/USD': 'XDGZUSD',
        'DOT/USD': 'DOTUSD',
        'MATIC/USD': 'MATICUSD',
        'AVAX/USD': 'AVAXUSD',
        'LINK/USD': 'LINKUSD',
        'ATOM/USD': 'ATOMUSD',
        'UNI/USD': 'UNIUSD',
        'LTC/USD': 'XLTCZUSD',
        'BCH/USD': 'BCHUSD',
        'ALGO/USD': 'ALGOUSD',
        
        # DeFi tokens
        'AAVE/USD': 'AAVEUSD',
        'MKR/USD': 'MKRUSD',
        'COMP/USD': 'COMPUSD',
        'CRV/USD': 'CRVUSD',
        'SNX/USD': 'SNXUSD',
        
        # Layer 2 / Scaling
        'MATIC/USD': 'MATICUSD',
        'OP/USD': 'OPUSD',
        'ARB/USD': 'ARBUSD',
    }
    
    def __init__(self, api_key: str, api_secret: str, timeout: int = 30):
        """
        Initialize Kraken client
        
        Args:
            api_key: Kraken API key
            api_secret: Kraken API secret (private key)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Sentient-Trader-Kraken-Client/1.0'
        })
        
        logger.info("🔧 KrakenClient initialized")
        logger.info(f"🔗 API URL: {self.API_URL}")
        logger.info(f"🔑 API Key: {self.api_key[:8]}...")
    
    def _get_kraken_signature(self, urlpath: str, data: Dict, nonce: str) -> str:
        """
        Generate Kraken API signature for authentication
        
        Args:
            urlpath: API endpoint path
            data: Request data
            nonce: Unique nonce value
            
        Returns:
            Base64-encoded signature
        """
        postdata = urllib.parse.urlencode(data)
        encoded = (str(nonce) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()
    
    @retry_on_kraken_error(max_retries=3)
    def _public_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a public (unauthenticated) API request
        
        Args:
            endpoint: API endpoint (e.g., 'Ticker', 'OHLC')
            params: Query parameters
            
        Returns:
            API response as dict
        """
        url = f"{self.API_URL}/{self.API_VERSION}/public/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Kraken returns errors in the 'error' field
            if data.get('error'):
                error_msg = ', '.join(data['error'])
                logger.error(f"Kraken API error: {error_msg}")
                raise ValueError(f"Kraken API error: {error_msg}")
            
            return data.get('result', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Public API request failed for {endpoint}: {e}")
            raise
    
    @retry_on_kraken_error(max_retries=3)
    def _private_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a private (authenticated) API request
        
        Args:
            endpoint: API endpoint (e.g., 'Balance', 'AddOrder')
            params: Request parameters
            
        Returns:
            API response as dict
        """
        if params is None:
            params = {}
        
        # Add nonce to params
        nonce = str(int(time.time() * 1000))
        params['nonce'] = nonce
        
        urlpath = f"/{self.API_VERSION}/private/{endpoint}"
        url = f"{self.API_URL}{urlpath}"
        
        # Generate signature
        signature = self._get_kraken_signature(urlpath, params, nonce)
        
        headers = {
            'API-Key': self.api_key,
            'API-Sign': signature
        }
        
        try:
            response = self.session.post(
                url,
                headers=headers,
                data=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Check for errors
            if data.get('error'):
                error_msg = ', '.join(data['error'])
                logger.error(f"Kraken API error: {error_msg}")
                raise ValueError(f"Kraken API error: {error_msg}")
            
            return data.get('result', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Private API request failed for {endpoint}: {e}")
            raise
    
    # =========================================================================
    # MARKET DATA METHODS
    # =========================================================================
    
    def get_ticker_data(self, pair: str) -> Dict:
        """
        Get ticker information for a trading pair
        
        Args:
            pair: Trading pair (e.g., 'XXBTZUSD' or 'BTC/USD')
            
        Returns:
            Ticker data including price, volume, high, low
        """
        # Convert user-friendly format to Kraken format if needed
        kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
        
        try:
            data = self._public_request('Ticker', {'pair': kraken_pair})
            
            # Extract ticker data
            if kraken_pair in data:
                ticker = data[kraken_pair]
                return {
                    'pair': pair,
                    'last_price': float(ticker['c'][0]),  # Last trade price
                    'bid': float(ticker['b'][0]),  # Best bid
                    'ask': float(ticker['a'][0]),  # Best ask
                    'high_24h': float(ticker['h'][1]),  # 24h high
                    'low_24h': float(ticker['l'][1]),  # 24h low
                    'volume_24h': float(ticker['v'][1]),  # 24h volume
                    'vwap_24h': float(ticker['p'][1]),  # 24h volume-weighted average price
                    'trades_24h': int(ticker['t'][1])  # Number of trades
                }
            else:
                logger.error(f"No data found for pair: {kraken_pair}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching ticker data for {pair}: {e}")
            return {}
    
    def get_ticker_info(self, pair: str) -> Dict:
        """
        Get raw ticker information for a trading pair (raw Kraken format)
        
        Args:
            pair: Trading pair (e.g., 'XXBTZUSD' or 'BTC/USD')
            
        Returns:
            Raw ticker data in Kraken format (includes 'c' key for last price)
        """
        # Convert user-friendly format to Kraken format if needed
        kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
        
        try:
            data = self._public_request('Ticker', {'pair': kraken_pair})
            
            # Return raw ticker data if found
            if kraken_pair in data:
                return data[kraken_pair]
            else:
                logger.debug(f"No data found for pair: {kraken_pair}")
                return {}
                
        except Exception as e:
            logger.debug(f"Error fetching ticker info for {pair}: {e}")
            return {}
    
    def get_ohlc_data(self, pair: str, interval: int = 60, since: Optional[int] = None) -> List[Dict]:
        """
        Get OHLC (candlestick) data for a trading pair
        
        Args:
            pair: Trading pair
            interval: Time interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Return committed OHLC data since given ID
            
        Returns:
            List of OHLC candles
        """
        kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
        
        params = {
            'pair': kraken_pair,
            'interval': str(interval)
        }
        
        if since:
            params['since'] = str(since)
        
        try:
            data = self._public_request('OHLC', params)
            
            ohlc_data = []
            if kraken_pair in data:
                for candle in data[kraken_pair]:
                    ohlc_data.append({
                        'timestamp': int(candle[0]),
                        'datetime': datetime.fromtimestamp(int(candle[0])),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'vwap': float(candle[5]),
                        'volume': float(candle[6]),
                        'count': int(candle[7])
                    })
            
            return ohlc_data
            
        except Exception as e:
            logger.error(f"Error fetching OHLC data for {pair}: {e}")
            return []
    
    def get_orderbook(self, pair: str, count: int = 10) -> Dict:
        """
        Get order book (depth of market) for a trading pair
        
        Args:
            pair: Trading pair
            count: Maximum number of asks/bids to return
            
        Returns:
            Order book with bids and asks
        """
        kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
        
        try:
            data = self._public_request('Depth', {'pair': kraken_pair, 'count': count})
            
            if kraken_pair in data:
                book = data[kraken_pair]
                return {
                    'pair': pair,
                    'bids': [{'price': float(b[0]), 'volume': float(b[1]), 'timestamp': int(b[2])} 
                            for b in book['bids']],
                    'asks': [{'price': float(a[0]), 'volume': float(a[1]), 'timestamp': int(a[2])} 
                            for a in book['asks']]
                }
            
            return {'pair': pair, 'bids': [], 'asks': []}
            
        except Exception as e:
            logger.error(f"Error fetching order book for {pair}: {e}")
            return {'pair': pair, 'bids': [], 'asks': []}
    
    def get_recent_trades(self, pair: str, since: Optional[int] = None) -> List[Dict]:
        """
        Get recent trades for a trading pair
        
        Args:
            pair: Trading pair
            since: Return trades since this timestamp
            
        Returns:
            List of recent trades
        """
        kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
        
        params = {'pair': kraken_pair}
        if since:
            params['since'] = since
        
        try:
            data = self._public_request('Trades', params)
            
            trades = []
            if kraken_pair in data:
                for trade in data[kraken_pair]:
                    trades.append({
                        'price': float(trade[0]),
                        'volume': float(trade[1]),
                        'timestamp': int(trade[2]),
                        'datetime': datetime.fromtimestamp(int(trade[2])),
                        'side': 'buy' if trade[3] == 'b' else 'sell',
                        'order_type': trade[4],
                        'misc': trade[5]
                    })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching recent trades for {pair}: {e}")
            return []
    
    # =========================================================================
    # ACCOUNT & BALANCE METHODS
    # =========================================================================
    
    def get_account_balance(self) -> List[KrakenBalance]:
        """
        Get account balances for all currencies
        
        Returns:
            List of KrakenBalance objects
        """
        try:
            data = self._private_request('Balance')
            
            balances = []
            for currency, balance in data.items():
                # Remove 'Z' or 'X' prefix from currency codes
                clean_currency = currency.lstrip('ZX')
                
                balances.append(KrakenBalance(
                    currency=clean_currency,
                    balance=float(balance),
                    available=float(balance),  # Will be updated with open orders
                    hold=0.0
                ))
            
            # Get extended balance info (including holds)
            try:
                extended_data = self._private_request('BalanceEx')
                for currency, details in extended_data.items():
                    clean_currency = currency.lstrip('ZX')
                    for bal in balances:
                        if bal.currency == clean_currency:
                            bal.balance = float(details.get('balance', bal.balance))
                            bal.hold = float(details.get('hold_trade', 0.0))
                            bal.available = bal.balance - bal.hold
                            break
            except:
                # BalanceEx may not be available with all API key permissions
                pass
            
            return balances
            
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return []
    
    def get_total_balance_usd(self) -> float:
        """
        Get total account balance in USD equivalent
        
        Returns:
            Total balance in USD
        """
        try:
            balances = self.get_account_balance()
            total_usd = 0.0
            
            for balance in balances:
                if balance.currency == 'USD' or balance.currency == 'ZUSD':
                    total_usd += balance.balance
                elif balance.balance > 0:
                    # Get USD conversion rate for this asset
                    try:
                        # Strip Kraken suffixes (.F, .S, .M) from currency before constructing pair
                        currency = balance.currency.replace('.F', '').replace('.S', '').replace('.M', '')
                        pair = f"{currency}/USD"
                        ticker = self.get_ticker_data(pair)
                        if ticker:
                            total_usd += balance.balance * ticker['last_price']
                    except:
                        # If conversion fails, skip this currency
                        pass
            
            return total_usd
            
        except Exception as e:
            logger.error(f"Error calculating total balance: {e}")
            return 0.0
    
    # =========================================================================
    # ORDER MANAGEMENT METHODS
    # =========================================================================
    
    def place_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        validate: bool = False
    ) -> Optional[KrakenOrder]:
        """
        Place an order on Kraken
        
        Args:
            pair: Trading pair
            side: Order side (BUY or SELL)
            order_type: Order type (MARKET, LIMIT, etc.)
            volume: Order volume (amount of crypto to buy/sell)
            price: Limit price (required for limit orders)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            validate: If True, only validate the order without placing it
            
        Returns:
            KrakenOrder object if successful, None otherwise
        """
        kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
        
        params = {
            'pair': kraken_pair,
            'type': side.value,
            'ordertype': order_type.value,
            'volume': str(volume)
        }
        
        # Add price for limit orders (Kraken requires max 6 decimals)
        if order_type == OrderType.LIMIT and price:
            params['price'] = str(round(price, 6))
        
        # Add stop loss (Kraken requires max 6 decimals)
        if stop_loss:
            params['close[ordertype]'] = 'stop-loss'
            params['close[price]'] = str(round(stop_loss, 6))
        
        # Add take profit (Kraken requires max 6 decimals)
        if take_profit:
            params['close[ordertype]'] = 'take-profit'
            params['close[price]'] = str(round(take_profit, 6))
        
        # Validation mode
        if validate:
            params['validate'] = 'true'
        
        try:
            data = self._private_request('AddOrder', params)
            
            if validate:
                logger.info(f"✅ Order validation successful for {pair}")
                return None
            
            # Extract order info
            order_id = data.get('txid', [''])[0]
            
            logger.info(f"✅ Order placed successfully: {order_id}")
            
            return KrakenOrder(
                order_id=order_id,
                pair=pair,
                side=side.value,
                order_type=order_type.value,
                price=price,
                volume=volume,
                status='pending',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order
        
        Args:
            order_id: Order transaction ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = self._private_request('CancelOrder', {'txid': order_id})
            
            if data.get('count', 0) > 0:
                logger.info(f"✅ Order cancelled: {order_id}")
                return True
            else:
                logger.warning(f"⚠️ Order cancellation may have failed: {order_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_open_orders(self) -> List[KrakenOrder]:
        """
        Get all open orders
        
        Returns:
            List of open KrakenOrder objects
        """
        try:
            data = self._private_request('OpenOrders')
            
            orders = []
            open_orders = data.get('open', {})
            
            for order_id, order_data in open_orders.items():
                desc = order_data.get('descr', {})
                
                orders.append(KrakenOrder(
                    order_id=order_id,
                    pair=desc.get('pair', ''),
                    side=desc.get('type', ''),
                    order_type=desc.get('ordertype', ''),
                    price=float(desc.get('price', 0)) if desc.get('price') else None,
                    volume=float(order_data.get('vol', 0)),
                    status=order_data.get('status', ''),
                    timestamp=datetime.fromtimestamp(order_data.get('opentm', 0)),
                    executed_volume=float(order_data.get('vol_exec', 0))
                ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    def get_closed_orders(self, start: Optional[int] = None, end: Optional[int] = None) -> List[KrakenOrder]:
        """
        Get closed orders
        
        Args:
            start: Starting timestamp
            end: Ending timestamp
            
        Returns:
            List of closed KrakenOrder objects
        """
        params = {}
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        try:
            data = self._private_request('ClosedOrders', params)
            
            orders = []
            closed_orders = data.get('closed', {})
            
            for order_id, order_data in closed_orders.items():
                desc = order_data.get('descr', {})
                
                orders.append(KrakenOrder(
                    order_id=order_id,
                    pair=desc.get('pair', ''),
                    side=desc.get('type', ''),
                    order_type=desc.get('ordertype', ''),
                    price=float(desc.get('price', 0)) if desc.get('price') else None,
                    volume=float(order_data.get('vol', 0)),
                    status=order_data.get('status', ''),
                    timestamp=datetime.fromtimestamp(order_data.get('opentm', 0)),
                    executed_volume=float(order_data.get('vol_exec', 0)),
                    avg_price=float(order_data.get('price', 0))
                ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Error fetching closed orders: {e}")
            return []
    
    # =========================================================================
    # POSITION MANAGEMENT METHODS
    # =========================================================================
    
    def get_open_positions(self) -> List[KrakenPosition]:
        """
        Get all open positions
        
        Returns:
            List of KrakenPosition objects
        """
        try:
            # Get account balances
            balances = self.get_account_balance()
            
            positions = []
            for balance in balances:
                if balance.balance > 0 and balance.currency != 'USD':
                    # Get current price
                    pair = f"{balance.currency}/USD"
                    try:
                        ticker = self.get_ticker_data(pair)
                        current_price = ticker.get('last_price', 0)
                        
                        # Create position
                        position = KrakenPosition(
                            pair=pair,
                            side='long',  # Kraken spot trading is always long
                            volume=balance.balance,
                            cost=balance.balance * current_price,  # Approximate
                            fee=0.0,  # Unknown without trade history
                            margin=0.0,  # Spot trading has no margin
                            net_pnl=0.0,  # Would need trade history to calculate
                            current_price=current_price,
                            entry_price=0.0  # Unknown without trade history
                        )
                        
                        positions.append(position)
                    except:
                        # Skip if we can't get price data
                        pass
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            return []
    
    # =========================================================================
    # VALIDATION & UTILITY METHODS
    # =========================================================================
    
    def validate_connection(self) -> Tuple[bool, str]:
        """
        Validate Kraken API connection
        
        Returns:
            (success: bool, message: str)
        """
        try:
            # Test public API
            self._public_request('Time')
            
            # Test private API (authentication)
            self._private_request('Balance')
            
            logger.info("✅ Kraken API connection validated successfully")
            return True, "Connection successful"
            
        except Exception as e:
            logger.error(f"❌ Kraken API connection validation failed: {e}")
            return False, f"Connection failed: {str(e)}"
    
    def get_tradable_pairs(self) -> List[str]:
        """
        Get list of all tradable pairs on Kraken
        
        Returns:
            List of trading pair names
        """
        try:
            data = self._public_request('AssetPairs')
            return list(data.keys())
        except:
            return list(self.POPULAR_PAIRS.values())
    
    def get_server_time(self) -> datetime:
        """
        Get Kraken server time
        
        Returns:
            Server datetime
        """
        try:
            data = self._public_request('Time')
            return datetime.fromtimestamp(data['unixtime'])
        except:
            return datetime.now()


def validate_kraken_connection(api_key: str, api_secret: str) -> Tuple[bool, str]:
    """
    Validate Kraken API credentials
    
    Args:
        api_key: Kraken API key
        api_secret: Kraken API secret
        
    Returns:
        (success: bool, message: str)
    """
    try:
        client = KrakenClient(api_key=api_key, api_secret=api_secret)
        return client.validate_connection()
    except Exception as e:
        return False, f"Validation failed: {str(e)}"

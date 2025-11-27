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
from utils.crypto_pair_utils import normalize_crypto_pair



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
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
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
        'DOGE/USD': 'DOGUSD',  # Fixed: was XDGZUSD
        'DOT/USD': 'DOTUSD',
        'MATIC/USD': 'POLUSD',  # Fixed: MATIC renamed to POL on Kraken
        'POL/USD': 'POLUSD',    # Add POL as alternative
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
        'OP/USD': 'OPUSD',
        'ARB/USD': 'ARBUSD',
        
        # Additional altcoins
        'BAT/USD': 'BATUSD',
        'SC/USD': 'SCUSD',
        'HIPPO/USD': 'HIPPOUSD',
        'SWEAT/USD': 'SWEATUSD',
        'SAROS/USD': 'SAROSUSD',
        'CCD/USD': 'CCDUSD',
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
                # Use DEBUG for "Unknown asset pair" (common with watchlist items), ERROR for others
                if 'Unknown asset pair' in error_msg:
                    logger.debug(f"Kraken API: {error_msg}")
                else:
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
                # Use DEBUG for "Unknown asset pair" (common with watchlist items), ERROR for others
                if 'Unknown asset pair' in error_msg:
                    logger.debug(f"Kraken API: {error_msg}")
                else:
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
            pair: Trading pair (e.g., 'XXBTZUSD', 'BTC/USD', 'BTCUSD', 'btcusd', 'btc/usd')
            
        Returns:
            Ticker data including price, volume, high, low
        """
        # Normalize pair format globally (handles BTC/USD, BTCUSD, btcusd, btc/usd)
        normalized_pair = normalize_crypto_pair(pair)
        
        # Convert user-friendly format to Kraken format if needed
        kraken_pair = self.POPULAR_PAIRS.get(normalized_pair, normalized_pair)
        
        # If pair is not in POPULAR_PAIRS, try common format conversions
        if kraken_pair == normalized_pair:
            # Try common formats: ATOM/USD -> ATOMUSD
            if '/' in normalized_pair:
                kraken_pair = normalized_pair.replace('/', '').upper()
            else:
                kraken_pair = normalized_pair.upper()
        
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
                # Try to find the pair in the response (Kraken sometimes returns different keys)
                if data:
                    # Get the first available pair if exact match not found
                    first_key = list(data.keys())[0]
                    ticker = data[first_key]
                    logger.debug(f"Using alternative pair format: {first_key} for {pair}")
                    return {
                        'pair': pair,
                        'last_price': float(ticker['c'][0]),
                        'bid': float(ticker['b'][0]),
                        'ask': float(ticker['a'][0]),
                        'high_24h': float(ticker['h'][1]),
                        'low_24h': float(ticker['l'][1]),
                        'volume_24h': float(ticker['v'][1]),
                        'vwap_24h': float(ticker['p'][1]),
                        'trades_24h': int(ticker['t'][1])
                    }
                logger.debug(f"No data found for pair: {kraken_pair} (pair may not exist on Kraken)")
                return {}
                
        except ValueError as e:
            # Handle "Unknown asset pair" errors gracefully
            error_msg = str(e)
            if "Unknown asset pair" in error_msg or "EQuery:Unknown asset pair" in error_msg:
                logger.debug(f"Pair {pair} ({kraken_pair}) not available on Kraken: {error_msg}")
                return {}
            logger.error(f"Error fetching ticker data for {pair}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching ticker data for {pair}: {e}")
            return {}
    
    def get_ticker_info(self, pair: str) -> Dict:
        """
        Get raw ticker information for a trading pair (raw Kraken format)
        
        Args:
            pair: Trading pair (e.g., 'XXBTZUSD', 'BTC/USD', 'BTCUSD', 'btcusd', 'btc/usd')
            
        Returns:
            Raw ticker data in Kraken format (includes 'c' key for last price)
        """
        # Normalize pair format globally (handles BTC/USD, BTCUSD, btcusd, btc/usd)
        normalized_pair = normalize_crypto_pair(pair)
        
        # Convert user-friendly format to Kraken format if needed
        kraken_pair = self.POPULAR_PAIRS.get(normalized_pair, normalized_pair)
        
        # If pair is not in POPULAR_PAIRS, try common format conversions
        if kraken_pair == normalized_pair:
            # Try common formats: ATOM/USD -> ATOMUSD
            if '/' in normalized_pair:
                kraken_pair = normalized_pair.replace('/', '').upper()
            else:
                kraken_pair = normalized_pair.upper()
        
        try:
            data = self._public_request('Ticker', {'pair': kraken_pair})
            
            # Return raw ticker data if found
            if kraken_pair in data:
                return data[kraken_pair]
            else:
                # Try to find the pair in the response
                if data:
                    first_key = list(data.keys())[0]
                    logger.debug(f"Using alternative pair format: {first_key} for {pair}")
                    return data[first_key]
                logger.debug(f"No data found for pair: {kraken_pair}")
                return {}
                
        except ValueError as e:
            # Handle "Unknown asset pair" errors gracefully
            error_msg = str(e)
            if "Unknown asset pair" in error_msg or "EQuery:Unknown asset pair" in error_msg:
                logger.debug(f"Pair {pair} ({kraken_pair}) not available on Kraken: {error_msg}")
                return {}
            logger.debug(f"Error fetching ticker info for {pair}: {e}")
            return {}
        except Exception as e:
            logger.debug(f"Error fetching ticker info for {pair}: {e}")
            return {}
    
    def get_ticker_batch(self, pairs: List[str]) -> Tuple[bool, Dict]:
        """
        Get ticker information for multiple trading pairs in one request
        
        Args:
            pairs: List of trading pairs in any format (e.g., ['BTC/USD', 'BTCUSD', 'btcusd', 'eth/usd'])
                   Will be normalized and converted to Kraken format automatically
            
        Returns:
            Tuple of (success: bool, ticker_data: Dict)
            ticker_data maps normalized pair (e.g., 'BTC/USD') to ticker info
        """
        try:
            # Normalize and convert pairs to Kraken format
            normalized_pairs = []
            kraken_pairs = []
            pair_mapping = {}  # Maps Kraken format -> normalized format
            
            for pair in pairs:
                # Normalize pair (handles BTC/USD, BTCUSD, btcusd, btc/usd)
                normalized = normalize_crypto_pair(pair)
                
                # Get Kraken format from POPULAR_PAIRS or use fallback
                kraken_pair = self.POPULAR_PAIRS.get(normalized, normalized)
                
                # If not in POPULAR_PAIRS, try common format conversions
                if kraken_pair == normalized:
                    # Try common formats: ATOM/USD -> ATOMUSD
                    if '/' in normalized:
                        kraken_pair = normalized.replace('/', '').upper()
                    else:
                        kraken_pair = normalized.upper()
                
                normalized_pairs.append(normalized)
                kraken_pairs.append(kraken_pair)
                pair_mapping[kraken_pair] = normalized
            
            # Kraken accepts comma-separated pairs
            pair_str = ','.join(kraken_pairs)
            
            data = self._public_request('Ticker', {'pair': pair_str})
            
            if not data:
                return False, {}
            
            # Map results back to normalized pair names
            result = {}
            for kraken_pair, normalized_pair in pair_mapping.items():
                # Try exact match first
                if kraken_pair in data:
                    result[normalized_pair] = data[kraken_pair]
                else:
                    # Try to find match with different formatting
                    for key in data.keys():
                        if kraken_pair.upper() in key or key in kraken_pair.upper():
                            result[normalized_pair] = data[key]
                            break
            
            return True, result
            
        except Exception as e:
            logger.debug(f"Error fetching batch ticker data: {e}")
            return False, {}
    
    def get_ohlc_data(self, pair: str, interval: int = 60, since: Optional[int] = None) -> List[Dict]:
        """
        Get OHLC (candlestick) data for a trading pair
        
        Args:
            pair: Trading pair (e.g., 'ATOM/USD', 'ATOMUSD', 'atomusd', 'atom/usd')
            interval: Time interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
                     Can be int or string like "5" or "5m"
            since: Return committed OHLC data since given ID
            
        Returns:
            List of OHLC candles
        """
        # Normalize pair format globally (handles BTC/USD, BTCUSD, btcusd, btc/usd)
        normalized_pair = normalize_crypto_pair(pair)
        
        # Get the actual Kraken pair format
        kraken_pair = self.POPULAR_PAIRS.get(normalized_pair, normalized_pair)

        # If pair is not in POPULAR_PAIRS, try common format conversions
        if kraken_pair == normalized_pair:
            # Try common formats: ATOM/USD -> ATOMUSD
            if '/' in normalized_pair:
                kraken_pair = normalized_pair.replace('/', '').upper()
            else:
                kraken_pair = normalized_pair.upper()
        else:
            # Ensure pair is in Kraken format by removing slash if present
            if '/' in kraken_pair:
                kraken_pair = kraken_pair.replace('/', '').upper()
        
        # Convert interval to integer (handle string formats like "5" or "5m")
        if isinstance(interval, str):
            # Remove 'm' suffix if present (e.g., "5m" -> 5)
            interval_str = interval.replace('m', '').replace('M', '').strip()
            try:
                interval = int(interval_str)
            except ValueError:
                logger.error(f"Invalid interval format: {interval}, using default 60")
                interval = 60
        
        # Kraken OHLC valid intervals: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
        valid_intervals = [1, 5, 15, 30, 60, 240, 1440, 10080, 21600]
        if interval not in valid_intervals:
            # Round to nearest valid interval
            closest = min(valid_intervals, key=lambda x: abs(x - interval))
            logger.debug(f"Interval {interval} not valid, using closest: {closest}")
            interval = closest
        
        params = {
            'pair': kraken_pair,
            'interval': str(interval)
        }
        
        if since:
            params['since'] = str(since)
        
        try:
            data = self._public_request('OHLC', params)
            
            if not data:
                logger.debug(f"No OHLC data returned from API for {pair} ({kraken_pair})")
                return []

            ohlc_data = []
            
            # Kraken OHLC can return data with different key formats
            # Try exact match first, then try variants
            if kraken_pair in data:
                pair_key = kraken_pair
            else:
                # Try to find any matching key (Kraken might return ATOMUSD.d, etc.)
                pair_key = None
                for key in data.keys():
                    if key.startswith(kraken_pair) or kraken_pair.startswith(key.split('.')[0]):
                        pair_key = key
                        break
                
                if not pair_key:
                    logger.debug(f"No OHLC data found for {pair} (tried {kraken_pair})")
                    return []
            
            if pair_key in data:
                for candle in data[pair_key]:
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
            params['since'] = str(since)
        
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
    
    def _normalize_kraken_currency(self, currency_code: str) -> str:
        """
        Normalize Kraken currency code to standard format.
        Handles special cases like XXBT -> BTC, XXRP -> XRP.
        """
        # Specific mappings for Kraken weirdness
        MAPPINGS = {
            'XXBT': 'BTC',
            'XBT': 'BTC',
            'XXRP': 'XRP',
            'XDG': 'DOGE',
            'XXDG': 'DOGE',
            'XREP': 'REP',
            'XXMR': 'XMR',
            'XXLM': 'XLM',
            'XMLN': 'MLN',
        }
        
        if currency_code in MAPPINGS:
            return MAPPINGS[currency_code]
        
        # Standard ISO 4217 handling (remove leading X or Z for 4-letter codes)
        if len(currency_code) == 4 and (currency_code.startswith('X') or currency_code.startswith('Z')):
            return currency_code[1:]
            
        return currency_code

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
                # Use robust normalization instead of naive lstrip
                clean_currency = self._normalize_kraken_currency(currency)
                
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
                    clean_currency = self._normalize_kraken_currency(currency)
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
    
    def check_margin_enabled(self) -> bool:
        """
        Check if margin trading is enabled on the account
        
        Returns:
            True if margin trading is available, False otherwise
        """
        try:
            # Query trade balance which includes margin info
            data = self._private_request('TradeBalance', {'asset': 'ZUSD'})
            
            # If we can query trade balance without error, margin is likely enabled
            # Note: You need to enable margin trading in Kraken settings first
            logger.info("✅ Margin trading API access confirmed")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Permission denied" in error_msg or "Invalid key" in error_msg:
                logger.warning("⚠️ Margin trading not enabled or insufficient API permissions")
                logger.warning("   Enable margin trading at: https://www.kraken.com/u/funding/margin")
                return False
            logger.error(f"Error checking margin status: {e}")
            return False
    
    def get_margin_info(self) -> Dict:
        """
        Get margin trading account information
        
        Returns:
            Dictionary with margin balance, equity, margin level, etc.
        """
        try:
            data = self._private_request('TradeBalance', {'asset': 'ZUSD'})
            
            return {
                'equity': float(data.get('eb', 0)),  # Equivalent balance (combined balance of all currencies)
                'trade_balance': float(data.get('tb', 0)),  # Trade balance (equity - credit)
                'margin_used': float(data.get('m', 0)),  # Margin amount of open positions
                'unrealized_pnl': float(data.get('n', 0)),  # Unrealized net profit/loss
                'cost_basis': float(data.get('c', 0)),  # Cost basis of open positions
                'floating_valuation': float(data.get('v', 0)),  # Current floating valuation
                'margin_level': float(data.get('ml', 0)) if data.get('ml') else None,  # Margin level percentage
                'free_margin': float(data.get('mf', 0))  # Free margin
            }
            
        except Exception as e:
            logger.error(f"Error fetching margin info: {e}")
            return {}
    
    def _get_pair_decimals(self, pair: str) -> int:
        """
        Get the number of decimal places allowed for a trading pair's price
        
        Args:
            pair: Trading pair (e.g., "UNI/USD", "BTC/USD")
            
        Returns:
            Number of decimals allowed (default: 6)
        """
        try:
            kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
            data = self._public_request('AssetPairs', {'pair': kraken_pair})
            
            # Get first result (should only be one)
            for pair_info in data.values():
                # pair_decimals = number of decimals for price
                return pair_info.get('pair_decimals', 6)
            
            return 6  # Default fallback
        except:
            # If API call fails, use sensible defaults based on pair
            if 'USD' in pair or 'EUR' in pair or 'GBP' in pair:
                # Fiat pairs typically use 2-3 decimals for most altcoins
                if pair.startswith(('BTC', 'ETH')):
                    return 1  # BTC/USD, ETH/USD = 1 decimal
                else:
                    return 3  # Most altcoin/fiat = 3 decimals
            return 6  # Crypto/crypto pairs = 6 decimals

    def place_order(
        self,
        pair: str,
        side: OrderSide,
        order_type: OrderType,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[int] = None,
        validate: bool = False
    ) -> Optional[KrakenOrder]:
        """
        Place an order on Kraken (supports both spot and margin trading)
        
        Args:
            pair: Trading pair
            side: Order side (BUY or SELL)
            order_type: Order type (MARKET, LIMIT, etc.)
            volume: Order volume (amount of crypto to buy/sell)
            price: Limit price (required for limit orders)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            leverage: Leverage amount (2, 3, 4, 5 for margin trading). None = spot trading
            validate: If True, only validate the order without placing it
            
        Returns:
            KrakenOrder object if successful, None otherwise
            
        Note:
            - Leverage enables margin trading and short selling
            - For SELL orders with leverage > 1, you're opening a short position
            - Requires margin trading enabled on your Kraken account
        """
        kraken_pair = self.POPULAR_PAIRS.get(pair, pair)
        
        # Get correct decimal precision for this pair
        decimals = self._get_pair_decimals(pair)
        
        params = {
            'pair': kraken_pair,
            'type': side.value,
            'ordertype': order_type.value,
            'volume': str(volume)
        }
        
        # Add leverage for margin trading (enables short selling)
        # Note: Not all pairs support leverage on Kraken
        if leverage and leverage > 1:
            # Check if pair supports leverage by querying asset pair details
            try:
                pair_info = self._public_request('AssetPairs', {'pair': kraken_pair})
                pair_data = pair_info.get(kraken_pair, {})
                leverage_buy = pair_data.get('leverage_buy', [])
                leverage_sell = pair_data.get('leverage_sell', [])
                
                # Check if requested leverage is supported
                if side == OrderSide.BUY and leverage_buy and int(leverage) in leverage_buy:
                    params['leverage'] = str(int(leverage))
                    logger.info(f"🔧 Using {leverage}x leverage for {side.value} order on {pair}")
                elif side == OrderSide.SELL and leverage_sell and int(leverage) in leverage_sell:
                    params['leverage'] = str(int(leverage))
                    logger.info(f"🔧 Using {leverage}x leverage for {side.value} order on {pair}")
                else:
                    logger.warning(f"⚠️ {pair} does not support {leverage}x leverage on Kraken. Placing as spot order instead.")
                    logger.warning(f"   Supported leverage: BUY={leverage_buy}, SELL={leverage_sell}")
                    # Don't add leverage parameter - will execute as spot trade
            except Exception as e:
                logger.warning(f"⚠️ Could not validate leverage for {pair}. Placing as spot order: {e}")
                # Don't add leverage parameter - safer to execute as spot trade
        
        # Add price for limit orders (use pair-specific decimals)
        if order_type == OrderType.LIMIT and price:
            params['price'] = str(round(price, decimals))
        
        # Add stop loss and take profit (use pair-specific decimals)
        # Note: Kraken API only supports one close order at a time
        # Priority: stop-loss for safety, then take-profit if no stop-loss
        if stop_loss:
            params['close[ordertype]'] = 'stop-loss'
            params['close[price]'] = str(round(stop_loss, decimals))
            logger.info(f"Setting stop-loss at ${stop_loss:.{decimals}f} for {pair} ({decimals} decimals)")
        elif take_profit:
            # Only set take-profit if no stop-loss (safety first)
            params['close[ordertype]'] = 'take-profit'
            params['close[price]'] = str(round(take_profit, decimals))
            logger.info(f"Setting take-profit at ${take_profit:.{decimals}f} for {pair} ({decimals} decimals)")
        
        # Note: If both stop_loss and take_profit are provided, only stop_loss will be set
        # This is a limitation of Kraken's API - only one close order per entry order
        if stop_loss and take_profit:
            logger.warning(f"Both stop-loss and take-profit provided for {pair}. Only stop-loss will be set. Consider placing take-profit separately after entry.")
        
        # Validation mode
        if validate:
            params['validate'] = 'true'
        
        try:
            data = self._private_request('AddOrder', params)
            
            if validate:
                logger.info(f"✅ Order validation successful for {pair}")
                return None
            
            # Extract order info - Kraken returns txid as a list
            txid_list = data.get('txid', [])
            if not txid_list or len(txid_list) == 0:
                logger.error(f"❌ Order placement returned no transaction ID for {pair}")
                return None
            
            order_id = txid_list[0] if isinstance(txid_list, list) else str(txid_list)
            
            if not order_id or order_id == '':
                logger.error(f"❌ Order placement returned empty transaction ID for {pair}")
                return None
            
            logger.info(f"✅ Order placed successfully: {order_id} for {pair}")
            
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
            raise e  # Re-raise so UI can display specific error
    
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
    
    def get_trades_history(self, start: Optional[int] = None, end: Optional[int] = None) -> List[Dict]:
        """
        Get trades history (executed trades)
        
        Args:
            start: Starting timestamp (Unix timestamp)
            end: Ending timestamp (Unix timestamp)
            
        Returns:
            List of trade dictionaries with keys: trade_id, pair, side, volume, price, cost, fee, timestamp
        """
        params = {}
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        
        try:
            data = self._private_request('TradesHistory', params)
            
            trades = []
            trades_data = data.get('trades', {})
            
            for trade_id, trade_info in trades_data.items():
                trades.append({
                    'trade_id': trade_id,
                    'pair': trade_info.get('pair', ''),
                    'side': trade_info.get('type', ''),  # buy or sell
                    'volume': float(trade_info.get('vol', 0)),
                    'price': float(trade_info.get('price', 0)),
                    'cost': float(trade_info.get('cost', 0)),
                    'fee': float(trade_info.get('fee', 0)),
                    'timestamp': float(trade_info.get('time', 0)),
                    'datetime': datetime.fromtimestamp(float(trade_info.get('time', 0)))
                })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trades history: {e}")
            return []
    
    # =========================================================================
    # POSITION MANAGEMENT METHODS
    # =========================================================================
    
    def get_open_positions(self, calculate_real_cost: bool = True, min_value: float = 1.0) -> List[KrakenPosition]:
        """
        Get all open positions with accurate entry prices and P&L
        
        Args:
            calculate_real_cost: If True, fetches trade history to calculate real entry prices and costs
            min_value: Minimum position value in USD to include (filters out dust). Default $1.
        
        Returns:
            List of KrakenPosition objects
        """
        try:
            # Get account balances
            balances = self.get_account_balance()
            
            # Get trade history to calculate accurate entry prices (last 3 months)
            trades_by_pair = {}
            if calculate_real_cost:
                try:
                    three_months_ago = int((datetime.now() - timedelta(days=90)).timestamp())
                    trades = self.get_trades_history(start=three_months_ago)
                    
                    # Group trades by pair and side
                    for trade in trades:
                        pair = trade['pair']
                        if pair not in trades_by_pair:
                            trades_by_pair[pair] = {'buy': [], 'sell': []}
                        trades_by_pair[pair][trade['side']].append(trade)
                    
                    logger.debug(f"Loaded {len(trades)} trades from history for cost calculation")
                except Exception as e:
                    logger.warning(f"Could not fetch trade history for cost calculation: {e}")
                    trades_by_pair = {}
            
            positions = []
            skipped_assets = []
            
            logger.info(f"Processing {len(balances)} balance(s) from Kraken...")
            logger.debug(f"Dust filter: Skipping positions < ${min_value:.2f}")
            
            for balance in balances:
                if balance.balance > 0 and balance.currency != 'USD':
                    # Skip futures contracts (marked with .F, .S, .M suffixes)
                    currency = balance.currency
                    if any(currency.endswith(suffix) for suffix in ['.F', '.S', '.M', '.P']):
                        logger.info(f"⏭️ Skipping futures/staking balance: {currency} ({balance.balance:.6f})")
                        skipped_assets.append((currency, "Futures/Staking not supported"))
                        continue
                    
                    # Get current price
                    pair = f"{currency}/USD"
                    try:
                        ticker = self.get_ticker_data(pair)
                        current_price = ticker.get('last_price', 0)
                        
                        if current_price == 0:
                            logger.warning(f"⚠️ Skipping {pair}: No price data available (balance: {balance.balance:.6f})")
                            skipped_assets.append((currency, "No price data"))
                            continue
                        
                        # Calculate entry price and cost from trade history
                        entry_price = 0.0
                        total_cost = 0.0
                        total_volume = 0.0
                        total_fees = 0.0
                        
                        # Try multiple pair formats (BATUSD, BATUSDT, XBT+BAT+USD, etc.)
                        kraken_pair = self.POPULAR_PAIRS.get(pair, pair.replace('/', ''))
                        possible_pairs = [kraken_pair, pair.replace('/', ''), f"X{currency}ZUSD", f"{currency}USD"]
                        
                        for possible_pair in possible_pairs:
                            if possible_pair in trades_by_pair:
                                # Sort all trades by timestamp (FIFO matching)
                                all_trades = []
                                for trade in trades_by_pair[possible_pair]['buy']:
                                    all_trades.append({**trade, 'type': 'buy'})
                                for trade in trades_by_pair[possible_pair]['sell']:
                                    all_trades.append({**trade, 'type': 'sell'})
                                all_trades.sort(key=lambda t: t['timestamp'])
                                
                                # FIFO matching: Track remaining position
                                position_lots = []  # List of (volume, cost_per_unit) tuples
                                
                                for trade in all_trades:
                                    if trade['type'] == 'buy':
                                        # Add to position
                                        cost_with_fee = trade['cost'] + trade['fee']
                                        cost_per_unit = cost_with_fee / trade['volume']
                                        position_lots.append((trade['volume'], cost_per_unit, trade['fee']))
                                    else:  # sell
                                        # Remove from position (FIFO)
                                        remaining_to_sell = trade['volume']
                                        while remaining_to_sell > 0 and position_lots:
                                            lot_volume, lot_cost_per_unit, lot_fee = position_lots[0]
                                            if lot_volume <= remaining_to_sell:
                                                # Sell entire lot
                                                position_lots.pop(0)
                                                remaining_to_sell -= lot_volume
                                            else:
                                                # Partial sell
                                                position_lots[0] = (lot_volume - remaining_to_sell, lot_cost_per_unit, lot_fee)
                                                remaining_to_sell = 0
                                
                                # Calculate entry price from remaining lots
                                if position_lots:
                                    total_cost = sum(vol * cost for vol, cost, _ in position_lots)
                                    total_volume = sum(vol for vol, _, _ in position_lots)
                                    total_fees = sum(fee for _, _, fee in position_lots)
                                    
                                    # Verify volume matches balance (within 0.01% tolerance)
                                    volume_diff = abs(total_volume - balance.balance)
                                    if volume_diff / balance.balance > 0.0001:  # More than 0.01% difference
                                        logger.warning(f"{pair}: Volume mismatch! Calculated {total_volume:.8f} vs Balance {balance.balance:.8f} (diff: {volume_diff:.8f})")
                                        logger.warning(f"{pair}: Using balance volume and proportional cost adjustment")
                                        # Adjust cost proportionally to match actual balance
                                        total_cost = total_cost * (balance.balance / total_volume)
                                        total_volume = balance.balance
                                    
                                    entry_price = total_cost / total_volume
                                    logger.debug("{}: Calculated entry price ${:.6f} from {} remaining lots (FIFO matched)", pair, entry_price, len(position_lots))
                                    break
                        
                        # If we couldn't find trade history, use current price as entry (no P&L calculation)
                        if entry_price == 0.0 or total_volume == 0.0:
                            entry_price = current_price
                            total_cost = balance.balance * current_price
                            logger.debug(f"{pair}: No trade history found, using current price as entry")
                        
                        # Calculate P&L
                        current_value = balance.balance * current_price
                        position_cost = entry_price * balance.balance
                        net_pnl = current_value - position_cost - total_fees
                        
                        # Filter out dust positions (< min_value threshold)
                        if current_value < min_value:
                            logger.debug(f"⏭️ Skipping {pair}: Dust position worth ${current_value:.4f} (balance: {balance.balance:.8f})")
                            skipped_assets.append((currency, f"Dust (${current_value:.4f})"))
                            continue
                        
                        # Create position
                        position = KrakenPosition(
                            pair=pair,
                            side='long',  # Kraken spot trading is always long
                            volume=balance.balance,
                            cost=position_cost,
                            fee=total_fees,
                            margin=0.0,  # Spot trading has no margin
                            net_pnl=net_pnl,
                            current_price=current_price,
                            entry_price=entry_price
                        )
                        
                        logger.debug(f"✅ {pair}: Balance={balance.balance:.8f}, Value=${current_value:.2f}, Entry=${entry_price:.6f}")
                        positions.append(position)
                        
                    except Exception as e:
                        # Skip if we can't get price data
                        logger.warning(f"⚠️ Skipping {pair}: Error fetching data - {str(e)}")
                        skipped_assets.append((currency, f"Error: {str(e)}"))
                        pass
            
            # Summary logging
            logger.info(f"✅ Loaded {len(positions)} open position(s) with accurate entry prices")
            
            if skipped_assets:
                logger.warning(f"⚠️ Skipped {len(skipped_assets)} asset(s):")
                for currency, reason in skipped_assets:
                    logger.warning(f"   • {currency}: {reason}")
            
            return positions
            
        except Exception as e:
            logger.error(f"Error fetching open positions: {e}")
            return []
    
    def get_margin_positions(self) -> List[KrakenPosition]:
        """
        Get open margin positions (includes both long and short positions)
        
        Returns:
            List of KrakenPosition objects for margin positions
        """
        try:
            data = self._private_request('OpenPositions')
            
            positions = []
            open_positions = data.get('result', {})
            
            if not open_positions:
                logger.info("No open margin positions found")
                return []
            
            for pos_id, pos_data in open_positions.items():
                try:
                    # Extract position details
                    pair = pos_data.get('pair', '')
                    side = pos_data.get('type', '')  # 'buy' or 'sell'
                    volume = float(pos_data.get('vol', 0))
                    cost = float(pos_data.get('cost', 0))
                    fee = float(pos_data.get('fee', 0))
                    margin = float(pos_data.get('margin', 0))
                    net_pnl = float(pos_data.get('net', 0))
                    
                    # Calculate average entry price
                    entry_price = cost / volume if volume > 0 else 0
                    
                    # Get current price
                    current_price = 0
                    try:
                        # Normalize pair format
                        normalized_pair = pair.replace('X', '').replace('Z', '')
                        # Try to get ticker
                        ticker = self.get_ticker_data(normalized_pair)
                        current_price = ticker.get('last_price', 0) if ticker else 0
                    except:
                        pass
                    
                    if current_price == 0:
                        logger.warning(f"Could not get current price for margin position: {pair}")
                        current_price = entry_price  # Fallback
                    
                    position = KrakenPosition(
                        pair=pair,
                        side=side,  # 'buy' = long, 'sell' = short
                        volume=volume,
                        cost=cost,
                        fee=fee,
                        margin=margin,
                        net_pnl=net_pnl,
                        current_price=current_price,
                        entry_price=entry_price
                    )
                    
                    positions.append(position)
                    
                except Exception as e:
                    logger.warning(f"Error processing margin position {pos_id}: {e}")
                    continue
            
            logger.info(f"✅ Loaded {len(positions)} margin position(s)")
            return positions
            
        except Exception as e:
            error_msg = str(e)
            if "Invalid key" in error_msg or "Permission denied" in error_msg:
                logger.debug("No margin positions (margin trading may not be enabled)")
                return []
            logger.error(f"Error fetching margin positions: {e}")
            return []
    
    def get_all_positions(self) -> List[KrakenPosition]:
        """
        Get all positions (both spot and margin)
        
        Returns:
            Combined list of all KrakenPosition objects
        """
        spot_positions = self.get_open_positions()
        margin_positions = self.get_margin_positions()
        
        all_positions = spot_positions + margin_positions
        
        logger.info(f"📊 Total positions: {len(all_positions)} (Spot: {len(spot_positions)}, Margin: {len(margin_positions)})")
        
        return all_positions
    
    def get_staked_balances(self) -> List[Dict]:
        """
        Get staked/flex savings balances (read-only, cannot be traded)
        
        Returns:
            List of staked asset dictionaries with keys: currency, balance, type, current_price, value_usd
        """
        try:
            balances = self.get_account_balance()
            staked_assets = []
            
            for balance in balances:
                if balance.balance > 0:
                    currency = balance.currency
                    
                    # Check if it's a staked/flex asset
                    staked_type = None
                    clean_currency = currency
                    
                    if currency.endswith('.F'):
                        staked_type = 'Flex Staking'
                        clean_currency = currency.replace('.F', '')
                    elif currency.endswith('.S'):
                        staked_type = 'Locked Staking'
                        clean_currency = currency.replace('.S', '')
                    elif currency.endswith('.M'):
                        staked_type = 'Staking (M)'
                        clean_currency = currency.replace('.M', '')
                    elif currency.endswith('.P'):
                        staked_type = 'Parachain'
                        clean_currency = currency.replace('.P', '')
                    
                    if staked_type:
                        # Get current price
                        pair = f"{clean_currency}/USD"
                        current_price = 0.0
                        value_usd = 0.0
                        
                        try:
                            ticker = self.get_ticker_data(pair)
                            if ticker:
                                current_price = ticker.get('last_price', 0)
                                value_usd = balance.balance * current_price
                        except:
                            # If can't get price, just show the balance
                            pass
                        
                        staked_assets.append({
                            'currency': clean_currency,
                            'raw_currency': currency,
                            'balance': balance.balance,
                            'type': staked_type,
                            'current_price': current_price,
                            'value_usd': value_usd
                        })
            
            if staked_assets:
                logger.info(f"✅ Found {len(staked_assets)} staked asset(s)")
            
            return staked_assets
            
        except Exception as e:
            logger.error(f"Error fetching staked balances: {e}")
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
    
    def get_tradable_asset_pairs(self) -> List[Dict]:
        """
        Get list of all tradable asset pairs with full details
        
        Returns:
            List of dictionaries containing pair information with keys:
            - 'altname': Alternative name (e.g., 'BTC/USD')
            - 'base': Base asset (e.g., 'BTC')
            - 'quote': Quote asset (e.g., 'USD')
            - 'pair': Kraken pair name (e.g., 'XXBTZUSD')
        """
        try:
            data = self._public_request('AssetPairs')
            pairs_list = []
            
            for pair_name, pair_info in data.items():
                # Extract relevant information
                pair_dict = {
                    'pair': pair_name,
                    'altname': pair_info.get('altname', pair_name),
                    'base': pair_info.get('base', ''),
                    'quote': pair_info.get('quote', ''),
                    'wsname': pair_info.get('wsname', pair_info.get('altname', pair_name))
                }
                pairs_list.append(pair_dict)
            
            return pairs_list
        except Exception as e:
            logger.error(f"Error fetching asset pairs: {e}")
            # Return popular pairs as fallback
            fallback_pairs = []
            for altname, kraken_name in self.POPULAR_PAIRS.items():
                base, quote = altname.split('/') if '/' in altname else (altname, 'USD')
                fallback_pairs.append({
                    'pair': kraken_name,
                    'altname': altname,
                    'base': base,
                    'quote': quote,
                    'wsname': altname
                })
            return fallback_pairs
    
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
    
    def get_portfolio_analysis(self) -> Dict:
        """
        Get comprehensive portfolio analysis for AI recommendations
        
        Returns:
            Dictionary with portfolio metrics, position details, recommendations, and staked assets
        """
        try:
            positions = self.get_open_positions(calculate_real_cost=True)
            staked_assets = self.get_staked_balances()
            
            # Calculate staked value
            total_staked_value = sum(asset['value_usd'] for asset in staked_assets)
            
            if not positions and not staked_assets:
                return {
                    'total_value': 0.0,
                    'total_cost': 0.0,
                    'total_pnl': 0.0,
                    'total_pnl_pct': 0.0,
                    'num_positions': 0,
                    'positions': [],
                    'winners': [],
                    'losers': [],
                    'recommendations': [],
                    'staked_assets': [],
                    'total_staked_value': 0.0,
                    'combined_value': 0.0
                }
            
            total_value = 0.0
            total_cost = 0.0
            total_pnl = 0.0
            winners = []
            losers = []
            position_details = []
            
            for pos in positions:
                current_value = pos.volume * pos.current_price
                position_cost = pos.cost if pos.cost > 0 else (pos.volume * pos.entry_price)
                pnl = pos.net_pnl if pos.net_pnl != 0 else (current_value - position_cost)
                pnl_pct = (pnl / position_cost * 100) if position_cost > 0 else 0.0
                
                total_value += current_value
                total_cost += position_cost
                total_pnl += pnl
                
                pos_detail = {
                    'pair': pos.pair,
                    'volume': pos.volume,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'cost': position_cost,
                    'current_value': current_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'allocation_pct': 0.0  # Will calculate after getting total
                }
                
                position_details.append(pos_detail)
                
                if pnl > 0:
                    winners.append(pos_detail)
                elif pnl < 0:
                    losers.append(pos_detail)
            
            # Calculate allocation percentages
            for pos_detail in position_details:
                pos_detail['allocation_pct'] = (pos_detail['current_value'] / total_value * 100) if total_value > 0 else 0.0
            
            # Sort winners and losers by P&L percentage
            winners.sort(key=lambda x: x['pnl_pct'], reverse=True)
            losers.sort(key=lambda x: x['pnl_pct'])
            
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
            
            # Generate AI recommendations
            recommendations = []
            
            # Recommendation 1: Take profits on big winners
            for winner in winners[:3]:  # Top 3 winners
                if winner['pnl_pct'] > 20:
                    recommendations.append({
                        'action': 'TAKE_PROFIT',
                        'pair': winner['pair'],
                        'reason': f"Strong gain of +{winner['pnl_pct']:.1f}%. Consider taking partial profits.",
                        'priority': 'HIGH'
                    })
            
            # Recommendation 2: Cut losses on big losers
            for loser in losers[:3]:  # Top 3 losers
                if loser['pnl_pct'] < -15:
                    recommendations.append({
                        'action': 'CUT_LOSS',
                        'pair': loser['pair'],
                        'reason': f"Significant loss of {loser['pnl_pct']:.1f}%. Consider cutting losses or dollar-cost averaging.",
                        'priority': 'HIGH'
                    })
            
            # Recommendation 3: Rebalance if too concentrated
            for pos_detail in position_details:
                if pos_detail['allocation_pct'] > 30:
                    recommendations.append({
                        'action': 'REBALANCE',
                        'pair': pos_detail['pair'],
                        'reason': f"High concentration at {pos_detail['allocation_pct']:.1f}% of portfolio. Consider diversifying.",
                        'priority': 'MEDIUM'
                    })
            
            # Calculate combined portfolio value
            combined_value = total_value + total_staked_value
            
            return {
                'total_value': total_value,
                'total_cost': total_cost,
                'total_pnl': total_pnl,
                'total_pnl_pct': total_pnl_pct,
                'num_positions': len(positions),
                'num_winners': len(winners),
                'num_losers': len(losers),
                'positions': position_details,
                'winners': winners,
                'losers': losers,
                'recommendations': recommendations,
                'staked_assets': staked_assets,
                'total_staked_value': total_staked_value,
                'combined_value': combined_value  # Tradeable + Staked
            }
            
        except Exception as e:
            logger.error("Error analyzing portfolio: {}", str(e), exc_info=True)
            return {
                'total_value': 0.0,
                'total_cost': 0.0,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
                'num_positions': 0,
                'positions': [],
                'winners': [],
                'losers': [],
                'recommendations': [],
                'error': str(e)
            }


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

"""
Tradier API Client for Paper and Production Trading Integration
Handles account management, order placement, and position tracking
"""

import requests
import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from functools import wraps
from .trading_config import get_trading_mode_manager, TradingMode

logger = logging.getLogger(__name__)

def retry_on_timeout(max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    """
    Decorator to retry API calls on 504 Gateway Timeout errors with exponential backoff.
    
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
                    # Check if it's a 504 Gateway Timeout error
                    if e.response is not None and e.response.status_code == 504:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(
                                f"⏳ 504 Gateway Timeout on {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                                f"Retrying in {delay:.1f}s... (Tradier sandbox may be slow)"
                            )
                            time.sleep(delay)
                            delay *= backoff_factor
                            continue
                        else:
                            logger.error(
                                f"❌ 504 Gateway Timeout on {func.__name__} after {max_retries + 1} attempts. "
                                f"Tradier sandbox API is experiencing issues. This is a Tradier server problem, not your code."
                            )
                    # Re-raise non-504 errors immediately
                    raise
                except requests.exceptions.Timeout as e:
                    # Handle connection timeouts
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

class TradierClient:
    """Client for Tradier API integration with paper and production trading"""
    
    def __init__(self, account_id: str = None, access_token: str = None, api_url: str = None, trading_mode: TradingMode = None):
        # If no credentials provided, use trading mode manager
        if account_id is None or access_token is None:
            mode_manager = get_trading_mode_manager()
            if trading_mode:
                if not mode_manager.set_mode(trading_mode):
                    raise ValueError(f"No trading credentials available for {trading_mode.value} mode. Please configure environment variables.")
            
            creds = mode_manager.get_current_credentials()
            if not creds:
                raise ValueError("No trading credentials available. Please configure environment variables.")
            
            self.account_id = creds.account_id
            self.access_token = creds.access_token
            self.api_url = creds.api_url
            self.trading_mode = creds.mode
        else:
            # Use provided credentials
            self.account_id = account_id
            self.access_token = access_token
            self.api_url = api_url or "https://sandbox.tradier.com"
            self.trading_mode = trading_mode or TradingMode.PAPER
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"🔧 TradierClient initialized for {self.trading_mode.value} mode")
        logger.info(f"🔗 API URL: {self.api_url}")
        logger.info(f"👤 Account ID: {self.account_id[:8]}...")
        
        # Ensure we have a robust JSON parser for inconsistent responses
        # Some sandbox endpoints may return JSON as a string or with wrong content-type
        # so we normalize here.
        
    def _safe_json(self, response) -> dict:
        """Safely parse JSON from a requests response and always return a dict.
        - dict => returned as-is
        - list => wrapped under {"data": list}
        - string => attempt json.loads; otherwise fallback to parsing response.text
        - invalid => {}
        """
        try:
            # First check the raw response text for "undefined" before parsing
            raw_text = response.text or ""
            if raw_text.strip().lower() in ('undefined', 'null', 'none', ''):
                logger.info("Tradier returned '%s' - treating as empty result", raw_text.strip())
                return {}
            
            # Try the library JSON first
            try:
                data = response.json()
            except ValueError:
                # If JSON parsing fails, check if it's a plain string response
                if raw_text:
                    logger.warning("JSON parsing failed, raw text: %s", raw_text[:300])
                    return {}
                data = None

            # Handle Tradier's "undefined" string response (no data case)
            if isinstance(data, str) and data.strip().lower() in ('undefined', 'null', 'none', ''):
                logger.info("Tradier returned '%s' - treating as empty result", data.strip())
                return {}

            # Already a dict
            if isinstance(data, dict):
                return data
            # Already a list
            if isinstance(data, list):
                try:
                    logger.warning(
                        "Tradier returned list JSON; wrapping in dict. status=%s ct=%s size=%d",
                        getattr(response, 'status_code', 'n/a'),
                        response.headers.get('Content-Type', ''),
                        len(data)
                    )
                except Exception:
                    pass
                return {"data": data}
            # String JSON or plain string
            if isinstance(data, str):
                try:
                    snippet = (response.text or "")[:300]
                    logger.warning(
                        "Tradier returned JSON as string; attempting to load. status=%s ct=%s snippet=%s",
                        getattr(response, 'status_code', 'n/a'),
                        response.headers.get('Content-Type', ''),
                        snippet
                    )
                except Exception:
                    pass
                try:
                    loaded = json.loads(data)
                    if isinstance(loaded, dict):
                        return loaded
                    if isinstance(loaded, list):
                        return {"data": loaded}
                except Exception:
                    # If we can't parse the string, return empty dict
                    logger.warning("Could not parse string as JSON: %s", data[:100])
                    return {}

            # Last resort: parse raw text
            try:
                raw_loaded = json.loads(response.text or "")
                if isinstance(raw_loaded, dict):
                    return raw_loaded
                if isinstance(raw_loaded, list):
                    return {"data": raw_loaded}
            except Exception:
                pass

            # Give up
            try:
                snippet = (response.text or "")[:300]
            except Exception:
                snippet = ''
            logger.error(
                "Unable to parse Tradier response as JSON. status=%s ct=%s snippet=%s",
                getattr(response, 'status_code', 'n/a'),
                response.headers.get('Content-Type', ''),
                snippet
            )
            return {}
        except Exception as e:
            logger.error("_safe_json unexpected error: %s", e)
            return {}
    
    @retry_on_timeout(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def get_account_balance(self) -> Tuple[bool, Dict]:
        """Get account balance information"""
        try:
            response = self.session.get(f'{self.api_url}/v1/accounts/{self.account_id}/balances')
            response.raise_for_status()
            
            balance_data = self._safe_json(response)
            logger.info(f"Account balance retrieved: {balance_data}")
            return True, balance_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting account balance: {e}")
            return False, {"error": str(e)}
    
    @retry_on_timeout(max_retries=4, initial_delay=0.5, backoff_factor=2.0)
    def get_positions(self) -> Tuple[bool, List[Dict]]:
        """Get current positions with enhanced error handling"""
        try:
            response = self.session.get(
                f'{self.api_url}/v1/accounts/{self.account_id}/positions',
                timeout=10  # Add explicit timeout
            )
            response.raise_for_status()
            
            positions_data = self._safe_json(response)
            if not isinstance(positions_data, dict):
                logger.error("Positions payload is not a dict: %s (type: %s)", positions_data, type(positions_data))
                return True, []
            
            # Handle nested structure safely
            positions_wrapper = positions_data.get('positions', {})
            if not isinstance(positions_wrapper, dict):
                logger.debug("No positions found (wrapper is %s, likely empty account)", type(positions_wrapper).__name__)
                return True, []
            
            positions = positions_wrapper.get('position', [])
            
            # Ensure positions is always a list
            if not isinstance(positions, list):
                positions = [positions] if positions else []
            
            logger.info(f"Retrieved {len(positions)} positions")
            return True, positions
            
        except AttributeError as e:
            logger.error(f"Positions error details: {positions_data}", exc_info=True)
            return True, []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting positions: {e}")
            logger.warning("⚠️ Returning empty positions list due to API error (safe fallback)")
            return False, []
    
    @retry_on_timeout(max_retries=4, initial_delay=0.5, backoff_factor=2.0)
    def get_orders(self, status: str = "all") -> Tuple[bool, List[Dict]]:
        """Get order history with enhanced error handling"""
        try:
            params = {"status": status}
            response = self.session.get(
                f'{self.api_url}/v1/accounts/{self.account_id}/orders',
                params=params,
                timeout=10  # Add explicit timeout
            )
            response.raise_for_status()
            
            orders_data = self._safe_json(response)
            if not isinstance(orders_data, dict):
                logger.error("Orders payload is not a dict: %s (type: %s)", orders_data, type(orders_data))
                return True, []
            
            # Handle nested structure safely
            orders_wrapper = orders_data.get('orders', {})
            if not isinstance(orders_wrapper, dict):
                logger.debug("No orders found (wrapper is %s, likely empty account)", type(orders_wrapper).__name__)
                return True, []
            
            orders = orders_wrapper.get('order', [])
            
            # Ensure orders is always a list
            if not isinstance(orders, list):
                orders = [orders] if orders else []
            
            logger.info(f"Retrieved {len(orders)} orders")
            return True, orders
            
        except AttributeError as e:
            logger.error(f"Orders error details: {orders_data}", exc_info=True)
            return True, []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting orders: {e}")
            logger.warning("⚠️ Returning empty orders list due to API error (safe fallback)")
            return False, []
    
    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a single symbol (convenience method)"""
        try:
            params = {"symbols": symbol}
            
            response = self.session.get(f'{self.api_url}/v1/markets/quotes', params=params)
            response.raise_for_status()
            
            quotes_data = response.json()
            quotes = quotes_data.get('quotes', {}).get('quote', {})
            
            # If single quote, return it directly
            if isinstance(quotes, dict):
                return quotes
            elif isinstance(quotes, list) and len(quotes) > 0:
                return quotes[0]
            else:
                return {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {}
    
    @retry_on_timeout(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def get_quotes(self, symbols: List[str]) -> Tuple[bool, Dict]:
        """Get real-time quotes for symbols"""
        try:
            symbols_str = ",".join(symbols)
            params = {"symbols": symbols_str}
            
            response = self.session.get(f'{self.api_url}/v1/markets/quotes', params=params)
            response.raise_for_status()
            
            quotes_data = response.json()
            logger.info(f"Retrieved quotes for {len(symbols)} symbols")
            return True, quotes_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting quotes: {e}")
            return False, {"error": str(e)}
    
    def get_option_chains(self, symbol: str, expiration: str = None) -> Tuple[bool, Dict]:
        """Get option chains for a symbol"""
        try:
            params = {"symbol": symbol}
            if expiration:
                params["expiration"] = expiration
            
            response = self.session.get(f'{self.api_url}/v1/markets/options/chains', params=params)
            response.raise_for_status()
            
            chains_data = response.json()
            logger.info(f"Retrieved option chains for {symbol}")
            return True, chains_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting option chains: {e}")
            return False, {"error": str(e)}
    
    def place_equity_order(self, symbol: str, side: str, quantity: int, order_type: str = 'market', 
                          duration: str = 'day', price: float = None, stop: float = None) -> Dict:
        """Place an equity (stock) order - convenience method"""
        try:
            order_data = {
                'class': 'equity',
                'symbol': symbol.upper(),
                'side': side.lower(),
                'quantity': str(quantity),
                'type': order_type.lower(),
                'duration': duration.lower()
            }
            
            # Add price for limit orders
            if order_type.lower() == 'limit' and price:
                order_data['price'] = str(price)
            
            # Add stop price for stop orders
            if order_type.lower() == 'stop' and stop:
                order_data['stop'] = str(stop)
            
            response = self.session.post(
                f'{self.api_url}/v1/accounts/{self.account_id}/orders',
                data=order_data  # Use data instead of json for form-encoded
            )
            response.raise_for_status()
            
            order_response = response.json()
            logger.info(f"Equity order placed: {symbol} {side} {quantity} @ {order_type}")
            
            return order_response.get('order', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error placing equity order: {e}")
            return {}
    
    def place_order(self, order_data: Dict) -> Tuple[bool, Dict]:
        """Place an order - Tradier expects form-encoded data"""
        try:
            # Validate required fields for simple orders
            if order_data.get('class') not in ['otoco', 'oco', 'combo']:
                required_fields = ['class', 'side', 'quantity', 'type']
                
                # For options trades, require option_symbol instead of symbol
                if order_data.get('class') == 'option':
                    required_fields.append('option_symbol')
                else:
                    required_fields.append('symbol')
                
                for field in required_fields:
                    if field not in order_data:
                        return False, {"error": f"Missing required field: {field}"}
            
            # Remove account_id if present (not needed in form data)
            order_data_copy = order_data.copy()
            order_data_copy.pop('account_id', None)
            
            # Log what we're sending
            logger.info(f"🚀 Placing order with data: {order_data_copy}")
            
            # Tradier expects form-encoded data, not JSON
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Accept': 'application/json'
            }
            
            response = requests.post(
                f'{self.api_url}/v1/accounts/{self.account_id}/orders',
                headers=headers,
                data=order_data_copy  # Send as form data, not JSON
            )
            
            # Log response before checking status
            logger.info(f"📥 Response status: {response.status_code}")
            logger.info(f"📥 Response text: {response.text[:500]}")
            
            response.raise_for_status()
            
            order_response = response.json()
            logger.info(f"✅ Order placed successfully: {order_response}")
            return True, order_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error placing order: {e}")
            # Try to get error details from response
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_text = e.response.text
                    logger.error(f"❌ Error response text: {error_text}")
                    try:
                        error_detail = e.response.json()
                        logger.error(f"❌ Error details JSON: {error_detail}")
                        return False, {"error": str(e), "details": error_detail, "response_text": error_text}
                    except:
                        return False, {"error": str(e), "response_text": error_text}
                else:
                    return False, {"error": str(e)}
            except Exception as ex:
                logger.error(f"❌ Error parsing error response: {ex}")
                return False, {"error": str(e)}
    
    def place_bracket_order(self,
                           symbol: str,
                           side: str,
                           quantity: int,
                           entry_price: float,
                           take_profit_price: float,
                           stop_loss_price: float,
                           duration: str = 'gtc',
                           option_symbol: str = None,
                           tag: str = None) -> Tuple[bool, Dict]:
        """
        Place a bracket order (OTOCO - One-Triggers-OCO) with entry, take-profit, and stop-loss.
        This function constructs and sends a multi-leg order to Tradier.
        
        Note: For SELL orders, you must have an existing position. Short selling is not supported
        in paper trading accounts.
        """
        try:
            # Validate side
            if side.lower() not in ['buy', 'sell']:
                return False, {"error": f"Invalid side: {side}. Must be 'buy' or 'sell'"}
            
            # Warn about SELL orders (potential short selling)
            if side.lower() == 'sell':
                logger.warning(f"⚠️ Placing SELL bracket order for {symbol}. Ensure you have an existing position to avoid short selling rejection.")
            
            exit_side = 'sell' if side.lower() == 'buy' else 'buy'
            trade_symbol = option_symbol if option_symbol else symbol.upper()
            
            # Round all prices to 2 decimals (Tradier requirement)
            entry_price = round(float(entry_price), 2)
            take_profit_price = round(float(take_profit_price), 2)
            stop_loss_price = round(float(stop_loss_price), 2)
            
            # Validate price logic for BUY orders
            if side.lower() == 'buy':
                if take_profit_price <= entry_price:
                    logger.warning(f"⚠️ BUY order: take_profit (${take_profit_price}) should be > entry (${entry_price})")
                if stop_loss_price >= entry_price:
                    logger.warning(f"⚠️ BUY order: stop_loss (${stop_loss_price}) should be < entry (${entry_price})")
            
            # Validate price logic for SELL orders (closing position)
            if side.lower() == 'sell':
                if take_profit_price >= entry_price:
                    logger.warning(f"⚠️ SELL order: take_profit (${take_profit_price}) should be < entry (${entry_price})")
                if stop_loss_price <= entry_price:
                    logger.warning(f"⚠️ SELL order: stop_loss (${stop_loss_price}) should be > entry (${entry_price})")

            # 1. Build the order payload with explicit parameters
            order_data = {
                'class': 'otoco',

                # Leg 1: Entry Order
                'symbol[0]': trade_symbol,
                'side[0]': side.lower(),
                'quantity[0]': str(quantity),
                'type[0]': 'limit',
                'price[0]': str(entry_price),
                'duration[0]': duration.lower(),

                # Leg 2: Take-Profit Order
                'symbol[1]': trade_symbol,
                'side[1]': exit_side,
                'quantity[1]': str(quantity),
                'type[1]': 'limit',
                'price[1]': str(take_profit_price),
                'duration[1]': duration.lower(),

                # Leg 3: Stop-Loss Order
                'symbol[2]': trade_symbol,
                'side[2]': exit_side,
                'quantity[2]': str(quantity),
                'type[2]': 'stop',  # Use 'stop' for the stop-loss leg
                'stop[2]': str(stop_loss_price),
                'duration[2]': duration.lower(),
            }

            # Add tag if provided
            if tag:
                order_data['tag'] = tag

            logger.info(f"🎯 Placing OTOCO bracket order for {quantity} {trade_symbol} @ ${entry_price}")
            logger.info(f"   - Take-Profit: ${take_profit_price}")
            logger.info(f"   - Stop-Loss: ${stop_loss_price}")
            logger.info(f"   - Payload: {order_data}")

            # 2. Use the generic place_order method to send the request
            success, result = self.place_order(order_data)

            if success:
                logger.info(f"✅ OTOCO bracket order placed successfully: {result}")
            else:
                logger.error(f"❌ Failed to place OTOCO bracket order: {result}")

            return success, result

        except Exception as e:
            logger.error(f"❌ Unexpected error in place_bracket_order: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    def place_bracket_order_percentage(self,
                                      symbol: str,
                                      side: str,
                                      quantity: int,
                                      entry_price: float,
                                      take_profit_pct: float = 5.0,
                                      stop_loss_pct: float = 3.0,
                                      duration: str = 'gtc',
                                      option_symbol: str = None,
                                      tag: str = None) -> Tuple[bool, Dict]:
        """
        Place a bracket order using percentage-based take-profit and stop-loss.
        
        This is a convenience wrapper around place_bracket_order that calculates
        the take-profit and stop-loss prices based on percentages from the entry price.
        
        Args:
            symbol: Stock symbol or option symbol
            side: 'buy' or 'sell' for entry
            quantity: Number of shares/contracts
            entry_price: Limit price for entry
            take_profit_pct: Percentage gain for take-profit (default 5%)
            stop_loss_pct: Percentage loss for stop-loss (default 3%)
            duration: Order duration
            option_symbol: Optional OCC format symbol for options
            tag: Optional order tag
            
        Returns:
            Tuple of (success, response_dict)
            
        Example:
            # Buy at $150, take profit at +5% ($157.50), stop at -3% ($145.50)
            success, result = client.place_bracket_order_percentage(
                symbol='AAPL',
                side='buy',
                quantity=100,
                entry_price=150.00,
                take_profit_pct=5.0,
                stop_loss_pct=3.0
            )
        """
        try:
            # Calculate prices based on entry side
            if side.lower() == 'buy':
                # For buy orders: profit is above, stop is below
                take_profit_price = round(entry_price * (1 + take_profit_pct / 100), 2)
                stop_loss_price = round(entry_price * (1 - stop_loss_pct / 100), 2)
            else:
                # For sell orders: profit is below, stop is above
                take_profit_price = round(entry_price * (1 - take_profit_pct / 100), 2)
                stop_loss_price = round(entry_price * (1 + stop_loss_pct / 100), 2)
            
            logger.info(f"📊 Calculated bracket prices from {entry_price}:")
            logger.info(f"   Take-Profit: ${take_profit_price} ({'+' if side.lower() == 'buy' else '-'}{take_profit_pct}%)")
            logger.info(f"   Stop-Loss: ${stop_loss_price} ({'-' if side.lower() == 'buy' else '+'}{stop_loss_pct}%)")
            
            return self.place_bracket_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                duration=duration,
                option_symbol=option_symbol,
                tag=tag
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating bracket order: {e}")
            return False, {"error": str(e)}
    
    def cancel_order(self, order_id: str) -> Tuple[bool, Dict]:
        """Cancel an order"""
        try:
            response = self.session.delete(f'{self.api_url}/v1/accounts/{self.account_id}/orders/{order_id}')
            response.raise_for_status()
            
            cancel_response = response.json()
            logger.info(f"Order {order_id} cancelled successfully")
            return True, cancel_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error cancelling order: {e}")
            return False, {"error": str(e)}
    
    @retry_on_timeout(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def get_order_status(self, order_id: str) -> Tuple[bool, Dict]:
        """Get status of a specific order"""
        try:
            response = self.session.get(f'{self.api_url}/v1/accounts/{self.account_id}/orders/{order_id}')
            response.raise_for_status()
            
            order_data = response.json()
            logger.info(f"Retrieved order status for {order_id}")
            return True, order_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting order status: {e}")
            return False, {"error": str(e)}
    
    def get_options_chain(self, symbol: str, expiration: str = None) -> Tuple[bool, Dict]:
        """Get options chain for a symbol"""
        try:
            url = f'{self.api_url}/v1/markets/options/chains'
            params = {'symbol': symbol.upper()}
            if expiration:
                params['expiration'] = expiration
                
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return True, response.json()
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return False, {"error": str(e)}
    
    def validate_options_symbol(self, option_symbol: str) -> Tuple[bool, str]:
        """Validate if an options symbol exists"""
        try:
            # Extract underlying symbol from options symbol
            # Format: SYMBOL + YYMMDD + C/P + 8-digit strike
            underlying = ""
            for i, char in enumerate(option_symbol):
                if char.isdigit():
                    underlying = option_symbol[:i]
                    break
            
            if not underlying:
                return False, "Invalid options symbol format"
            
            # Get options chain for the underlying symbol
            success, data = self.get_options_chain(underlying)
            if not success:
                # Check if it's a 400 error (likely API limitation)
                error_msg = data.get('error', 'Unknown error')
                if "400" in str(error_msg):
                    return False, f"Options data not available for {underlying} (API limitation). Symbol format appears correct: {option_symbol}"
                return False, f"Could not fetch options chain: {error_msg}"
            
            # Check if the symbol exists in the chain
            options = data.get('options', {}).get('option', [])
            if not isinstance(options, list):
                options = [options] if options else []
            
            for option in options:
                if option.get('symbol') == option_symbol:
                    return True, "Valid options symbol"
            
            return False, "Options symbol not found in available contracts"
            
        except Exception as e:
            logger.error(f"Error validating options symbol: {e}")
            return False, f"Validation error: {str(e)}"
    
    def convert_signal_to_order(self, signal: Dict) -> Dict:
        """Convert Option Alpha signal to Tradier order format"""
        try:
            ticker = signal.get('ticker', '').upper()
            action = signal.get('action', '').upper()
            strike = signal.get('strike', 0)
            expiry = signal.get('expiry', '')
            qty = signal.get('qty', 1)
            
            # Map Option Alpha actions to Tradier order types
            order_mapping = {
                'SELL_PUT': {
                    'side': 'sell_to_open',
                    'class': 'option',
                    'type': 'limit'
                },
                'SELL_CALL': {
                    'side': 'sell_to_open', 
                    'class': 'option',
                    'type': 'limit'
                },
                'BUY_PUT': {
                    'side': 'buy_to_open',
                    'class': 'option', 
                    'type': 'limit'
                },
                'BUY_CALL': {
                    'side': 'buy_to_open',
                    'class': 'option',
                    'type': 'limit'
                }
            }
            
            if action not in order_mapping:
                raise ValueError(f"Unsupported action: {action}")
            
            order_config = order_mapping[action]
            
            # Format expiration date for Tradier
            exp_date = datetime.strptime(expiry, '%Y-%m-%d').strftime('%Y%m%d')
            
            # Construct option symbol (simplified - you may need more sophisticated logic)
            option_symbol = f"{ticker}{exp_date}C{int(strike*1000):08d}" if 'CALL' in action else f"{ticker}{exp_date}P{int(strike*1000):08d}"
            
            order_data = {
                'class': order_config['class'],
                'symbol': option_symbol,
                'side': order_config['side'],
                'quantity': str(qty),
                'type': order_config['type'],
                'duration': 'day',
                'price': str(strike * 0.95),  # Default to 95% of strike as limit price
                'tag': f"AI_BOT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            }
            
            logger.info(f"Converted signal to order: {order_data}")
            return order_data
            
        except Exception as e:
            logger.error(f"Error converting signal to order: {e}")
            return {"error": str(e)}
    
    def execute_signal(self, signal: Dict) -> Tuple[bool, Dict]:
        """Execute a trading signal by converting and placing order"""
        try:
            # Convert signal to order format
            order_data = self.convert_signal_to_order(signal)
            
            if "error" in order_data:
                return False, order_data
            
            # Place the order
            success, result = self.place_order(order_data)
            
            if success:
                logger.info(f"Successfully executed signal: {signal.get('ticker')} {signal.get('action')}")
                return True, {
                    "status": "success",
                    "order_id": result.get('order', {}).get('id'),
                    "message": "Order placed successfully"
                }
            else:
                return False, result
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False, {"error": str(e)}
    
    def get_account_summary(self) -> Tuple[bool, Dict]:
        """Get comprehensive account summary"""
        try:
            # Get balance
            balance_success, balance_data = self.get_account_balance()
            
            # Get positions
            positions_success, positions = self.get_positions()
            
            # Get recent orders
            orders_success, orders = self.get_orders(status="all")
            
            summary = {
                "account_id": self.account_id,
                "balance": balance_data if balance_success else {"error": "Failed to retrieve"},
                "positions": positions if positions_success else [],
                "recent_orders": orders[:10] if orders_success else [],  # Last 10 orders
                "timestamp": datetime.now().isoformat()
            }
            
            return True, summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return False, {"error": str(e)}

# Utility functions for common operations
def create_tradier_client_from_env(trading_mode: TradingMode = None) -> Optional[TradierClient]:
    """Create TradierClient from environment variables using trading mode manager"""
    try:
        return TradierClient(trading_mode=trading_mode)
    except ValueError as e:
        logger.error(f"Failed to create Tradier client: {e}")
        return None

def create_tradier_client_for_mode(trading_mode: TradingMode) -> Optional[TradierClient]:
    """Create TradierClient for a specific trading mode"""
    try:
        return TradierClient(trading_mode=trading_mode)
    except ValueError as e:
        logger.error(f"Failed to create Tradier client for {trading_mode.value}: {e}")
        return None

def validate_tradier_connection(trading_mode: TradingMode = None) -> Tuple[bool, str]:
    """Validate Tradier API connection for current or specified mode"""
    client = create_tradier_client_from_env(trading_mode)
    if not client:
        mode_name = trading_mode.value if trading_mode else "current"
        return False, f"Failed to create Tradier client for {mode_name} mode - check environment variables"
    
    success, balance = client.get_account_balance()
    if success:
        mode_name = client.trading_mode.value
        return True, f"Tradier {mode_name} connection successful"
    else:
        mode_name = client.trading_mode.value
        return False, f"Tradier {mode_name} connection failed: {balance.get('error', 'Unknown error')}"

def validate_all_trading_modes() -> Dict[str, Tuple[bool, str]]:
    """Validate connections for all available trading modes"""
    mode_manager = get_trading_mode_manager()
    results = {}
    
    for mode in mode_manager.get_available_modes():
        success, message = validate_tradier_connection(mode)
        results[mode.value] = (success, message)
    
    return results

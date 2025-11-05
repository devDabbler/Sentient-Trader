"""
Warrior Trading Detector Service
Implements Ross Cameron's Gap & Go scalping strategy

Features:
- Premarket gapper detection (4-10% gap)
- Relative volume filter (2-3x)
- Price filter ($2-$20)
- 1-minute chart patterns
- Real-time entry alerts
- Stop/target calculation
- Market-wide scanning (S&P 500, NASDAQ 100, custom universes)
"""

from loguru import logger
import yfinance as yf
import os
from datetime import datetime, time as dt_time, timedelta
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.warrior_trading_strategy import (
    WarriorSetupType,
    GapAndGoSetup,
    MicroPullbackSetup,
    RedToGreenSetup,
    BullFlagSetup,
    MomentumScalpSetup,
    WarriorTradingSignal
)



class WarriorTradingDetector:
    """Detects Warrior Trading setups based on Gap & Go strategy"""
    
    def __init__(self, 
                 min_gap_pct: float = 4.0,
                 max_gap_pct: float = 10.0,
                 min_price: float = 2.0,
                 max_price: float = 20.0,
                 min_volume_ratio: float = 2.0,
                 max_volume_ratio: float = 3.0,
                 config = None,
                 watchlist: Optional[List[str]] = None):
        """
        Initialize detector
        
        Args:
            min_gap_pct: Minimum premarket gap percentage (default 4%)
            max_gap_pct: Maximum premarket gap percentage (default 10%)
            min_price: Minimum stock price (default $2)
            max_price: Maximum stock price (default $20)
            min_volume_ratio: Minimum relative volume (default 2x)
            max_volume_ratio: Maximum relative volume (default 3x)
            config: Configuration object (optional)
            watchlist: List of ticker symbols (optional, for fallback)
        """
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume_ratio = min_volume_ratio
        self.max_volume_ratio = max_volume_ratio
        self.config = config
        self.watchlist = watchlist or []
        
        # Cache for premarket data
        self._premarket_cache: Dict[str, Dict] = {}
        
    def get_premarket_gappers(self, tickers: Optional[List[str]] = None) -> List[Dict]:
        """
        PLACEHOLDER: Get premarket gappers (4-10% gap, 2-3x volume)
        
        TODO: Integrate with real Level 2 scanner later
        For now, uses extended hours data from yfinance
        
        Args:
            tickers: Optional list of tickers to scan. If None, uses watchlist
            
        Returns:
            List of gapper dictionaries with gap %, volume ratio, etc.
        """
        gappers = []
        
        try:
            # Use default watchlist if none provided
            if not tickers:
                tickers = [
                    'AAPL', 'AMD', 'TSLA', 'NVDA', 'PLTR', 'SOFI', 'RIVN',
                    'MARA', 'RIOT', 'NOK', 'AMC', 'GME', 'SNAP', 'HOOD',
                    'NIO', 'LCID', 'PLUG', 'FCEL', 'TLRY', 'SNDL'
                ]
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    
                    # Get extended hours data (includes premarket)
                    hist = stock.history(period="2d", interval="1m", prepost=True)
                    
                    if hist.empty or len(hist) < 1:
                        continue
                    
                    # Get previous day's close
                    prev_close = hist.iloc[-1]['Close'] if len(hist) > 0 else None
                    
                    # Get current/premarket price
                    info = stock.info
                    current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                    
                    if not current_price or not prev_close:
                        continue
                    
                    # Calculate gap
                    gap_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    # Check gap filter
                    if not (self.min_gap_pct <= abs(gap_pct) <= self.max_gap_pct):
                        continue
                    
                    # Check price filter
                    if not (self.min_price <= current_price <= self.max_price):
                        continue
                    
                    # Get volume data
                    volume = info.get('regularMarketVolume', 0) or info.get('volume', 0)
                    avg_volume = info.get('averageVolume', 0) or info.get('averageVolume10days', 0)
                    
                    if avg_volume == 0:
                        continue
                    
                    volume_ratio = volume / avg_volume if avg_volume > 0 else 0
                    
                    # Check volume filter
                    if not (self.min_volume_ratio <= volume_ratio <= self.max_volume_ratio):
                        continue
                    
                    gappers.append({
                        'ticker': ticker,
                        'gap_pct': gap_pct,
                        'current_price': current_price,
                        'prev_close': prev_close,
                        'volume_ratio': volume_ratio,
                        'volume': volume,
                        'avg_volume': avg_volume
                    })
                    
                    logger.debug(f"Found gapper: {ticker} - Gap: {gap_pct:.2f}%, Vol: {volume_ratio:.1f}x")
                    
                except Exception as e:
                    logger.debug(f"Error checking {ticker}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in get_premarket_gappers: {e}")
            
        return gappers
    
    def _get_market_universe(self) -> List[str]:
        """
        Get stock universe to scan based on config.
        
        Returns list of ticker symbols to screen.
        Options:
        - WATCHLIST: Use existing watchlist only
        - SP500: S&P 500 stocks
        - NASDAQ100: NASDAQ 100 stocks
        - ALL: Combined universe
        - CUSTOM: Load from file
        """
        universe_type = getattr(self.config, 'SCAN_UNIVERSE', 'WATCHLIST') if self.config else 'WATCHLIST'
        
        if universe_type == 'WATCHLIST':
            return self.watchlist
        
        elif universe_type == 'SP500':
            return self._load_sp500_symbols()
        
        elif universe_type == 'NASDAQ100':
            return self._load_nasdaq100_symbols()
        
        elif universe_type == 'ALL':
            # Combine multiple sources
            sp500 = self._load_sp500_symbols()
            nasdaq = self._load_nasdaq100_symbols()
            return list(set(sp500 + nasdaq + self.watchlist))
        
        elif universe_type == 'CUSTOM':
            return self._load_custom_universe()
        
        else:
            # Default fallback
            logger.warning(f"Unknown SCAN_UNIVERSE type '{universe_type}', falling back to watchlist")
            return self.watchlist
    
    def _load_sp500_symbols(self) -> List[str]:
        """Load S&P 500 symbol list"""
        file_path = 'data/sp500_symbols.txt'
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(symbols)} S&P 500 symbols")
                return symbols
            except Exception as e:
                logger.error(f"Error loading S&P 500 symbols: {e}")
                return []
        else:
            logger.warning(f"S&P 500 symbols file not found: {file_path}")
            return []
    
    def _load_nasdaq100_symbols(self) -> List[str]:
        """Load NASDAQ 100 symbol list"""
        file_path = 'data/nasdaq100_symbols.txt'
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(symbols)} NASDAQ 100 symbols")
                return symbols
            except Exception as e:
                logger.error(f"Error loading NASDAQ 100 symbols: {e}")
                return []
        else:
            logger.warning(f"NASDAQ 100 symbols file not found: {file_path}")
            return []
    
    def _load_custom_universe(self) -> List[str]:
        """Load custom universe from config file"""
        custom_path = getattr(self.config, 'CUSTOM_UNIVERSE_FILE', 'data/custom_universe.txt') if self.config else 'data/custom_universe.txt'
        if os.path.exists(custom_path):
            try:
                with open(custom_path, 'r') as f:
                    # Filter out comments and empty lines
                    symbols = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                logger.info(f"Loaded {len(symbols)} custom symbols")
                return symbols
            except Exception as e:
                logger.error(f"Error loading custom universe: {e}")
                return []
        else:
            logger.warning(f"Custom universe file not found: {custom_path}")
            return []
    
    def _get_previous_closes_yfinance(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get previous day closes using yfinance.
        
        Batch process with threading for speed.
        """
        prev_closes = {}
        
        # Use threading for parallel requests
        def fetch_close(ticker):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="5d")  # Get last 5 days
                if not hist.empty:
                    return ticker, hist['Close'].iloc[-1]
            except Exception as e:
                logger.debug(f"Error fetching close for {ticker}: {e}")
            return ticker, None
        
        # Parallel fetch with thread pool
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_close, t): t for t in tickers}
            
            for future in as_completed(futures):
                ticker, close_price = future.result()
                if close_price:
                    prev_closes[ticker] = float(close_price)
        
        logger.debug(f"Fetched previous closes for {len(prev_closes)}/{len(tickers)} tickers")
        return prev_closes
    
    def _get_premarket_quotes_tradier(self, tickers: List[str], tradier_client) -> Dict[str, Dict]:
        """
        Get premarket quotes from Tradier API.
        
        Uses existing Tradier client integration.
        Tradier supports bulk quotes via GET /markets/quotes.
        """
        quotes_data = {}
        
        try:
            # Tradier bulk quote format: ?symbols=AAPL,TSLA,AMD
            symbols_str = ','.join(tickers)
            
            # Call Tradier API
            response = tradier_client.get_quotes(symbols_str)
            
            # Parse response
            if response and 'quotes' in response:
                quotes = response['quotes'].get('quote', [])
                
                # Handle single vs multiple quotes
                if isinstance(quotes, dict):
                    quotes = [quotes]
                
                for quote in quotes:
                    ticker = quote.get('symbol')
                    if ticker:
                        quotes_data[ticker] = {
                            'last': quote.get('last', 0),
                            'bid': quote.get('bid', 0),
                            'ask': quote.get('ask', 0),
                            'volume': quote.get('volume', 0),
                            'average_volume': quote.get('average_volume', 0),
                            'timestamp': quote.get('trade_date', None)
                        }
            
            logger.debug(f"Fetched Tradier quotes for {len(quotes_data)}/{len(tickers)} tickers")
        
        except Exception as e:
            logger.error(f"Tradier API error: {e}")
        
        return quotes_data
    
    def _get_premarket_quotes_yfinance(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Fallback: Get quotes from yfinance.
        
        Less reliable for premarket but works if Tradier unavailable.
        """
        quotes_data = {}
        
        # Batch download
        try:
            data = yf.download(tickers, period='1d', interval='1m', 
                              prepost=True, group_by='ticker', progress=False)
            
            for ticker in tickers:
                try:
                    if ticker in data:
                        ticker_data = data[ticker]
                        if not ticker_data.empty:
                            last_price = ticker_data['Close'].iloc[-1]
                            volume = ticker_data['Volume'].sum()
                            
                            # Get average volume from info (separate call)
                            stock = yf.Ticker(ticker)
                            info = stock.info
                            avg_volume = info.get('averageVolume', 0)
                            
                            quotes_data[ticker] = {
                                'last': float(last_price),
                                'bid': float(last_price),  # Approximate
                                'ask': float(last_price),  # Approximate
                                'volume': int(volume),
                                'average_volume': avg_volume,
                                'timestamp': ticker_data.index[-1]
                            }
                except Exception:
                    continue
            
            logger.debug(f"Fetched yfinance quotes for {len(quotes_data)}/{len(tickers)} tickers")
        
        except Exception as e:
            logger.error(f"yfinance error: {e}")
        
        return quotes_data
    
    def scan_market_for_gappers(
        self,
        universe: Optional[List[str]] = None,
        tradier_client = None,
        use_yfinance: bool = True
    ) -> List[Dict]:
        """
        Scan entire market for premarket gappers matching Warrior Trading criteria.
        
        Uses hybrid approach:
        1. yfinance: Get previous day close (free, reliable)
        2. Tradier: Get premarket quotes (real-time, already integrated)
        3. Filter by Warrior Trading criteria:
           - Price range ($2-$20)
           - Gap percentage (2%-20%)
           - Volume ratio (1.5x-10x average)
        
        Args:
            universe: List of tickers to scan (if None, uses config)
            tradier_client: TradierClient instance for real-time quotes
            use_yfinance: Use yfinance for historical data (default True)
        
        Returns:
            List of qualified gapper dictionaries with:
            - ticker: Symbol
            - gap_pct: Gap percentage
            - current_price: Premarket price
            - prev_close: Previous day close
            - volume_ratio: Current/average volume
            - volume: Current volume
            - avg_volume: Average volume
        """
        qualified_gappers = []
        
        # Get universe to scan
        if not universe:
            universe = self._get_market_universe()
        
        if not universe:
            logger.warning("No tickers in universe to scan")
            return qualified_gappers
        
        logger.info(f"🔍 Scanning {len(universe)} tickers for premarket gappers...")
        
        # Apply max results limit from config
        max_results = getattr(self.config, 'MAX_SCAN_RESULTS', 50) if self.config else 50
        
        # Batch process in chunks (avoid rate limits)
        chunk_size = getattr(self.config, 'SCAN_CHUNK_SIZE', 50) if self.config else 50
        
        for i in range(0, len(universe), chunk_size):
            chunk = universe[i:i + chunk_size]
            logger.debug(f"Processing chunk {i//chunk_size + 1}: {len(chunk)} tickers")
            
            # Step 1: Get previous day closes (yfinance - batch)
            prev_closes = {}
            if use_yfinance:
                prev_closes = self._get_previous_closes_yfinance(chunk)
            
            # Step 2: Get premarket quotes (Tradier - batch)
            use_tradier = getattr(self.config, 'USE_TRADIER_QUOTES', True) if self.config else True
            if use_tradier and tradier_client:
                quotes_data = self._get_premarket_quotes_tradier(chunk, tradier_client)
            else:
                # Fallback to yfinance
                quotes_data = self._get_premarket_quotes_yfinance(chunk)
            
            # Step 3: Process each ticker and apply filters
            for ticker in chunk:
                # Skip if missing data
                if ticker not in prev_closes or ticker not in quotes_data:
                    continue
                
                prev_close = prev_closes[ticker]
                quote = quotes_data[ticker]
                current_price = quote.get('last', quote.get('bid', 0))
                
                # Skip if no price data
                if not current_price or not prev_close or prev_close == 0:
                    continue
                
                # Filter 1: Price range
                if not (self.min_price <= current_price <= self.max_price):
                    continue
                
                # Filter 2: Calculate gap percentage
                gap_pct = ((current_price - prev_close) / prev_close) * 100
                
                if not (self.min_gap_pct <= abs(gap_pct) <= self.max_gap_pct):
                    continue
                
                # Filter 3: Volume ratio
                volume = quote.get('volume', 0)
                avg_volume = quote.get('average_volume', 0)
                
                if avg_volume == 0:
                    continue
                
                volume_ratio = volume / avg_volume if avg_volume > 0 else 0
                
                if not (self.min_volume_ratio <= volume_ratio <= self.max_volume_ratio):
                    continue
                
                # Qualified gapper!
                qualified_gappers.append({
                    'ticker': ticker,
                    'gap_pct': gap_pct,
                    'current_price': current_price,
                    'prev_close': prev_close,
                    'volume_ratio': volume_ratio,
                    'volume': volume,
                    'avg_volume': avg_volume,
                    'timestamp': quote.get('timestamp', None)
                })
                
                logger.debug(f"✅ Found gapper: {ticker} - Gap: {gap_pct:.2f}%, Vol: {volume_ratio:.1f}x")
                
                # Stop if we hit max results
                if len(qualified_gappers) >= max_results:
                    logger.info(f"Reached max results limit ({max_results}), stopping scan")
                    return sorted(qualified_gappers, 
                                key=lambda x: abs(x['gap_pct']), 
                                reverse=True)
        
        # Sort by gap percentage (highest first)
        logger.info(f"✅ Market scan found {len(qualified_gappers)} qualified gappers")
        return sorted(qualified_gappers, 
                    key=lambda x: abs(x['gap_pct']), 
                    reverse=True)
    
    def detect_gap_and_go(self, ticker: str, gapper_data: Dict) -> Optional[WarriorTradingSignal]:
        """
        Detect Gap & Go setup
        
        Entry: Breakout above premarket high after 9:30 AM
        Stop: Low of breakout candle (1%)
        Target: 2% profit (scale out)
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get 1-minute data for today
            hist = stock.history(period="1d", interval="1m", prepost=True)
            
            if hist.empty:
                return None
            
            current_price = gapper_data['current_price']
            gap_pct = gapper_data['gap_pct']
            
            # Find premarket high (before 9:30 AM ET)
            # Convert to ET timezone (simplified - assumes UTC-5)
            hist['Time'] = hist.index
            premarket_bars = hist[hist.index.hour < 9]
            
            if premarket_bars.empty:
                premarket_high = current_price
            else:
                premarket_high = premarket_bars['High'].max()
            
            # Look for breakout above premarket high after 9:30
            market_open_bars = hist[hist.index.hour >= 9]
            
            if market_open_bars.empty:
                return None
            
            # Check latest candle
            latest_bar = market_open_bars.iloc[-1]
            
            # Bullish gap and go: price breaks above premarket high
            if current_price > premarket_high * 1.01:  # 1% buffer
                entry_price = current_price
                stop_loss = latest_bar['Low'] * 0.99  # 1% below low
                risk = entry_price - stop_loss
                profit_target = entry_price * 1.02  # 2% target
                rr_ratio = (profit_target - entry_price) / risk if risk > 0 else 2.0
                
                confidence = min(95.0, 70.0 + (gap_pct * 2) + (gapper_data['volume_ratio'] * 5))
                
                return WarriorTradingSignal(
                    ticker=ticker,
                    setup_type=WarriorSetupType.GAP_AND_GO,
                    direction="LONG",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    risk_reward_ratio=rr_ratio,
                    confidence=confidence,
                    reasoning=f"Gap & Go: {gap_pct:+.1f}% gap, {gapper_data['volume_ratio']:.1f}x volume, breakout above premarket high",
                    metadata={
                        'gap_pct': gap_pct,
                        'premarket_high': premarket_high,
                        'volume_ratio': gapper_data['volume_ratio']
                    },
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error detecting Gap & Go for {ticker}: {e}")
            
        return None
    
    def detect_micro_pullback(self, ticker: str) -> Optional[WarriorTradingSignal]:
        """
        Detect 1-minute Micro Pullback setup
        
        Entry: Small pullback (0.2-0.5%) in uptrend, bounce off 9 EMA
        Stop: Low of pullback candle
        Target: 2R (2x risk)
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d", interval="1m")
            
            if hist.empty or len(hist) < 20:
                return None
            
            # Calculate 9 EMA
            hist['EMA9'] = hist['Close'].ewm(span=9, adjust=False).mean()
            
            latest = hist.iloc[-1]
            prev = hist.iloc[-2]
            
            # Check for uptrend (price above EMA)
            if latest['Close'] < latest['EMA9']:
                return None
            
            # Check for small pullback
            pullback_pct = ((prev['Low'] - prev['High']) / prev['High']) * 100
            
            if not (-0.5 <= pullback_pct <= -0.2):  # 0.2-0.5% pullback
                return None
            
            # Volume spike on entry
            avg_volume = hist['Volume'].tail(20).mean()
            if latest['Volume'] < avg_volume * 1.5:
                return None
            
            # Entry on bounce
            if latest['Close'] > latest['Open'] and latest['Close'] > prev['Close']:
                entry_price = latest['Close']
                stop_loss = prev['Low']
                risk = entry_price - stop_loss
                profit_target = entry_price + (risk * 2)  # 2R
                rr_ratio = 2.0
                
                confidence = min(90.0, 65.0 + (abs(pullback_pct) * 20))
                
                return WarriorTradingSignal(
                    ticker=ticker,
                    setup_type=WarriorSetupType.MICRO_PULLBACK,
                    direction="LONG",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    risk_reward_ratio=rr_ratio,
                    confidence=confidence,
                    reasoning=f"Micro Pullback: {pullback_pct:.2f}% pullback, bounce off 9 EMA, volume spike",
                    metadata={
                        'pullback_pct': pullback_pct,
                        'ema_support': latest['EMA9']
                    },
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error detecting Micro Pullback for {ticker}: {e}")
            
        return None
    
    def detect_red_to_green(self, ticker: str) -> Optional[WarriorTradingSignal]:
        """
        Detect Red-to-Green reversal setup
        
        Entry: Green candle after red candles, above premarket low
        Stop: Below premarket low
        Target: 2% profit
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d", interval="1m", prepost=True)
            
            if hist.empty or len(hist) < 5:
                return None
            
            # Find premarket low
            premarket_bars = hist[hist.index.hour < 9]
            if premarket_bars.empty:
                return None
            
            premarket_low = premarket_bars['Low'].min()
            
            # Check last few candles for red-to-green pattern
            recent = hist.tail(5)
            
            # Count red candles
            red_count = sum(1 for _, row in recent.iterrows() if row['Close'] < row['Open'])
            
            if red_count < 2:
                return None
            
            # Check latest is green
            latest = recent.iloc[-1]
            if latest['Close'] <= latest['Open']:
                return None
            
            # Volume confirmation
            avg_volume = hist['Volume'].tail(20).mean()
            if latest['Volume'] < avg_volume * 1.3:
                return None
            
            # Price above premarket low
            if latest['Close'] < premarket_low:
                return None
            
            entry_price = latest['Close']
            stop_loss = premarket_low * 0.99  # 1% below premarket low
            risk = entry_price - stop_loss
            profit_target = entry_price * 1.02  # 2% target
            rr_ratio = (profit_target - entry_price) / risk if risk > 0 else 2.0
            
            confidence = min(85.0, 60.0 + (red_count * 5) + 10)
            
            return WarriorTradingSignal(
                ticker=ticker,
                setup_type=WarriorSetupType.RED_TO_GREEN,
                direction="LONG",
                entry_price=entry_price,
                stop_loss=stop_loss,
                profit_target=profit_target,
                risk_reward_ratio=rr_ratio,
                confidence=confidence,
                reasoning=f"Red-to-Green: {red_count} red candles, green reversal, volume spike",
                metadata={
                    'premarket_low': premarket_low,
                    'red_candles': red_count
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error detecting Red-to-Green for {ticker}: {e}")
            
        return None
    
    def detect_bull_flag(self, ticker: str) -> Optional[WarriorTradingSignal]:
        """
        Detect Bull Flag breakout setup
        
        Entry: Breakout above flag high
        Stop: Below flag low
        Target: Flag pole height
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d", interval="5m")
            
            if hist.empty or len(hist) < 30:
                return None
            
            # Find flag pole (strong move up)
            # Look for 3+ consecutive green candles with volume
            hist['IsGreen'] = hist['Close'] > hist['Open']
            hist['VolumeRatio'] = hist['Volume'] / hist['Volume'].rolling(20).mean()
            
            # Find flag pole high
            flag_pole_high = hist['High'].rolling(10).max().iloc[-1]
            
            # Find consolidation (flag)
            recent = hist.tail(20)
            flag_low = recent['Low'].min()
            flag_high = recent['High'].max()
            
            # Check for breakout
            latest = hist.iloc[-1]
            if latest['Close'] > flag_high * 1.01:  # 1% breakout
                # Volume confirmation
                if latest['VolumeRatio'] < 2.0:
                    return None
                
                entry_price = latest['Close']
                stop_loss = flag_low * 0.99
                risk = entry_price - stop_loss
                
                # Target = flag pole height
                flag_pole_height = flag_pole_high - flag_low
                profit_target = entry_price + flag_pole_height
                rr_ratio = flag_pole_height / risk if risk > 0 else 2.0
                
                confidence = min(90.0, 70.0 + (latest['VolumeRatio'] * 10))
                
                return WarriorTradingSignal(
                    ticker=ticker,
                    setup_type=WarriorSetupType.BULL_FLAG,
                    direction="LONG",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    profit_target=profit_target,
                    risk_reward_ratio=rr_ratio,
                    confidence=confidence,
                    reasoning=f"Bull Flag: Breakout above flag, volume {latest['VolumeRatio']:.1f}x",
                    metadata={
                        'flag_pole_high': flag_pole_high,
                        'flag_low': flag_low,
                        'volume_ratio': latest['VolumeRatio']
                    },
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error detecting Bull Flag for {ticker}: {e}")
            
        return None
    
    def scan_for_setups(self, tickers: Optional[List[str]] = None, 
                       trading_window_start: dt_time = dt_time(9, 30),
                       trading_window_end: dt_time = dt_time(10, 0),
                       bypass_window_check: bool = False) -> List[WarriorTradingSignal]:
        """
        Scan for all Warrior Trading setups
        
        Args:
            tickers: List of tickers to scan (uses default if None)
            trading_window_start: Start of trading window (default 9:30 AM)
            trading_window_end: End of trading window (default 10:00 AM)
            bypass_window_check: If True, skip trading window check (for test mode)
            
        Returns:
            List of WarriorTradingSignal objects
        """
        signals = []
        
        # Check if within trading window (unless bypassed for test mode)
        if not bypass_window_check:
            now = datetime.now().time()
            
            # Check if premarket is enabled in config
            enable_premarket = getattr(self.config, 'enable_premarket', False) or \
                             getattr(self.config, 'ENABLE_PREMARKET', False)
            
            if enable_premarket:
                # Get premarket start time
                premarket_start_hour = getattr(self.config, 'premarket_start_hour', 
                                              getattr(self.config, 'PREMARKET_START_HOUR', 7))
                premarket_start_minute = getattr(self.config, 'premarket_start_minute',
                                                getattr(self.config, 'PREMARKET_START_MINUTE', 0))
                premarket_start = dt_time(premarket_start_hour, premarket_start_minute)
                
                # For Gap & Go, scan during premarket (7:00 AM - 10:00 AM)
                # This allows identifying gappers before market open
                extended_window_start = premarket_start
                extended_window_end = trading_window_end  # Keep 10:00 AM end time
                
                if not (extended_window_start <= now <= extended_window_end):
                    logger.debug(f"Outside extended trading window (premarket enabled): {now} not in {extended_window_start}-{extended_window_end}")
                    return signals
                else:
                    if now < trading_window_start:
                        logger.info(f"🌅 Premarket scanning active: {now} (market opens at {trading_window_start})")
            else:
                # Standard trading window check (9:30 AM - 10:00 AM)
                if not (trading_window_start <= now <= trading_window_end):
                    logger.debug(f"Outside trading window: {now} not in {trading_window_start}-{trading_window_end}")
                    return signals
        
        # Get premarket gappers first
        gappers = self.get_premarket_gappers(tickers)
        
        for gapper in gappers:
            ticker = gapper['ticker']
            
            # Try Gap & Go first (primary strategy)
            gap_go_signal = self.detect_gap_and_go(ticker, gapper)
            if gap_go_signal:
                signals.append(gap_go_signal)
                continue  # Don't check other setups if Gap & Go found
            
            # Try other setups
            micro_signal = self.detect_micro_pullback(ticker)
            if micro_signal:
                signals.append(micro_signal)
            
            red_green_signal = self.detect_red_to_green(ticker)
            if red_green_signal:
                signals.append(red_green_signal)
            
            bull_flag_signal = self.detect_bull_flag(ticker)
            if bull_flag_signal:
                signals.append(bull_flag_signal)
        
        logger.info(f"Found {len(signals)} Warrior Trading setups")
        return signals


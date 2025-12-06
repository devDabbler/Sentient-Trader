"""
Crypto Opportunity Scanner
Scans cryptocurrency markets for trading opportunities

Adapted from TopTradesScanner for crypto-specific analysis
Fetches from multiple sources: CoinGecko, CoinMarketCap, and Kraken
"""

from loguru import logger
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from clients.kraken_client import KrakenClient
from services.crypto_data_aggregator import CryptoDataAggregator, AggregatedCryptoData
from utils.crypto_pair_utils import normalize_crypto_pair, extract_base_asset
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio


@dataclass
class CryptoOpportunity:
    """Container for a crypto trading opportunity"""
    symbol: str  # e.g., 'BTC/USD'
    base_asset: str  # e.g., 'BTC'
    score: float
    current_price: float
    change_pct_24h: float
    volume_24h: float
    volume_ratio: float
    volatility_24h: float
    reason: str
    strategy: str  # 'scalp', 'momentum', 'swing'
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    fear_greed_index: Optional[int] = None
    # Whale tracking data
    whale_activity_score: Optional[float] = None  # 0-100 whale activity score
    whale_alert: Optional[str] = None  # Whale alert description if any
    exchange_flow_direction: Optional[str] = None  # 'INFLOW', 'OUTFLOW', 'NEUTRAL'
    whale_confidence: Optional[str] = None  # Confidence in whale signal validity


class CryptoOpportunityScanner:
    """Scans cryptocurrency markets for trading opportunities"""
    
    def __init__(self, kraken_client: KrakenClient, config=None):
        """
        Initialize crypto scanner
        
        Args:
            kraken_client: KrakenClient instance
            config: Trading configuration
        """
        self.client = kraken_client
        self.config = config
        self.aggregator = CryptoDataAggregator()
        
        # Popular crypto pairs from config or defaults - expanded for better coverage
        if config and hasattr(config, 'CRYPTO_WATCHLIST'):
            self.watchlist = config.CRYPTO_WATCHLIST
        else:
            self.watchlist = [
                # Major Layer 1 Blockchains
                'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'AVAX/USD',
                'DOT/USD', 'ATOM/USD', 'NEAR/USD', 'APT/USD', 'SUI/USD',
                'TON/USD', 'SEI/USD', 'INJ/USD', 'TIA/USD', 'KASPA/USD',
                
                # Layer 2 & Scaling
                'MATIC/USD', 'ARB/USD', 'OP/USD', 'LINEA/USD', 'SCROLL/USD',
                'STX/USD', 'RSK/USD', 'MANTA/USD', 'BLAST/USD', 'ZK/USD',
                'MODE/USD', 'BASE/USD', 'TAIKO/USD',
                
                # DeFi Leaders
                'LINK/USD', 'UNI/USD', 'AAVE/USD', 'CURVE/USD', 'LIDO/USD',
                'COMPOUND/USD', 'MAKER/USD', 'SNX/USD', 'DYDX/USD', 'GMX/USD',
                'PENDLE/USD', 'EIGEN/USD', 'ETHENA/USD', 'MORPHO/USD',
                
                # AI & Data (HOT Sector)
                'RENDER/USD', 'FET/USD', 'AGIX/USD', 'OCEAN/USD', 'ARKM/USD',
                'TAO/USD', 'AKT/USD', 'RNDR/USD', 'AIOZ/USD', 'NMR/USD',
                'VIRTUAL/USD', 'AI16Z/USD', 'GOAT/USD', 'GRIFFAIN/USD',
                
                # RWA & Tokenization (Hot Sector)
                'ONDO/USD', 'CPOOL/USD', 'MAPLE/USD', 'CENTRI/USD', 'PROPC/USD',
                
                # New Launches & High Potential
                'JTO/USD', 'PYTH/USD', 'STRK/USD', 'BLUR/USD', 'JUP/USD',
                'W/USD', 'ETHFI/USD', 'ALT/USD', 'AEVO/USD', 'ENA/USD',
                'ZRO/USD', 'LISTA/USD', 'NOT/USD', 'DOGS/USD', 'CATI/USD',
                'HMSTR/USD', 'NEIRO/USD', 'TURBO/USD', 'BRETT/USD',
                
                # Gaming & Metaverse
                'GALA/USD', 'SAND/USD', 'MANA/USD', 'ENJ/USD', 'THETA/USD',
                'AXIE/USD', 'FLOW/USD', 'ILV/USD', 'PRIME/USD', 'BEAM/USD',
                'PORTAL/USD', 'PIXEL/USD', 'XAI/USD', 'RONIN/USD',
                
                # Privacy & Security
                'MONERO/USD', 'ZCASH/USD', 'DASH/USD', 'SCRT/USD',
                
                # Solana Ecosystem
                'MARINADE/USD', 'MAGIC/USD', 'COPE/USD', 'RAY/USD', 'ORCA/USD',
                'JITO/USD', 'DRIFT/USD', 'TENSOR/USD', 'PARCL/USD',
                
                # Cosmos Ecosystem
                'OSMO/USD', 'JUNO/USD', 'STARS/USD', 'EVMOS/USD', 'DYM/USD',
                
                # Meme Coins (high volatility - major opportunities)
                'SHIB/USD', 'DOGE/USD', 'PEPE/USD', 'FLOKI/USD', 'BONK/USD',
                'WIF/USD', 'POPCAT/USD', 'MEW/USD', 'BOME/USD', 'SLERF/USD',
                'PONKE/USD', 'MYRO/USD', 'MICHI/USD', 'GIGA/USD', 'SPX/USD',
                'MOG/USD', 'LADYS/USD', 'WOJAK/USD', 'MUMU/USD',
                
                # Infrastructure & Utilities
                'FIL/USD', 'AR/USD', 'GRT/USD', 'LPT/USD', 'ANKR/USD',
                'POKT/USD', 'FLUX/USD', 'HNT/USD', 'MOBILE/USD', 'IOT/USD',
                
                # Exchange Tokens
                'BNB/USD', 'CRO/USD', 'OKB/USD', 'LEO/USD', 'KCS/USD',
                
                # Additional Emerging
                'VET/USD', 'TRX/USD', 'ALGO/USD', 'HBAR/USD', 'PERP/USD', 'GNS/USD',
                'XLM/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD', 'ETC/USD'
            ]
        
        # Track dynamic trending coins (populated by fetch_trending_coins)
        self.trending_coins: List[str] = []
        self.last_trending_fetch = None
        
        logger.info(f"Crypto Scanner initialized with {len(self.watchlist)} pairs")
        logger.info("   • Multi-source data: CoinGecko, CoinMarketCap, Kraken")
        logger.info("   • Dynamic trending coin detection enabled")
    
    def fetch_trending_coins(self, force_refresh: bool = False) -> List[str]:
        """
        Fetch trending coins from CoinGecko trending API.
        Caches results for 15 minutes to avoid rate limits.
        
        Returns:
            List of trending coin symbols (e.g., ['WIF/USD', 'PEPE/USD'])
        """
        import requests
        from datetime import datetime, timedelta
        
        # Check cache (15 minute TTL)
        if not force_refresh and self.last_trending_fetch:
            if datetime.now() - self.last_trending_fetch < timedelta(minutes=15):
                return self.trending_coins
        
        try:
            logger.info("🔥 Fetching trending coins from CoinGecko...")
            
            # CoinGecko trending API
            response = requests.get(
                "https://api.coingecko.com/api/v3/search/trending",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                trending = []
                
                # Extract trending coins
                for item in data.get('coins', []):
                    coin = item.get('item', {})
                    symbol = coin.get('symbol', '').upper()
                    if symbol:
                        trending.append(f"{symbol}/USD")
                
                # Also fetch top gainers for more opportunities
                gainers_response = requests.get(
                    "https://api.coingecko.com/api/v3/coins/markets",
                    params={
                        'vs_currency': 'usd',
                        'order': 'percent_change_24h_desc',
                        'per_page': 50,
                        'page': 1,
                        'sparkline': False
                    },
                    timeout=10
                )
                
                if gainers_response.status_code == 200:
                    gainers = gainers_response.json()
                    for coin in gainers:
                        symbol = coin.get('symbol', '').upper()
                        change = coin.get('price_change_percentage_24h', 0)
                        # Only add if significant move (>5%)
                        if symbol and abs(change or 0) > 5:
                            pair = f"{symbol}/USD"
                            if pair not in trending:
                                trending.append(pair)
                
                self.trending_coins = trending
                self.last_trending_fetch = datetime.now()
                logger.info(f"✅ Found {len(trending)} trending/moving coins")
                return trending
            else:
                logger.warning(f"CoinGecko trending API returned {response.status_code}")
                return self.trending_coins
                
        except Exception as e:
            logger.warning(f"Failed to fetch trending coins: {e}")
            return self.trending_coins
    
    def get_combined_watchlist(self) -> List[str]:
        """
        Get combined watchlist including static watchlist + trending coins.
        Removes duplicates.
        """
        # Fetch trending if needed
        trending = self.fetch_trending_coins()
        
        # Combine and deduplicate
        combined = list(set(self.watchlist + trending))
        return combined
    
    def scan_opportunities(
        self,
        strategy: str = 'ALL',
        top_n: int = 10,
        min_score: float = 60.0,
        use_parallel: bool = True
    ) -> List[CryptoOpportunity]:
        """
        Scan for crypto trading opportunities
        
        Args:
            strategy: 'SCALP', 'MOMENTUM', 'SWING', or 'ALL'
            top_n: Number of top opportunities to return
            min_score: Minimum score threshold
            use_parallel: Use parallel processing for 5-8x speedup (default: True)
            
        Returns:
            List of CryptoOpportunity objects
        """
        if use_parallel:
            return self._scan_opportunities_parallel(strategy, top_n, min_score)
        else:
            return self._scan_opportunities_sequential(strategy, top_n, min_score)
    
    def _scan_opportunities_sequential(
        self,
        strategy: str,
        top_n: int,
        min_score: float
    ) -> List[CryptoOpportunity]:
        """Sequential scanning (fallback method)"""
        logger.info(f"Scanning {len(self.watchlist)} crypto pairs for {strategy} opportunities (sequential)...")
        
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                opportunity = self._analyze_crypto_pair(symbol, strategy)
                
                if opportunity and opportunity.score >= min_score:
                    opportunities.append(opportunity)
                    
                # Rate limiting - be nice to Kraken API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Found {len(opportunities)} opportunities (showing top {top_n})")
        
        return opportunities[:top_n]
    
    def _scan_opportunities_parallel(
        self,
        strategy: str,
        top_n: int,
        min_score: float,
        max_workers: int = 8
    ) -> List[CryptoOpportunity]:
        """Parallel scanning using ThreadPoolExecutor (5-8x faster)"""
        logger.info(f"Scanning {len(self.watchlist)} crypto pairs for {strategy} opportunities (parallel, {max_workers} workers)...")
        
        opportunities = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_symbol = {
                executor.submit(self._analyze_crypto_pair, symbol, strategy): symbol
                for symbol in self.watchlist
            }
            
            # Process results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    opportunity = future.result()
                    
                    if opportunity and opportunity.score >= min_score:
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
        
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Found {len(opportunities)} opportunities (showing top {top_n}) in {elapsed_time:.2f}s")
        
        return opportunities[:top_n]
    
    def _analyze_crypto_pair(
        self,
        symbol: str,
        strategy: str
    ) -> Optional[CryptoOpportunity]:
        """
        Analyze a single crypto pair for trading opportunities
        
        Args:
            symbol: Crypto pair (e.g., 'BTC/USD')
            strategy: Target strategy
            
        Returns:
            CryptoOpportunity or None
        """
        try:
            # Normalize pair format globally (handles BTC/USD, BTCUSD, btcusd, btc/usd)
            normalized_symbol = normalize_crypto_pair(symbol)
            
            # Get market data
            ticker = self.client.get_ticker_data(normalized_symbol)
            
            if not ticker:
                return None
            
            # Get OHLC data for technical analysis
            ohlc_5m = self.client.get_ohlc_data(normalized_symbol, interval=5)  # 5-minute
            ohlc_1h = self.client.get_ohlc_data(normalized_symbol, interval=60)  # 1-hour
            
            if not ohlc_5m or not ohlc_1h:
                return None
            
            # Calculate metrics
            current_price = ticker['last_price']
            change_pct_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
            volume_24h = ticker['volume_24h']
            
            # Calculate volume ratio (current vs average)
            avg_volume = sum([candle['volume'] for candle in ohlc_1h[-24:]]) / 24 if len(ohlc_1h) >= 24 else volume_24h
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate volatility
            volatility_24h = self._calculate_volatility([candle['close'] for candle in ohlc_1h[-24:]])
            
            # Calculate technical indicators
            rsi = self._calculate_rsi([candle['close'] for candle in ohlc_5m[-14:]])
            ema_8 = self._calculate_ema([candle['close'] for candle in ohlc_5m[-20:]], 8)
            ema_20 = self._calculate_ema([candle['close'] for candle in ohlc_1h[-40:]], 20)
            
            # Score the opportunity
            score, reason, confidence, risk_level, target_strategy = self._score_opportunity(
                symbol=normalized_symbol,
                current_price=current_price,
                change_pct_24h=change_pct_24h,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                rsi=rsi,
                ema_8=ema_8,
                ema_20=ema_20,
                strategy=strategy
            )
            
            # Parse base asset
            base_asset = extract_base_asset(normalized_symbol)
            
            # Create opportunity
            opportunity = CryptoOpportunity(
                symbol=normalized_symbol,
                base_asset=base_asset,
                score=score,
                current_price=current_price,
                change_pct_24h=change_pct_24h,
                volume_24h=volume_24h,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                reason=reason,
                strategy=target_strategy,
                confidence=confidence,
                risk_level=risk_level
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _analyze_crypto_pair_multi_source(
        self,
        symbol: str,
        strategy: str,
        aggregated_coins: List[AggregatedCryptoData]
    ) -> Optional[CryptoOpportunity]:
        """
        Analyze a single crypto pair using multiple sources
        Tries Kraken first, falls back to CoinGecko/CoinMarketCap data
        
        Args:
            symbol: Crypto pair (e.g., 'BTC/USD')
            strategy: Target strategy
            aggregated_coins: List of AggregatedCryptoData from CoinGecko/CoinMarketCap
            
        Returns:
            CryptoOpportunity or None
        """
        try:
            # Normalize pair format globally
            normalized_symbol = normalize_crypto_pair(symbol)
            base_asset = extract_base_asset(normalized_symbol)
            
            # Try to get data from aggregated coins first
            aggregated_data = None
            for coin in aggregated_coins:
                if coin.symbol.upper() == base_asset:
                    aggregated_data = coin
                    break
            
            # Try Kraken first (for technical analysis) - use normalized symbol
            ticker = self.client.get_ticker_data(normalized_symbol)
            ohlc_5m = None
            ohlc_1h = None
            
            if ticker:
                current_price = ticker['last_price']
                # Get OHLC data for technical analysis - use normalized symbol
                ohlc_5m = self.client.get_ohlc_data(normalized_symbol, interval=5)
                ohlc_1h = self.client.get_ohlc_data(normalized_symbol, interval=60)
            elif aggregated_data:
                # Use aggregated data from CoinGecko/CoinMarketCap
                current_price = aggregated_data.price_usd
            else:
                return None
            
            # Calculate metrics from available data
            if ticker and ohlc_1h:
                # Use Kraken data (most accurate)
                change_pct_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100 if ticker['low_24h'] > 0 else 0
                volume_24h = ticker['volume_24h']
                
                # Calculate volume ratio
                avg_volume = sum([candle['volume'] for candle in ohlc_1h[-24:]]) / 24 if len(ohlc_1h) >= 24 else volume_24h
                volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
                
                # Calculate volatility
                volatility_24h = self._calculate_volatility([candle['close'] for candle in ohlc_1h[-24:]])
                
                # Calculate technical indicators
                rsi = self._calculate_rsi([candle['close'] for candle in ohlc_5m[-14:]]) if ohlc_5m and len(ohlc_5m) >= 14 else 50.0
                ema_8 = self._calculate_ema([candle['close'] for candle in ohlc_5m[-20:]], 8) if ohlc_5m and len(ohlc_5m) >= 20 else current_price
                ema_20 = self._calculate_ema([candle['close'] for candle in ohlc_1h[-40:]], 20) if ohlc_1h and len(ohlc_1h) >= 40 else current_price
            elif aggregated_data:
                # Use aggregated data (CoinGecko/CoinMarketCap)
                change_pct_24h = aggregated_data.change_24h
                volume_24h = aggregated_data.volume_24h
                
                # Estimate volume ratio (use 1.0 as default)
                volume_ratio = 1.0
                
                # Estimate volatility from 24h change
                volatility_24h = abs(change_pct_24h) if change_pct_24h else 0.0
                
                # Default technical indicators (can't calculate without OHLC)
                rsi = 50.0
                ema_8 = current_price
                ema_20 = current_price
            else:
                return None
            
            # Score the opportunity - use normalized symbol
            score, reason, confidence, risk_level, target_strategy = self._score_opportunity(
                symbol=normalized_symbol,
                current_price=current_price,
                change_pct_24h=change_pct_24h,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                rsi=rsi,
                ema_8=ema_8,
                ema_20=ema_20,
                strategy=strategy
            )
            
            # Parse base asset
            base_asset = extract_base_asset(normalized_symbol)
            
            # Create opportunity - use normalized symbol
            opportunity = CryptoOpportunity(
                symbol=normalized_symbol,
                base_asset=base_asset,
                score=score,
                current_price=current_price,
                change_pct_24h=change_pct_24h,
                volume_24h=volume_24h,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                reason=reason,
                strategy=target_strategy,
                confidence=confidence,
                risk_level=risk_level
            )
            
            return opportunity
            
        except Exception as e:
            logger.debug(f"Error analyzing {symbol} (multi-source): {e}")
            return None
    
    def _score_opportunity(
        self,
        symbol: str,
        current_price: float,
        change_pct_24h: float,
        volume_ratio: float,
        volatility_24h: float,
        rsi: float,
        ema_8: float,
        ema_20: float,
        strategy: str
    ) -> Tuple[float, str, str, str, str]:
        """
        Score a crypto opportunity based on strategy
        
        Returns:
            (score, reason, confidence, risk_level, target_strategy)
        """
        score = 0.0
        reasons = []
        target_strategy = 'scalp'
        
        # Price momentum score
        if abs(change_pct_24h) > 8:
            score += 25
            reasons.append(f"Strong 24h move ({change_pct_24h:.1f}%)")
        elif abs(change_pct_24h) > 4:
            score += 15
            reasons.append(f"Good momentum ({change_pct_24h:.1f}%)")
        
        # Volume score
        if volume_ratio > 2.5:
            score += 25
            reasons.append(f"High volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 1.5:
            score += 15
            reasons.append(f"Above avg volume ({volume_ratio:.1f}x)")
        
        # RSI score
        if rsi < 30:
            score += 20
            reasons.append(f"Oversold RSI ({rsi:.0f})")
            target_strategy = 'momentum'
        elif rsi > 70:
            score += 15
            reasons.append(f"Overbought RSI ({rsi:.0f}) - potential reversal")
        elif 40 <= rsi <= 60:
            score += 10
            reasons.append("Neutral RSI - range bound")
        
        # Trend score (EMA)
        if current_price > ema_8 > ema_20:
            score += 20
            reasons.append("Bullish trend (price > EMA8 > EMA20)")
            if target_strategy == 'scalp':
                target_strategy = 'swing'
        elif current_price < ema_8 < ema_20:
            score += 15
            reasons.append("Bearish trend - short opportunity")
        
        # Volatility score (match strategy preference)
        if strategy in ['SCALP', 'ALL']:
            if 3 <= volatility_24h <= 8:
                score += 15
                reasons.append(f"Good scalping volatility ({volatility_24h:.1f}%)")
                target_strategy = 'scalp'
        
        if strategy in ['MOMENTUM', 'ALL']:
            if volatility_24h > 8:
                score += 20
                reasons.append(f"High momentum potential ({volatility_24h:.1f}%)")
                target_strategy = 'momentum'
        
        if strategy in ['SWING', 'ALL']:
            if volatility_24h < 5 and abs(change_pct_24h) > 2:
                score += 15
                reasons.append(f"Swing setup - steady trend")
                target_strategy = 'swing'
        
        # Determine confidence
        if score >= 80:
            confidence = 'HIGH'
        elif score >= 60:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Determine risk level
        if volatility_24h > 12:
            risk_level = 'EXTREME'
        elif volatility_24h > 8:
            risk_level = 'HIGH'
        elif volatility_24h > 4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        reason = '; '.join(reasons) if reasons else 'No significant signals'
        
        return score, reason, confidence, risk_level, target_strategy
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility (standard deviation)"""
        if len(prices) < 2:
            return 0.0
        
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std_dev = variance ** 0.5
        
        return (std_dev / mean) * 100 if mean > 0 else 0.0
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Neutral default
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return sum(prices) / len(prices) if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """
        Calculate Average Directional Index (ADX) for trend strength
        Hyperopt showed ADX threshold of 15-25 works well for crypto
        
        Returns:
            ADX value (0-100), higher = stronger trend
        """
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return 25.0  # Default moderate trend
        
        try:
            # Calculate True Range and Directional Movement
            tr_list = []
            plus_dm_list = []
            minus_dm_list = []
            
            for i in range(1, len(highs)):
                high = highs[i]
                low = lows[i]
                prev_high = highs[i-1]
                prev_low = lows[i-1]
                prev_close = closes[i-1]
                
                # True Range
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_list.append(tr)
                
                # Directional Movement
                plus_dm = high - prev_high if (high - prev_high) > (prev_low - low) and (high - prev_high) > 0 else 0
                minus_dm = prev_low - low if (prev_low - low) > (high - prev_high) and (prev_low - low) > 0 else 0
                plus_dm_list.append(plus_dm)
                minus_dm_list.append(minus_dm)
            
            if len(tr_list) < period:
                return 25.0
            
            # Smoothed averages (Wilder's smoothing)
            atr = sum(tr_list[:period]) / period
            plus_di = sum(plus_dm_list[:period]) / period
            minus_di = sum(minus_dm_list[:period]) / period
            
            for i in range(period, len(tr_list)):
                atr = (atr * (period - 1) + tr_list[i]) / period
                plus_di = (plus_di * (period - 1) + plus_dm_list[i]) / period
                minus_di = (minus_di * (period - 1) + minus_dm_list[i]) / period
            
            # Calculate +DI and -DI
            if atr > 0:
                plus_di_pct = (plus_di / atr) * 100
                minus_di_pct = (minus_di / atr) * 100
            else:
                return 25.0
            
            # Calculate DX
            di_sum = plus_di_pct + minus_di_pct
            if di_sum > 0:
                dx = abs(plus_di_pct - minus_di_pct) / di_sum * 100
            else:
                return 25.0
            
            # ADX is smoothed DX (simplified - just return DX for now)
            return min(dx, 100.0)
            
        except Exception as e:
            logger.debug(f"ADX calculation error: {e}")
            return 25.0  # Default moderate trend
    
    def get_crypto_market_overview(self) -> Dict:
        """
        Get overview of crypto market conditions
        
        Returns:
            Dict with market statistics
        """
        try:
            total_volume = 0.0
            avg_change = 0.0
            avg_volatility = 0.0
            bullish_count = 0
            bearish_count = 0
            
            for symbol in self.watchlist[:10]:  # Sample top 10
                try:
                    ticker = self.client.get_ticker_data(symbol)
                    if ticker:
                        total_volume += ticker['volume_24h']
                        change = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
                        avg_change += change
                        
                        if ticker['last_price'] > ticker['vwap_24h']:
                            bullish_count += 1
                        else:
                            bearish_count += 1
                    
                    time.sleep(0.3)  # Rate limiting
                    
                except:
                    continue
            
            sample_size = min(10, len(self.watchlist))
            avg_change /= sample_size if sample_size > 0 else 1
            
            market_sentiment = 'BULLISH' if bullish_count > bearish_count else 'BEARISH'
            
            return {
                'total_volume_sampled': total_volume,
                'avg_change_pct': avg_change,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'market_sentiment': market_sentiment,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {}


    def scan_buzzing_cryptos(
        self,
        top_n: int = 10,
        min_volume_ratio: float = 2.0,
        use_parallel: bool = True,
        use_multi_source: bool = True
    ) -> List[CryptoOpportunity]:
        """
        Scan for buzzing/trending cryptocurrencies with high social momentum
        Focus on volume surges and rapid price movement
        
        Args:
            top_n: Number of top buzzing cryptos to return
            min_volume_ratio: Minimum volume ratio (vs average)
            use_parallel: Use parallel processing for 5-8x speedup (default: True)
            use_multi_source: Fetch from CoinGecko, CoinMarketCap, and Kraken (default: True)
            
        Returns:
            List of buzzing CryptoOpportunity objects
        """
        logger.info(f"🔥 Scanning for {top_n} buzzing cryptocurrencies...")
        
        if use_multi_source:
            return self._scan_buzzing_multi_source(top_n, min_volume_ratio, use_parallel)
        
        buzzing_opportunities = []
        start_time = time.time()
        
        # Use parallel or sequential processing
        if use_parallel:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_crypto_pair, symbol, 'MOMENTUM'): symbol
                    for symbol in self.watchlist
                }
                
                for future in as_completed(future_to_symbol):
                    try:
                        opportunity = future.result()
                        
                        if opportunity and opportunity.volume_ratio >= min_volume_ratio:
                            buzz_bonus = min((opportunity.volume_ratio - 1.0) * 10, 30)
                            opportunity.score += buzz_bonus
                            opportunity.reason += f" | 🔥 BUZZING (Vol: {opportunity.volume_ratio:.1f}x)"
                            buzzing_opportunities.append(opportunity)
                            
                    except Exception as e:
                        logger.error(f"Error analyzing for buzz: {e}")
                        continue
        else:
            for symbol in self.watchlist:
                try:
                    opportunity = self._analyze_crypto_pair(symbol, 'MOMENTUM')
                    
                    if opportunity and opportunity.volume_ratio >= min_volume_ratio:
                        buzz_bonus = min((opportunity.volume_ratio - 1.0) * 10, 30)
                        opportunity.score += buzz_bonus
                        opportunity.reason += f" | 🔥 BUZZING (Vol: {opportunity.volume_ratio:.1f}x)"
                        buzzing_opportunities.append(opportunity)
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} for buzz: {e}")
                    continue
        
        # Sort by volume ratio (primary indicator of buzz)
        buzzing_opportunities.sort(key=lambda x: x.volume_ratio, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Found {len(buzzing_opportunities)} buzzing cryptos in {elapsed_time:.2f}s")
        
        return buzzing_opportunities[:top_n]
    
    def _scan_buzzing_multi_source(
        self,
        top_n: int,
        min_volume_ratio: float,
        use_parallel: bool = True
    ) -> List[CryptoOpportunity]:
        """
        Scan for buzzing cryptos using multiple sources
        Fetches from CoinGecko, CoinMarketCap, Kraken + trending
        """
        logger.info(f"🔍 Scanning buzzing cryptos from multiple sources + trending...")
        start_time = time.time()
        
        # Fetch trending coins first
        trending_symbols = self.fetch_trending_coins()
        
        # Fetch from CoinGecko and CoinMarketCap with LOWER thresholds
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            aggregated_coins = loop.run_until_complete(
                self.aggregator.fetch_all_coins(
                    min_volume_24h=25000,  # LOWERED: $25k volume (was $100k)
                    max_coins=500  # INCREASED: 500 coins (was 300)
                )
            )
            loop.close()
            
            logger.info(f"✅ Fetched {len(aggregated_coins)} coins from CoinGecko/CoinMarketCap")
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
            aggregated_coins = []
        
        # Create symbol list from aggregated coins
        symbols_from_sources = [f"{coin.symbol}/USD" for coin in aggregated_coins]
        
        # Combine watchlist + trending + aggregated (remove duplicates)
        all_symbols = list(set(symbols_from_sources + self.watchlist + trending_symbols))
        
        logger.info(f"📊 Analyzing {len(all_symbols)} total symbols for buzzing cryptos (incl. {len(trending_symbols)} trending)...")
        
        buzzing_opportunities = []
        
        if use_parallel:
            max_workers = 12  # INCREASED
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_crypto_pair_multi_source, symbol, 'MOMENTUM', aggregated_coins): symbol
                    for symbol in all_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        opportunity = future.result()
                        
                        # LOWERED: min_volume_ratio * 0.75 for more catches
                        if opportunity and opportunity.volume_ratio >= (min_volume_ratio * 0.75):
                            buzz_bonus = min((opportunity.volume_ratio - 1.0) * 10, 30)
                            opportunity.score += buzz_bonus
                            opportunity.reason += f" | 🔥 BUZZING (Vol: {opportunity.volume_ratio:.1f}x)"
                            buzzing_opportunities.append(opportunity)
                            
                    except Exception as e:
                        logger.debug(f"Error analyzing {symbol} for buzz: {e}")
                        continue
        else:
            for symbol in all_symbols:
                try:
                    opportunity = self._analyze_crypto_pair_multi_source(symbol, 'MOMENTUM', aggregated_coins)
                    
                    if opportunity and opportunity.volume_ratio >= min_volume_ratio:
                        buzz_bonus = min((opportunity.volume_ratio - 1.0) * 10, 30)
                        opportunity.score += buzz_bonus
                        opportunity.reason += f" | 🔥 BUZZING (Vol: {opportunity.volume_ratio:.1f}x)"
                        buzzing_opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol} for buzz: {e}")
                    continue
        
        # Sort by volume ratio (primary indicator of buzz)
        buzzing_opportunities.sort(key=lambda x: x.volume_ratio, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Found {len(buzzing_opportunities)} buzzing cryptos in {elapsed_time:.2f}s")
        
        return buzzing_opportunities[:top_n]
    
    def scan_hottest_cryptos(
        self,
        top_n: int = 10,
        min_momentum: float = 3.0,
        use_parallel: bool = True,
        use_multi_source: bool = True
    ) -> List[CryptoOpportunity]:
        """
        Scan for the hottest cryptocurrencies with strongest momentum
        Focus on price action and volatility
        
        Args:
            top_n: Number of hottest cryptos to return
            min_momentum: Minimum 24h change % (absolute value)
            use_parallel: Use parallel processing for 5-8x speedup (default: True)
            use_multi_source: Fetch from CoinGecko, CoinMarketCap, and Kraken (default: True)
            
        Returns:
            List of hottest CryptoOpportunity objects
        """
        logger.info(f"🌶️ Scanning for {top_n} hottest cryptocurrencies...")
        
        if use_multi_source:
            return self._scan_hottest_multi_source(top_n, min_momentum, use_parallel)
        
        hot_opportunities = []
        start_time = time.time()
        
        # Use parallel or sequential processing
        if use_parallel:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_crypto_pair, symbol, 'MOMENTUM'): symbol
                    for symbol in self.watchlist
                }
                
                for future in as_completed(future_to_symbol):
                    try:
                        opportunity = future.result()
                        
                        if opportunity and abs(opportunity.change_pct_24h) >= min_momentum:
                            momentum_bonus = min(abs(opportunity.change_pct_24h) * 2, 40)
                            opportunity.score += momentum_bonus
                            
                            direction = "🚀" if opportunity.change_pct_24h > 0 else "📉"
                            opportunity.reason += f" | 🌶️ HOTTEST {direction} ({opportunity.change_pct_24h:+.1f}%)"
                            hot_opportunities.append(opportunity)
                            
                    except Exception as e:
                        logger.error(f"Error analyzing for heat: {e}")
                        continue
        else:
            for symbol in self.watchlist:
                try:
                    opportunity = self._analyze_crypto_pair(symbol, 'MOMENTUM')
                    
                    if opportunity and abs(opportunity.change_pct_24h) >= min_momentum:
                        momentum_bonus = min(abs(opportunity.change_pct_24h) * 2, 40)
                        opportunity.score += momentum_bonus
                        
                        direction = "🚀" if opportunity.change_pct_24h > 0 else "📉"
                        opportunity.reason += f" | 🌶️ HOTTEST {direction} ({opportunity.change_pct_24h:+.1f}%)"
                        hot_opportunities.append(opportunity)
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} for heat: {e}")
                    continue
        
        # Sort by absolute momentum (primary indicator of heat)
        hot_opportunities.sort(key=lambda x: abs(x.change_pct_24h), reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Found {len(hot_opportunities)} hottest cryptos in {elapsed_time:.2f}s")
        
        return hot_opportunities[:top_n]
    
    def _scan_hottest_multi_source(
        self,
        top_n: int,
        min_momentum: float,
        use_parallel: bool = True
    ) -> List[CryptoOpportunity]:
        """
        Scan for hottest cryptos using multiple sources
        Fetches from CoinGecko, CoinMarketCap, Kraken + trending
        """
        logger.info(f"🔍 Scanning hottest cryptos from multiple sources + trending...")
        start_time = time.time()
        
        # Fetch trending coins first
        trending_symbols = self.fetch_trending_coins()
        
        # Fetch from CoinGecko and CoinMarketCap with LOWER thresholds
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            aggregated_coins = loop.run_until_complete(
                self.aggregator.fetch_all_coins(
                    min_volume_24h=15000,  # LOWERED: $15k volume (was $50k)
                    max_coins=500  # INCREASED: 500 coins (was 300)
                )
            )
            loop.close()
            
            logger.info(f"✅ Fetched {len(aggregated_coins)} coins from CoinGecko/CoinMarketCap")
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
            aggregated_coins = []
        
        # Create symbol list from aggregated coins
        symbols_from_sources = [f"{coin.symbol}/USD" for coin in aggregated_coins]
        
        # Combine watchlist + trending + aggregated (remove duplicates)
        all_symbols = list(set(symbols_from_sources + self.watchlist + trending_symbols))
        
        logger.info(f"📊 Analyzing {len(all_symbols)} total symbols for hottest cryptos (incl. {len(trending_symbols)} trending)...")
        
        hot_opportunities = []
        
        if use_parallel:
            max_workers = 12  # INCREASED
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_crypto_pair_multi_source, symbol, 'MOMENTUM', aggregated_coins): symbol
                    for symbol in all_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        opportunity = future.result()
                        
                        if opportunity and abs(opportunity.change_pct_24h) >= min_momentum:
                            momentum_bonus = min(abs(opportunity.change_pct_24h) * 2, 40)
                            opportunity.score += momentum_bonus
                            
                            direction = "🚀" if opportunity.change_pct_24h > 0 else "📉"
                            opportunity.reason += f" | 🌶️ HOTTEST {direction} ({opportunity.change_pct_24h:+.1f}%)"
                            hot_opportunities.append(opportunity)
                            
                    except Exception as e:
                        logger.debug(f"Error analyzing {symbol} for heat: {e}")
                        continue
        else:
            for symbol in all_symbols:
                try:
                    opportunity = self._analyze_crypto_pair_multi_source(symbol, 'MOMENTUM', aggregated_coins)
                    
                    if opportunity and abs(opportunity.change_pct_24h) >= min_momentum:
                        momentum_bonus = min(abs(opportunity.change_pct_24h) * 2, 40)
                        opportunity.score += momentum_bonus
                        
                        direction = "🚀" if opportunity.change_pct_24h > 0 else "📉"
                        opportunity.reason += f" | 🌶️ HOTTEST {direction} ({opportunity.change_pct_24h:+.1f}%)"
                        hot_opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol} for heat: {e}")
                    continue
        
        # Sort by absolute momentum (primary indicator of heat)
        hot_opportunities.sort(key=lambda x: abs(x.change_pct_24h), reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Found {len(hot_opportunities)} hottest cryptos in {elapsed_time:.2f}s")
        
        return hot_opportunities[:top_n]
    
    def scan_breakout_cryptos(
        self,
        top_n: int = 10,
        use_parallel: bool = True,
        use_multi_source: bool = True
    ) -> List[CryptoOpportunity]:
        """
        Scan for cryptocurrencies breaking out of consolidation
        Focus on price breaking above EMAs with volume confirmation
        
        Args:
            top_n: Number of breakout opportunities to return
            use_parallel: Use parallel processing for 5-8x speedup (default: True)
            use_multi_source: Fetch from CoinGecko, CoinMarketCap, and Kraken (default: True)
            
        Returns:
            List of breakout CryptoOpportunity objects
        """
        logger.info(f"💥 Scanning for {top_n} breakout cryptocurrencies...")
        
        if use_multi_source:
            return self._scan_breakout_multi_source(top_n, use_parallel)
        
        breakout_opportunities = []
        start_time = time.time()
        
        # Use parallel or sequential processing
        if use_parallel:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_breakout_single, symbol): symbol
                    for symbol in self.watchlist
                }
                
                for future in as_completed(future_to_symbol):
                    try:
                        opportunity = future.result()
                        if opportunity:
                            breakout_opportunities.append(opportunity)
                    except Exception as e:
                        logger.error(f"Error analyzing for breakout: {e}")
                        continue
        else:
            for symbol in self.watchlist:
                try:
                    opportunity = self._analyze_breakout_single(symbol)
                    if opportunity:
                        breakout_opportunities.append(opportunity)
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} for breakout: {e}")
                    continue
        
        # Sort by score
        breakout_opportunities.sort(key=lambda x: x.score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Found {len(breakout_opportunities)} breakout opportunities in {elapsed_time:.2f}s")
        
        return breakout_opportunities[:top_n]
    
    def _scan_breakout_multi_source(
        self,
        top_n: int,
        use_parallel: bool = True
    ) -> List[CryptoOpportunity]:
        """
        Scan for breakout cryptos using multiple sources
        Fetches from CoinGecko, CoinMarketCap, Kraken + trending coins
        Enhanced with lower thresholds for more opportunities
        """
        logger.info(f"🔍 Scanning breakout cryptos from multiple sources + trending...")
        start_time = time.time()
        
        # Fetch trending coins first
        trending_symbols = self.fetch_trending_coins()
        
        # Fetch from CoinGecko and CoinMarketCap with LOWER thresholds
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            aggregated_coins = loop.run_until_complete(
                self.aggregator.fetch_all_coins(
                    min_volume_24h=10000,  # LOWERED: $10k volume (was $50k)
                    max_coins=500  # INCREASED: up to 500 coins (was 300)
                )
            )
            loop.close()
            
            logger.info(f"✅ Fetched {len(aggregated_coins)} coins from CoinGecko/CoinMarketCap")
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
            aggregated_coins = []
        
        # Create symbol list from aggregated coins
        symbols_from_sources = [f"{coin.symbol}/USD" for coin in aggregated_coins]
        
        # Combine watchlist + trending + aggregated (remove duplicates)
        all_symbols = list(set(symbols_from_sources + self.watchlist + trending_symbols))
        
        logger.info(f"📊 Analyzing {len(all_symbols)} total symbols for breakout cryptos (incl. {len(trending_symbols)} trending)...")
        
        breakout_opportunities = []
        
        if use_parallel:
            max_workers = 12  # INCREASED: more workers for faster scanning
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_breakout_single, symbol): symbol
                    for symbol in all_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        opportunity = future.result()
                        if opportunity:
                            breakout_opportunities.append(opportunity)
                    except Exception as e:
                        logger.debug(f"Error analyzing {symbol} for breakout: {e}")
                        continue
        else:
            for symbol in all_symbols:
                try:
                    opportunity = self._analyze_breakout_single(symbol)
                    if opportunity:
                        breakout_opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol} for breakout: {e}")
                    continue
        
        # Sort by score
        breakout_opportunities.sort(key=lambda x: x.score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Found {len(breakout_opportunities)} breakout opportunities in {elapsed_time:.2f}s")
        
        return breakout_opportunities[:top_n]
    
    def _analyze_breakout_single(self, symbol: str) -> Optional[CryptoOpportunity]:
        """
        Analyze a single symbol for breakout patterns.
        Enhanced with MULTIPLE breakout detection patterns for more opportunities.
        """
        try:
            # Normalize pair format globally (handles BTC/USD, BTCUSD, btcusd, btc/usd)
            normalized_symbol = normalize_crypto_pair(symbol)
            
            # Get data for breakout analysis
            ticker = self.client.get_ticker_data(normalized_symbol)
            if not ticker:
                return None
            
            ohlc_1h = self.client.get_ohlc_data(normalized_symbol, interval=60)
            if not ohlc_1h or len(ohlc_1h) < 20:  # LOWERED: 20 candles (was 40)
                return None
            
            current_price = ticker['last_price']
            prices = [candle['close'] for candle in ohlc_1h]
            highs = [candle['high'] for candle in ohlc_1h]
            lows = [candle['low'] for candle in ohlc_1h]
            volumes = [candle['volume'] for candle in ohlc_1h]
            
            # Calculate EMAs
            ema_8 = self._calculate_ema(prices, 8)
            ema_20 = self._calculate_ema(prices, 20)
            ema_50 = self._calculate_ema(prices, min(50, len(prices))) if len(prices) >= 20 else ema_20
            
            # Calculate RSI
            rsi = self._calculate_rsi(prices[-14:]) if len(prices) >= 14 else 50.0
            
            volume_24h = ticker['volume_24h']
            avg_volume = sum(volumes[-24:]) / len(volumes[-24:]) if len(volumes) >= 24 else volume_24h
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
            
            change_pct_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100 if ticker['low_24h'] > 0 else 0
            volatility_24h = self._calculate_volatility(prices[-24:]) if len(prices) >= 24 else 0
            
            # Calculate ADX for trend strength (hyperopt showed 15-25 works well)
            adx = self._calculate_adx(highs, lows, prices) if len(highs) >= 15 else 25.0
            
            # ============================================
            # ENHANCED BREAKOUT DETECTION PATTERNS
            # (Hyperopt-optimized thresholds applied)
            # ============================================
            
            breakout_type = None
            score = 0
            reason_parts = []
            
            # ADX filter: Skip if trend is too weak (hyperopt adx_threshold ~15-25)
            # But don't filter out oversold bounces which work in choppy markets
            min_adx_for_trend = 15
            
            # Pattern 1: Classic EMA Breakout (hyperopt-optimized)
            # Using hyperopt values: volume_factor ~1.55, ema alignment
            if current_price > ema_8 > ema_20 and volume_ratio > 1.5 and current_price > ema_50 * 1.01:
                breakout_type = "EMA_BREAKOUT"
                score = 70
                reason_parts.append(f"EMA alignment (P>{ema_8:.4f}>{ema_20:.4f})")
            
            # Pattern 2: Volume Spike Breakout (hyperopt-tuned)
            # Lowered volume threshold slightly for more opportunities
            if volume_ratio > 2.0 and change_pct_24h > 3:
                if breakout_type:
                    score += 15
                else:
                    breakout_type = "VOLUME_SPIKE"
                    score = 65
                reason_parts.append(f"🔥 Vol spike {volume_ratio:.1f}x")
            
            # Pattern 3: Momentum Surge (hyperopt-tuned) - catching big moves early
            # RSI cap at 62 from hyperopt (buy_rsi_high) - don't chase overbought
            if change_pct_24h > 8 and rsi > 50 and rsi < 62:
                if breakout_type:
                    score += 10
                else:
                    breakout_type = "MOMENTUM_SURGE"
                    score = 68
                reason_parts.append(f"📈 Momentum +{change_pct_24h:.1f}%")
            
            # Pattern 4: Oversold Bounce (new) - catching reversals
            if rsi < 35 and current_price > ema_8 and volume_ratio > 1.2:
                if not breakout_type:
                    breakout_type = "OVERSOLD_BOUNCE"
                    score = 62
                reason_parts.append(f"📊 Oversold RSI:{rsi:.0f}")
            
            # Pattern 5: Resistance Break (hyperopt-tuned)
            # Volume threshold aligned with hyperopt volume_factor ~1.55
            recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            if current_price > recent_high * 0.98 and volume_ratio > 1.55:
                if breakout_type:
                    score += 12
                else:
                    breakout_type = "RESISTANCE_BREAK"
                    score = 72
                reason_parts.append(f"🚀 Breaking ${recent_high:.4f} resistance")
            
            # Pattern 6: Consolidation Breakout (new) - low volatility -> surge
            if len(prices) >= 48:
                recent_volatility = self._calculate_volatility(prices[-12:])
                prior_volatility = self._calculate_volatility(prices[-48:-12])
                if prior_volatility > 0 and recent_volatility > prior_volatility * 1.5 and change_pct_24h > 0:
                    if breakout_type:
                        score += 8
                    else:
                        breakout_type = "CONSOLIDATION_BREAK"
                        score = 66
                    reason_parts.append("⚡ Volatility expansion")
            
            # Pattern 7: Meme/Trending Coin Boost (new)
            base_asset = extract_base_asset(normalized_symbol)
            meme_coins = ['PEPE', 'DOGE', 'SHIB', 'WIF', 'BONK', 'FLOKI', 'POPCAT', 'MEW', 'BRETT', 'TURBO', 'NEIRO', 'MOG']
            if base_asset in meme_coins and volume_ratio > 1.5:
                if breakout_type:
                    score += 5
                else:
                    if change_pct_24h > 5:
                        breakout_type = "MEME_MOMENTUM"
                        score = 60
                        reason_parts.append("🐸 Meme momentum")
            
            # If no breakout pattern detected, return None
            if not breakout_type:
                return None
            
            # Add volume bonus
            if volume_ratio > 2.0:
                score += min((volume_ratio - 1.0) * 10, 15)
            
            # Add momentum bonus
            if change_pct_24h > 5:
                score += min(change_pct_24h * 0.5, 10)
            
            # ADX trend strength bonus/penalty (hyperopt-optimized)
            # Strong trend (ADX > 25) = bonus, weak trend (ADX < 15) = penalty
            if adx > 30:
                score += 8  # Strong trend bonus
                reason_parts.append(f"📈 Strong trend ADX:{adx:.0f}")
            elif adx > 20:
                score += 4  # Moderate trend bonus
            elif adx < 15 and breakout_type not in ['OVERSOLD_BOUNCE']:
                score -= 5  # Weak trend penalty (except for bounces)
            
            # Build reason string
            reason = f"💥 {breakout_type}: " + " | ".join(reason_parts) + f" | Vol: {volume_ratio:.1f}x | ADX:{adx:.0f}"
            
            opportunity = CryptoOpportunity(
                symbol=normalized_symbol,
                base_asset=base_asset,
                score=min(score, 100),  # Cap at 100
                current_price=current_price,
                change_pct_24h=change_pct_24h,
                volume_24h=volume_24h,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                reason=reason,
                strategy='momentum' if change_pct_24h > 5 else 'scalp',
                confidence='HIGH' if score >= 80 else 'MEDIUM' if score >= 65 else 'LOW',
                risk_level='HIGH' if volatility_24h > 10 else 'MEDIUM' if volatility_24h > 5 else 'LOW'
            )
            
            return opportunity
            
        except Exception as e:
            logger.debug(f"Error analyzing {symbol} for breakout: {e}")
            return None
    
    def get_scanner_stats(self) -> Dict:
        """Get scanner statistics and coverage info"""
        return {
            'total_coins_scanned': len(self.watchlist),
            'trending_coins': len(self.trending_coins),
            'categories': {
                'layer_1_blockchains': 14,
                'layer_2_scaling': 13,
                'defi_leaders': 14,
                'ai_data': 14,
                'rwa_tokenization': 5,
                'new_launches': 19,
                'gaming_metaverse': 14,
                'privacy_security': 4,
                'solana_ecosystem': 9,
                'cosmos_ecosystem': 5,
                'meme_coins': 19,
                'infrastructure': 10,
                'exchange_tokens': 5,
                'additional_emerging': 11
            },
            'breakout_patterns': [
                'EMA_BREAKOUT', 'VOLUME_SPIKE', 'MOMENTUM_SURGE', 
                'OVERSOLD_BOUNCE', 'RESISTANCE_BREAK', 'CONSOLIDATION_BREAK',
                'MEME_MOMENTUM'
            ],
            'scan_method': 'Multi-Pattern Technical Analysis + Volume + Momentum + Trending',
            'update_frequency': 'Real-time (Kraken API + CoinGecko Trending)',
            'strategies': ['SCALP', 'MOMENTUM', 'SWING', 'ALL'],
            'note': 'Enhanced crypto market coverage with 130+ coins + dynamic trending detection'
        }


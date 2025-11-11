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
                
                # Layer 2 & Scaling
                'MATIC/USD', 'ARB/USD', 'OP/USD', 'LINEA/USD', 'SCROLL/USD',
                'STX/USD', 'RSK/USD',
                
                # DeFi Leaders
                'LINK/USD', 'UNI/USD', 'AAVE/USD', 'CURVE/USD', 'LIDO/USD',
                'COMPOUND/USD', 'MAKER/USD', 'SNX/USD', 'DYDX/USD', 'GMX/USD',
                
                # Emerging & High Potential
                'RENDER/USD', 'FET/USD', 'AGIX/USD', 'OCEAN/USD', 'ARKM/USD',
                'JTO/USD', 'PYTH/USD', 'ONDO/USD', 'STRK/USD', 'BLUR/USD',
                
                # Gaming & Metaverse
                'GALA/USD', 'SAND/USD', 'MANA/USD', 'ENJ/USD', 'THETA/USD',
                'AXIE/USD', 'FLOW/USD', 'ILV/USD',
                
                # Privacy & Security
                'MONERO/USD', 'ZCASH/USD', 'DASH/USD',
                
                # Solana Ecosystem
                'MARINADE/USD', 'MAGIC/USD', 'COPE/USD',
                
                # Cosmos Ecosystem
                'OSMO/USD', 'JUNO/USD', 'STARS/USD', 'EVMOS/USD',
                
                # Meme Coins (high volatility)
                'SHIB/USD', 'DOGE/USD', 'PEPE/USD', 'FLOKI/USD', 'BONK/USD',
                
                # Additional Emerging
                'VET/USD', 'TRX/USD', 'ALGO/USD', 'HBAR/USD', 'PERP/USD', 'GNS/USD'
            ]
        
        logger.info(f"Crypto Scanner initialized with {len(self.watchlist)} pairs")
        logger.info("   • Multi-source data: CoinGecko, CoinMarketCap, Kraken")
    
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
        Fetches from CoinGecko, CoinMarketCap, and Kraken
        """
        logger.info(f"🔍 Scanning buzzing cryptos from multiple sources...")
        start_time = time.time()
        
        # Fetch from CoinGecko and CoinMarketCap
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            aggregated_coins = loop.run_until_complete(
                self.aggregator.fetch_all_coins(
                    min_volume_24h=100000,  # Minimum $100k volume for buzzing
                    max_coins=300  # Get up to 300 coins
                )
            )
            loop.close()
            
            logger.info(f"✅ Fetched {len(aggregated_coins)} coins from CoinGecko/CoinMarketCap")
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
            aggregated_coins = []
        
        # Create symbol list from aggregated coins
        symbols_from_sources = [f"{coin.symbol}/USD" for coin in aggregated_coins]
        
        # Combine with watchlist (remove duplicates)
        all_symbols = list(set(symbols_from_sources + self.watchlist))
        
        logger.info(f"📊 Analyzing {len(all_symbols)} total symbols for buzzing cryptos...")
        
        buzzing_opportunities = []
        
        if use_parallel:
            max_workers = 8
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_crypto_pair_multi_source, symbol, 'MOMENTUM', aggregated_coins): symbol
                    for symbol in all_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        opportunity = future.result()
                        
                        if opportunity and opportunity.volume_ratio >= min_volume_ratio:
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
        Fetches from CoinGecko, CoinMarketCap, and Kraken
        """
        logger.info(f"🔍 Scanning hottest cryptos from multiple sources...")
        start_time = time.time()
        
        # Fetch from CoinGecko and CoinMarketCap
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            aggregated_coins = loop.run_until_complete(
                self.aggregator.fetch_all_coins(
                    min_volume_24h=50000,  # Minimum $50k volume
                    max_coins=300  # Get up to 300 coins
                )
            )
            loop.close()
            
            logger.info(f"✅ Fetched {len(aggregated_coins)} coins from CoinGecko/CoinMarketCap")
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
            aggregated_coins = []
        
        # Create symbol list from aggregated coins
        symbols_from_sources = [f"{coin.symbol}/USD" for coin in aggregated_coins]
        
        # Combine with watchlist (remove duplicates)
        all_symbols = list(set(symbols_from_sources + self.watchlist))
        
        logger.info(f"📊 Analyzing {len(all_symbols)} total symbols for hottest cryptos...")
        
        hot_opportunities = []
        
        if use_parallel:
            max_workers = 8
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
        Fetches from CoinGecko, CoinMarketCap, and Kraken
        """
        logger.info(f"🔍 Scanning breakout cryptos from multiple sources...")
        start_time = time.time()
        
        # Fetch from CoinGecko and CoinMarketCap
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            aggregated_coins = loop.run_until_complete(
                self.aggregator.fetch_all_coins(
                    min_volume_24h=50000,  # Minimum $50k volume
                    max_coins=300  # Get up to 300 coins
                )
            )
            loop.close()
            
            logger.info(f"✅ Fetched {len(aggregated_coins)} coins from CoinGecko/CoinMarketCap")
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
            aggregated_coins = []
        
        # Create symbol list from aggregated coins
        symbols_from_sources = [f"{coin.symbol}/USD" for coin in aggregated_coins]
        
        # Combine with watchlist (remove duplicates)
        all_symbols = list(set(symbols_from_sources + self.watchlist))
        
        logger.info(f"📊 Analyzing {len(all_symbols)} total symbols for breakout cryptos...")
        
        breakout_opportunities = []
        
        if use_parallel:
            max_workers = 8
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
        """Analyze a single symbol for breakout pattern"""
        try:
            # Normalize pair format globally (handles BTC/USD, BTCUSD, btcusd, btc/usd)
            normalized_symbol = normalize_crypto_pair(symbol)
            
            # Get data for breakout analysis
            ticker = self.client.get_ticker_data(normalized_symbol)
            if not ticker:
                return None
            
            ohlc_1h = self.client.get_ohlc_data(normalized_symbol, interval=60)
            if not ohlc_1h or len(ohlc_1h) < 40:
                return None
            
            current_price = ticker['last_price']
            prices = [candle['close'] for candle in ohlc_1h]
            
            # Calculate EMAs
            ema_8 = self._calculate_ema(prices, 8)
            ema_20 = self._calculate_ema(prices, 20)
            ema_50 = self._calculate_ema(prices, 50) if len(prices) >= 50 else ema_20
            
            # Breakout conditions:
            # 1. Price > EMA8 > EMA20
            # 2. Volume surge
            # 3. Recent consolidation (low volatility before breakout)
            
            volume_24h = ticker['volume_24h']
            avg_volume = sum([candle['volume'] for candle in ohlc_1h[-24:]]) / 24 if len(ohlc_1h) >= 24 else volume_24h
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
            
            # Check for breakout pattern
            is_breakout = (
                current_price > ema_8 > ema_20 and  # Bullish alignment
                volume_ratio > 1.5 and  # Volume confirmation
                current_price > ema_50 * 1.02  # Breaking key resistance
            )
            
            if is_breakout:
                # Create opportunity
                change_pct_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
                volatility_24h = self._calculate_volatility(prices[-24:])
                
                # Calculate score
                score = 70  # Base breakout score
                score += min((volume_ratio - 1.0) * 15, 20)  # Volume bonus
                score += min(((current_price / ema_50) - 1.0) * 200, 10)  # Strength bonus
                
                # Parse base asset
                base_asset = extract_base_asset(normalized_symbol)
                
                opportunity = CryptoOpportunity(
                    symbol=normalized_symbol,
                    base_asset=base_asset,
                    score=score,
                    current_price=current_price,
                    change_pct_24h=change_pct_24h,
                    volume_24h=volume_24h,
                    volume_ratio=volume_ratio,
                    volatility_24h=volatility_24h,
                    reason=f"💥 BREAKOUT: Price > EMA8 > EMA20 | Vol: {volume_ratio:.1f}x | Above EMA50",
                    strategy='momentum',
                    confidence='HIGH' if score >= 85 else 'MEDIUM',
                    risk_level='MEDIUM'
                )
                
                return opportunity
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} for breakout: {e}")
            return None
    
    def get_scanner_stats(self) -> Dict:
        """Get scanner statistics and coverage info"""
        return {
            'total_coins_scanned': len(self.watchlist),
            'categories': {
                'layer_1_blockchains': 10,
                'layer_2_scaling': 7,
                'defi_leaders': 10,
                'emerging_high_potential': 10,
                'gaming_metaverse': 8,
                'privacy_security': 3,
                'solana_ecosystem': 3,
                'cosmos_ecosystem': 4,
                'meme_coins': 5,
                'additional_emerging': 6
            },
            'scan_method': 'Technical Analysis + Volume + Momentum',
            'update_frequency': 'Real-time (Kraken API)',
            'strategies': ['SCALP', 'MOMENTUM', 'SWING', 'ALL'],
            'note': 'Comprehensive crypto market coverage with 69+ coins across all major categories'
        }


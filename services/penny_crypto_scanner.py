"""
Penny Crypto Scanner - Find sub-$1 cryptocurrencies with monster runner potential
Scans for low-price cryptos including sub-penny (0.0000000+) coins
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
import hashlib
import asyncio


@dataclass
class PennyCryptoOpportunity:
    """Container for a penny crypto trading opportunity"""
    symbol: str  # e.g., 'SHIB/USD'
    base_asset: str  # e.g., 'SHIB'
    current_price: float
    price_decimals: int  # Number of decimal places (e.g., 8 for 0.00000001)
    change_pct_24h: float
    change_pct_7d: float
    volume_24h: float
    volume_ratio: float
    volatility_24h: float
    rsi: float
    momentum_score: float
    runner_potential_score: float  # 0-100 score for monster runner potential
    reason: str
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'
    entry_price: float
    target_1: float  # 50% target
    target_2: float  # 100% target
    target_3: float  # 200%+ target


class PennyCryptoScanner:
    """Scans cryptocurrency markets for sub-$1 coins with monster runner potential"""
    
    # Popular penny cryptos to scan - Kraken supported pairs only
    PENNY_WATCHLIST = [
        # Meme coins (high volatility runners) - Kraken supported
        'SHIB/USD', 'DOGE/USD', 'PEPE/USD', 'FLOKI/USD', 'BONK/USD', 'WIF/USD',
        'MEME/USD',
        
        # Layer 2 & Scaling - Kraken supported
        'MATIC/USD', 'ARB/USD', 'OP/USD',
        
        # Altcoins under $1 - Kraken supported
        'XRP/USD', 'ADA/USD', 'ATOM/USD', 'ALGO/USD', 'VET/USD', 'TRX/USD',
        'NEAR/USD', 'HBAR/USD',
        
        # Gaming & Metaverse - Kraken supported
        'GALA/USD', 'SAND/USD', 'MANA/USD', 'ENJ/USD', 'THETA/USD',
        
        # DeFi & Infrastructure - Kraken supported
        'LINK/USD', 'UNI/USD', 'AAVE/USD', 'LIDO/USD',
        
        # Emerging & Low-cap runners - Kraken supported
        'RENDER/USD', 'FET/USD', 'AGIX/USD', 'PYTH/USD', 'ONDO/USD', 'STRK/USD',
        
        # Privacy & Security - Kraken supported (Monero is XMR)
        'XMR/USD', 'ZEC/USD', 'DASH/USD',
        
        # Solana ecosystem - Kraken supported
        'SOL/USD',
        
        # Bitcoin layer 2 - Kraken supported
        'STX/USD',
        
        # Cosmos ecosystem - Kraken supported
        'OSMO/USD',
        
        # Additional emerging coins - Kraken supported
        'DYDX/USD', 'PERP/USD'
    ]
    
    def __init__(self, kraken_client: KrakenClient, config=None):
        """
        Initialize penny crypto scanner
        
        Args:
            kraken_client: KrakenClient instance
            config: Trading configuration
        """
        self.client = kraken_client
        self.config = config
        self.aggregator = CryptoDataAggregator()
        
        # Use custom watchlist from config or defaults
        if config and hasattr(config, 'PENNY_CRYPTO_WATCHLIST'):
            self.watchlist = config.PENNY_CRYPTO_WATCHLIST
        else:
            self.watchlist = self.PENNY_WATCHLIST
        
        logger.info(f"Penny Crypto Scanner initialized with {len(self.watchlist)} pairs")
        logger.info("   • Multi-source data: CoinGecko, CoinMarketCap, Kraken")
    
    def scan_penny_cryptos(
        self,
        max_price: float = 1.0,
        top_n: int = 10,
        min_runner_score: float = 60.0,
        use_parallel: bool = True,
        use_multi_source: bool = True
    ) -> List[PennyCryptoOpportunity]:
        """
        Scan for penny cryptos under $1 with monster runner potential
        
        Args:
            max_price: Maximum price filter (default: $1.00)
            top_n: Number of top opportunities to return
            min_runner_score: Minimum runner potential score
            use_parallel: Use parallel processing for 5-8x speedup (default: True)
            use_multi_source: Fetch from CoinGecko, CoinMarketCap, and Kraken (default: True)
            
        Returns:
            List of PennyCryptoOpportunity objects sorted by runner potential
        """
        if use_multi_source:
            return self._scan_multi_source(max_price, top_n, min_runner_score, use_parallel)
        elif use_parallel:
            return self._scan_parallel(max_price, top_n, min_runner_score)
        else:
            return self._scan_sequential(max_price, top_n, min_runner_score)
    
    def _scan_sequential(
        self,
        max_price: float,
        top_n: int,
        min_runner_score: float
    ) -> List[PennyCryptoOpportunity]:
        """Sequential scanning (fallback method)"""
        logger.info(f"Scanning {len(self.watchlist)} penny cryptos (sequential)...")
        
        opportunities = []
        
        for symbol in self.watchlist:
            try:
                opportunity = self._analyze_penny_crypto(symbol, max_price)
                
                if opportunity and opportunity.runner_potential_score >= min_runner_score:
                    opportunities.append(opportunity)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by runner potential score
        opportunities.sort(key=lambda x: x.runner_potential_score, reverse=True)
        
        logger.info(f"Found {len(opportunities)} penny cryptos (showing top {top_n})")
        
        return opportunities[:top_n]
    
    def _scan_multi_source(
        self,
        max_price: float,
        top_n: int,
        min_runner_score: float,
        use_parallel: bool = True
    ) -> List[PennyCryptoOpportunity]:
        """
        Scan using multiple sources (CoinGecko, CoinMarketCap, Kraken)
        Fetches top penny coins from all platforms and analyzes them
        """
        logger.info(f"🔍 Scanning penny cryptos from multiple sources (max_price=${max_price})...")
        start_time = time.time()
        
        # Fetch from CoinGecko and CoinMarketCap
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            aggregated_coins = loop.run_until_complete(
                self.aggregator.fetch_all_coins(
                    max_price=max_price,
                    min_volume_24h=10000,  # Minimum $10k volume
                    max_coins=500  # Get up to 500 penny coins
                )
            )
            loop.close()
            
            logger.info(f"✅ Fetched {len(aggregated_coins)} penny coins from CoinGecko/CoinMarketCap")
        except Exception as e:
            logger.error(f"Error fetching from aggregator: {e}")
            aggregated_coins = []
        
        # Create symbol list from aggregated coins
        symbols_from_sources = [f"{coin.symbol}/USD" for coin in aggregated_coins]
        
        # Combine with watchlist (remove duplicates)
        all_symbols = list(set(symbols_from_sources + self.watchlist))
        
        logger.info(f"📊 Analyzing {len(all_symbols)} total symbols (from watchlist + multi-source)...")
        
        opportunities = []
        
        if use_parallel:
            max_workers = 8
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_penny_crypto_multi_source, symbol, max_price, aggregated_coins): symbol
                    for symbol in all_symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        opportunity = future.result()
                        
                        if opportunity and opportunity.runner_potential_score >= min_runner_score:
                            opportunities.append(opportunity)
                            
                    except Exception as e:
                        logger.debug(f"Error analyzing {symbol}: {e}")
                        continue
        else:
            for symbol in all_symbols:
                try:
                    opportunity = self._analyze_penny_crypto_multi_source(symbol, max_price, aggregated_coins)
                    
                    if opportunity and opportunity.runner_potential_score >= min_runner_score:
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol}: {e}")
                    continue
        
        # Sort by runner potential score
        opportunities.sort(key=lambda x: x.runner_potential_score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Found {len(opportunities)} penny cryptos (showing top {top_n}) in {elapsed_time:.2f}s")
        
        return opportunities[:top_n]
    
    def _scan_parallel(
        self,
        max_price: float,
        top_n: int,
        min_runner_score: float,
        max_workers: int = 8
    ) -> List[PennyCryptoOpportunity]:
        """Parallel scanning using ThreadPoolExecutor (5-8x faster)"""
        logger.info(f"Scanning {len(self.watchlist)} penny cryptos (parallel, {max_workers} workers)...")
        
        opportunities = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._analyze_penny_crypto, symbol, max_price): symbol
                for symbol in self.watchlist
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    opportunity = future.result()
                    
                    if opportunity and opportunity.runner_potential_score >= min_runner_score:
                        opportunities.append(opportunity)
                        
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
        
        # Sort by runner potential score
        opportunities.sort(key=lambda x: x.runner_potential_score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Found {len(opportunities)} penny cryptos (showing top {top_n}) in {elapsed_time:.2f}s")
        
        return opportunities[:top_n]
    
    def _analyze_penny_crypto(
        self,
        symbol: str,
        max_price: float
    ) -> Optional[PennyCryptoOpportunity]:
        """
        Analyze a single penny crypto for monster runner potential
        
        Args:
            symbol: Crypto pair (e.g., 'SHIB/USD')
            max_price: Maximum price filter
            
        Returns:
            PennyCryptoOpportunity or None
        """
        try:
            # Normalize pair format globally (handles BTC/USD, BTCUSD, btcusd, btc/usd)
            normalized_symbol = normalize_crypto_pair(symbol)
            
            # Get market data
            ticker = self.client.get_ticker_data(normalized_symbol)
            
            if not ticker:
                return None
            
            current_price = ticker['last_price']
            
            # Filter by price
            if current_price >= max_price:
                return None
            
            # Get OHLC data for technical analysis
            ohlc_5m = self.client.get_ohlc_data(normalized_symbol, interval=5)  # 5-minute
            ohlc_1h = self.client.get_ohlc_data(normalized_symbol, interval=60)  # 1-hour
            
            if not ohlc_5m or not ohlc_1h:
                return None
            
            # Calculate metrics
            change_pct_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
            
            # Get 7-day change (estimate from 1h data if available)
            change_pct_7d = 0.0
            if len(ohlc_1h) >= 168:  # 7 days of hourly data
                week_ago_price = ohlc_1h[-168]['open']
                change_pct_7d = ((current_price - week_ago_price) / week_ago_price) * 100
            
            volume_24h = ticker['volume_24h']
            
            # Calculate volume ratio
            avg_volume = sum([candle['volume'] for candle in ohlc_1h[-24:]]) / 24 if len(ohlc_1h) >= 24 else volume_24h
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate volatility
            volatility_24h = self._calculate_volatility([candle['close'] for candle in ohlc_1h[-24:]])
            
            # Calculate technical indicators
            rsi = self._calculate_rsi([candle['close'] for candle in ohlc_5m[-14:]])
            momentum_score = self._calculate_momentum([candle['close'] for candle in ohlc_1h[-24:]])
            
            # Calculate price decimals (for sub-penny display)
            price_decimals = self._count_decimals(current_price)
            
            # Score runner potential
            runner_score, reason, confidence, risk_level = self._score_runner_potential(
                symbol=normalized_symbol,
                current_price=current_price,
                change_pct_24h=change_pct_24h,
                change_pct_7d=change_pct_7d,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                rsi=rsi,
                momentum_score=momentum_score
            )
            
            # Calculate targets (50%, 100%, 200%+)
            entry_price = current_price
            target_1 = entry_price * 1.5  # 50% gain
            target_2 = entry_price * 2.0  # 100% gain
            target_3 = entry_price * 3.0  # 200% gain
            
            # Parse base asset
            base_asset = extract_base_asset(normalized_symbol)
            
            # Create opportunity
            opportunity = PennyCryptoOpportunity(
                symbol=normalized_symbol,
                base_asset=base_asset,
                current_price=current_price,
                price_decimals=price_decimals,
                change_pct_24h=change_pct_24h,
                change_pct_7d=change_pct_7d,
                volume_24h=volume_24h,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                rsi=rsi,
                momentum_score=momentum_score,
                runner_potential_score=runner_score,
                reason=reason,
                confidence=confidence,
                risk_level=risk_level,
                entry_price=entry_price,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3
            )
            
            return opportunity
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _analyze_penny_crypto_multi_source(
        self,
        symbol: str,
        max_price: float,
        aggregated_coins: List[AggregatedCryptoData]
    ) -> Optional[PennyCryptoOpportunity]:
        """
        Analyze a single penny crypto using multiple sources
        Tries Kraken first, falls back to CoinGecko/CoinMarketCap data
        
        Args:
            symbol: Crypto pair (e.g., 'SHIB/USD')
            max_price: Maximum price filter
            aggregated_coins: List of AggregatedCryptoData from CoinGecko/CoinMarketCap
            
        Returns:
            PennyCryptoOpportunity or None
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
            
            # Filter by price
            if current_price >= max_price:
                return None
            
            # Calculate metrics from available data
            if ticker and ohlc_1h:
                # Use Kraken data (most accurate)
                change_pct_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100 if ticker['low_24h'] > 0 else 0
                change_pct_7d = 0.0
                if len(ohlc_1h) >= 168:
                    week_ago_price = ohlc_1h[-168]['open']
                    change_pct_7d = ((current_price - week_ago_price) / week_ago_price) * 100 if week_ago_price > 0 else 0
                
                volume_24h = ticker['volume_24h']
                
                # Calculate volume ratio
                avg_volume = sum([candle['volume'] for candle in ohlc_1h[-24:]]) / 24 if len(ohlc_1h) >= 24 else volume_24h
                volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1.0
                
                # Calculate volatility
                volatility_24h = self._calculate_volatility([candle['close'] for candle in ohlc_1h[-24:]])
                
                # Calculate technical indicators
                rsi = self._calculate_rsi([candle['close'] for candle in ohlc_5m[-14:]]) if ohlc_5m and len(ohlc_5m) >= 14 else 50.0
                momentum_score = self._calculate_momentum([candle['close'] for candle in ohlc_1h[-24:]]) if ohlc_1h and len(ohlc_1h) >= 24 else 50.0
            elif aggregated_data:
                # Use aggregated data (CoinGecko/CoinMarketCap)
                change_pct_24h = aggregated_data.change_24h
                change_pct_7d = aggregated_data.change_7d
                volume_24h = aggregated_data.volume_24h
                
                # Estimate volume ratio (use 1.0 as default)
                volume_ratio = 1.0
                
                # Estimate volatility from 24h change
                volatility_24h = abs(change_pct_24h) if change_pct_24h else 0.0
                
                # Default technical indicators (can't calculate without OHLC)
                rsi = 50.0
                momentum_score = 50.0 + (change_pct_24h / 2) if change_pct_24h else 50.0
                momentum_score = max(0, min(100, momentum_score))
            else:
                return None
            
            # Calculate price decimals (for sub-penny display)
            price_decimals = self._count_decimals(current_price)
            
            # Score runner potential - use normalized symbol
            runner_score, reason, confidence, risk_level = self._score_runner_potential(
                symbol=normalized_symbol,
                current_price=current_price,
                change_pct_24h=change_pct_24h,
                change_pct_7d=change_pct_7d,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                rsi=rsi,
                momentum_score=momentum_score
            )
            
            # Calculate targets (50%, 100%, 200%+)
            entry_price = current_price
            target_1 = entry_price * 1.5  # 50% gain
            target_2 = entry_price * 2.0  # 100% gain
            target_3 = entry_price * 3.0  # 200% gain
            
            # Create opportunity - use normalized symbol
            opportunity = PennyCryptoOpportunity(
                symbol=normalized_symbol,
                base_asset=base_asset,
                current_price=current_price,
                price_decimals=price_decimals,
                change_pct_24h=change_pct_24h,
                change_pct_7d=change_pct_7d,
                volume_24h=volume_24h,
                volume_ratio=volume_ratio,
                volatility_24h=volatility_24h,
                rsi=rsi,
                momentum_score=momentum_score,
                runner_potential_score=runner_score,
                reason=reason,
                confidence=confidence,
                risk_level=risk_level,
                entry_price=entry_price,
                target_1=target_1,
                target_2=target_2,
                target_3=target_3
            )
            
            return opportunity
            
        except Exception as e:
            logger.debug(f"Error analyzing {symbol} (multi-source): {e}")
            return None
    
    def _score_runner_potential(
        self,
        symbol: str,
        current_price: float,
        change_pct_24h: float,
        change_pct_7d: float,
        volume_ratio: float,
        volatility_24h: float,
        rsi: float,
        momentum_score: float
    ) -> Tuple[float, str, str, str]:
        """
        Score monster runner potential for penny cryptos
        
        Returns:
            (runner_score, reason, confidence, risk_level)
        """
        score = 0.0
        reasons = []
        
        # Price momentum (24h) - KEY for runners
        if abs(change_pct_24h) > 15:
            score += 30
            reasons.append(f"🚀 EXTREME 24h move ({change_pct_24h:+.1f}%)")
        elif abs(change_pct_24h) > 8:
            score += 25
            reasons.append(f"🔥 Strong 24h move ({change_pct_24h:+.1f}%)")
        elif abs(change_pct_24h) > 4:
            score += 15
            reasons.append(f"📈 Good momentum ({change_pct_24h:+.1f}%)")
        
        # 7-day trend
        if change_pct_7d > 20:
            score += 20
            reasons.append(f"📊 Strong 7d trend ({change_pct_7d:+.1f}%)")
        elif change_pct_7d > 10:
            score += 15
            reasons.append(f"📈 Positive 7d trend ({change_pct_7d:+.1f}%)")
        
        # Volume surge - CRITICAL for runners
        if volume_ratio > 3.0:
            score += 25
            reasons.append(f"💥 EXTREME volume surge ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 2.0:
            score += 20
            reasons.append(f"🔥 High volume ({volume_ratio:.1f}x avg)")
        elif volume_ratio > 1.5:
            score += 12
            reasons.append(f"📊 Above avg volume ({volume_ratio:.1f}x)")
        
        # Volatility - Higher volatility = more runner potential
        if volatility_24h > 15:
            score += 20
            reasons.append(f"⚡ EXTREME volatility ({volatility_24h:.1f}%)")
        elif volatility_24h > 8:
            score += 15
            reasons.append(f"🔥 High volatility ({volatility_24h:.1f}%)")
        elif volatility_24h > 4:
            score += 10
            reasons.append(f"📈 Good volatility ({volatility_24h:.1f}%)")
        
        # RSI - Oversold = potential bounce/runner
        if rsi < 20:
            score += 20
            reasons.append(f"🎯 EXTREME oversold RSI ({rsi:.0f})")
        elif rsi < 30:
            score += 15
            reasons.append(f"🎯 Oversold RSI ({rsi:.0f})")
        elif rsi > 70:
            score += 10
            reasons.append(f"⚠️ Overbought RSI ({rsi:.0f}) - potential pullback")
        
        # Momentum score
        if momentum_score > 70:
            score += 15
            reasons.append(f"🚀 EXTREME momentum ({momentum_score:.0f})")
        elif momentum_score > 50:
            score += 10
            reasons.append(f"📈 Strong momentum ({momentum_score:.0f})")
        
        # Determine confidence
        if score >= 85:
            confidence = 'HIGH'
        elif score >= 65:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Determine risk level (penny cryptos are inherently risky)
        if volatility_24h > 20 or volume_ratio > 4:
            risk_level = 'EXTREME'
        elif volatility_24h > 12 or volume_ratio > 2.5:
            risk_level = 'HIGH'
        elif volatility_24h > 6:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        reason = '; '.join(reasons) if reasons else 'No significant signals'
        
        return score, reason, confidence, risk_level
    
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
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate momentum score (0-100)"""
        if len(prices) < 2:
            return 50.0
        
        # Calculate rate of change
        price_change = prices[-1] - prices[0]
        price_range = max(prices) - min(prices)
        
        if price_range == 0:
            return 50.0
        
        momentum = (price_change / price_range) * 100
        # Normalize to 0-100
        momentum = max(0, min(100, momentum + 50))
        
        return momentum
    
    def _count_decimals(self, price: float) -> int:
        """Count number of decimal places in price"""
        if price == 0:
            return 0
        
        price_str = f"{price:.15f}".rstrip('0')
        if '.' in price_str:
            return len(price_str.split('.')[1])
        return 0
    
    def scan_sub_penny_cryptos(
        self,
        max_price: float = 0.01,
        top_n: int = 10,
        use_parallel: bool = True,
        use_multi_source: bool = True
    ) -> List[PennyCryptoOpportunity]:
        """
        Scan specifically for sub-penny cryptos (under $0.01)
        These have the highest runner potential
        
        Uses multi-source data from CoinGecko, CoinMarketCap, and Kraken
        (same as scan_penny_cryptos for consistency)
        
        Args:
            max_price: Maximum price filter (default: $0.01)
            top_n: Number of top opportunities to return
            use_parallel: Use parallel processing (default: True)
            use_multi_source: Fetch from CoinGecko, CoinMarketCap, and Kraken (default: True)
            
        Returns:
            List of sub-penny PennyCryptoOpportunity objects
        """
        if use_multi_source:
            # Use the same multi-source approach as scan_penny_cryptos
            return self._scan_multi_source(max_price, top_n, min_runner_score=50.0, use_parallel=use_parallel)
        
        logger.info(f"🔍 Scanning for sub-penny cryptos (under ${max_price})...")
        
        opportunities = []
        start_time = time.time()
        
        if use_parallel:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_symbol = {
                    executor.submit(self._analyze_penny_crypto, symbol, max_price): symbol
                    for symbol in self.watchlist
                }
                
                for future in as_completed(future_to_symbol):
                    try:
                        opportunity = future.result()
                        if opportunity:
                            opportunities.append(opportunity)
                    except Exception as e:
                        logger.error(f"Error analyzing: {e}")
                        continue
        else:
            for symbol in self.watchlist:
                try:
                    opportunity = self._analyze_penny_crypto(symbol, max_price)
                    if opportunity:
                        opportunities.append(opportunity)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
        
        # Sort by runner potential score
        opportunities.sort(key=lambda x: x.runner_potential_score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Found {len(opportunities)} sub-penny cryptos in {elapsed_time:.2f}s")
        
        return opportunities[:top_n]
    
    def get_scanner_stats(self) -> Dict:
        """Get scanner statistics and coverage info"""
        return {
            'total_coins_scanned': len(self.watchlist),
            'categories': {
                'meme_coins': 10,
                'layer_2': 5,
                'altcoins': 10,
                'gaming': 8,
                'defi': 8,
                'emerging': 10,
                'privacy': 3,
                'solana_ecosystem': 4,
                'bitcoin_layer2': 2,
                'cosmos': 4,
                'other': 5
            },
            'scan_method': 'Technical + Volume + Momentum Analysis',
            'update_frequency': 'Real-time (Kraken API)',
            'note': 'Scans established coins. For emerging runners, monitor volume surges and social sentiment.'
        }
    
    async def scan_trending_runners(self, top_n: int = 10) -> List[Dict]:
        """
        Scan for monster runners using CoinGecko trending + sentiment analysis
        
        This combines:
        1. CoinGecko trending API (emerging coins gaining attention)
        2. Social sentiment (Reddit + Twitter)
        3. Technical analysis (volume, momentum, volatility)
        
        Args:
            top_n: Number of trending coins to analyze
            
        Returns:
            List of dicts with trending + technical + sentiment analysis
        """
        try:
            from services.crypto_sentiment_analyzer import CryptoSentimentAnalyzer
            
            logger.info(f"🔍 Scanning for trending monster runners (top {top_n})...")
            
            sentiment_analyzer = CryptoSentimentAnalyzer()
            
            # Get trending coins with sentiment
            trending_with_sentiment = await sentiment_analyzer.get_trending_with_sentiment(top_n)
            
            results = []
            
            for trend_data in trending_with_sentiment:
                try:
                    symbol = trend_data['symbol']
                    
                    # Try to get technical analysis for this coin
                    # Convert symbol to Kraken format (e.g., BTC -> BTC/USD)
                    kraken_symbol = f"{symbol}/USD"
                    
                    technical = await self._get_technical_for_symbol(kraken_symbol)
                    
                    # Combine trending + sentiment + technical
                    result = {
                        'symbol': symbol,
                        'name': trend_data['name'],
                        'price_usd': trend_data['price_usd'],
                        'market_cap_rank': trend_data['market_cap_rank'],
                        'trending_score': trend_data['trending_score'],
                        'trending_sentiment': trend_data['trending_sentiment'],
                        'social_sentiment': trend_data['social_sentiment'],
                        'runner_potential': trend_data['runner_potential'],
                        'combined_score': trend_data['combined_score'],
                        'technical': technical,
                        'overall_runner_score': self._calculate_overall_runner_score(
                            trend_data['runner_potential']['runner_score'],
                            technical.get('runner_potential_score', 0) if technical else 0
                        )
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort by overall runner score
            results.sort(key=lambda x: x['overall_runner_score'], reverse=True)
            
            logger.info(f"✅ Found {len(results)} trending runners")
            return results
            
        except Exception as e:
            logger.error(f"Error scanning trending runners: {e}")
            return []
    
    async def _get_technical_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get technical analysis for a symbol"""
        try:
            opportunity = self._analyze_penny_crypto(symbol, max_price=1.0)
            
            if opportunity:
                return {
                    'current_price': opportunity.current_price,
                    'price_decimals': opportunity.price_decimals,
                    'change_24h': opportunity.change_pct_24h,
                    'change_7d': opportunity.change_pct_7d,
                    'volume_24h': opportunity.volume_24h,
                    'volume_ratio': opportunity.volume_ratio,
                    'volatility': opportunity.volatility_24h,
                    'rsi': opportunity.rsi,
                    'momentum_score': opportunity.momentum_score,
                    'runner_potential_score': opportunity.runner_potential_score,
                    'risk_level': opportunity.risk_level
                }
            return None
            
        except Exception as e:
            logger.debug(f"Technical analysis error for {symbol}: {e}")
            return None
    
    def _calculate_overall_runner_score(self, sentiment_score: float, technical_score: float) -> float:
        """
        Calculate overall runner score combining sentiment + technical
        
        Args:
            sentiment_score: Sentiment-based runner score (0-100)
            technical_score: Technical-based runner score (0-100)
            
        Returns:
            Combined runner score (0-100)
        """
        # Weight: 60% sentiment (trending is key), 40% technical (confirmation)
        return (sentiment_score * 0.6) + (technical_score * 0.4)

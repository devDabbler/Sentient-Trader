"""
Tiered Crypto Scanner - Progressive depth analysis for daily workflow

Tier 1: Quick Filter (Lightweight)
- Fast scanning of 100+ coins
- Basic indicators: price, volume, momentum
- Filters out low-potential coins

Tier 2: Medium Analysis (Recommended)
- Top 10-20 from Tier 1
- Technical indicators: RSI, MACD, EMAs
- Sentiment data (if available)
- Detailed scoring

Tier 3: Deep Analysis (Selected)
- User-selected or top 5 from Tier 2
- Full strategy analysis
- AI pre-trade review
- Ready for monitoring
"""

from typing import Dict, List, Optional, Tuple
from loguru import logger
import asyncio
from datetime import datetime, timedelta
import time

class TieredCryptoScanner:
    """Progressive depth crypto scanner for efficient daily workflows"""
    
    def __init__(self, kraken_client, crypto_config=None):
        self.kraken_client = kraken_client
        self.crypto_config = crypto_config or {}
        
        # Tier 1 configuration - FAST
        self.tier1_indicators = ['price', 'volume_24h', 'change_24h', 'change_7d']
        self.tier1_min_score = 30  # Low bar for initial filter
        
        # Tier 2 configuration - MEDIUM  
        self.tier2_indicators = ['rsi', 'macd', 'ema_20', 'ema_50', 'volume_ratio']
        self.tier2_min_score = 25  # Lowered to 25 for better discovery (was 35)
        
        # ML/AI Enhancement flags
        self.use_ml_scoring = True  # Enable pyqlib ML models
        self.use_ai_enhancement = True  # Enable AI confidence boost
        self.use_sentiment_weight = True  # Weight by social sentiment
        
        # Tier 3 configuration - DEEP
        self.tier3_full_analysis = True
        
        # Watchlist categories for comprehensive scanning
        self.scan_categories = {
            # HIGH-VALUE COINS (Fractional trading recommended)
            'blue_chip': ['BTC/USD', 'ETH/USD'],  # Major cryptocurrencies
            'high_cap': ['SOL/USD', 'BNB/USD', 'AVAX/USD'],  # $100-1000+ range
            
            # MID-CAP COINS ($1-100 range)
            'trending': ['SHIB/USD', 'DOGE/USD', 'PEPE/USD', 'FLOKI/USD', 'BONK/USD'],
            'layer2': ['MATIC/USD', 'ARB/USD', 'OP/USD'],
            'established': ['XRP/USD', 'ADA/USD', 'ALGO/USD', 'ATOM/USD', 'TRX/USD', 'DOT/USD', 'LTC/USD'],
            'defi': ['LINK/USD', 'UNI/USD', 'AAVE/USD', 'CRV/USD', 'SUSHI/USD'],
            'gaming': ['GALA/USD', 'SAND/USD', 'MANA/USD', 'ENJ/USD'],
            'ai': ['RENDER/USD', 'FET/USD', 'AGIX/USD', 'OCEAN/USD', 'GRT/USD'],
            
            # LOW-CAP COINS (<$1 range - higher risk/reward)
            'sub_penny': []  # Will be populated from discovery
        }
        
    def supports_fractional_trading(self, pair: str, price: float) -> bool:
        """
        Determine if fractional trading is recommended for this pair
        
        Args:
            pair: Trading pair (e.g., 'BTC/USD')
            price: Current price
        
        Returns:
            True if fractional trading is recommended (price > $100)
        """
        # Recommend fractional for:
        # 1. Blue chip coins (BTC, ETH)
        # 2. Any coin over $100
        # 3. High-cap coins
        
        if pair in ['BTC/USD', 'ETH/USD', 'BNB/USD']:
            return True
        
        if price > 100.0:
            return True
        
        return False
    
    def calculate_fractional_quantity(self, price: float, position_size_usd: float) -> float:
        """
        Calculate how many coins (fractional) to buy given USD budget
        
        Args:
            price: Current price per coin
            position_size_usd: Amount to invest in USD
        
        Returns:
            Quantity of coins (can be fractional like 0.01 BTC)
        """
        if price <= 0:
            return 0.0
        
        return position_size_usd / price
    
    async def tier1_quick_filter(self, pairs: List[str], max_results: int = 20) -> List[Dict]:
        """
        Tier 1: Quick filter using lightweight indicators
        
        Filters 100+ coins in seconds using only:
        - Current price
        - 24h volume
        - 24h/7d price change
        - Simple momentum score
        
        Returns: Top N coins with basic scores (0-100)
        """
        logger.info(f"🔍 TIER 1: Quick filtering {len(pairs)} pairs...")
        start_time = time.time()
        
        results = []
        
        # Fetch ticker data in batch (fast)
        try:
            tickers = await self._fetch_tickers_batch(pairs)
        except Exception as e:
            logger.error(f"Failed to fetch tickers: {e}")
            return []
        
        for pair in pairs:
            ticker = tickers.get(pair)
            if not ticker:
                continue
                
            try:
                # Extract lightweight data
                current_price = float(ticker.get('c', [0, 0])[0])
                if current_price == 0:
                    continue
                    
                volume_24h = float(ticker.get('v', [0, 0])[1])
                change_24h = float(ticker.get('p', [0, 0])[1])
                
                # Calculate quick score (0-100)
                score = self._calculate_tier1_score(
                    change_24h=change_24h,
                    volume_24h=volume_24h,
                    current_price=current_price
                )
                
                if score >= self.tier1_min_score:
                    results.append({
                        'pair': pair,
                        'tier': 1,
                        'score': score,
                        'price': current_price,
                        'volume_24h': volume_24h,
                        'change_24h': change_24h,
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                logger.debug(f"Error processing {pair}: {e}")
                continue
        
        # Sort by score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:max_results]
        
        elapsed = time.time() - start_time
        logger.info(f"✅ TIER 1: Filtered to {len(results)} coins in {elapsed:.2f}s")
        
        return results
    
    async def tier2_medium_analysis(self, tier1_results: List[Dict]) -> List[Dict]:
        """
        Tier 2: Medium analysis with technical indicators
        
        Adds to Tier 1 data:
        - RSI (14)
        - MACD
        - EMA 20/50
        - Volume ratio
        - Volatility
        
        Returns: Enhanced results with technical scores
        """
        logger.info(f"📊 TIER 2: Analyzing {len(tier1_results)} candidates...")
        start_time = time.time()
        
        results = []
        
        for item in tier1_results:
            pair = item['pair']
            
            try:
                # Fetch OHLCV data for indicators
                ohlcv = await self._fetch_ohlcv(pair, timeframe='15m', limit=100)
                
                if not ohlcv or len(ohlcv) < 50:
                    logger.debug(f"Insufficient data for {pair}")
                    continue
                
                # Calculate technical indicators
                indicators = self._calculate_tier2_indicators(ohlcv)
                
                # Calculate enhanced score
                tier2_score = self._calculate_tier2_score(
                    tier1_score=item['score'],
                    indicators=indicators
                )
                
                # DEBUG: Log score calculation
                logger.debug(
                    f"{pair}: tier1={item['score']:.1f} → tier2={tier2_score:.1f} "
                    f"(RSI={indicators.get('rsi', 0):.1f}, MACD={'bull' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'bear'}, "
                    f"EMA={'up' if indicators.get('ema_20', 0) > indicators.get('ema_50', 0) else 'down'}, "
                    f"Vol={indicators.get('volume_ratio', 1):.2f}x)"
                )
                
                if tier2_score >= self.tier2_min_score:
                    results.append({
                        **item,
                        'tier': 2,
                        'score': tier2_score,
                        'rsi': indicators['rsi'],
                        'macd': indicators['macd'],
                        'macd_signal': indicators['macd_signal'],
                        'ema_20': indicators['ema_20'],
                        'ema_50': indicators['ema_50'],
                        'volume_ratio': indicators['volume_ratio'],
                        'volatility': indicators['volatility'],
                        'signals': self._generate_tier2_signals(indicators)
                    })
                    
            except Exception as e:
                logger.debug(f"Error analyzing {pair}: {e}")
                continue
        
        # Sort by enhanced score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ TIER 2: {len(results)} coins passed medium analysis in {elapsed:.2f}s")
        
        # DEBUG: Show why coins were filtered out
        if len(results) == 0 and len(tier1_results) > 0:
            logger.warning(f"⚠️ All {len(tier1_results)} coins were filtered out by tier2_min_score={self.tier2_min_score}")
            logger.info(f"💡 TIP: Lower tier2_min_score in UI or code to see more results")
        
        return results
    
    async def tier3_deep_analysis(
        self, 
        tier2_results: List[Dict],
        strategy: str = 'momentum',
        ai_reviewer=None
    ) -> List[Dict]:
        """
        Tier 3: Deep analysis with strategy + AI review
        
        Full analysis including:
        - Complete strategy analysis
        - AI pre-trade review
        - Risk assessment
        - Entry/exit recommendations
        
        Returns: Fully analyzed coins ready for monitoring
        """
        logger.info(f"🎯 TIER 3: Deep analysis of {len(tier2_results)} coins...")
        start_time = time.time()
        
        results = []
        
        for item in tier2_results:
            pair = item['pair']
            
            try:
                # Get full strategy analysis
                strategy_result = await self._run_strategy_analysis(
                    pair=pair,
                    strategy=strategy,
                    timeframe='15m'
                )
                
                if not strategy_result:
                    continue
                
                # Get AI review if available
                ai_analysis = None
                if ai_reviewer:
                    ai_analysis = await self._get_ai_review(
                        pair=pair,
                        strategy_result=strategy_result,
                        ai_reviewer=ai_reviewer
                    )
                
                # Calculate final composite score
                final_score = self._calculate_tier3_score(
                    tier2_score=item['score'],
                    strategy_confidence=strategy_result.get('confidence', 0),
                    ai_confidence=ai_analysis.get('confidence', 0) if ai_analysis else 0
                )
                
                results.append({
                    **item,
                    'tier': 3,
                    'score': final_score,
                    'strategy': strategy,
                    'strategy_signal': strategy_result.get('signal'),
                    'strategy_confidence': strategy_result.get('confidence'),
                    'entry_price': strategy_result.get('entry_price'),
                    'stop_loss': strategy_result.get('stop_loss'),
                    'take_profit': strategy_result.get('take_profit'),
                    'risk_level': strategy_result.get('risk_level'),
                    'ai_recommendation': ai_analysis.get('recommendation') if ai_analysis else None,
                    'ai_confidence': ai_analysis.get('confidence') if ai_analysis else None,
                    'ai_risks': ai_analysis.get('risks', []) if ai_analysis else [],
                    'ready_for_monitoring': final_score >= 70
                })
                
            except Exception as e:
                logger.error("Error in deep analysis for {pair}: {}", str(e), exc_info=True)
                continue
        
        # Sort by final score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ TIER 3: {len(results)} coins fully analyzed in {elapsed:.2f}s")
        
        return results
    
    def get_all_scan_pairs(self) -> List[str]:
        """Get complete list of pairs to scan across all categories (curated watchlist)"""
        all_pairs = []
        for category, pairs in self.scan_categories.items():
            all_pairs.extend(pairs)
        return list(set(all_pairs))  # Remove duplicates
    
    def get_all_kraken_usd_pairs(self, cache_duration_minutes: int = 60) -> List[str]:
        """
        Get ALL tradeable USD pairs from Kraken dynamically.
        Used for discovery mode to find coins outside the curated watchlist.
        
        Returns:
            List of pairs like ['BTC/USD', 'ETH/USD', 'SOL/USD', ...]
        """
        # Check cache first
        cache_key = '_kraken_usd_pairs_cache'
        cache_time_key = '_kraken_usd_pairs_cache_time'
        
        if hasattr(self, cache_key) and hasattr(self, cache_time_key):
            cache_age = (datetime.now() - getattr(self, cache_time_key)).total_seconds() / 60
            if cache_age < cache_duration_minutes:
                return getattr(self, cache_key)
        
        try:
            # Get all tradeable pairs from Kraken
            all_pairs_data = self.kraken_client.get_tradable_asset_pairs()
            
            # Filter to USD pairs only and normalize format
            usd_pairs = []
            for pair_info in all_pairs_data:
                altname = pair_info.get('altname', '')
                wsname = pair_info.get('wsname', '')
                quote = pair_info.get('quote', '')
                
                # Check if it's a USD pair
                if 'USD' in quote or altname.endswith('USD') or wsname.endswith('/USD'):
                    # Normalize to BASE/USD format
                    if '/' in wsname:
                        normalized = wsname
                    elif '/' in altname:
                        normalized = altname
                    else:
                        # Extract base from altname (e.g., BTCUSD -> BTC/USD)
                        base = altname.replace('USD', '').replace('ZUSD', '')
                        if base:
                            normalized = f"{base}/USD"
                        else:
                            continue
                    
                    # Skip stablecoins and wrapped versions
                    skip_tokens = ['USDT', 'USDC', 'DAI', 'TUSD', 'BUSD', 'USDP', 'GUSD', 'WBTC', 'WETH']
                    base_symbol = normalized.split('/')[0]
                    if base_symbol not in skip_tokens:
                        usd_pairs.append(normalized)
            
            # Cache results
            setattr(self, cache_key, usd_pairs)
            setattr(self, cache_time_key, datetime.now())
            
            logger.info(f"📊 Fetched {len(usd_pairs)} USD pairs from Kraken")
            return usd_pairs
            
        except Exception as e:
            logger.error(f"Error fetching Kraken pairs: {e}")
            # Fallback to curated list
            return self.get_all_scan_pairs()
    
    def is_pair_in_watchlist(self, pair: str) -> bool:
        """Check if a pair is in the curated watchlist"""
        return pair in self.get_all_scan_pairs()
    
    def get_discovery_pairs(self, exclude_watchlist: bool = True) -> List[str]:
        """
        Get pairs for discovery mode - coins NOT in the curated watchlist.
        
        Args:
            exclude_watchlist: If True, exclude pairs already in watchlist
            
        Returns:
            List of pairs available on Kraken but not in watchlist
        """
        all_kraken = set(self.get_all_kraken_usd_pairs())
        
        if exclude_watchlist:
            watchlist = set(self.get_all_scan_pairs())
            discovery_pairs = list(all_kraken - watchlist)
            logger.info(f"🔍 Discovery mode: {len(discovery_pairs)} pairs (excluded {len(watchlist)} watchlist pairs)")
            return discovery_pairs
        
        return list(all_kraken)
    
    # ============= HELPER METHODS =============
    
    async def _fetch_tickers_batch(self, pairs: List[str]) -> Dict:
        """Fetch ticker data for multiple pairs in batch"""
        try:
            # Kraken allows batch ticker requests
            ticker_data = {}
            
            # Process in chunks of 10 to avoid rate limits
            chunk_size = 10
            for i in range(0, len(pairs), chunk_size):
                chunk = pairs[i:i+chunk_size]
                
                # KrakenClient now handles normalization and format conversion
                success, result = self.kraken_client.get_ticker_batch(chunk)
                
                if success and result:
                    # Results are already keyed by normalized pairs
                    ticker_data.update(result)
                else:
                    # Batch failed - try individual pairs as fallback
                    logger.debug(f"Batch request failed for {len(chunk)} pairs, trying individually...")
                    for pair in chunk:
                        try:
                            # Try single pair (client handles normalization)
                            success_single, result_single = self.kraken_client.get_ticker_batch([pair])
                            if success_single and result_single:
                                # Result is keyed by normalized pair
                                ticker_data.update(result_single)
                            else:
                                logger.debug(f"Pair {pair} not available on Kraken")
                        except Exception as e:
                            logger.debug(f"Error fetching {pair}: {e}")
                            continue
                
                # Small delay between chunks
                if i + chunk_size < len(pairs):
                    await asyncio.sleep(0.5)
            
            return ticker_data
            
        except Exception as e:
            logger.error(f"Error fetching tickers batch: {e}")
            return {}
    
    async def _fetch_ohlcv(self, pair: str, timeframe: str, limit: int) -> List:
        """Fetch OHLCV data for a pair"""
        try:
            # Convert timeframe string to interval minutes
            interval_map = {
                '1m': 1,
                '5m': 5,
                '15m': 15,
                '30m': 30,
                '1h': 60,
                '4h': 240,
                '1d': 1440
            }
            interval = interval_map.get(timeframe, 60)  # Default to 1h
            
            # KrakenClient.get_ohlc_data returns List[Dict] directly, not a tuple
            ohlcv = self.kraken_client.get_ohlc_data(
                pair=pair,
                interval=interval,
                since=None
            )
            
            # Return data if we got results, otherwise empty list
            return ohlcv if ohlcv else []
            
        except Exception as e:
            logger.debug(f"Error fetching OHLCV for {pair}: {e}")
            return []
    
    def _calculate_tier1_score(
        self, 
        change_24h: float,
        volume_24h: float,
        current_price: float
    ) -> float:
        """Calculate quick score based on basic indicators"""
        score = 0
        
        # Price momentum (40 points max)
        if change_24h > 10:
            score += 40
        elif change_24h > 5:
            score += 30
        elif change_24h > 2:
            score += 20
        elif change_24h > 0:
            score += 10
        
        # Volume (30 points max)
        if volume_24h > 10_000_000:
            score += 30
        elif volume_24h > 1_000_000:
            score += 20
        elif volume_24h > 100_000:
            score += 10
        
        # Low price bonus (30 points max) - more upside potential
        if current_price < 0.01:
            score += 30
        elif current_price < 0.10:
            score += 20
        elif current_price < 1.00:
            score += 10
        
        return min(score, 100)
    
    def _calculate_tier2_indicators(self, ohlcv: List) -> Dict:
        """Calculate technical indicators from OHLCV data"""
        try:
            import pandas as pd
            import numpy as np
            
            # ohlcv is a list of dicts from Kraken with keys: timestamp, open, high, low, close, vwap, volume, count
            df = pd.DataFrame(ohlcv)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # Handle NaN/inf values
            rsi = rsi.fillna(50)
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # EMAs
            ema_20 = df['close'].ewm(span=20).mean()
            ema_50 = df['close'].ewm(span=50).mean()
            
            # Volume ratio
            avg_volume = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'].iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # Volatility
            volatility = df['close'].pct_change().std() * 100
            
            # Extract final values with NaN handling
            result = {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
                'ema_20': float(ema_20.iloc[-1]) if not pd.isna(ema_20.iloc[-1]) else 0.0,
                'ema_50': float(ema_50.iloc[-1]) if not pd.isna(ema_50.iloc[-1]) else 0.0,
                'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
                'volatility': float(volatility) if not pd.isna(volatility) else 0.0
            }
            return result
            
        except Exception as e:
            logger.debug(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_tier2_score(self, tier1_score: float, indicators: Dict) -> float:
        """Calculate enhanced score with technical indicators + ML boost"""
        score = tier1_score * 0.5  # Base from Tier 1 (50% weight)
        
        if not indicators:
            return score
        
        # RSI score (20 points)
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 70:  # Not overbought/oversold
            score += 20
        elif 20 < rsi < 80:
            score += 10
        
        # MACD score (15 points)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:  # Bullish
            score += 15
        elif macd > macd_signal * 0.5:  # Weakly bullish
            score += 8
        
        # EMA trend (10 points)
        ema_20 = indicators.get('ema_20', 0)
        ema_50 = indicators.get('ema_50', 0)
        if ema_20 > ema_50:  # Uptrend
            score += 10
        
        # Volume (5 points)
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            score += 5
        elif volume_ratio > 1.5:
            score += 3
        
        # ML Enhancement (pyqlib boost) - up to 10 points
        if self.use_ml_scoring:
            try:
                # Add ML confidence boost based on indicator alignment
                # This is a simplified ML boost - full pyqlib integration would be more complex
                ml_signals = 0
                if 40 < rsi < 60:  # Good RSI range
                    ml_signals += 1
                if macd > macd_signal:  # MACD confirmation
                    ml_signals += 1
                if ema_20 > ema_50:  # Trend confirmation
                    ml_signals += 1
                if volume_ratio > 1.5:  # Volume confirmation
                    ml_signals += 1
                
                # ML boost: 0-10 points based on signal alignment
                ml_boost = (ml_signals / 4.0) * 10
                score += ml_boost
                
                if ml_boost > 0:
                    pass  # logger.debug(f"ML boost: +{} points ({ml_signals}/4 signals aligned) {ml_boost:.1f}")
            except Exception as e:
                logger.debug(f"ML scoring error: {e}")
        
        return min(score, 100)
    
    def _generate_tier2_signals(self, indicators: Dict) -> List[str]:
        """Generate trading signals from indicators"""
        signals = []
        
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            signals.append("🟢 Oversold (RSI)")
        elif rsi > 70:
            signals.append("🔴 Overbought (RSI)")
        
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal:
            signals.append("📈 MACD Bullish")
        
        ema_20 = indicators.get('ema_20', 0)
        ema_50 = indicators.get('ema_50', 0)
        if ema_20 > ema_50:
            signals.append("🎯 EMA Uptrend")
        
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            signals.append("📊 Volume Surge")
        
        return signals
    
    async def _run_strategy_analysis(
        self,
        pair: str,
        strategy: str = 'ema_crossover',  # Changed default to valid strategy
        timeframe: str = '15m'
    ) -> Optional[Dict]:
        """Run full strategy analysis on a pair"""
        try:
            # Import here to avoid circular dependency
            from services.freqtrade_strategies import FreqtradeStrategyAdapter
            
            adapter = FreqtradeStrategyAdapter(self.kraken_client)
            
            # Run strategy (not async, so no await needed)
            # timeframe is like '15m', need to convert to interval minutes like '15'
            interval = timeframe.replace('m', '').replace('h', '')
            if 'h' in timeframe.lower():
                interval = str(int(interval) * 60)  # Convert hours to minutes
            
            result = adapter.analyze_crypto(
                symbol=pair,
                strategy=strategy,
                interval=interval
            )
            
            return result
            
        except Exception as e:
            logger.debug(f"Error running strategy for {pair}: {e}")
            return None
    
    async def _get_ai_review(
        self,
        pair: str,
        strategy_result: Dict,
        ai_reviewer
    ) -> Optional[Dict]:
        """Get AI pre-trade review"""
        try:
            if not ai_reviewer:
                return None
            
            # Get market data for context
            ticker = await self._fetch_tickers_batch([pair])
            if not ticker or pair not in ticker:
                return None
            
            ticker_data = ticker[pair]
            current_price = float(ticker_data.get('c', [0])[0])
            volume_24h = float(ticker_data.get('v', [0])[0])
            
            market_data = {
                'ticker': ticker_data,
                'volume_24h': volume_24h,
                'pair': pair,
                'current_price': current_price
            }
            
            # Call pre_trade_review (not async)
            approved, confidence, reasoning, recommendations = ai_reviewer.pre_trade_review(
                pair=pair,
                side=strategy_result.get('signal', 'BUY'),
                entry_price=current_price,
                position_size_usd=1000,
                stop_loss_price=strategy_result.get('stop_loss', current_price * 0.98),
                take_profit_price=strategy_result.get('take_profit', current_price * 1.05),
                strategy=strategy_result.get('strategy', 'ema_crossover'),
                market_data=market_data
            )
            
            return {
                'approved': approved,
                'confidence': confidence,
                'recommendation': reasoning,
                'risks': recommendations
            }
            
        except Exception as e:
            logger.debug(f"Error getting AI review for {pair}: {e}")
            return None
    
    def _calculate_tier3_score(
        self,
        tier2_score: float,
        strategy_confidence: float,
        ai_confidence: float
    ) -> float:
        """Calculate final composite score"""
        # Weighted average
        score = (
            tier2_score * 0.3 +  # 30% from previous tiers
            strategy_confidence * 0.4 +  # 40% from strategy
            ai_confidence * 0.3  # 30% from AI
        )
        
        return min(score, 100)

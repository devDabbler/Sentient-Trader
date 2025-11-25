"""
Tiered Stock/Options Scanner - Progressive depth analysis for daily workflow

Similar to the crypto tiered scanner but for stocks/options.

Tier 1: Quick Filter (Lightweight)
- Fast scanning of 100+ stocks
- Basic indicators: price, volume, momentum
- Filters out low-potential stocks

Tier 2: Medium Analysis (Technical)
- Top 10-20 from Tier 1
- Technical indicators: RSI, MACD, EMAs
- Volume analysis, volatility
- Detailed scoring

Tier 3: Deep Analysis (AI/ML)
- User-selected or top 5 from Tier 2
- Full strategy analysis
- AI pre-trade review
- Options analysis
- Ready for monitoring
"""

from typing import Dict, List, Optional, Tuple
from loguru import logger
import asyncio
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
import numpy as np


class TieredStockScanner:
    """Progressive depth stock/options scanner for efficient daily workflows"""
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai
        
        # Tier 1 configuration - FAST
        self.tier1_indicators = ['price', 'volume', 'change_pct', 'volume_ratio']
        self.tier1_min_score = 30  # Low bar for initial filter
        
        # Tier 2 configuration - MEDIUM  
        self.tier2_indicators = ['rsi', 'macd', 'ema_20', 'ema_50', 'volume_ratio', 'atr']
        self.tier2_min_score = 25  # Lowered for better discovery
        
        # AI/ML Enhancement flags
        self.use_ml_scoring = True
        self.use_ai_enhancement = use_ai
        
        # Tier 3 configuration - DEEP
        self.tier3_full_analysis = True
        
        # Stock categories for comprehensive scanning
        self.scan_categories = {
            # Mega caps (options-friendly)
            'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX'],
            
            # High-beta tech
            'high_beta_tech': ['PLTR', 'SOFI', 'HOOD', 'COIN', 'RBLX', 'SNAP', 'SHOP', 'NET', 'CRWD'],
            
            # Momentum/Meme stocks
            'momentum': ['GME', 'AMC', 'BBBY', 'BB', 'NOK'],
            
            # EV/Clean energy
            'ev_energy': ['NIO', 'LCID', 'RIVN', 'PLUG', 'FCEL', 'CHPT', 'ENPH', 'SEDG'],
            
            # Crypto-related
            'crypto_related': ['MARA', 'RIOT', 'COIN', 'MSTR', 'HUT', 'CLSK'],
            
            # AI stocks
            'ai_stocks': ['NVDA', 'AMD', 'PLTR', 'AI', 'SOUN', 'BBAI', 'PATH'],
            
            # Biotech (catalyst-driven)
            'biotech': ['MRNA', 'BNTX', 'NVAX', 'VKTX', 'ALNY', 'CRSP'],
            
            # Financial
            'financial': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'PYPL'],
            
            # Energy
            'energy': ['XOM', 'CVX', 'OXY', 'SLB', 'COP'],
            
            # ETFs for reference
            'etfs': ['SPY', 'QQQ', 'IWM', 'ARKK', 'XLF', 'XLE', 'XLK'],
            
            # Options-friendly high IV
            'high_iv': ['TSLA', 'NVDA', 'AMD', 'GME', 'AMC', 'COIN', 'MARA'],
            
            # Penny stocks (high risk/reward)
            'penny_stocks': ['SNDL', 'GSAT', 'SENS', 'SIRI'],
        }
        
        # Cache for data
        self._ticker_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    def get_all_scan_tickers(self) -> List[str]:
        """Get complete list of tickers to scan across all categories"""
        all_tickers = []
        for category, tickers in self.scan_categories.items():
            all_tickers.extend(tickers)
        return list(set(all_tickers))  # Remove duplicates
    
    def get_tickers_by_category(self, category: str) -> List[str]:
        """Get tickers for a specific category"""
        return self.scan_categories.get(category, [])
    
    def tier1_quick_filter(self, tickers: List[str], max_results: int = 20) -> List[Dict]:
        """
        Tier 1: Quick filter using lightweight indicators
        
        Filters 100+ stocks in seconds using only:
        - Current price
        - Volume and volume ratio
        - Price change %
        - Simple momentum score
        
        Returns: Top N stocks with basic scores (0-100)
        """
        logger.info(f"🔍 TIER 1: Quick filtering {len(tickers)} stocks...")
        start_time = time.time()
        
        results = []
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_quick_data, ticker): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                original_input = future_to_ticker[future]
                try:
                    data = future.result(timeout=10)
                    if data and data.get('price', 0) > 0:
                        # Calculate quick score
                        score = self._calculate_tier1_score(data)
                        
                        if score >= self.tier1_min_score:
                            results.append({
                                'ticker': data['ticker'],  # Use ticker from data (always a string)
                                'tier': 1,
                                'score': score,
                                'price': data['price'],
                                'change_pct': data['change_pct'],
                                'volume': data['volume'],
                                'volume_ratio': data['volume_ratio'],
                                'market_cap': data.get('market_cap'),
                                'sector': data.get('sector'),
                                'timestamp': datetime.now().isoformat()
                            })
                except Exception as e:
                    # Extract ticker for error logging
                    ticker_str = original_input.get('ticker', str(original_input)) if isinstance(original_input, dict) else original_input
                    logger.debug(f"Error processing {ticker_str}: {e}")
                    continue
        
        # Sort by score and limit results
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:max_results]
        
        elapsed = time.time() - start_time
        logger.info(f"✅ TIER 1: Filtered to {len(results)} stocks in {elapsed:.2f}s")
        
        return results
    
    def tier2_medium_analysis(self, tier1_results: List[Dict]) -> List[Dict]:
        """
        Tier 2: Medium analysis with technical indicators
        
        Adds to Tier 1 data:
        - RSI (14)
        - MACD
        - EMA 20/50
        - Volume analysis
        - ATR volatility
        
        Returns: Enhanced results with technical scores
        """
        logger.info(f"📊 TIER 2: Analyzing {len(tier1_results)} candidates...")
        start_time = time.time()
        
        results = []
        
        for item in tier1_results:
            ticker = item['ticker']
            
            try:
                # Fetch historical data for indicators
                hist_data = self._fetch_historical_data(ticker, period='3mo')
                
                if hist_data is None or len(hist_data) < 50:
                    logger.debug(f"Insufficient data for {ticker}")
                    continue
                
                # Calculate technical indicators
                indicators = self._calculate_tier2_indicators(hist_data)
                
                # Calculate enhanced score
                tier2_score = self._calculate_tier2_score(
                    tier1_score=item['score'],
                    indicators=indicators
                )
                
                logger.debug(
                    f"{ticker}: tier1={item['score']:.1f} → tier2={tier2_score:.1f} "
                    f"(RSI={indicators.get('rsi', 0):.1f}, MACD={'bull' if indicators.get('macd', 0) > indicators.get('macd_signal', 0) else 'bear'})"
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
                        'ema_200': indicators.get('ema_200'),
                        'volume_ratio': indicators['volume_ratio'],
                        'atr': indicators['atr'],
                        'volatility': indicators['volatility'],
                        'signals': self._generate_tier2_signals(indicators),
                        'trend': self._determine_trend(indicators)
                    })
                    
            except Exception as e:
                logger.debug(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by enhanced score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ TIER 2: {len(results)} stocks passed medium analysis in {elapsed:.2f}s")
        
        if len(results) == 0 and len(tier1_results) > 0:
            logger.warning(f"⚠️ All {len(tier1_results)} stocks were filtered out by tier2_min_score={self.tier2_min_score}")
        
        return results
    
    def tier3_deep_analysis(
        self, 
        tier2_results: List[Dict],
        include_options: bool = True,
        ai_reviewer=None
    ) -> List[Dict]:
        """
        Tier 3: Deep analysis with AI review and options analysis
        
        Full analysis including:
        - Complete technical analysis
        - AI pre-trade review (if available)
        - Options chain analysis
        - Risk assessment
        - Entry/exit recommendations
        
        Returns: Fully analyzed stocks ready for monitoring
        """
        logger.info(f"🎯 TIER 3: Deep analysis of {len(tier2_results)} stocks...")
        start_time = time.time()
        
        results = []
        
        for item in tier2_results:
            ticker = item['ticker']
            
            try:
                # Get full comprehensive analysis
                deep_analysis = self._run_deep_analysis(ticker)
                
                if not deep_analysis:
                    continue
                
                # Get AI review if available
                ai_analysis = None
                if ai_reviewer and self.use_ai_enhancement:
                    ai_analysis = self._get_ai_review(ticker, item, ai_reviewer)
                
                # Get options analysis if requested
                options_analysis = None
                if include_options:
                    options_analysis = self._analyze_options(ticker)
                
                # Calculate final composite score
                final_score = self._calculate_tier3_score(
                    tier2_score=item['score'],
                    deep_analysis=deep_analysis,
                    ai_confidence=ai_analysis.get('confidence', 0) if ai_analysis else 0
                )
                
                results.append({
                    **item,
                    'tier': 3,
                    'score': final_score,
                    'support': deep_analysis.get('support'),
                    'resistance': deep_analysis.get('resistance'),
                    'stop_loss': deep_analysis.get('stop_loss'),
                    'take_profit': deep_analysis.get('take_profit'),
                    'risk_level': deep_analysis.get('risk_level', 'MEDIUM'),
                    'ai_recommendation': ai_analysis.get('recommendation') if ai_analysis else None,
                    'ai_confidence': ai_analysis.get('confidence') if ai_analysis else None,
                    'ai_reasoning': ai_analysis.get('reasoning') if ai_analysis else None,
                    'options_iv_rank': options_analysis.get('iv_rank') if options_analysis else None,
                    'options_signal': options_analysis.get('signal') if options_analysis else None,
                    'ready_for_monitoring': final_score >= 70
                })
                
            except Exception as e:
                logger.error(f"Error in deep analysis for {ticker}: {e}")
                continue
        
        # Sort by final score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ TIER 3: {len(results)} stocks fully analyzed in {elapsed:.2f}s")
        
        return results
    
    # ============= HELPER METHODS =============
    
    def _fetch_quick_data(self, ticker) -> Optional[Dict]:
        """Fetch quick data for a single ticker"""
        try:
            # Handle dict input (watchlist items) - extract ticker string
            if isinstance(ticker, dict):
                ticker = ticker.get('ticker', '')
            
            # Ensure ticker is a valid string
            if not ticker or not isinstance(ticker, str):
                return None
            
            ticker = ticker.upper()
            stock = yf.Ticker(ticker)
            
            # Get fast info
            fast_info = stock.fast_info
            info = {}
            try:
                info = stock.info
            except:
                pass
            
            # Get recent history for volume
            hist = stock.history(period='5d')
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price / prev_close - 1) * 100)
            
            current_volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            return {
                'ticker': ticker,
                'price': float(current_price),
                'change_pct': float(change_pct),
                'volume': int(current_volume),
                'avg_volume': float(avg_volume),
                'volume_ratio': float(current_volume / avg_volume) if avg_volume > 0 else 0,
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector', 'Unknown'),
            }
            
        except Exception as e:
            logger.debug(f"Error fetching quick data for {ticker}: {e}")
            return None
    
    def _fetch_historical_data(self, ticker, period: str = '3mo') -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data"""
        try:
            # Handle dict input (watchlist items) - extract ticker string
            if isinstance(ticker, dict):
                ticker = ticker.get('ticker', '')
            if not ticker or not isinstance(ticker, str):
                return None
            ticker = ticker.upper()
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            return hist if not hist.empty else None
        except Exception as e:
            logger.debug(f"Error fetching historical data for {ticker}: {e}")
            return None
    
    def _calculate_tier1_score(self, data: Dict) -> float:
        """Calculate quick score based on basic indicators"""
        score = 0
        
        # Price momentum (40 points max)
        change_pct = data.get('change_pct', 0)
        if change_pct > 10:
            score += 40
        elif change_pct > 5:
            score += 30
        elif change_pct > 2:
            score += 20
        elif change_pct > 0:
            score += 10
        
        # Volume ratio (35 points max)
        volume_ratio = data.get('volume_ratio', 1)
        if volume_ratio > 3:
            score += 35
        elif volume_ratio > 2:
            score += 25
        elif volume_ratio > 1.5:
            score += 15
        elif volume_ratio > 1:
            score += 5
        
        # Market cap bonus for different plays (25 points max)
        market_cap = data.get('market_cap', 0)
        if market_cap:
            if market_cap < 300_000_000:  # Under 300M (penny stock territory)
                score += 25  # High upside potential
            elif market_cap < 2_000_000_000:  # Small cap
                score += 20
            elif market_cap < 10_000_000_000:  # Mid cap
                score += 15
            else:  # Large cap (stable for options)
                score += 10
        
        return min(score, 100)
    
    def _calculate_tier2_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators from historical data"""
        try:
            # Ensure numeric types
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(50)
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            
            # EMAs
            ema_20 = df['Close'].ewm(span=20).mean()
            ema_50 = df['Close'].ewm(span=50).mean()
            ema_200 = df['Close'].ewm(span=200).mean() if len(df) >= 200 else None
            
            # Volume ratio
            avg_volume = df['Volume'].rolling(window=20).mean()
            volume_ratio = df['Volume'].iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # ATR (Average True Range)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            
            # Volatility
            volatility = df['Close'].pct_change().std() * 100
            
            current_price = df['Close'].iloc[-1]
            
            return {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                'macd': float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0.0,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
                'ema_20': float(ema_20.iloc[-1]) if not pd.isna(ema_20.iloc[-1]) else current_price,
                'ema_50': float(ema_50.iloc[-1]) if not pd.isna(ema_50.iloc[-1]) else current_price,
                'ema_200': float(ema_200.iloc[-1]) if ema_200 is not None and not pd.isna(ema_200.iloc[-1]) else None,
                'volume_ratio': float(volume_ratio) if not pd.isna(volume_ratio) else 1.0,
                'atr': float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0,
                'volatility': float(volatility) if not pd.isna(volatility) else 0.0,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.debug(f"Error calculating indicators: {e}")
            return {}
    
    def _calculate_tier2_score(self, tier1_score: float, indicators: Dict) -> float:
        """Calculate enhanced score with technical indicators"""
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
        current_price = indicators.get('current_price', 0)
        
        if ema_20 > ema_50:  # Uptrend
            score += 10
        
        # Price above EMAs bonus (5 points)
        if current_price > ema_20 > ema_50:
            score += 5
        
        # Volume surge (5 points)
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            score += 5
        elif volume_ratio > 1.5:
            score += 3
        
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
        else:
            signals.append("📉 MACD Bearish")
        
        ema_20 = indicators.get('ema_20', 0)
        ema_50 = indicators.get('ema_50', 0)
        if ema_20 > ema_50:
            signals.append("🎯 EMA Uptrend")
        else:
            signals.append("⬇️ EMA Downtrend")
        
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 2:
            signals.append("📊 Volume Surge")
        
        return signals
    
    def _determine_trend(self, indicators: Dict) -> str:
        """Determine overall trend from indicators"""
        ema_20 = indicators.get('ema_20', 0)
        ema_50 = indicators.get('ema_50', 0)
        ema_200 = indicators.get('ema_200')
        current_price = indicators.get('current_price', 0)
        
        if ema_200 and current_price > ema_20 > ema_50 > ema_200:
            return "STRONG UPTREND"
        elif current_price > ema_20 > ema_50:
            return "UPTREND"
        elif ema_200 and current_price < ema_20 < ema_50 < ema_200:
            return "STRONG DOWNTREND"
        elif current_price < ema_20 < ema_50:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def _run_deep_analysis(self, ticker: str) -> Optional[Dict]:
        """Run deep analysis on a stock"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='6mo')
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            
            # Calculate support/resistance
            recent_lows = hist['Low'].rolling(window=20).min()
            recent_highs = hist['High'].rolling(window=20).max()
            
            support = float(recent_lows.iloc[-1])
            resistance = float(recent_highs.iloc[-1])
            
            # Calculate stop loss and take profit
            atr = (hist['High'] - hist['Low']).rolling(window=14).mean().iloc[-1]
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
            
            # Determine risk level
            volatility = hist['Close'].pct_change().std() * 100
            if volatility > 5:
                risk_level = "HIGH"
            elif volatility > 2.5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                'support': support,
                'resistance': resistance,
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'risk_level': risk_level,
                'volatility': float(volatility)
            }
            
        except Exception as e:
            logger.debug(f"Error in deep analysis for {ticker}: {e}")
            return None
    
    def _get_ai_review(self, ticker: str, item: Dict, ai_reviewer) -> Optional[Dict]:
        """Get AI pre-trade review"""
        try:
            if not ai_reviewer:
                return None
            
            # Call the AI reviewer
            approved, confidence, reasoning, recommendations = ai_reviewer.review_stock(
                ticker=ticker,
                price=item.get('price', 0),
                change_pct=item.get('change_pct', 0),
                volume_ratio=item.get('volume_ratio', 1),
                rsi=item.get('rsi', 50),
                trend=item.get('trend', 'UNKNOWN')
            )
            
            return {
                'approved': approved,
                'confidence': confidence,
                'reasoning': reasoning,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.debug(f"Error getting AI review for {ticker}: {e}")
            return None
    
    def _analyze_options(self, ticker: str) -> Optional[Dict]:
        """Analyze options for a stock"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get options expiration dates
            expirations = stock.options
            if not expirations:
                return None
            
            # Get nearest expiration
            nearest_exp = expirations[0]
            chain = stock.option_chain(nearest_exp)
            
            # Calculate implied volatility rank (simplified)
            calls = chain.calls
            puts = chain.puts
            
            if calls.empty:
                return None
            
            avg_iv = calls['impliedVolatility'].mean() * 100
            
            # Simplified IV rank (would need historical IV for accurate rank)
            if avg_iv > 80:
                iv_rank = 90
                signal = "HIGH IV - Consider selling premium"
            elif avg_iv > 50:
                iv_rank = 60
                signal = "MODERATE IV - Mixed strategies"
            else:
                iv_rank = 30
                signal = "LOW IV - Consider buying premium"
            
            return {
                'iv_rank': iv_rank,
                'avg_iv': avg_iv,
                'signal': signal,
                'nearest_expiration': nearest_exp
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing options for {ticker}: {e}")
            return None
    
    def _calculate_tier3_score(
        self,
        tier2_score: float,
        deep_analysis: Dict,
        ai_confidence: float
    ) -> float:
        """Calculate final composite score"""
        # Weighted average
        score = (
            tier2_score * 0.4 +  # 40% from previous tiers
            (100 - (deep_analysis.get('risk_level', 'MEDIUM') == 'HIGH') * 20) * 0.3 +  # 30% risk adjustment
            ai_confidence * 0.3  # 30% from AI
        )
        
        return min(score, 100)


# Singleton instance
_scanner_instance = None


def get_tiered_stock_scanner(use_ai: bool = True) -> TieredStockScanner:
    """Get singleton instance of TieredStockScanner"""
    global _scanner_instance
    
    if _scanner_instance is None:
        _scanner_instance = TieredStockScanner(use_ai=use_ai)
    
    return _scanner_instance

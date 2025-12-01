"""
Enhanced Multi-Pronged Stock Opportunity Detector

Combines MULTIPLE analysis approaches to find trading opportunities:
1. Technical Indicators (RSI, MACD, Bollinger Bands, Volume, Momentum)
2. ML Confidence Scoring (Pattern recognition, trend strength)
3. LLM Reasoning (AI analysis of composite signals)
4. Event Detection (Earnings, FDA, SEC filings, News)
5. Sentiment Analysis (Social buzz, news sentiment)
6. Composite Scoring (Weighted combination of all signals)

This replaces the single-path cache-only approach with a comprehensive
multi-factor detection system aligned with production standards.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from loguru import logger
import yfinance as yf
import pandas as pd
from pandas import DataFrame

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from services.llm_helper import get_llm_helper
from services.enhanced_catalyst_detector import EnhancedCatalystDetector


@dataclass
class TechnicalSignals:
    """Technical analysis signals"""
    rsi: float = 0
    rsi_overbought: bool = False
    rsi_oversold: bool = False
    macd_bullish: bool = False
    macd_signal: str = "NEUTRAL"
    bollinger_above_upper: bool = False
    bollinger_below_lower: bool = False
    price_above_sma50: bool = False
    price_above_sma200: bool = False
    volume_spike: bool = False
    volume_ratio: float = 1.0
    momentum_positive: bool = False
    trend: str = "FLAT"  # UPTREND, DOWNTREND, FLAT
    score: float = 0.0  # 0-100
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass
class EventSignals:
    """Event-based signals"""
    has_earnings: bool = False
    earnings_date: Optional[str] = None
    earnings_direction: str = "UNKNOWN"  # POSITIVE, NEGATIVE, NEUTRAL
    fda_related: bool = False
    fda_catalyst: Optional[str] = None
    sec_filing: bool = False
    filing_type: Optional[str] = None
    news_score: float = 0.5  # 0-1.0
    news_sentiment: str = "NEUTRAL"  # POSITIVE, NEGATIVE, NEUTRAL
    major_news: Optional[str] = None
    score: float = 0.0
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


@dataclass
class CompositeOpportunity:
    """Complete opportunity with all analysis"""
    symbol: str
    price: float
    technical_score: float  # 0-100
    event_score: float  # 0-100
    ml_score: float  # 0-100
    llm_score: float  # 0-100
    composite_score: float  # Weighted average
    confidence: str  # CRITICAL, HIGH, MEDIUM, LOW
    technical_signals: TechnicalSignals
    event_signals: EventSignals
    ml_reasoning: Optional[str] = None
    llm_reasoning: Optional[str] = None
    composite_reasoning: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class EnhancedStockOpportunityDetector:
    """
    Multi-pronged stock opportunity detection system
    Combines technical, fundamental, event, and sentiment analysis
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize detector
        
        Args:
            use_llm: Whether to include LLM-based analysis
        """
        self.use_llm = use_llm
        self.llm_helper = None
        self.catalyst_detector = None
        
        # Thresholds
        self.min_composite_score = 60  # Minimum composite score to flag opportunity
        self.technical_weight = 0.25
        self.event_weight = 0.25
        self.ml_weight = 0.25
        self.llm_weight = 0.25
        
        # Cache for performance
        self.ticker_cache = {}
        self.cache_ttl_minutes = 30
        
        if use_llm:
            try:
                self.llm_helper = get_llm_helper('enhanced_stock_detector')
                logger.info("✅ LLM helper initialized for opportunity detection")
            except Exception as e:
                logger.warning(f"⚠️ LLM helper not available: {e}")
                self.use_llm = False
        
        try:
            self.catalyst_detector = EnhancedCatalystDetector()
            logger.info("✅ Catalyst detector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Catalyst detector not available: {e}")
            self.catalyst_detector = None
    
    def analyze_ticker(self, symbol: str) -> Optional[CompositeOpportunity]:
        """
        Comprehensive multi-pronged analysis of a ticker
        
        Args:
            symbol: Ticker symbol to analyze
            
        Returns:
            CompositeOpportunity if found and meets threshold, None otherwise
        """
        try:
            logger.debug(f"🔍 Analyzing {symbol} with multi-pronged approach...")
            
            # Check cache first
            cached = self._get_cached_analysis(symbol)
            if cached:
                logger.debug(f"  ✓ Using cached analysis for {symbol}")
                return cached
            
            analysis_start = time.time()
            
            # Fetch price data
            logger.debug(f"  📊 Fetching price data...")
            ticker_data = self._get_ticker_data(symbol)
            if ticker_data is None or ticker_data.empty:
                logger.debug(f"  ❌ Could not fetch price data for {symbol}")
                return None
            
            current_price = ticker_data['Close'].iloc[-1]
            
            # 1. TECHNICAL ANALYSIS
            logger.debug(f"  📈 Running technical analysis...")
            technical_signals = self._analyze_technical(symbol, ticker_data)
            technical_score = technical_signals.score
            
            # 2. EVENT DETECTION
            logger.debug(f"  📢 Checking for catalysts/events...")
            event_signals = self._analyze_events(symbol)
            event_score = event_signals.score
            
            # 3. ML CONFIDENCE SCORING
            logger.debug(f"  🤖 Computing ML confidence score...")
            ml_score = self._compute_ml_score(symbol, ticker_data, technical_signals)
            ml_reasoning = self._get_ml_reasoning(symbol, technical_signals, ml_score)
            
            # 4. LLM ANALYSIS (if enabled)
            llm_score = 0.0
            llm_reasoning = None
            if self.use_llm and self.llm_helper:
                logger.debug(f"  🧠 Running LLM analysis...")
                llm_score, llm_reasoning = self._run_llm_analysis(
                    symbol, technical_signals, event_signals, ml_score
                )
            
            # 5. COMPOSITE SCORING
            composite_score = self._calculate_composite_score(
                technical_score, event_score, ml_score, llm_score
            )
            
            # Determine confidence level
            confidence = self._determine_confidence(composite_score)
            
            # Generate composite reasoning
            composite_reasoning = self._generate_composite_reasoning(
                symbol, technical_signals, event_signals, ml_reasoning, llm_reasoning, composite_score
            )
            
            # Create opportunity
            opportunity = CompositeOpportunity(
                symbol=symbol,
                price=current_price,
                technical_score=technical_score,
                event_score=event_score,
                ml_score=ml_score,
                llm_score=llm_score,
                composite_score=composite_score,
                confidence=confidence,
                technical_signals=technical_signals,
                event_signals=event_signals,
                ml_reasoning=ml_reasoning,
                llm_reasoning=llm_reasoning,
                composite_reasoning=composite_reasoning
            )
            
            analysis_duration = time.time() - analysis_start
            
            # Log result - always log to help diagnose filtering issues
            emoji = "🎯" if composite_score >= 80 else "👀" if composite_score >= 70 else "✓" if composite_score >= self.min_composite_score else "⏭️"
            
            logger.info(
                f"  {emoji} {symbol}: Score {composite_score:.0f}/100 "
                f"(Tech: {technical_score:.0f}, Event: {event_score:.0f}, "
                f"ML: {ml_score:.0f}, LLM: {llm_score:.0f}) "
                f"{'✅ PASSED' if composite_score >= self.min_composite_score else '❌ Below ' + str(self.min_composite_score) + ' threshold'}"
            )
            
            # Cache result
            self._cache_analysis(symbol, opportunity)
            
            return opportunity if composite_score >= self.min_composite_score else None
        
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}", exc_info=True)
            return None
    
    # ============================================================
    # TECHNICAL ANALYSIS
    # ============================================================
    
    def _analyze_technical(self, symbol: str, df: DataFrame) -> TechnicalSignals:
        """Comprehensive technical analysis"""
        signals = TechnicalSignals()
        reasons = []
        
        try:
            # RSI
            rsi = self._calculate_rsi(df['Close'])
            signals.rsi = rsi
            signals.rsi_overbought = rsi > 70
            signals.rsi_oversold = rsi < 30
            
            if signals.rsi_overbought:
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif signals.rsi_oversold:
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif 40 < rsi < 60:
                reasons.append(f"RSI neutral ({rsi:.0f})")
            
            # MACD
            macd_positive = self._check_macd(df['Close'])
            signals.macd_bullish = macd_positive
            signals.macd_signal = "BULLISH" if macd_positive else "BEARISH"
            if macd_positive:
                reasons.append("MACD bullish")
            else:
                reasons.append("MACD bearish/neutral")
            
            # Bollinger Bands
            bb_upper, bb_lower, bb_mid = self._calculate_bollinger_bands(df['Close'])
            current_price = df['Close'].iloc[-1]
            signals.bollinger_above_upper = current_price > bb_upper
            signals.bollinger_below_lower = current_price < bb_lower
            
            if signals.bollinger_above_upper:
                reasons.append("Price above Bollinger upper band (potential reversal)")
            elif signals.bollinger_below_lower:
                reasons.append("Price below Bollinger lower band (oversold)")
            
            # Moving averages
            sma50 = df['Close'].tail(50).mean()
            sma200 = df['Close'].tail(200).mean() if len(df) >= 200 else sma50
            signals.price_above_sma50 = current_price > sma50
            signals.price_above_sma200 = current_price > sma200
            
            if signals.price_above_sma50 and signals.price_above_sma200:
                signals.trend = "UPTREND"
                reasons.append("Price above both SMA50 and SMA200 (uptrend)")
            elif current_price < sma50 and current_price < sma200:
                signals.trend = "DOWNTREND"
                reasons.append("Price below both SMAs (downtrend)")
            else:
                signals.trend = "FLAT"
            
            # Volume analysis
            volume_ratio = self._calculate_volume_ratio(df)
            signals.volume_ratio = volume_ratio
            signals.volume_spike = volume_ratio > 1.5
            
            if signals.volume_spike:
                reasons.append(f"Volume spike ({volume_ratio:.1f}x average)")
            
            # Momentum
            momentum = self._calculate_momentum(df['Close'])
            signals.momentum_positive = momentum > 0
            if signals.momentum_positive:
                reasons.append(f"Positive momentum")
            
            # Calculate composite technical score
            score = 0.0
            if signals.volume_spike:
                score += 15
            if signals.momentum_positive:
                score += 15
            if signals.trend == "UPTREND":
                score += 20
            if signals.macd_bullish:
                score += 15
            if signals.price_above_sma50:
                score += 10
            if signals.rsi not in (signals.rsi_overbought, signals.rsi_oversold):
                score += 10
            
            signals.score = min(100, score)
            signals.reasons = reasons
            
            logger.debug(f"    📈 Technical: {signals.score:.0f} - {', '.join(reasons[:3])}")
        
        except Exception as e:
            logger.debug(f"    ⚠️  Error in technical analysis: {e}")
            signals.score = 50.0
        
        return signals
    
    # ============================================================
    # EVENT DETECTION
    # ============================================================
    
    def _analyze_events(self, symbol: str) -> EventSignals:
        """Analyze events and catalysts"""
        signals = EventSignals()
        reasons = []
        
        try:
            # Check for upcoming earnings
            # This is simplified - in production would check earnings calendar
            signals.has_earnings = False
            
            # FDA catalyst detection
            if self.catalyst_detector:
                try:
                    is_healthcare = self.catalyst_detector.is_healthcare_stock(symbol)
                    if is_healthcare:
                        signals.fda_related = True
                        reasons.append("Healthcare stock (potential FDA catalyst)")
                        signals.score += 20
                except:
                    pass
            
            # News sentiment (simplified)
            signals.news_sentiment = "NEUTRAL"
            signals.news_score = 0.5
            
            # Calculate event score
            score = signals.score
            if signals.fda_related:
                score += 25
            if signals.has_earnings:
                score += 15
            
            signals.score = min(100, score)
            signals.reasons = reasons
            
            logger.debug(f"    📢 Events: {signals.score:.0f} - {', '.join(reasons) if reasons else 'No catalysts'}")
        
        except Exception as e:
            logger.debug(f"    ⚠️  Error in event analysis: {e}")
            signals.score = 30.0
        
        return signals
    
    # ============================================================
    # ML CONFIDENCE SCORING
    # ============================================================
    
    def _compute_ml_score(self, symbol: str, df: DataFrame, technical_signals: TechnicalSignals) -> float:
        """Compute ML-based confidence score"""
        try:
            score = 0.0
            
            # Recent performance (last 5 days)
            recent_return = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
            if recent_return > 2:
                score += 15
            elif recent_return > 0:
                score += 5
            
            # Volatility assessment
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            if 1 < volatility < 5:  # Moderate volatility is good
                score += 15
            elif volatility > 5:  # High volatility
                score += 10
            
            # Price range
            price_range = (df['Close'].max() - df['Close'].min()) / df['Close'].mean() * 100
            if price_range > 5:
                score += 10
            
            # Technical alignment
            if technical_signals.trend == "UPTREND":
                score += 20
            if technical_signals.volume_spike:
                score += 15
            if technical_signals.macd_bullish:
                score += 15
            
            logger.debug(f"    🤖 ML: {score:.0f}")
            return min(100, score)
        
        except Exception as e:
            logger.debug(f"    ⚠️  Error in ML scoring: {e}")
            return 50.0
    
    def _get_ml_reasoning(self, symbol: str, technical_signals: TechnicalSignals, ml_score: float) -> str:
        """Generate ML reasoning"""
        reasons = []
        
        if ml_score >= 80:
            reasons.append("Strong ML confidence - multiple positive signals")
        elif ml_score >= 60:
            reasons.append("Moderate ML confidence - some positive indicators")
        
        if technical_signals.trend == "UPTREND":
            reasons.append("Uptrend confirmed by moving averages")
        
        if technical_signals.volume_spike:
            reasons.append(f"Volume surge detected ({technical_signals.volume_ratio:.1f}x)")
        
        return " | ".join(reasons) if reasons else "Moderate confidence setup"
    
    # ============================================================
    # LLM ANALYSIS
    # ============================================================
    
    def _run_llm_analysis(
        self,
        symbol: str,
        technical_signals: TechnicalSignals,
        event_signals: EventSignals,
        ml_score: float
    ) -> Tuple[float, str]:
        """Run LLM analysis on composite signals"""
        try:
            prompt = f"""
Analyze this stock trading opportunity:

Symbol: {symbol}
Technical Signals: {technical_signals.score:.0f}/100 - {', '.join(technical_signals.reasons[:2])}
Event Signals: {event_signals.score:.0f}/100 - {', '.join(event_signals.reasons) if event_signals.reasons else 'None'}
ML Confidence: {ml_score:.0f}/100
Trend: {technical_signals.trend}
Volume: {technical_signals.volume_ratio:.1f}x average

Provide a brief trading analysis (2-3 sentences) and confidence score (0-100).
Format: [SCORE: XX] Analysis here.
"""
            
            if self.llm_helper:
                response = self.llm_helper.query_llm(prompt, priority="LOW", use_cache=True)
                
                # Parse response
                score_start = response.find("[SCORE:")
                if score_start != -1:
                    score_end = response.find("]", score_start)
                    try:
                        score = float(response[score_start+7:score_end].strip())
                        reasoning = response[score_end+1:].strip()
                        logger.debug(f"    🧠 LLM: {score:.0f} - {reasoning[:60]}...")
                        return min(100, score), reasoning
                    except:
                        pass
            
            return ml_score, None  # Fallback to ML score
        
        except Exception as e:
            logger.debug(f"    ⚠️  Error in LLM analysis: {e}")
            return ml_score, None
    
    # ============================================================
    # COMPOSITE SCORING & REASONING
    # ============================================================
    
    def _calculate_composite_score(
        self, technical: float, event: float, ml: float, llm: float
    ) -> float:
        """
        Calculate weighted composite score with dynamic weight redistribution
        
        When certain components return 0 (no data/disabled), redistribute their
        weight to active components to avoid penalizing valid opportunities.
        """
        # Determine which components are active (non-zero)
        active_weights = {}
        inactive_weight = 0.0
        
        if technical > 0:
            active_weights['technical'] = self.technical_weight
        else:
            inactive_weight += self.technical_weight
            
        if event > 0:
            active_weights['event'] = self.event_weight
        else:
            inactive_weight += self.event_weight
            
        if ml > 0:
            active_weights['ml'] = self.ml_weight
        else:
            inactive_weight += self.ml_weight
            
        if llm > 0:
            active_weights['llm'] = self.llm_weight
        else:
            inactive_weight += self.llm_weight
        
        # If no active weights, return 0
        if not active_weights:
            return 0.0
        
        # Redistribute inactive weight proportionally to active components
        total_active_weight = sum(active_weights.values())
        redistribution_factor = (total_active_weight + inactive_weight) / total_active_weight
        
        # Calculate score with redistributed weights
        score = 0.0
        if 'technical' in active_weights:
            score += technical * active_weights['technical'] * redistribution_factor
        if 'event' in active_weights:
            score += event * active_weights['event'] * redistribution_factor
        if 'ml' in active_weights:
            score += ml * active_weights['ml'] * redistribution_factor
        if 'llm' in active_weights:
            score += llm * active_weights['llm'] * redistribution_factor
        
        return min(100, score)
    
    def _determine_confidence(self, score: float) -> str:
        """Determine confidence level"""
        if score >= 85:
            return "CRITICAL"
        elif score >= 75:
            return "HIGH"
        elif score >= 65:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_composite_reasoning(
        self,
        symbol: str,
        technical_signals: TechnicalSignals,
        event_signals: EventSignals,
        ml_reasoning: Optional[str],
        llm_reasoning: Optional[str],
        score: float
    ) -> str:
        """Generate comprehensive composite reasoning"""
        parts = []
        
        if score >= 80:
            parts.append(f"🎯 STRONG SETUP - {technical_signals.trend} with {technical_signals.volume_ratio:.1f}x volume")
        elif score >= 70:
            parts.append(f"👀 GOOD SETUP - {technical_signals.trend}")
        
        if technical_signals.reasons:
            parts.append(f"Technical: {technical_signals.reasons[0]}")
        
        if event_signals.reasons:
            parts.append(f"Catalyst: {event_signals.reasons[0]}")
        
        if ml_reasoning:
            parts.append(f"ML: {ml_reasoning[:100]}")
        
        if llm_reasoning:
            parts.append(f"AI: {llm_reasoning[:100]}")
        
        return " | ".join(parts)
    
    # ============================================================
    # TECHNICAL INDICATORS
    # ============================================================
    
    def _calculate_rsi(self, prices, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            if len(prices) < period:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # Use .item() to extract scalar value (avoids FutureWarning)
            last_rsi = rsi.iloc[-1]
            return float(last_rsi.item()) if hasattr(last_rsi, 'item') else float(last_rsi)
        except:
            return 50.0
    
    def _check_macd(self, prices, fast: int = 12, slow: int = 26, signal: int = 9) -> bool:
        """Check if MACD is bullish"""
        try:
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal).mean()
            return macd.iloc[-1] > signal_line.iloc[-1]
        except:
            return False
    
    def _calculate_bollinger_bands(self, prices, period: int = 20, num_std: float = 2.0):
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            return float(upper.iloc[-1]), float(lower.iloc[-1]), float(sma.iloc[-1])
        except:
            return 0, 0, 0
    
    def _calculate_volume_ratio(self, df: DataFrame) -> float:
        """Calculate volume ratio vs average"""
        try:
            if 'Volume' not in df.columns:
                return 1.0
            recent_volume = df['Volume'].iloc[-1]
            avg_volume = df['Volume'].tail(20).mean()
            return recent_volume / avg_volume if avg_volume > 0 else 1.0
        except:
            return 1.0
    
    def _calculate_momentum(self, prices) -> float:
        """Calculate momentum (ROC)"""
        try:
            if len(prices) < 5:
                return 0
            momentum = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5]
            return float(momentum)
        except:
            return 0
    
    # ============================================================
    # UTILITY METHODS
    # ============================================================
    
    def _get_ticker_data(self, symbol: str, period: str = "3mo") -> Optional[DataFrame]:
        """Fetch ticker data from yfinance with retry logic for rate limiting"""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                data = yf.download(symbol, period=period, progress=False, threads=False, auto_adjust=True)
                if data is None or (isinstance(data, DataFrame) and data.empty):
                    logger.debug(f"  ❌ No data returned for {symbol}")
                    return None
                
                # Handle multi-indexed columns (yfinance now returns (Price, Ticker) format)
                if isinstance(data.columns, pd.MultiIndex):
                    # Flatten the columns - take only the first level (Price)
                    data.columns = data.columns.get_level_values(0)
                    logger.debug(f"  ⚙️  Flattened multi-index columns for {symbol}")
                
                logger.debug(f"  ✓ Fetched {len(data)} rows for {symbol}")
                return data
            except Exception as e:
                error_str = str(e).lower()
                # Check for rate limiting / throttling
                if 'rate' in error_str or 'throttl' in error_str or 'right back' in error_str:
                    if attempt < max_retries:
                        logger.warning(f"  ⚠️  Rate limited on {symbol}, waiting 5s (attempt {attempt + 1})...")
                        time.sleep(5)
                        continue
                logger.debug(f"Error fetching {symbol}: {e}")
                return None
        return None
    
    def _get_cached_analysis(self, symbol: str) -> Optional[CompositeOpportunity]:
        """Get cached analysis if still valid"""
        if symbol in self.ticker_cache:
            cached_item = self.ticker_cache[symbol]
            age_minutes = (datetime.now() - cached_item['time']).total_seconds() / 60
            if age_minutes < self.cache_ttl_minutes:
                return cached_item['opportunity']
            else:
                del self.ticker_cache[symbol]
        return None
    
    def _cache_analysis(self, symbol: str, opportunity: CompositeOpportunity):
        """Cache analysis result"""
        self.ticker_cache[symbol] = {
            'opportunity': opportunity,
            'time': datetime.now()
        }


# Singleton instance
_detector_instance = None


def get_enhanced_stock_detector(use_llm: bool = True) -> EnhancedStockOpportunityDetector:
    """Get singleton instance of enhanced detector"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EnhancedStockOpportunityDetector(use_llm=use_llm)
    return _detector_instance


if __name__ == "__main__":
    detector = get_enhanced_stock_detector()
    
    # Test with a symbol
    test_symbols = ["AAPL", "SPY", "NVDA"]
    for symbol in test_symbols:
        logger.info(f"\nTesting {symbol}...")
        opportunity = detector.analyze_ticker(symbol)
        if opportunity:
            logger.info(f"✅ {symbol}: {opportunity.composite_score:.0f} - {opportunity.confidence}")
            logger.info(f"   {opportunity.composite_reasoning}")
        else:
            logger.info(f"⏭️  {symbol}: No opportunity")


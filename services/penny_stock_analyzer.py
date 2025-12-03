"""
Penny Stock Scoring and Analysis Module

This module provides comprehensive scoring algorithms for penny stocks,
including momentum, valuation, catalyst, and composite scores.

IMPORTANT: Uses lazy import of yfinance to prevent Task Scheduler hangs
"""
print("[TRACE] penny_stock_analyzer.py: Starting module load...", flush=True)

# Lazy import - only load when actually fetching data
def _get_yf():
    import yfinance as yf
    return yf

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from services.penny_stock_constants import PENNY_THRESHOLDS, is_penny_stock
from services.fda_catalyst_detector import FDACatalystDetector



@dataclass
class StockScores:
    """Container for all stock scores"""
    momentum_score: float
    valuation_score: float
    catalyst_score: float
    composite_score: float
    confidence_level: str
    reasoning: str
    risk_narrative: str


class PennyStockScorer:
    """Calculates comprehensive scores for penny stocks"""
    
    # Scoring weights
    SCORE_WEIGHTS = {
        'MOMENTUM': 0.35,
        'VALUATION': 0.25,
        'CATALYST': 0.20,
        'TECHNICAL': 0.10,
        'NEWS_SENTIMENT': 0.10
    }
    
    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'VERY_HIGH': 80,
        'HIGH': 65,
        'MEDIUM': 50,
        'LOW': 35
    }
    
    @staticmethod
    def calculate_momentum_score(data: Dict) -> float:
        """
        Calculate momentum score (0-100) based on price action, volume, and technical indicators
        
        Args:
            data: Dictionary containing stock data
                - pct_change: Price change percentage
                - volume: Current volume
                - avg_volume: Average volume
                - rsi: RSI indicator
                - technical_score: Technical analysis score
                - buzz: Social buzz level ('high', 'med', 'low')
        
        Returns:
            Momentum score from 0 to 100
        """
        score = 50  # Base score
        
        # Price momentum (30 points)
        pct_change = data.get('pct_change', 0)
        if pct_change > 0.20:
            score += 30
        elif pct_change > 0.10:
            score += 20
        elif pct_change > 0.05:
            score += 10
        elif pct_change > 0:
            score += 5
        elif pct_change < -0.10:
            score -= 20
        elif pct_change < -0.05:
            score -= 10
        
        # Volume vs average (25 points)
        volume = data.get('volume', 0)
        avg_volume = data.get('avg_volume', 0)
        if avg_volume > 0:
            vol_ratio = volume / avg_volume
            if vol_ratio > 3:
                score += 25
            elif vol_ratio > 2:
                score += 20
            elif vol_ratio > 1.5:
                score += 15
            elif vol_ratio > 1:
                score += 10
            elif vol_ratio < 0.5:
                score -= 15
        
        # Social buzz (20 points)
        buzz = str(data.get('buzz', '')).lower()
        if buzz == 'high':
            score += 20
        elif buzz in ['med', 'medium']:
            score += 10
        elif buzz == 'low':
            score += 5
        
        # Technical score (15 points)
        tech_score = data.get('technical_score', 50)
        score += (tech_score - 50) * 0.3
        
        # RSI consideration (10 points)
        rsi = data.get('rsi', 50)
        if rsi < 30:
            score += 10  # Oversold = opportunity
        elif rsi > 70:
            score -= 5  # Overbought = caution
        
        return max(0, min(100, round(score)))
    
    @staticmethod
    def calculate_valuation_score(data: Dict) -> float:
        """
        Calculate valuation score (0-100) based on fundamentals
        
        Args:
            data: Dictionary containing stock data
                - pe_ratio: Price to earnings ratio
                - revenue_growth: Revenue growth percentage
                - profit_margin: Profit margin percentage
                - analyst_rating: Analyst rating string
                - cash_debt: Cash/debt status
        
        Returns:
            Valuation score from 0 to 100
        """
        score = 50  # Base score
        data_points = 0
        
        # P/E Ratio (25 points) - handle string values like 'Infinity'
        pe_ratio = data.get('pe_ratio', -1)
        try:
            pe_ratio = float(pe_ratio) if pe_ratio is not None else -1
            if pe_ratio == float('inf') or pe_ratio == float('-inf'):
                pe_ratio = -1  # Treat infinity as no data
        except (ValueError, TypeError):
            pe_ratio = -1  # Handle string values like 'Infinity'
        
        if pe_ratio > 0:
            data_points += 1
            if pe_ratio < 10:
                score += 25  # Very cheap
            elif pe_ratio < 15:
                score += 20
            elif pe_ratio < 20:
                score += 15
            elif pe_ratio < 30:
                score += 5
            elif pe_ratio > 50:
                score -= 15  # Expensive
        
        # Revenue Growth (25 points)
        revenue_growth = data.get('revenue_growth', None)
        if revenue_growth is not None:
            data_points += 1
            if revenue_growth > 50:
                score += 25
            elif revenue_growth > 30:
                score += 20
            elif revenue_growth > 15:
                score += 15
            elif revenue_growth > 5:
                score += 10
            elif revenue_growth < -10:
                score -= 20
        
        # Profit Margin (20 points)
        profit_margin = data.get('profit_margin', None)
        if profit_margin is not None:
            data_points += 1
            if profit_margin > 20:
                score += 20
            elif profit_margin > 10:
                score += 15
            elif profit_margin > 5:
                score += 10
            elif profit_margin > 0:
                score += 5
            elif profit_margin < -20:
                score -= 20
        
        # Analyst Rating (15 points)
        rating = str(data.get('analyst_rating', '')).lower()
        if 'strong buy' in rating:
            score += 15
        elif 'buy' in rating:
            score += 10
        elif 'hold' in rating:
            score += 0
        elif 'sell' in rating:
            score -= 10
        
        # Cash/Debt situation (15 points)
        cash_debt = str(data.get('cash_debt', '')).lower()
        if 'strong' in cash_debt or 'excellent' in cash_debt:
            score += 15
        elif 'good' in cash_debt or 'positive' in cash_debt:
            score += 10
        elif 'weak' in cash_debt or 'poor' in cash_debt:
            score -= 15
        
        # If we have very few data points, reduce confidence
        if data_points < 2:
            score = 50  # Not enough data for valuation
        
        return max(0, min(100, round(score)))
    
    @staticmethod
    def calculate_catalyst_score(data: Dict) -> float:
        """
        Calculate catalyst score (0-100) based on news, events, and insider activity
        
        Args:
            data: Dictionary containing stock data
                - news_sentiment: News sentiment string
                - news_count: Number of news articles
                - catalyst: Catalyst description
                - insider: Insider activity
        
        Returns:
            Catalyst score from 0 to 100
        """
        score = 50  # Base score
        
        # News sentiment (35 points)
        news_sentiment = str(data.get('news_sentiment', '')).lower()
        news_count = data.get('news_count', 0)
        
        if 'very positive' in news_sentiment:
            score += 35
        elif 'positive' in news_sentiment:
            score += 25
        elif 'neutral' in news_sentiment:
            score += 0
        elif 'very negative' in news_sentiment:
            score -= 35
        elif 'negative' in news_sentiment:
            score -= 20
        
        # News volume bonus (10 points)
        if news_count > 10:
            score += 10
        elif news_count > 5:
            score += 7
        elif news_count > 2:
            score += 5
        elif news_count == 0:
            score -= 10
        
        # Catalyst field (30 points)
        catalyst = str(data.get('catalyst', '')).lower()
        if len(catalyst) > 20:
            if 'fda' in catalyst or 'approval' in catalyst:
                score += 30
            elif 'earnings' in catalyst or 'revenue' in catalyst:
                score += 25
            elif 'partnership' in catalyst or 'deal' in catalyst:
                score += 25
            elif 'merger' in catalyst or 'acquisition' in catalyst:
                score += 25
            elif 'contract' in catalyst or 'award' in catalyst:
                score += 20
            elif 'launch' in catalyst or 'release' in catalyst:
                score += 15
            else:
                score += 10  # Generic catalyst
        
        # Insider activity (15 points)
        insider = str(data.get('insider', '')).lower()
        if 'buy' in insider or 'accumulation' in insider:
            score += 15
        elif 'sell' in insider or 'distribution' in insider:
            score -= 15
        
        # Verified status (10 points)
        verified = data.get('verified', '')
        if verified == '✅':
            score += 10
        elif verified == '⚠️':
            score += 5
        elif verified == '❌':
            score -= 10
        
        return max(0, min(100, round(score)))
    
    @classmethod
    def calculate_composite_score(cls, momentum: float, valuation: float, 
                                  catalyst: float, technical: float, 
                                  news_sentiment: str) -> float:
        """
        Calculate composite score with weighted components
        
        Args:
            momentum: Momentum score (0-100)
            valuation: Valuation score (0-100)
            catalyst: Catalyst score (0-100)
            technical: Technical score (0-100)
            news_sentiment: News sentiment string
        
        Returns:
            Composite score from 0 to 100
        """
        # Convert news sentiment to numeric score
        news_score = 50
        sentiment = str(news_sentiment).lower()
        if 'very positive' in sentiment:
            news_score = 90
        elif 'positive' in sentiment:
            news_score = 70
        elif 'neutral' in sentiment:
            news_score = 50
        elif 'very negative' in sentiment:
            news_score = 10
        elif 'negative' in sentiment:
            news_score = 30
        
        composite = (
            momentum * cls.SCORE_WEIGHTS['MOMENTUM'] +
            valuation * cls.SCORE_WEIGHTS['VALUATION'] +
            catalyst * cls.SCORE_WEIGHTS['CATALYST'] +
            technical * cls.SCORE_WEIGHTS['TECHNICAL'] +
            news_score * cls.SCORE_WEIGHTS['NEWS_SENTIMENT']
        )
        
        return round(composite)
    
    @classmethod
    def get_confidence_level(cls, composite_score: float) -> str:
        """Get confidence level from composite score"""
        if composite_score >= cls.CONFIDENCE_THRESHOLDS['VERY_HIGH']:
            return 'VERY HIGH'
        if composite_score >= cls.CONFIDENCE_THRESHOLDS['HIGH']:
            return 'HIGH'
        if composite_score >= cls.CONFIDENCE_THRESHOLDS['MEDIUM']:
            return 'MEDIUM'
        if composite_score >= cls.CONFIDENCE_THRESHOLDS['LOW']:
            return 'LOW'
        return 'VERY LOW'
    
    @staticmethod
    def generate_confidence_reasoning(data: Dict, scores: Dict[str, float]) -> str:
        """Generate detailed confidence reasoning"""
        reasons = []
        
        # Momentum factors
        if scores['momentum'] >= 70:
            reasons.append('Strong momentum')
        elif scores['momentum'] <= 30:
            reasons.append('Weak momentum')
        
        # Valuation factors
        if scores['valuation'] >= 70:
            reasons.append('Attractive valuation')
        elif scores['valuation'] <= 30:
            reasons.append('Overvalued')
        
        # Catalyst factors
        if scores['catalyst'] >= 70:
            reasons.append('Strong catalysts')
        elif scores['catalyst'] <= 30:
            reasons.append('Limited catalysts')
        
        # Technical factors
        tech_score = data.get('technical_score', 50)
        if tech_score >= 70:
            reasons.append('Bullish technicals')
        elif tech_score <= 30:
            reasons.append('Bearish technicals')
        
        # Risk factors
        risk = data.get('risk', '')
        if risk == 'H' or risk == 'High':
            reasons.append('High risk profile')
        
        # News sentiment
        sentiment = str(data.get('news_sentiment', '')).lower()
        if 'positive' in sentiment:
            reasons.append('Positive news flow')
        elif 'negative' in sentiment:
            reasons.append('Negative news')
        
        # Analyst support
        rating = str(data.get('analyst_rating', '')).lower()
        if 'buy' in rating:
            reasons.append('Analyst support')
        
        if not reasons:
            return 'Insufficient data for high confidence'
        
        return '; '.join(reasons)
    
    @staticmethod
    def generate_risk_narrative(data: Dict, scores: Dict[str, float]) -> str:
        """Generate risk narrative"""
        risks = []
        opportunities = []
        
        # Risk factors
        risk = data.get('risk', '')
        if risk in ['H', 'High']:
            risks.append('High general risk')
        
        dilution = str(data.get('dilution', '')).lower()
        if dilution == 'high':
            risks.append('Dilution risk')
        
        float_m = data.get('float_m', 0)
        if float_m < 10:
            risks.append('Low float (high volatility)')
        elif float_m > 500:
            risks.append('Large float (limited upside)')
        
        profit_margin = data.get('profit_margin', 0)
        if profit_margin is not None and profit_margin < 0:
            risks.append('Unprofitable')
        
        pe_ratio = data.get('pe_ratio', -1)
        if pe_ratio > 50:
            risks.append('High P/E ratio')
        
        # Opportunities
        if scores.get('momentum', 0) >= 70:
            opportunities.append('Strong momentum')
        if scores.get('catalyst', 0) >= 70:
            opportunities.append('Major catalysts')
        
        revenue_growth = data.get('revenue_growth', 0)
        if revenue_growth is not None and revenue_growth > 30:
            opportunities.append('Rapid revenue growth')
        
        rsi = data.get('rsi', 50)
        if rsi < 30:
            opportunities.append('Oversold (potential bounce)')
        
        # Construct narrative
        narrative = ''
        if risks:
            narrative += f"RISKS: {', '.join(risks)}. "
        if opportunities:
            narrative += f"OPPORTUNITIES: {', '.join(opportunities)}."
        
        return narrative if narrative else 'Balanced risk/reward profile'
    
    @classmethod
    def calculate_all_scores(cls, data: Dict) -> StockScores:
        """
        Calculate all scores for a stock
        
        Args:
            data: Dictionary containing all stock data
        
        Returns:
            StockScores object with all calculated scores
        """
        # Calculate individual scores
        momentum = cls.calculate_momentum_score(data)
        valuation = cls.calculate_valuation_score(data)
        catalyst = cls.calculate_catalyst_score(data)
        
        # Calculate composite score
        technical = data.get('technical_score', 50)
        news_sentiment = data.get('news_sentiment', 'Neutral')
        composite = cls.calculate_composite_score(
            momentum, valuation, catalyst, technical, news_sentiment
        )
        
        # Create scores dict for reasoning generation
        scores = {
            'momentum': momentum,
            'valuation': valuation,
            'catalyst': catalyst,
            'composite': composite
        }
        
        # Generate insights
        confidence = cls.get_confidence_level(composite)
        reasoning = cls.generate_confidence_reasoning(data, scores)
        risk_narrative = cls.generate_risk_narrative(data, scores)
        
        return StockScores(
            momentum_score=momentum,
            valuation_score=valuation,
            catalyst_score=catalyst,
            composite_score=composite,
            confidence_level=confidence,
            reasoning=reasoning,
            risk_narrative=risk_narrative
        )


class PennyStockAnalyzer:
    """Comprehensive penny stock analysis"""
    
    # Penny stock classification thresholds (using centralized constants)
    PENNY_STOCK_CRITERIA = {
        'MAX_PRICE': PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE,
        'MAX_MARKET_CAP': PENNY_THRESHOLDS.MAX_MARKET_CAP,
        'MIN_FLOAT': PENNY_THRESHOLDS.MIN_FLOAT,
        'MICRO_CAP_THRESHOLD': PENNY_THRESHOLDS.MICRO_CAP_THRESHOLD,
    }
    
    def __init__(self):
        self.scorer = PennyStockScorer()
        self.fda_detector = FDACatalystDetector()
    
    @classmethod
    def classify_stock_type(cls, price: float, market_cap: float, float_shares: float, 
                           exchange: str = "") -> Dict[str, any]:
        """
        Classify stock as penny stock based on multiple factors, not just price.
        
        Args:
            price: Current stock price
            market_cap: Market capitalization
            float_shares: Number of shares in float
            exchange: Exchange listing (NASDAQ, NYSE, OTC, etc.)
            
        Returns:
            Dict with classification details
        """
        is_penny = False
        risk_level = "LOW"
        classification = "STANDARD"
        reasons = []
        
        # Price check
        price_is_low = price < cls.PENNY_STOCK_CRITERIA['MAX_PRICE']
        
        # Market cap check
        is_micro_cap = market_cap < cls.PENNY_STOCK_CRITERIA['MAX_MARKET_CAP']
        is_nano_cap = market_cap < cls.PENNY_STOCK_CRITERIA['MICRO_CAP_THRESHOLD']
        
        # Float check
        has_low_float = float_shares > 0 and float_shares < cls.PENNY_STOCK_CRITERIA['MIN_FLOAT']
        
        # Exchange check (OTC is higher risk)
        is_otc = 'OTC' in exchange.upper() or 'PINK' in exchange.upper()
        
        # Classification logic
        if price_is_low and is_micro_cap:
            is_penny = True
            classification = "PENNY_STOCK"
            reasons.append(f"Price ${price:.2f} < ${cls.PENNY_STOCK_CRITERIA['MAX_PRICE']}")
            reasons.append(f"Market cap ${market_cap/1e6:.1f}M < ${cls.PENNY_STOCK_CRITERIA['MAX_MARKET_CAP']/1e6:.0f}M")
            
            if is_nano_cap:
                risk_level = "VERY_HIGH"
                classification = "NANO_CAP"
                reasons.append("Nano-cap (<$50M)")
            elif is_otc:
                risk_level = "VERY_HIGH"
                reasons.append("OTC/Pink Sheets listing")
            else:
                risk_level = "HIGH"
        elif price_is_low and not is_micro_cap:
            # Low price but sizeable market cap (like BYND scenario)
            classification = "LOW_PRICED"
            risk_level = "MEDIUM"
            reasons.append(f"Low price ${price:.2f} but market cap ${market_cap/1e6:.1f}M")
        elif is_micro_cap and not price_is_low:
            classification = "MICRO_CAP"
            risk_level = "HIGH"
            reasons.append(f"Micro-cap ${market_cap/1e6:.1f}M")
        
        if has_low_float:
            reasons.append(f"Low float {float_shares/1e6:.1f}M shares")
            if risk_level in ["LOW", "MEDIUM"]:
                risk_level = "HIGH"
        
        return {
            'is_penny_stock': is_penny,
            'classification': classification,
            'risk_level': risk_level,
            'reasons': reasons,
            'price_is_low': price_is_low,
            'is_micro_cap': is_micro_cap,
            'is_otc': is_otc
        }
    
    def analyze_stock(self, ticker: str, existing_data: Optional[Dict] = None) -> Dict:
        """
        Perform comprehensive analysis on a penny stock
        
        Args:
            ticker: Stock ticker symbol
            existing_data: Optional pre-fetched data to avoid redundant API calls
        
        Returns:
            Dictionary with complete analysis
        """
        try:
            # Fetch stock data (lazy import to prevent Task Scheduler hangs)
            yf = _get_yf()
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period='3mo')
            
            if hist.empty:
                logger.warning(f"No historical data for {ticker}")
                return {'error': 'No data available'}
            
            # Calculate basic metrics
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            pct_change = (current_price - prev_close) / prev_close if prev_close else 0
            
            # Calculate technical indicators
            rsi = self._calculate_rsi(hist['Close'])
            atr = self._calculate_atr(hist)
            
            # Calculate ATR-based stops and targets
            atr_stops = self._calculate_atr_stops_and_targets(current_price, atr)
            
            # Stock classification
            market_cap = info.get('marketCap', 0)
            float_shares = info.get('floatShares', 0)
            exchange = info.get('exchange', '')
            
            stock_classification = self.classify_stock_type(
                current_price, market_cap, float_shares, exchange
            )
            
            # Detect FDA/Healthcare catalysts
            is_healthcare, healthcare_sector = self.fda_detector.is_healthcare_stock(ticker, info)
            fda_catalyst_info = self.fda_detector.get_catalyst_summary(ticker)
            
            # Helper to safely get numeric values (yfinance sometimes returns 'Infinity' strings)
            def safe_numeric(value, default=None):
                if value is None:
                    return default
                try:
                    num = float(value)
                    if num == float('inf') or num == float('-inf') or num != num:  # Check for inf/nan
                        return default
                    return num
                except (ValueError, TypeError):
                    return default
            
            # Prepare data dictionary
            pe_raw = info.get('trailingPE', info.get('forwardPE', None))
            revenue_growth_raw = info.get('revenueGrowth', 0)
            profit_margin_raw = info.get('profitMargins', 0)
            
            data = {
                'ticker': ticker,
                'price': current_price,
                'pct_change': pct_change,
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                'avg_volume': int(hist['Volume'].mean()) if 'Volume' in hist else 0,
                'rsi': rsi,
                'atr': atr,
                'technical_score': self._calculate_technical_score(hist),
                'pe_ratio': safe_numeric(pe_raw, -1),
                'revenue_growth': safe_numeric(revenue_growth_raw, 0) * 100 if revenue_growth_raw else None,
                'profit_margin': safe_numeric(profit_margin_raw, 0) * 100 if profit_margin_raw else None,
                'market_cap': market_cap,
                'float_m': float_shares / 1_000_000 if float_shares else 0,
                'exchange': exchange,
                'is_healthcare': is_healthcare,
                'healthcare_sector': healthcare_sector,
                'fda_catalyst': fda_catalyst_info.get('catalyst_description', ''),
                **stock_classification,
                **atr_stops
            }
            
            # Merge with existing data if provided
            if existing_data:
                data.update(existing_data)
            
            # Calculate all scores
            scores = self.scorer.calculate_all_scores(data)
            
            # Apply FDA catalyst boost if detected
            catalyst_score_boost = 0
            catalyst_description_enhanced = scores.reasoning
            
            if fda_catalyst_info.get('has_catalyst'):
                catalyst_score_boost = fda_catalyst_info.get('score_boost', 0)
                catalyst_desc = fda_catalyst_info.get('catalyst_description', '')
                
                # Enhance composite score with FDA boost
                boosted_composite = min(100, scores.composite_score + (catalyst_score_boost * 0.6))
                scores = StockScores(
                    momentum_score=scores.momentum_score,
                    valuation_score=scores.valuation_score,
                    catalyst_score=min(100, scores.catalyst_score + catalyst_score_boost),
                    composite_score=boosted_composite,
                    confidence_level=self.scorer.get_confidence_level(boosted_composite),
                    reasoning=f"{scores.reasoning} | 💊 FDA: {fda_catalyst_info.get('catalyst_type', '').replace('_', ' ')}",
                    risk_narrative=scores.risk_narrative
                )
                
                logger.info(f"🎯 FDA BOOST for {ticker}: +{catalyst_score_boost:.0f} points → Composite: {scores.composite_score:.0f}")
            
            # Combine everything
            result = {
                **data,
                'momentum_score': scores.momentum_score,
                'valuation_score': scores.valuation_score,
                'catalyst_score': scores.catalyst_score,
                'composite_score': scores.composite_score,
                'confidence_level': scores.confidence_level,
                'reasoning': scores.reasoning,
                'risk_narrative': scores.risk_narrative,
                'last_updated': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            prices = pd.to_numeric(prices, errors='coerce').dropna()
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return round(rsi.iloc[-1], 2)
        except:
            return 50.0
    
    @staticmethod
    def _calculate_atr(hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = hist['High']
            low = hist['Low']
            close = hist['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
        except:
            return 0.0
    
    def _calculate_atr_stops_and_targets(self, current_price: float, atr: float, 
                                        trading_style: str = "SCALP") -> dict:
        """
        Calculate ATR-based stops and targets based on trading style.
        
        Args:
            current_price: Current stock price
            atr: Average True Range value
            trading_style: SCALP (tight), SWING (medium), POSITION (wide)
            
        Returns:
            Dict with stop_loss, target, stop_pct, target_pct, risk_reward
        """
        try:
            # Adjust multipliers based on trading style
            if trading_style == "SCALP":
                stop_multiplier = 1.0  # Tight 1.0 ATR stop
                rr_ratio = 2.0  # 2:1 R/R
            elif trading_style == "SWING":
                stop_multiplier = 1.5  # Medium 1.5 ATR stop
                rr_ratio = 2.5  # 2.5:1 R/R
            else:  # POSITION
                stop_multiplier = 2.0  # Wide 2.0 ATR stop
                rr_ratio = 3.0  # 3:1 R/R
            
            # For penny stocks, ensure minimum stop isn't too wide
            stop_distance = atr * stop_multiplier
            stop_pct = (stop_distance / current_price) * 100
            
            # Cap penny stock stops at 12% max (avoid -68% stops like BYND example)
            if stop_pct > 12.0:
                stop_pct = 12.0
                stop_distance = current_price * 0.12
            
            stop_loss = current_price - stop_distance
            target = current_price + (stop_distance * rr_ratio)
            target_pct = (target - current_price) / current_price * 100
            
            # Alternative: percentage-based stops for very low ATR
            if atr < current_price * 0.02:  # If ATR < 2% of price
                # Use fixed percentage stops
                if trading_style == "SCALP":
                    stop_pct = 3.0
                    target_pct = 6.0
                elif trading_style == "SWING":
                    stop_pct = 5.0
                    target_pct = 12.0
                else:
                    stop_pct = 8.0
                    target_pct = 20.0
                
                stop_loss = current_price * (1 - stop_pct / 100)
                target = current_price * (1 + target_pct / 100)
                rr_ratio = target_pct / stop_pct
            
            return {
                'atr_stop_loss': round(stop_loss, 2),
                'atr_target': round(target, 2),
                'atr_stop_pct': round(stop_pct, 2),
                'atr_target_pct': round(target_pct, 2),
                'atr_risk_reward': round(rr_ratio, 2),
                'trading_style': trading_style
            }
        except Exception as e:
            logger.error(f"Error calculating ATR stops: {e}")
            # Fallback to conservative fixed percentages
            return {
                'atr_stop_loss': round(current_price * 0.92, 2),  # 8% stop
                'atr_target': round(current_price * 1.16, 2),  # 16% target
                'atr_stop_pct': 8.0,
                'atr_target_pct': 16.0,
                'atr_risk_reward': 2.0,
                'trading_style': trading_style
            }
    
    @staticmethod
    def _calculate_technical_score(hist: pd.DataFrame) -> float:
        """Calculate overall technical score"""
        try:
            score = 50
            
            # Moving averages
            close = hist['Close']
            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()
            
            current_price = close.iloc[-1]
            
            # Price vs MA20
            if not ma20.empty and not pd.isna(ma20.iloc[-1]):
                if current_price > ma20.iloc[-1]:
                    score += 15
                else:
                    score -= 10
            
            # Price vs MA50
            if not ma50.empty and not pd.isna(ma50.iloc[-1]):
                if current_price > ma50.iloc[-1]:
                    score += 15
                else:
                    score -= 10
            
            # Volume trend
            vol = hist['Volume']
            if len(vol) > 20:
                recent_vol = vol.iloc[-5:].mean()
                avg_vol = vol.iloc[-20:].mean()
                if recent_vol > avg_vol * 1.5:
                    score += 20
            
            return max(0, min(100, round(score)))
        except:
            return 50.0

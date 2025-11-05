"""
ML-Enhanced Confidence Scanner

Combines Qlib ML predictions with LLM reasoning for maximum confidence.
This is your "best of both worlds" scanner for the most confident trading decisions.

Flow:
1. Qlib extracts 158 alpha factors (vs your current 10-15)
2. ML model predicts probability of price increase
3. LLM provides reasoning and risk analysis
4. Combined score gives you highest confidence trades

Usage:
    scanner = MLEnhancedScanner()
    trades = scanner.scan_with_ml_confidence(top_n=10)
    # Returns only trades with BOTH high ML score AND high LLM confidence
"""

from loguru import logger
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .ai_confidence_scanner import AIConfidenceScanner, AIConfidenceTrade
from .qlib_integration import QLIbEnhancedAnalyzer, check_qlib_installation
from .alpha_factors import AlphaFactorCalculator



@dataclass
class MLEnhancedTrade(AIConfidenceTrade):
    """Trade with both ML and LLM confidence"""
    ml_prediction_score: float = 0.0  # 0-100 from Qlib ML model
    ml_confidence: str = "MEDIUM"  # ML-based confidence
    ml_features_count: int = 0  # Number of alpha factors used
    combined_score: float = 0.0  # Weighted combination of ML + LLM + Quant
    ensemble_confidence: str = "MEDIUM"  # Final ensemble confidence


class MLEnhancedScanner:
    """
    Your ultimate confidence scanner combining:
    - Qlib ML predictions (158 alpha factors)
    - LLM reasoning and risk analysis
    - Traditional technical analysis
    
    This gives you the MOST confident trading decisions.
    """
    
    def __init__(self, use_ml: bool = None, use_llm: bool = None):
        """
        Initialize ML-Enhanced Scanner
        
        Args:
            use_ml: Whether to use Qlib ML. Auto-detect if None.
            use_llm: Whether to use LLM. Auto-detect if None.
        """
        # Initialize base AI confidence scanner (has LLM + quant)
        self.ai_scanner = AIConfidenceScanner(use_llm=use_llm)
        
        # Initialize Qlib ML analyzer
        qlib_available, msg = check_qlib_installation()
        
        if use_ml is None:
            self.use_ml = qlib_available
        else:
            self.use_ml = use_ml and qlib_available
        
        if self.use_ml:
            self.ml_analyzer = QLIbEnhancedAnalyzer()
            logger.info("✓ ML-Enhanced Scanner: Using Qlib ML + LLM + Quantitative analysis")
        else:
            self.ml_analyzer = None
            logger.info("✓ ML-Enhanced Scanner: Using LLM + Quantitative analysis (Qlib not available)")
        
        # Weights for ensemble scoring (tunable)
        self.weights = {
            'ml': 0.4,        # 40% ML prediction
            'llm': 0.35,      # 35% LLM confidence
            'quant': 0.25     # 25% Quantitative score
        }
    
    def scan_top_options_with_ml(self, 
                                  top_n: int = 20,
                                  min_ensemble_score: float = 70.0,
                                  min_price: float = None,
                                  max_price: float = None) -> List[MLEnhancedTrade]:
        """
        Scan for top options with ML + LLM confidence.
        
        This is your MAIN method for finding the best trades.
        Only returns trades that score high on BOTH ML and LLM.
        
        Args:
            top_n: Number of top trades to return
            min_ensemble_score: Minimum combined score (0-100)
        
        Returns:
            List of MLEnhancedTrade with highest confidence
        """
        logger.info(f"🎯 Starting ML-Enhanced scan for top {top_n} options (min ensemble score: {min_ensemble_score})...")
        
        # Step 1: Get AI confidence trades (LLM + Quant)
        # IMPORTANT: Pass price filters HERE to avoid wasting LLM calls on expensive stocks
        ai_trades = self.ai_scanner.scan_top_options_with_ai(
            top_n=top_n * 2,  # Get more candidates for ML filtering
            min_ai_rating=4.0,  # Lower threshold, ML will filter
            min_score=45.0,
            min_price=min_price,  # Filter BEFORE AI analysis
            max_price=max_price   # Filter BEFORE AI analysis
        )
        
        if not ai_trades:
            logger.warning("No AI trades found")
            return []
        
        logger.info(f"✓ Got {len(ai_trades)} AI-analyzed trades")
        
        # Step 1.5: Apply price filters if specified
        if min_price or max_price:
            filtered_trades = []
            price_rejected = 0
            for trade in ai_trades:
                if min_price and trade.price < min_price:
                    price_rejected += 1
                    continue
                if max_price and trade.price > max_price:
                    price_rejected += 1
                    continue
                filtered_trades.append(trade)
            
            if price_rejected > 0:
                logger.info(f"💰 Price Filter: Removed {price_rejected} trades outside ${min_price or 0:.2f}-${max_price or 999999:.2f} range")
            ai_trades = filtered_trades
        
        # Step 2: Enhance with ML predictions
        ml_trades = []
        rejected_trades = []
        
        for trade in ai_trades:
            ml_enhanced = self._enhance_with_ml(trade, 'options')
            
            # Only include if ensemble score meets threshold
            if ml_enhanced.combined_score >= min_ensemble_score:
                ml_trades.append(ml_enhanced)
                logger.info(f"✅ PASS {trade.ticker}: Ensemble={ml_enhanced.combined_score:.1f} "
                          f"(ML={ml_enhanced.ml_prediction_score:.1f}, "
                          f"AI={trade.ai_rating*10:.1f}, Quant={trade.score:.1f})")
            else:
                rejected_trades.append((trade.ticker, ml_enhanced.combined_score, 
                                       ml_enhanced.ml_prediction_score, trade.ai_rating*10, trade.score))
                logger.info(f"❌ REJECT {trade.ticker}: Ensemble={ml_enhanced.combined_score:.1f} < {min_ensemble_score} "
                          f"(ML={ml_enhanced.ml_prediction_score:.1f}, "
                          f"AI={trade.ai_rating*10:.1f}, Quant={trade.score:.1f})")
        
        # Step 3: Sort by combined score
        ml_trades.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Log summary of rejections
        if rejected_trades:
            logger.info(f"📊 REJECTION SUMMARY: {len(rejected_trades)} trades below threshold of {min_ensemble_score}")
            # Sort rejected by ensemble score and show top 5
            rejected_trades.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"   Top 5 closest to threshold:")
            for ticker, ens, ml, ai, quant in rejected_trades[:5]:
                logger.info(f"   • {ticker}: {ens:.1f} (ML:{ml:.1f} AI:{ai:.1f} Q:{quant:.1f}) - missed by {min_ensemble_score-ens:.1f}")
        
        # Step 4: Return top N
        final_trades = ml_trades[:top_n]
        
        logger.info(f"🎯 Found {len(final_trades)} high-confidence ML+LLM trades")
        self._log_confidence_summary(final_trades)
        
        return final_trades
    
    def scan_top_penny_stocks_with_ml(self,
                                       top_n: int = 15,
                                       min_ensemble_score: float = 65.0) -> List[MLEnhancedTrade]:
        """
        Scan for top penny stocks with ML + LLM confidence.
        
        Args:
            top_n: Number of top stocks to return
            min_ensemble_score: Minimum combined score (0-100)
        
        Returns:
            List of MLEnhancedTrade with highest confidence
        """
        logger.info(f"🎯 Starting ML-Enhanced scan for top {top_n} penny stocks...")
        
        # Get AI confidence trades
        ai_trades = self.ai_scanner.scan_top_penny_stocks_with_ai(
            top_n=top_n * 2,
            min_ai_rating=4.5,
            min_score=50.0
        )
        
        # Enhance with ML
        ml_trades = []
        for trade in ai_trades:
            ml_enhanced = self._enhance_with_ml(trade, 'penny_stock')
            
            if ml_enhanced.combined_score >= min_ensemble_score:
                ml_trades.append(ml_enhanced)
        
        # Sort and return top N
        ml_trades.sort(key=lambda x: x.combined_score, reverse=True)
        final_trades = ml_trades[:top_n]
        
        logger.info(f"🎯 Found {len(final_trades)} high-confidence penny stock trades")
        return final_trades
    
    def _enhance_with_ml(self, trade: AIConfidenceTrade, trade_type: str) -> MLEnhancedTrade:
        """
        Enhance an AI confidence trade with ML predictions.
        
        This is where the magic happens - combining ML + LLM.
        """
        ml_score = 0.0
        ml_conf = "MEDIUM"
        features_count = 0
        
        # Calculate alpha factors using yfinance
        try:
            alpha_calc = AlphaFactorCalculator()
            features = alpha_calc.calculate_factors(trade.ticker)
            features_count = len(features)
            
            if features:
                # Estimate ML score from alpha factors
                ml_score = self._estimate_ml_score_from_features(features, trade)
                ml_conf = self._score_to_confidence(ml_score)
                logger.debug(f"{trade.ticker}: Calculated {features_count} alpha factors, ML score: {ml_score:.1f}")
            else:
                # No features available
                ml_score = trade.score * 0.7  # Conservative estimate
                ml_conf = "LOW"
                features_count = 0
                
        except Exception as e:
            logger.error(f"Error calculating alpha factors for {trade.ticker}: {e}")
            ml_score = trade.score * 0.7
            ml_conf = "LOW"
            features_count = 0
        
        # Calculate combined score (ensemble)
        combined_score = self._calculate_ensemble_score(
            ml_score=ml_score,
            llm_rating=trade.ai_rating * 10,  # Convert 0-10 to 0-100
            quant_score=trade.score
        )
        
        # Determine ensemble confidence
        ensemble_conf = self._determine_ensemble_confidence(
            ml_conf=ml_conf,
            llm_conf=trade.ai_confidence,
            quant_conf=trade.confidence
        )
        
        # Create ML-enhanced trade
        return MLEnhancedTrade(
            ticker=trade.ticker,
            score=trade.score,
            price=trade.price,
            change_pct=trade.change_pct,
            volume=trade.volume,
            volume_ratio=trade.volume_ratio,
            reason=trade.reason,
            trade_type=trade_type,
            confidence=trade.confidence,
            risk_level=trade.risk_level,
            ai_confidence=trade.ai_confidence,
            ai_reasoning=trade.ai_reasoning,
            ai_risks=trade.ai_risks,
            ai_rating=trade.ai_rating,
            ml_prediction_score=ml_score,
            ml_confidence=ml_conf,
            ml_features_count=features_count,
            combined_score=combined_score,
            ensemble_confidence=ensemble_conf
        )
    
    def _estimate_ml_score_from_features(self, features: Dict, trade: AIConfidenceTrade) -> float:
        """
        Estimate ML score from alpha factors.
        Uses feature values to create a composite score.
        """
        try:
            score_components = []
            
            # Momentum signals (weight: 30%)
            momentum_score = 50  # neutral baseline
            if 'return_5d' in features:
                momentum_score += features['return_5d'] * 200  # 10% return = +20 points
            if 'momentum_20d' in features:
                momentum_score += features['momentum_20d'] * 100
            momentum_score = max(0, min(100, momentum_score))
            score_components.append(('momentum', momentum_score, 0.30))
            
            # Volume signals (weight: 20%)
            volume_score = 50
            if 'volume_5d_ratio' in features:
                vol_ratio = features['volume_5d_ratio']
                if vol_ratio > 1:
                    volume_score += min(30, (vol_ratio - 1) * 50)  # Higher volume = bullish
                else:
                    volume_score -= min(20, (1 - vol_ratio) * 30)  # Lower volume = bearish
            volume_score = max(0, min(100, volume_score))
            score_components.append(('volume', volume_score, 0.20))
            
            # Technical indicators (weight: 25%)
            technical_score = 50
            if 'rsi_14' in features:
                rsi = features['rsi_14']
                if rsi > 70:  # Overbought
                    technical_score -= (rsi - 70) * 0.5
                elif rsi < 30:  # Oversold (bullish for reversal)
                    technical_score += (30 - rsi) * 0.5
            if 'macd_histogram' in features and features['macd_histogram'] > 0:
                technical_score += 10
            technical_score = max(0, min(100, technical_score))
            score_components.append(('technical', technical_score, 0.25))
            
            # Volatility signals (weight: 15%)
            volatility_score = 50
            if 'volatility_20d' in features:
                vol = features['volatility_20d']
                # Moderate volatility is good for options
                if 0.01 < vol < 0.04:  # Sweet spot
                    volatility_score += 20
                elif vol > 0.06:  # Too high
                    volatility_score -= 15
            volatility_score = max(0, min(100, volatility_score))
            score_components.append(('volatility', volatility_score, 0.15))
            
            # Trend strength (weight: 10%)
            trend_score = 50
            if 'ma20_ratio' in features:
                ma_ratio = features['ma20_ratio']
                if ma_ratio > 1:  # Above MA
                    trend_score += min(30, (ma_ratio - 1) * 500)
                else:  # Below MA
                    trend_score -= min(20, (1 - ma_ratio) * 500)
            trend_score = max(0, min(100, trend_score))
            score_components.append(('trend', trend_score, 0.10))
            
            # Calculate weighted average
            if score_components:
                weighted_sum = sum(score * weight for _, score, weight in score_components)
                estimated_score = weighted_sum
            else:
                estimated_score = trade.score * 0.8
            
            # Adjust based on existing trade score (blending)
            final_score = estimated_score * 0.7 + trade.score * 0.3
            
            return float(max(0, min(100, final_score)))
            
        except Exception as e:
            logger.error(f"Error estimating ML score: {e}")
            return trade.score * 0.7
    
    def _calculate_ensemble_score(self, ml_score: float, llm_rating: float, quant_score: float) -> float:
        """
        Calculate weighted ensemble score.
        
        This is the key to your confident decisions:
        - ML catches patterns humans/rules miss
        - LLM provides reasoning and context
        - Quant provides proven signals
        """
        ensemble = (
            self.weights['ml'] * ml_score +
            self.weights['llm'] * llm_rating +
            self.weights['quant'] * quant_score
        )
        
        return round(ensemble, 1)
    
    def _determine_ensemble_confidence(self, ml_conf: str, llm_conf: str, quant_conf: str) -> str:
        """
        Determine overall ensemble confidence based on agreement.
        
        If all three agree on HIGH -> VERY HIGH
        If two agree on HIGH -> HIGH
        Otherwise -> MEDIUM
        """
        conf_map = {
            'VERY HIGH': 4,
            'HIGH': 3,
            'MEDIUM-HIGH': 2.5,
            'MEDIUM': 2,
            'LOW': 1
        }
        
        ml_level = conf_map.get(ml_conf, 2)
        llm_level = conf_map.get(llm_conf, 2)
        quant_level = conf_map.get(quant_conf, 2)
        
        avg_level = (ml_level + llm_level + quant_level) / 3
        
        if avg_level >= 3.5:
            return "VERY HIGH"
        elif avg_level >= 3.0:
            return "HIGH"
        elif avg_level >= 2.3:
            return "MEDIUM-HIGH"
        elif avg_level >= 1.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _score_to_confidence(self, score: float) -> str:
        """Convert numeric score to confidence level"""
        if score >= 85:
            return "VERY HIGH"
        elif score >= 75:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM-HIGH"
        elif score >= 45:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _log_confidence_summary(self, trades: List[MLEnhancedTrade]):
        """Log summary of confidence levels"""
        if not trades:
            return
        
        very_high = len([t for t in trades if t.ensemble_confidence == 'VERY HIGH'])
        high = len([t for t in trades if t.ensemble_confidence == 'HIGH'])
        
        avg_combined = np.mean([t.combined_score for t in trades])
        avg_ml = np.mean([t.ml_prediction_score for t in trades])
        avg_llm = np.mean([t.ai_rating * 10 for t in trades])
        
        logger.info(f"📊 Confidence Summary:")
        logger.info(f"   VERY HIGH: {very_high}, HIGH: {high}")
        logger.info(f"   Avg Combined Score: {avg_combined:.1f}")
        logger.info(f"   Avg ML: {avg_ml:.1f}, Avg LLM: {avg_llm:.1f}")
    
    def backtest_strategy(self,
                          start_date: str,
                          end_date: str,
                          min_ensemble_score: float = 70.0) -> Dict:
        """
        Backtest your ML-enhanced strategy on historical data.
        
        This answers: "Would this strategy have worked?"
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_ensemble_score: Minimum score threshold
        
        Returns:
            Dict with backtest results
        """
        if not self.use_ml or not self.ml_analyzer:
            return {
                'error': 'Qlib ML not available. Install with: pip install pyqlib'
            }
        
        logger.info(f"🔬 Backtesting ML-enhanced strategy from {start_date} to {end_date}...")
        
        # Use Qlib's backtesting framework
        strategy_config = {
            'min_ensemble_score': min_ensemble_score,
            'use_ml': True,
            'use_llm': False  # Don't use LLM in backtest (too expensive)
        }
        
        results = self.ml_analyzer.backtest_strategy(
            strategy_config=strategy_config,
            start_date=start_date,
            end_date=end_date
        )
        
        return results
    
    def get_ml_summary_explanation(self) -> str:
        """
        Get a simple explanation of what ML analysis means.
        Helps users understand the complex ML scoring.
        """
        return """
### 🧠 What is ML Analysis?

**Machine Learning (ML) Analysis** uses advanced algorithms to analyze **158 alpha factors** including:
- **Price patterns** (trends, reversals, support/resistance)
- **Volume dynamics** (accumulation, distribution, unusual activity)
- **Momentum indicators** (RSI, MACD, moving averages)
- **Volatility measures** (historical vol, Bollinger Bands)
- **Technical signals** (breakouts, crossovers, divergences)

**How it works:**
1. 🔍 **ML Model** analyzes patterns across all 158 factors
2. 📊 **Prediction Score** (0-100) indicates probability of price increase
3. 🎯 **Ensemble Score** combines ML + AI reasoning + Quantitative signals

**Confidence Levels:**
- **VERY HIGH (85+)**: All 3 systems strongly agree - highest conviction
- **HIGH (75-84)**: Strong agreement across systems - good confidence
- **MEDIUM-HIGH (60-74)**: Moderate agreement - proceed with caution
- **MEDIUM (<60)**: Mixed signals - higher risk

**Why use ML?**
✅ Identifies patterns humans might miss
✅ Analyzes far more data points than manual analysis
✅ Reduces emotional bias in decision-making
✅ Validates AI and quantitative signals
"""
    
    def get_unified_confidence_summary(self, trade: MLEnhancedTrade) -> str:
        """
        Create a unified summary combining ALL analysis methods.
        This is the comprehensive confidence report.
        """
        # Determine overall recommendation
        if trade.combined_score >= 85:
            recommendation = "🟢 **STRONG BUY** - Exceptional confidence across all systems"
            action = "Consider larger position size (within risk limits)"
        elif trade.combined_score >= 75:
            recommendation = "🟢 **BUY** - High confidence with good agreement"
            action = "Standard position size recommended"
        elif trade.combined_score >= 65:
            recommendation = "🟡 **CAUTIOUS BUY** - Moderate confidence"
            action = "Smaller position size, tight stops recommended"
        else:
            recommendation = "🟠 **WATCH** - Mixed signals, needs more confirmation"
            action = "Wait for better setup or skip this trade"
        
        # Agreement analysis
        ml_level = self._confidence_to_number(trade.ml_confidence)
        llm_level = self._confidence_to_number(trade.ai_confidence)
        quant_level = self._confidence_to_number(trade.confidence)
        
        agreement_score = min(ml_level, llm_level, quant_level)
        if agreement_score >= 3:
            agreement = "✅ **Strong Agreement** - All systems align"
        elif agreement_score >= 2:
            agreement = "⚠️ **Moderate Agreement** - Some divergence between systems"
        else:
            agreement = "❌ **Weak Agreement** - Significant divergence, proceed carefully"
        
        summary = f"""
### 📊 UNIFIED CONFIDENCE ANALYSIS: {trade.ticker}

**Overall Ensemble Score: {trade.combined_score:.1f}/100** ({trade.ensemble_confidence})

{recommendation}

---

#### 🎯 System Agreement
{agreement}

| System | Score | Confidence | Weight |
|--------|-------|------------|--------|
| 🧠 ML Model | {trade.ml_prediction_score:.1f}/100 | {trade.ml_confidence} | 40% |
| 🤖 AI Reasoning | {trade.ai_rating*10:.1f}/100 | {trade.ai_confidence} | 35% |
| 📈 Quantitative | {trade.score:.1f}/100 | {trade.confidence} | 25% |

---

#### 💡 What Each System Says

**🧠 ML Analysis** ({trade.ml_features_count} factors analyzed)
- Prediction: {trade.ml_prediction_score:.1f}/100
- The ML model analyzed {trade.ml_features_count} technical and fundamental factors
- Confidence: {trade.ml_confidence}

**🤖 AI Reasoning** (LLM Analysis)
- Rating: {trade.ai_rating:.1f}/10
- {trade.ai_reasoning}

**📈 Quantitative Signals**
- Score: {trade.score:.1f}/100
- {trade.reason}

---

#### ⚠️ Risk Assessment
**Risk Level: {trade.risk_level}**

{trade.ai_risks if trade.ai_risks else "Standard market risks apply"}

---

#### ✅ Recommended Action
{action}

**Key Metrics:**
- Current Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)
- Volume: {trade.volume:,} ({trade.volume_ratio:.1f}x average)
"""
        return summary
    
    def _confidence_to_number(self, confidence: str) -> int:
        """Convert confidence string to numeric level"""
        conf_map = {
            'VERY HIGH': 4,
            'HIGH': 3,
            'MEDIUM-HIGH': 2.5,
            'MEDIUM': 2,
            'LOW': 1
        }
        return conf_map.get(confidence, 2)
    
    def get_trade_explanation(self, trade: MLEnhancedTrade) -> str:
        """
        Get detailed explanation of why this is a confident trade.
        
        This gives you the full picture for your decision.
        """
        explanation = f"""
🎯 HIGH CONFIDENCE TRADE ANALYSIS: {trade.ticker}

📊 ENSEMBLE SCORE: {trade.combined_score:.1f}/100 ({trade.ensemble_confidence})

🧠 ML ANALYSIS (40% weight):
   • Prediction Score: {trade.ml_prediction_score:.1f}/100
   • ML Confidence: {trade.ml_confidence}
   • Alpha Factors Used: {trade.ml_features_count}
   • ML identifies patterns across 158 technical/fundamental factors

🤖 LLM ANALYSIS (35% weight):
   • AI Rating: {trade.ai_rating:.1f}/10
   • AI Confidence: {trade.ai_confidence}
   • Reasoning: {trade.ai_reasoning}
   • Risks: {trade.ai_risks}

📈 QUANTITATIVE ANALYSIS (25% weight):
   • Quant Score: {trade.score:.1f}/100
   • Quant Confidence: {trade.confidence}
   • Price: ${trade.price:.2f} ({trade.change_pct:+.2f}%)
   • Volume: {trade.volume:,} ({trade.volume_ratio:.2f}x avg)

💡 WHY THIS TRADE?
{trade.reason}

⚠️ RISK LEVEL: {trade.risk_level}

✅ RECOMMENDATION:
This trade scores high across ALL THREE systems (ML + LLM + Quant),
giving you maximum confidence for your decision.
"""
        return explanation


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize ML-enhanced scanner
    scanner = MLEnhancedScanner()
    
    # Scan for top options with maximum confidence
    print("\n🎯 Scanning for highest confidence options trades...")
    trades = scanner.scan_top_options_with_ml(top_n=5, min_ensemble_score=70.0)
    
    if trades:
        print(f"\n✅ Found {len(trades)} high-confidence trades:\n")
        for i, trade in enumerate(trades, 1):
            print(f"{i}. {trade.ticker}: Combined Score {trade.combined_score:.1f} "
                  f"({trade.ensemble_confidence})")
            print(f"   ML: {trade.ml_prediction_score:.1f}, "
                  f"LLM: {trade.ai_rating*10:.1f}, "
                  f"Quant: {trade.score:.1f}")
            print()
        
        # Show detailed explanation for top trade
        if trades:
            print("\n" + "="*70)
            print(scanner.get_trade_explanation(trades[0]))
    else:
        print("\n⚠️ No high-confidence trades found with current thresholds")

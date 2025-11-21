"""Trading style-specific analysis for different trading strategies."""

from loguru import logger
from typing import Dict, List, Optional, Tuple
from models.analysis import StockAnalysis
from analyzers.technical import TechnicalAnalyzer
from analyzers.news import NewsAnalyzer
import yfinance as yf
import pandas as pd
import numpy as np



class TradingStyleAnalyzer:
    """Provides specialized analysis for different trading styles"""
    
    @staticmethod
    def analyze_ai_style(analysis: StockAnalysis, hist: pd.DataFrame) -> Dict:
        """AI-powered analysis using machine learning signals"""
        try:
            results = {
                'style': 'AI',
                'score': 0,
                'signals': [],
                'recommendations': [],
                'risk_level': 'MEDIUM',
                'ml_prediction': None,
                'confidence': 0
            }
            
            ml_score = 50
            
            # Momentum Factor
            if abs(analysis.change_pct) > 5:
                ml_score += 20
                results['signals'].append(f"✅ Strong momentum: {analysis.change_pct:+.2f}%")
            elif abs(analysis.change_pct) > 2:
                ml_score += 10
            
            # Volume Profile
            volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 1
            if volume_ratio > 2:
                ml_score += 15
                results['signals'].append(f"✅ Exceptional volume: {volume_ratio:.1f}x")
            elif volume_ratio > 1.5:
                ml_score += 10
            
            # Technical Alignment
            if 30 < analysis.rsi < 70:
                ml_score += 5
                results['signals'].append(f"✅ RSI optimal: {analysis.rsi:.1f}")
            
            if analysis.macd_signal == "BULLISH":
                ml_score += 5
                results['signals'].append("✅ MACD bullish")
            
            # Sentiment
            if analysis.sentiment_score > 0.3:
                ml_score += 15
                results['signals'].append(f"✅ Very positive sentiment: {analysis.sentiment_score:.2f}")
            elif analysis.sentiment_score > 0.1:
                ml_score += 10
            
            ml_score = min(100, ml_score)
            results['score'] = ml_score
            results['confidence'] = ml_score
            
            if ml_score >= 75:
                results['ml_prediction'] = "STRONG BUY"
                results['recommendations'].append("🟢 **AI SIGNAL: STRONG BUY**")
            elif ml_score >= 60:
                results['ml_prediction'] = "BUY"
                results['recommendations'].append("🟢 **AI SIGNAL: BUY**")
            elif ml_score >= 45:
                results['ml_prediction'] = "HOLD"
                results['recommendations'].append("🟡 **AI SIGNAL: HOLD**")
            else:
                results['ml_prediction'] = "SELL"
                results['recommendations'].append("🔴 **AI SIGNAL: SELL**")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {'style': 'AI', 'score': 0, 'signals': [f"❌ Error: {str(e))}"]}
    
    @staticmethod
    def analyze_scalp_style(analysis: StockAnalysis, hist: pd.DataFrame) -> Dict:
        """Scalping analysis for very short-term trades"""
        try:
            results = {
                'style': 'SCALP',
                'score': 0,
                'signals': [],
                'recommendations': [],
                'risk_level': 'VERY HIGH',
                'entry_zones': [],
                'targets': []
            }
            
            scalp_score = 0
            current_price = analysis.price
            
            # Liquidity Check
            volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 1
            if analysis.volume > 1_000_000 and volume_ratio > 1.5:
                scalp_score += 30
                results['signals'].append(f"✅ Excellent liquidity: {analysis.volume:,}")
            elif analysis.volume > 500_000:
                scalp_score += 20
            else:
                results['signals'].append(f"❌ Low liquidity - NOT suitable")
            
            # Volatility
            if abs(analysis.change_pct) > 1:
                scalp_score += 20
                results['signals'].append(f"✅ Strong momentum: {analysis.change_pct:+.2f}%")
            
            # Price Level
            if current_price > 10:
                scalp_score += 15
                results['signals'].append(f"✅ Good price: ${current_price:.2f}")
            
            results['score'] = min(100, scalp_score)
            
            if scalp_score >= 70:
                results['recommendations'].append("🟢 **EXCELLENT for scalping**")
            elif scalp_score >= 50:
                results['recommendations'].append("🟡 **GOOD for scalping**")
            else:
                results['recommendations'].append("🔴 **POOR for scalping**")
            
            # Targets
            target1 = current_price * 1.002
            target2 = current_price * 1.005
            results['targets'] = [
                f"T1: ${target1:.2f} (+0.2%)",
                f"T2: ${target2:.2f} (+0.5%)"
            ]
            
            results['recommendations'].append("⚡ Use 1-5min charts, tight stops (0.2-0.3%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in scalp analysis: {e}")
            return {'style': 'SCALP', 'score': 0, 'signals': [f"❌ Error: {str(e))}"]}
    
    @staticmethod
    def analyze_warrior_scalping_style(analysis: StockAnalysis, hist: pd.DataFrame) -> Dict:
        """Warrior Scalping - aggressive momentum strategy"""
        try:
            results = {
                'style': 'WARRIOR_SCALPING',
                'score': 0,
                'signals': [],
                'recommendations': [],
                'risk_level': 'EXTREME',
                'setup_type': None,
                'targets': []
            }
            
            warrior_score = 0
            current_price = analysis.price
            
            # Gap Analysis
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
                today_open = hist['Open'].iloc[-1]
                gap_pct = ((today_open / prev_close - 1) * 100) if prev_close > 0 else 0
                
                if abs(gap_pct) > 5:
                    warrior_score += 25
                    results['signals'].append(f"✅ MAJOR GAP: {gap_pct:+.2f}%")
                    results['setup_type'] = "GAPPER"
                elif abs(gap_pct) > 2:
                    warrior_score += 15
            
            # Relative Volume
            volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 1
            if volume_ratio > 3:
                warrior_score += 25
                results['signals'].append(f"✅ EXPLOSIVE VOLUME: {volume_ratio:.1f}x")
            elif volume_ratio > 2:
                warrior_score += 20
            
            # Momentum
            if abs(analysis.change_pct) > 10:
                warrior_score += 20
                results['signals'].append(f"✅ EXTREME MOMENTUM: {analysis.change_pct:+.2f}%")
            elif abs(analysis.change_pct) > 5:
                warrior_score += 15
            
            # News Catalyst
            if len(analysis.recent_news) > 5:
                warrior_score += 15
                results['signals'].append(f"✅ HIGH NEWS VOLUME: {len(analysis.recent_news))}")
            
            results['score'] = min(100, warrior_score)
            
            if warrior_score >= 75:
                results['recommendations'].append("🔥 **WARRIOR SETUP CONFIRMED**")
            elif warrior_score >= 60:
                results['recommendations'].append("⚡ **STRONG WARRIOR SETUP**")
            else:
                results['recommendations'].append("❌ **POOR SETUP**")
            
            # Targets
            target1 = current_price * 1.01
            target2 = current_price * 1.02
            target3 = current_price * 1.03
            results['targets'] = [
                f"T1: ${target1:.2f} (+1%)",
                f"T2: ${target2:.2f} (+2%)",
                f"T3: ${target3:.2f} (+3%)"
            ]
            
            results['recommendations'].append("⚔️ Trade first 30-60min, scale out at targets")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in warrior scalping: {e}")
            return {'style': 'WARRIOR_SCALPING', 'score': 0, 'signals': [f"❌ Error: {str(e))}"]}
    
    @staticmethod
    def analyze_buy_and_hold_style(analysis: StockAnalysis, hist: pd.DataFrame) -> Dict:
        """Buy & Hold analysis for long-term investment"""
        try:
            results = {
                'style': 'BUY_AND_HOLD',
                'score': 0,
                'signals': [],
                'recommendations': [],
                'risk_level': 'LOW',
                'valuation': {},
                'targets': []
            }
            
            hold_score = 0
            current_price = analysis.price
            
            # Long-Term Trend
            if len(hist) >= 200:
                ma200 = hist['Close'].tail(200).mean()
                price_vs_ma200 = ((current_price / ma200 - 1) * 100)
                
                if price_vs_ma200 > 10:
                    hold_score += 25
                    results['signals'].append(f"✅ Strong uptrend: {price_vs_ma200:+.1f}% above 200-MA")
                elif price_vs_ma200 > 0:
                    hold_score += 20
                
                results['valuation']['200_day_ma'] = f"${ma200:.2f}"
            
            # Fundamentals
            try:
                ticker_obj = yf.Ticker(analysis.ticker)
                info = ticker_obj.info
                
                pe_ratio = info.get('trailingPE')
                if pe_ratio and 10 <= pe_ratio <= 25:
                    hold_score += 10
                    results['signals'].append(f"✅ Reasonable P/E: {pe_ratio:.2f}")
                    results['valuation']['pe_ratio'] = f"{pe_ratio:.2f}"
                
                market_cap = info.get('marketCap')
                if market_cap:
                    market_cap_b = market_cap / 1e9
                    if market_cap_b > 10:
                        hold_score += 10
                        results['signals'].append(f"✅ Large cap: ${market_cap_b:.1f}B")
                    results['valuation']['market_cap'] = f"${market_cap_b:.2f}B"
                
                dividend_yield = info.get('dividendYield')
                if dividend_yield:
                    dividend_pct = dividend_yield * 100
                    if dividend_pct > 3:
                        hold_score += 5
                        results['signals'].append(f"💰 Strong dividend: {dividend_pct:.2f}%")
                    results['valuation']['dividend_yield'] = f"{dividend_pct:.2f}%"
                
            except Exception as e:
                logger.warning(f"Limited fundamental data: {e}")
            
            # Volatility
            if len(hist) >= 30:
                returns = hist['Close'].pct_change().tail(30)
                volatility = returns.std() * np.sqrt(252) * 100
                
                if volatility < 20:
                    hold_score += 20
                    results['signals'].append(f"✅ Low volatility: {volatility:.1f}%")
                elif volatility < 40:
                    hold_score += 15
            
            # Sentiment
            if analysis.sentiment_score > 0.2:
                hold_score += 15
                results['signals'].append(f"✅ Positive sentiment: {analysis.sentiment_score:.2f}")
            
            results['score'] = min(100, hold_score)
            
            if hold_score >= 75:
                results['recommendations'].append("🟢 **EXCELLENT long-term investment**")
                results['risk_level'] = "LOW"
            elif hold_score >= 60:
                results['recommendations'].append("🟡 **GOOD long-term hold**")
                results['risk_level'] = "MEDIUM"
            else:
                results['recommendations'].append("🔴 **POOR for buy & hold**")
                results['risk_level'] = "HIGH"
            
            # Long-term targets
            target_6m = current_price * 1.10
            target_1y = current_price * 1.20
            target_2y = current_price * 1.50
            results['targets'] = [
                f"6-month: ${target_6m:.2f} (+10%)",
                f"1-year: ${target_1y:.2f} (+20%)",
                f"2-year: ${target_2y:.2f} (+50%)"
            ]
            
            results['recommendations'].append("💎 Hold 6+ months, reinvest dividends, DCA on dips")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in buy & hold analysis: {e}")
            return {'style': 'BUY_AND_HOLD', 'score': 0, 'signals': [f"❌ Error: {str(e))}"]}

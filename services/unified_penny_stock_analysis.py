"""
Unified Penny Stock Analysis Module

Integrates all enhancements:
- ATR-based stops
- Stock classification (not just price)
- ML explainability
- Options liquidity checks
- Catalyst quality analysis
- Stock liquidity checks
- Backtesting validation
"""

from loguru import logger
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from services.penny_stock_analyzer import PennyStockAnalyzer, PennyStockScorer
from services.ml_explainability import MLExplainer, OptionsLiquidityChecker
from services.enhanced_catalyst_detector import EnhancedCatalystDetector, StockLiquidityChecker
from services.penny_stock_backtest import PennyStockBacktester
from analyzers.technical import TechnicalAnalyzer



class UnifiedPennyStockAnalysis:
    """Comprehensive penny stock analysis with all enhancements"""
    
    def __init__(self):
        self.penny_analyzer = PennyStockAnalyzer()
        self.backtester = PennyStockBacktester()
    
    def analyze_comprehensive(self, ticker: str, trading_style: str = "SCALP",
                             include_backtest: bool = False,
                             check_options: bool = False) -> Dict:
        """
        Perform comprehensive penny stock analysis.
        
        Args:
            ticker: Stock ticker
            trading_style: SCALP, SWING, or POSITION
            include_backtest: Whether to run backtest (slower)
            check_options: Whether to check options liquidity
            
        Returns:
            Dict with complete analysis
        """
        try:
            logger.info(f"🔍 UnifiedPennyStockAnalysis.analyze_comprehensive starting for {ticker} (style: {trading_style})")
            
            # 1. Basic penny stock analysis
            logger.info(f"📊 Step 1: Running base penny stock analysis for {ticker}...")
            base_analysis = self.penny_analyzer.analyze_stock(ticker)
            
            if not base_analysis:
                logger.error(f"❌ Base analysis returned None for {ticker}")
                return {'ticker': ticker, 'error': 'Base analysis returned None'}
            
            # Check if result is a dict
            if not isinstance(base_analysis, dict):
                logger.error(f"❌ Base analysis returned non-dict type: {type(base_analysis)}")
                return {'ticker': ticker, 'error': f'Base analysis returned {type(base_analysis)}'}
            
            if 'error' in base_analysis:
                logger.error(f"❌ Base analysis error for {ticker}: {base_analysis['error']}")
                return base_analysis
            
            # Ensure ticker is in result
            if 'ticker' not in base_analysis:
                base_analysis['ticker'] = ticker
            
            logger.info(f"✅ Base analysis completed for {ticker}: price=${base_analysis.get('price', 'N/A')}, classification={base_analysis.get('classification', 'N/A')}")
            
            # 2. Stock classification (using enhanced classifier)
            classification = base_analysis.get('classification', 'UNKNOWN')
            risk_level = base_analysis.get('risk_level', 'UNKNOWN')
            logger.info(f"📋 Step 2: Classification = {classification}, Risk Level = {risk_level}")
            
            # 3. Stock liquidity check
            logger.info(f"💧 Step 3: Checking stock liquidity for {ticker}...")
            try:
                liquidity_check = StockLiquidityChecker.check_stock_liquidity(
                    ticker=ticker,
                    current_price=base_analysis.get('price', 0),
                    volume=base_analysis.get('volume', 0),
                    avg_volume=base_analysis.get('avg_volume', 0)
                )
            except Exception as liq_error:
                logger.warning(f"Liquidity check failed for {ticker}: {liq_error}")
                liquidity_check = {'overall_risk': 'UNKNOWN', 'risk_factors': [], 'warnings': []}
            pass  # logger.info(f"✅ Liquidity check completed: overall_risk={liquidity_check.get('overall_risk', 'N/A'})")
            
            # 4. ATR-based stops (already in base_analysis)
            logger.info(f"🎯 Step 4: Extracting ATR stops from base analysis...")
            atr_stops = {
                'atr_stop_loss': base_analysis.get('atr_stop_loss'),
                'atr_target': base_analysis.get('atr_target'),
                'atr_stop_pct': base_analysis.get('atr_stop_pct'),
                'atr_target_pct': base_analysis.get('atr_target_pct'),
                'atr_risk_reward': base_analysis.get('atr_risk_reward'),
            }
            logger.info(f"✅ ATR stops: stop=${atr_stops.get('atr_stop_loss', 'N/A')}, stop_pct={atr_stops.get('atr_stop_pct', 'N/A')}%")
            
            # 5. Trading recommendation based on stops
            stop_pct = base_analysis.get('atr_stop_pct', 8.0)
            if stop_pct > 12.0:
                stop_recommendation = f"⚠️ WIDE STOP ({stop_pct:.1f}%) - Consider smaller position or skip"
            elif stop_pct > 8.0:
                stop_recommendation = f"⚠️ MODERATE STOP ({stop_pct:.1f}%) - Use defined risk"
            else:
                stop_recommendation = f"✅ TIGHT STOP ({stop_pct:.1f}%) - Good risk/reward"
            
            # 6. Options analysis (if requested)
            options_analysis = None
            if check_options:
                options_analysis = self._analyze_options(ticker, base_analysis)
            
            # 7. Backtest (if requested)
            backtest_results = None
            if include_backtest:
                backtest_results = self._run_backtest(ticker, trading_style)
            
            # 8. ML Explainability (if ML score available)
            ml_explanation = None
            technical_score = base_analysis.get('technical_score', 50)
            ml_score = base_analysis.get('ml_score', technical_score)  # Fallback to technical
            
            if 'ml_features' in base_analysis:
                ml_explanation = MLExplainer.explain_score_divergence(
                    technical_score=technical_score,
                    ml_score=ml_score,
                    features=base_analysis['ml_features']
                )
            
            # 9. Final recommendation
            final_recommendation = self._generate_final_recommendation(
                base_analysis=base_analysis,
                liquidity_check=liquidity_check,
                stop_pct=stop_pct,
                risk_level=risk_level,
                ml_explanation=ml_explanation,
                backtest_results=backtest_results
            )
            
            # Combine everything
            result = {
                **base_analysis,
                'liquidity_check': liquidity_check,
                'stop_recommendation': stop_recommendation,
                'options_analysis': options_analysis,
                'ml_explanation': ml_explanation,
                'backtest_results': backtest_results,
                'final_recommendation': final_recommendation,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ UnifiedPennyStockAnalysis.analyze_comprehensive completed for {ticker}")
            pass  # logger.info(f"   Result keys: {list(result.keys(}... (total: {len(result))} keys)")[:10]))
            logger.info("   Final recommendation: {}", str(final_recommendation.get('decision', 'N/A') if final_recommendation else 'N/A'))
            
            return result
            
        except Exception as e:
            logger.error("❌ ERROR in comprehensive analysis for {ticker}: {}", str(e), exc_info=True)
            return {'ticker': ticker, 'error': str(e)}
    
    def _analyze_options(self, ticker: str, base_analysis: Dict) -> Optional[Dict]:
        """Analyze options liquidity and IV"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get IV rank if available
            iv_rank = base_analysis.get('iv_rank', 50.0)
            
            # Get options chain
            options_dates = stock.options
            if not options_dates:
                return {
                    'available': False,
                    'reason': 'No options available for this stock'
                }
            
            # Get nearest expiration
            opt_chain = stock.option_chain(options_dates[0])
            
            # Check call liquidity
            call_liquidity = OptionsLiquidityChecker.check_options_liquidity(
                opt_chain.calls, option_type='call'
            )
            
            # Check put liquidity
            put_liquidity = OptionsLiquidityChecker.check_options_liquidity(
                opt_chain.puts, option_type='put'
            )
            
            # Get premium strategy recommendation
            premium_rec = OptionsLiquidityChecker.recommend_premium_strategy(
                iv_rank=iv_rank,
                liquidity_check=call_liquidity
            )
            
            return {
                'available': True,
                'iv_rank': iv_rank,
                'call_liquidity': call_liquidity,
                'put_liquidity': put_liquidity,
                'premium_strategy': premium_rec,
                'expiration_dates': list(options_dates)[:5]  # First 5
            }
            
        except Exception as e:
            logger.error(f"Error analyzing options for {ticker}: {e}")
            return {
                'available': False,
                'reason': f'Error: {str(e)}'
            }
    
    def _run_backtest(self, ticker: str, trading_style: str) -> Optional[Dict]:
        """Run backtest for validation"""
        try:
            # Determine backtest parameters based on trading style
            if trading_style == "SCALP":
                stop_pct = 3.0
                target_pct = 6.0
                max_days = 5
            elif trading_style == "SWING":
                stop_pct = 5.0
                target_pct = 12.0
                max_days = 15
            else:  # POSITION
                stop_pct = 8.0
                target_pct = 20.0
                max_days = 30
            
            # Backtest last 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            results = self.backtester.backtest_percentage_strategy(
                ticker=ticker,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                stop_pct=stop_pct,
                target_pct=target_pct,
                max_holding_days=max_days
            )
            
            if results:
                return {
                    'available': True,
                    'win_rate': results.win_rate,
                    'expectancy': results.expectancy,
                    'expectancy_pct': results.expectancy_pct,
                    'profit_factor': results.profit_factor,
                    'max_drawdown': results.max_drawdown,
                    'total_trades': results.total_trades,
                    'is_profitable': results.expectancy > 0,
                    'recommendation': self._interpret_backtest(results)
                }
            else:
                return {'available': False, 'reason': 'Insufficient historical data'}
                
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            return {'available': False, 'reason': f'Error: {str(e)}'}
    
    @staticmethod
    def _interpret_backtest(results) -> str:
        """Interpret backtest results"""
        if results.expectancy > 0 and results.profit_factor > 1.5 and results.win_rate > 50:
            return "✅ STRONG HISTORICAL PERFORMANCE - Strategy validated"
        elif results.expectancy > 0 and results.profit_factor > 1.0:
            return "⚠️ MODERATE HISTORICAL PERFORMANCE - Proceed with caution"
        else:
            return "❌ NEGATIVE HISTORICAL PERFORMANCE - Strategy not validated"
    
    def _generate_final_recommendation(self, base_analysis: Dict, liquidity_check: Dict,
                                      stop_pct: float, risk_level: str,
                                      ml_explanation: Optional[Dict],
                                      backtest_results: Optional[Dict]) -> Dict:
        """Generate final trading recommendation"""
        
        warnings = []
        blockers = []
        signals = []
        
        # Score components
        composite_score = base_analysis.get('composite_score', 0)
        technical_score = base_analysis.get('technical_score', 0)
        
        # 1. Check liquidity blockers
        overall_liq_risk = liquidity_check.get('overall_risk', 'UNKNOWN')
        if overall_liq_risk == "CRITICAL":
            blockers.append("❌ ILLIQUID STOCK - Cannot execute safely")
        elif overall_liq_risk == "HIGH":
            warnings.append("⚠️ Low liquidity - Use limit orders only")
        
        # 2. Check stop width
        if stop_pct > 12.0:
            warnings.append(f"⚠️ Wide stop ({stop_pct:.1f}%) - Conflicts with penny stock best practices")
        
        # 3. Check risk classification
        if risk_level == "VERY_HIGH":
            warnings.append("⚠️ VERY HIGH RISK - Nano-cap or OTC listing")
        
        # 4. Check ML/Technical agreement
        if ml_explanation and ml_explanation['agreement'] == "WEAK":
            warnings.append("⚠️ ML and Technical signals disagree - High uncertainty")
        
        # 5. Check backtest
        if backtest_results and backtest_results.get('available'):
            if not backtest_results['is_profitable']:
                warnings.append("⚠️ Strategy has negative historical expectancy")
            else:
                signals.append(f"✅ Positive historical expectancy: {backtest_results['expectancy_pct']:.1f}%")
        
        # 6. Score-based signals
        if composite_score >= 75:
            signals.append("✅ High composite score")
        elif composite_score < 40:
            blockers.append("❌ Low composite score")
        
        # Final decision
        if blockers:
            decision = "AVOID"
            emoji = "❌"
            reason = " | ".join(blockers)
        elif len(warnings) >= 3:
            decision = "AVOID"
            emoji = "❌"
            reason = "Too many warning signals"
        elif composite_score >= 70 and len(warnings) <= 1:
            decision = "BUY"
            emoji = "✅"
            reason = "Strong signals, manageable risk"
        elif composite_score >= 50:
            decision = "WAIT"
            emoji = "⚠️"
            reason = "Mixed signals - wait for better setup"
        else:
            decision = "AVOID"
            emoji = "❌"
            reason = "Weak signals"
        
        return {
            'decision': decision,
            'emoji': emoji,
            'reason': reason,
            'warnings': warnings,
            'blockers': blockers,
            'signals': signals,
            'composite_score': composite_score,
            'summary': f"{emoji} {decision}: {reason}"
        }


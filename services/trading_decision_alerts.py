"""
Trading Decision Alert System

Helper functions to generate buy/sell/speculation alerts based on
analysis results and market conditions.
"""

import logging
from typing import Optional, Dict
from models.alerts import TradingAlert, AlertType, AlertPriority
from services.alert_system import AlertSystem

logger = logging.getLogger(__name__)


class TradingDecisionAlerts:
    """Generates trading decision alerts for buy/sell/speculation opportunities"""
    
    def __init__(self, alert_system: AlertSystem):
        """
        Initialize trading decision alerts
        
        Args:
            alert_system: AlertSystem instance for notifications
        """
        self.alert_system = alert_system
    
    def create_buy_signal(self, ticker: str, analysis, entry_price: float,
                         target_price: float, stop_loss: float,
                         reasoning: str = "") -> TradingAlert:
        """
        Create a BUY signal alert
        
        Args:
            ticker: Ticker symbol
            analysis: Analysis object with confidence score
            entry_price: Recommended entry price
            target_price: Target price
            stop_loss: Stop loss price
            reasoning: Explanation for the buy signal
            
        Returns:
            TradingAlert instance
        """
        risk = entry_price - stop_loss
        reward = target_price - entry_price
        risk_reward = reward / risk if risk > 0 else 0
        
        # Determine priority based on confidence
        confidence = getattr(analysis, 'confidence_score', 0)
        if confidence >= 90:
            priority = AlertPriority.CRITICAL
        elif confidence >= 80:
            priority = AlertPriority.HIGH
        elif confidence >= 70:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW
        
        # Build message
        message = f"🟢 BUY SIGNAL - Entry: ${entry_price:.2f} | Target: ${target_price:.2f} | Stop: ${stop_loss:.2f} | R/R: {risk_reward:.2f}:1"
        
        alert = TradingAlert(
            ticker=ticker,
            alert_type=AlertType.BUY_SIGNAL,
            priority=priority,
            message=message,
            confidence_score=confidence,
            details={
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'risk_reward': risk_reward,
                'position_size': self._calculate_position_size(risk_reward, confidence),
                'reasoning': reasoning or self._build_reasoning(analysis, 'buy')
            }
        )
        
        self.alert_system.trigger_alert(alert)
        return alert
    
    def create_sell_signal(self, ticker: str, current_price: float,
                          profit_pct: float, reasoning: str = "") -> TradingAlert:
        """
        Create a SELL signal alert
        
        Args:
            ticker: Ticker symbol
            current_price: Current price
            profit_pct: Profit percentage
            reasoning: Explanation for the sell signal
            
        Returns:
            TradingAlert instance
        """
        # Determine priority based on profit
        if profit_pct >= 20:
            priority = AlertPriority.HIGH
        elif profit_pct >= 10:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW
        
        # Build message
        message = f"🔴 SELL SIGNAL - Current: ${current_price:.2f} | Profit: {profit_pct:+.1f}%"
        
        alert = TradingAlert(
            ticker=ticker,
            alert_type=AlertType.SELL_SIGNAL,
            priority=priority,
            message=message,
            confidence_score=0.0,
            details={
                'current_price': current_price,
                'profit_pct': profit_pct,
                'reasoning': reasoning or "Take profit signal triggered"
            }
        )
        
        self.alert_system.trigger_alert(alert)
        return alert
    
    def create_speculation_opportunity(self, ticker: str, analysis,
                                      entry_price: float, target_price: float,
                                      stop_loss: float, reasoning: str = "") -> TradingAlert:
        """
        Create a SPECULATION opportunity alert (higher risk)
        
        Args:
            ticker: Ticker symbol
            analysis: Analysis object
            entry_price: Recommended entry price
            target_price: Target price
            stop_loss: Stop loss price
            reasoning: Explanation for the speculation
            
        Returns:
            TradingAlert instance
        """
        risk = entry_price - stop_loss
        reward = target_price - entry_price
        risk_reward = reward / risk if risk > 0 else 0
        
        confidence = getattr(analysis, 'confidence_score', 0)
        
        # Speculation is always MEDIUM or LOW priority
        priority = AlertPriority.MEDIUM if confidence >= 70 else AlertPriority.LOW
        
        # Build message
        message = f"🟡 SPECULATION - High Risk/Reward Setup | Entry: ${entry_price:.2f} | Target: ${target_price:.2f} | R/R: {risk_reward:.2f}:1"
        
        alert = TradingAlert(
            ticker=ticker,
            alert_type=AlertType.SPECULATION_OPPORTUNITY,
            priority=priority,
            message=message,
            confidence_score=confidence,
            details={
                'entry_price': entry_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'risk_reward': risk_reward,
                'position_size': "Small (1-2% of portfolio)",
                'reasoning': reasoning or "Speculative setup with high potential"
            }
        )
        
        self.alert_system.trigger_alert(alert)
        return alert
    
    def create_review_required(self, ticker: str, analysis,
                              reason: str, priority: AlertPriority = AlertPriority.MEDIUM) -> TradingAlert:
        """
        Create a REVIEW REQUIRED alert for manual evaluation
        
        Args:
            ticker: Ticker symbol
            analysis: Analysis object
            reason: Reason for manual review
            priority: Alert priority
            
        Returns:
            TradingAlert instance
        """
        confidence = getattr(analysis, 'confidence_score', 0)
        
        # Build message
        message = f"👀 REVIEW REQUIRED - {reason}"
        
        alert = TradingAlert(
            ticker=ticker,
            alert_type=AlertType.REVIEW_REQUIRED,
            priority=priority,
            message=message,
            confidence_score=confidence,
            details={
                'reason': reason,
                'price': getattr(analysis, 'price', 0),
                'rsi': getattr(analysis, 'rsi', 0),
                'trend': getattr(analysis, 'trend', 'Unknown'),
                'reasoning': self._build_reasoning(analysis, 'review')
            }
        )
        
        self.alert_system.trigger_alert(alert)
        return alert
    
    def _calculate_position_size(self, risk_reward: float, confidence: float) -> str:
        """
        Calculate recommended position size based on risk/reward and confidence
        
        Args:
            risk_reward: Risk/reward ratio
            confidence: Confidence score
            
        Returns:
            Position size recommendation string
        """
        if confidence >= 90 and risk_reward >= 3:
            return "Full (5-10% of portfolio)"
        elif confidence >= 80 and risk_reward >= 2:
            return "Large (3-5% of portfolio)"
        elif confidence >= 70 and risk_reward >= 1.5:
            return "Medium (2-3% of portfolio)"
        else:
            return "Small (1-2% of portfolio)"
    
    def _build_reasoning(self, analysis, signal_type: str) -> str:
        """
        Build reasoning text from analysis
        
        Args:
            analysis: Analysis object
            signal_type: Type of signal ('buy', 'sell', 'review')
            
        Returns:
            Reasoning string
        """
        reasons = []
        
        if signal_type == 'buy':
            if getattr(analysis, 'ema_reclaim', False):
                reasons.append("EMA reclaim confirmed")
            
            if hasattr(analysis, 'timeframe_alignment') and analysis.timeframe_alignment:
                if analysis.timeframe_alignment.get('aligned'):
                    reasons.append(f"Multi-timeframe aligned ({analysis.timeframe_alignment.get('alignment_score', 0):.0f}%)")
            
            if hasattr(analysis, 'sector_rs') and analysis.sector_rs:
                rs_score = analysis.sector_rs.get('rs_score', 0)
                if rs_score > 70:
                    reasons.append(f"Strong sector leader (RS: {rs_score:.0f})")
            
            if hasattr(analysis, 'demarker') and analysis.demarker:
                if analysis.demarker <= 0.30:
                    reasons.append(f"DeMarker oversold ({analysis.demarker:.2f})")
            
            if hasattr(analysis, 'rsi') and analysis.rsi:
                if 30 <= analysis.rsi <= 40:
                    reasons.append(f"RSI in buy zone ({analysis.rsi:.0f})")
        
        elif signal_type == 'review':
            confidence = getattr(analysis, 'confidence_score', 0)
            reasons.append(f"Confidence: {confidence:.0f}%")
            
            if hasattr(analysis, 'trend'):
                reasons.append(f"Trend: {analysis.trend}")
            
            if hasattr(analysis, 'rsi') and analysis.rsi:
                reasons.append(f"RSI: {analysis.rsi:.0f}")
        
        return " | ".join(reasons) if reasons else "Multiple technical indicators aligned"
    
    def analyze_and_alert(self, ticker: str, analysis, current_price: float) -> Optional[TradingAlert]:
        """
        Analyze a stock and automatically generate appropriate alert
        
        Args:
            ticker: Ticker symbol
            analysis: Analysis object
            current_price: Current stock price
            
        Returns:
            TradingAlert if generated, None otherwise
        """
        confidence = getattr(analysis, 'confidence_score', 0)
        
        # High confidence buy signal
        if confidence >= 85 and getattr(analysis, 'ema_reclaim', False):
            # Calculate targets based on Fibonacci or ATR
            target_price = current_price * 1.10  # 10% target
            stop_loss = current_price * 0.95     # 5% stop
            
            return self.create_buy_signal(
                ticker=ticker,
                analysis=analysis,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss
            )
        
        # Medium confidence - review required
        elif 70 <= confidence < 85:
            return self.create_review_required(
                ticker=ticker,
                analysis=analysis,
                reason="Good setup but needs confirmation",
                priority=AlertPriority.MEDIUM
            )
        
        # Speculative opportunity
        elif 60 <= confidence < 70 and getattr(analysis, 'demarker', 1) <= 0.30:
            target_price = current_price * 1.15  # 15% target
            stop_loss = current_price * 0.93     # 7% stop
            
            return self.create_speculation_opportunity(
                ticker=ticker,
                analysis=analysis,
                entry_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning="DeMarker oversold with moderate confidence"
            )
        
        return None


# Convenience function
def get_trading_decision_alerts(alert_system: AlertSystem) -> TradingDecisionAlerts:
    """
    Get TradingDecisionAlerts instance
    
    Args:
        alert_system: AlertSystem instance
        
    Returns:
        TradingDecisionAlerts instance
    """
    return TradingDecisionAlerts(alert_system)

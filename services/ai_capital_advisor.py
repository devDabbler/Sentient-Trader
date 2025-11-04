"""
AI Capital Advisor - Dynamic Position Sizing & Capital Allocation

Uses AI to:
1. Assess available capital and risk capacity
2. Analyze trade quality and opportunity strength
3. Recommend optimal position sizes dynamically
4. Adjust allocations based on portfolio state and market conditions
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class CapitalRecommendation:
    """AI-generated capital allocation recommendation"""
    ticker: str
    recommended_position_size_pct: float  # % of total capital
    recommended_position_value: float  # Dollar amount
    recommended_shares: int  # Number of shares
    confidence_adjustment: float  # Multiplier based on signal quality
    risk_adjustment: float  # Multiplier based on risk assessment
    reasoning: str  # AI explanation
    warnings: List[str]  # Any concerns or caveats
    
    @property
    def is_approved(self) -> bool:
        """Whether AI approves this allocation"""
        return len(self.warnings) == 0 and self.recommended_shares > 0


class AICapitalAdvisor:
    """
    AI-powered capital allocation advisor
    
    Dynamically adjusts position sizes based on:
    - Available capital and reserves
    - Trade confidence and quality
    - Current portfolio exposure
    - Risk/reward profile
    - Market conditions
    """
    
    def __init__(self, llm_analyzer=None):
        """
        Initialize AI Capital Advisor
        
        Args:
            llm_analyzer: LLM strategy analyzer for AI reasoning
        """
        self.llm_analyzer = llm_analyzer
        
        # Default allocation rules (can be overridden by AI)
        self.base_position_pct = 10.0  # Base position size
        self.min_position_pct = 5.0    # Minimum position
        self.max_position_pct = 20.0   # Maximum position
        
        # Confidence-based adjustments
        self.confidence_multipliers = {
            'VERY_HIGH': 1.5,   # 90%+ confidence
            'HIGH': 1.25,       # 75-90% confidence
            'MEDIUM': 1.0,      # 60-75% confidence
            'LOW': 0.75,        # 50-60% confidence
            'VERY_LOW': 0.5     # <50% confidence
        }
        
        # Risk-based adjustments
        self.risk_multipliers = {
            'L': 1.2,   # Low risk - increase position
            'M': 1.0,   # Medium risk - standard position
            'H': 0.7,   # High risk - reduce position
            'VH': 0.5   # Very high risk - minimal position
        }
    
    def recommend_position_size(
        self,
        ticker: str,
        price: float,
        signal_confidence: float,
        risk_level: str,
        available_capital: float,
        total_capital: float,
        current_positions: int = 0,
        max_positions: int = 10,
        ensemble_score: float = None,
        ai_reasoning: str = None,
        use_ai_reasoning: bool = True
    ) -> CapitalRecommendation:
        """
        Get AI recommendation for position size
        
        Args:
            ticker: Stock symbol
            price: Current stock price
            signal_confidence: Trading signal confidence (0-100)
            risk_level: Risk assessment (L/M/H/VH)
            available_capital: Available cash
            total_capital: Total portfolio value
            current_positions: Number of open positions
            max_positions: Maximum allowed positions
            ensemble_score: ML+LLM+Quant combined score
            ai_reasoning: AI analysis of the trade
            use_ai_reasoning: Whether to use LLM for dynamic sizing
        
        Returns:
            CapitalRecommendation with sizing details
        """
        warnings = []
        
        # Step 1: Calculate base position size
        base_pct = self._calculate_base_position_pct(
            current_positions, max_positions, available_capital, total_capital
        )
        
        # Step 2: Apply confidence adjustment
        confidence_level = self._get_confidence_level(signal_confidence)
        confidence_mult = self.confidence_multipliers.get(confidence_level, 1.0)
        
        # Step 3: Apply risk adjustment
        risk_mult = self.risk_multipliers.get(risk_level, 1.0)
        
        # Step 4: Calculate adjusted position percentage
        adjusted_pct = base_pct * confidence_mult * risk_mult
        
        # Step 5: Apply limits
        adjusted_pct = max(self.min_position_pct, min(adjusted_pct, self.max_position_pct))
        
        # Step 6: Calculate dollar amount and shares
        position_value = (adjusted_pct / 100.0) * total_capital
        position_value = min(position_value, available_capital)  # Can't exceed available
        
        shares = int(position_value / price) if price > 0 else 0
        actual_value = shares * price
        
        # Step 7: Validate affordability
        if actual_value > available_capital:
            shares = int(available_capital / price)
            actual_value = shares * price
            adjusted_pct = (actual_value / total_capital) * 100
            warnings.append(f"Position reduced to fit available capital (${available_capital:.2f})")
        
        if shares == 0:
            warnings.append(f"Stock price ${price:.2f} too high for available capital ${available_capital:.2f}")
        
        # Step 8: Check minimum viable position
        if actual_value < 50:  # Less than $50 position
            warnings.append(f"Position value ${actual_value:.2f} too small (min $50 recommended)")
        
        # Step 9: Get AI reasoning if available
        reasoning = self._generate_reasoning(
            ticker, price, shares, actual_value, adjusted_pct,
            signal_confidence, risk_level, confidence_mult, risk_mult,
            available_capital, total_capital, current_positions, max_positions,
            ensemble_score, ai_reasoning, use_ai_reasoning
        )
        
        return CapitalRecommendation(
            ticker=ticker,
            recommended_position_size_pct=adjusted_pct,
            recommended_position_value=actual_value,
            recommended_shares=shares,
            confidence_adjustment=confidence_mult,
            risk_adjustment=risk_mult,
            reasoning=reasoning,
            warnings=warnings
        )
    
    def _calculate_base_position_pct(
        self,
        current_positions: int,
        max_positions: int,
        available_capital: float,
        total_capital: float
    ) -> float:
        """
        Calculate base position size based on portfolio state
        
        Strategy:
        - Fewer positions = larger individual positions
        - More positions = smaller individual positions
        - Low capital = conservative sizing
        """
        # Calculate ideal position size for diversification
        if max_positions > 0:
            ideal_pct = 100.0 / max_positions
        else:
            ideal_pct = self.base_position_pct
        
        # Adjust based on current exposure
        if current_positions > 0:
            # Already have positions, be more conservative
            remaining_slots = max_positions - current_positions
            if remaining_slots > 0:
                available_pct = (available_capital / total_capital) * 100
                ideal_pct = min(ideal_pct, available_pct / remaining_slots)
        
        # Apply capital utilization factor
        utilization = ((total_capital - available_capital) / total_capital) * 100
        if utilization > 80:
            # Portfolio highly utilized, reduce new positions
            ideal_pct *= 0.7
        elif utilization < 20:
            # Portfolio underutilized, can be more aggressive
            ideal_pct *= 1.2
        
        return max(self.min_position_pct, min(ideal_pct, self.max_position_pct))
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Map confidence score to level"""
        if confidence >= 90:
            return 'VERY_HIGH'
        elif confidence >= 75:
            return 'HIGH'
        elif confidence >= 60:
            return 'MEDIUM'
        elif confidence >= 50:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _generate_reasoning(
        self,
        ticker: str,
        price: float,
        shares: int,
        value: float,
        pct: float,
        confidence: float,
        risk: str,
        conf_mult: float,
        risk_mult: float,
        available: float,
        total: float,
        positions: int,
        max_pos: int,
        ensemble_score: float,
        ai_reasoning: str,
        use_ai: bool
    ) -> str:
        """Generate explanation for position sizing"""
        
        if use_ai and self.llm_analyzer and ai_reasoning:
            # Use LLM for dynamic reasoning
            return self._get_ai_reasoning(
                ticker, price, shares, value, pct, confidence, risk,
                available, total, positions, max_pos, ensemble_score, ai_reasoning
            )
        else:
            # Use rule-based reasoning
            return self._get_rule_based_reasoning(
                ticker, shares, value, pct, confidence, risk,
                conf_mult, risk_mult, available, total, positions
            )
    
    def _get_rule_based_reasoning(
        self,
        ticker: str,
        shares: int,
        value: float,
        pct: float,
        confidence: float,
        risk: str,
        conf_mult: float,
        risk_mult: float,
        available: float,
        total: float,
        positions: int
    ) -> str:
        """Generate rule-based reasoning"""
        
        parts = []
        parts.append(f"Position: {shares} shares @ ${value:.2f} ({pct:.1f}% of portfolio)")
        
        if conf_mult != 1.0:
            direction = "increased" if conf_mult > 1.0 else "decreased"
            parts.append(f"Confidence adjustment: {direction} by {conf_mult:.1%} (signal: {confidence:.1f}%)")
        
        if risk_mult != 1.0:
            direction = "reduced" if risk_mult < 1.0 else "increased"
            parts.append(f"Risk adjustment: {direction} by {abs(1-risk_mult):.1%} (risk: {risk})")
        
        utilization = ((total - available) / total) * 100
        parts.append(f"Portfolio utilization: {utilization:.1f}% ({positions} positions)")
        
        return " | ".join(parts)
    
    def _get_ai_reasoning(
        self,
        ticker: str,
        price: float,
        shares: int,
        value: float,
        pct: float,
        confidence: float,
        risk: str,
        available: float,
        total: float,
        positions: int,
        max_pos: int,
        ensemble_score: float,
        ai_reasoning: str
    ) -> str:
        """Get AI-generated reasoning for position sizing"""
        
        if not self.llm_analyzer:
            return self._get_rule_based_reasoning(
                ticker, shares, value, pct, confidence, risk,
                1.0, 1.0, available, total, positions
            )
        
        try:
            prompt = f"""You are a capital allocation advisor. Analyze this trade and explain the recommended position size.

TRADE DETAILS:
- Ticker: {ticker}
- Price: ${price:.2f}
- Signal Confidence: {confidence:.1f}%
- Ensemble Score: {ensemble_score:.1f}% (ML+LLM+Quant)
- Risk Level: {risk}

PORTFOLIO STATE:
- Total Capital: ${total:.2f}
- Available Capital: ${available:.2f}
- Current Positions: {positions}/{max_pos}
- Utilization: {((total-available)/total*100):.1f}%

RECOMMENDATION:
- Shares: {shares}
- Position Value: ${value:.2f}
- Position Size: {pct:.1f}% of portfolio

AI TRADE ANALYSIS:
{ai_reasoning[:500]}

Provide a 1-2 sentence explanation of why this position size is appropriate given the trade quality, available capital, and portfolio state. Be concise and actionable."""

            response = self.llm_analyzer.analyze_with_llm(prompt)
            
            if response and len(response) > 20:
                return response[:300]  # Limit length
            
        except Exception as e:
            logger.error(f"Error getting AI reasoning: {e}")
        
        # Fallback to rule-based
        return self._get_rule_based_reasoning(
            ticker, shares, value, pct, confidence, risk,
            1.0, 1.0, available, total, positions
        )
    
    def assess_portfolio_capacity(
        self,
        total_capital: float,
        available_capital: float,
        current_positions: int,
        max_positions: int,
        reserved_pct: float = 5.0
    ) -> Dict:
        """
        Assess overall portfolio capacity for new trades
        
        Returns:
            Dict with capacity metrics and recommendations
        """
        utilization = ((total_capital - available_capital) / total_capital) * 100
        reserved_amount = (reserved_pct / 100) * total_capital
        usable_capital = available_capital - reserved_amount
        
        position_slots_used = (current_positions / max_positions) * 100 if max_positions > 0 else 0
        
        # Determine capacity status
        if usable_capital < 50:
            status = "CRITICAL"
            message = "Insufficient capital for new positions"
        elif utilization > 90:
            status = "FULL"
            message = "Portfolio near maximum utilization"
        elif position_slots_used > 90:
            status = "LIMITED"
            message = "Near maximum position count"
        elif utilization > 70:
            status = "MODERATE"
            message = "Moderate capacity for selective trades"
        else:
            status = "GOOD"
            message = "Good capacity for new positions"
        
        return {
            'status': status,
            'message': message,
            'total_capital': total_capital,
            'available_capital': available_capital,
            'usable_capital': max(0, usable_capital),
            'reserved_capital': reserved_amount,
            'utilization_pct': utilization,
            'positions_used': current_positions,
            'positions_available': max(0, max_positions - current_positions),
            'position_slots_pct': position_slots_used,
            'can_trade': usable_capital >= 50 and current_positions < max_positions
        }

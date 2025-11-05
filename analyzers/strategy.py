"""Strategy recommendation engine for options trading."""

from loguru import logger
from typing import List, Optional
from models.analysis import StockAnalysis, StrategyRecommendation
from models.reddit_strategies import (
    get_all_custom_strategies, 
    get_strategies_by_experience,
    CustomStrategy
)



class StrategyAdvisor:
    """Intelligent strategy recommendation engine"""
    
    STRATEGIES = {
        "SELL_PUT": {
            "name": "Cash-Secured Put (Sell Put)",
            "description": "Sell a put option to collect premium. You're obligated to buy the stock if it drops below the strike.",
            "risk_level": "Medium",
            "max_loss": "Strike price - Premium received",
            "max_gain": "Premium received (limited)",
            "best_for": ["Bullish or neutral outlook", "High IV", "Want to own stock at lower price"],
            "experience": "Beginner-Friendly",
            "capital_req": "High (need cash to secure)",
            "typical_win_rate": "65-75%",
            "examples": ["Sell 1x 30D put at strike - collect premium", "Cash-secured with 100 shares worth allocation"],
            "notes": "Good income strategy if you're willing to own the stock; monitor assignment risk around earnings.",
            "example_trade": {"dte": 30, "strike_offset_pct": -0.05, "qty": 2, "estimated_risk": 300}
        },
        "SELL_CALL": {
            "name": "Covered Call (Sell Call)",
            "description": "Sell a call option against stock you own to collect premium. Stock may be called away if price rises.",
            "risk_level": "Low-Medium",
            "max_loss": "Unlimited if stock drops (but you own stock)",
            "max_gain": "Premium + (Strike - Stock Purchase Price)",
            "best_for": ["Own the stock", "Neutral to slightly bullish", "Generate income"],
            "experience": "Beginner-Friendly",
            "capital_req": "High (need to own 100 shares)",
            "typical_win_rate": "70-80%"
        },
        "BUY_CALL": {
            "name": "Long Call (Buy Call)",
            "description": "Buy a call option for the right to buy stock at strike price. Bullish directional bet.",
            "risk_level": "Medium-High",
            "max_loss": "Premium paid (limited)",
            "max_gain": "Unlimited",
            "best_for": ["Strong bullish conviction", "Low to medium IV", "Limited capital for directional bet"],
            "experience": "Beginner-Friendly",
            "capital_req": "Low-Medium",
            "typical_win_rate": "30-45%",
            "examples": ["Buy a near-the-money call 30D for directional upside"],
            "notes": "Good for directional bullish bets when you expect a move.",
            "example_trade": {"dte": 30, "strike_offset_pct": 0.02, "qty": 1, "estimated_risk": 150}
        },
        "BUY_PUT": {
            "name": "Long Put (Buy Put)",
            "description": "Buy a put option for the right to sell stock at strike price. Bearish directional bet or hedge.",
            "risk_level": "Medium-High",
            "max_loss": "Premium paid (limited)",
            "max_gain": "Strike price - Premium (large potential)",
            "best_for": ["Bearish conviction", "Portfolio hedge", "Low to medium IV"],
            "experience": "Beginner-Friendly",
            "capital_req": "Low-Medium",
            "typical_win_rate": "30-45%"
        },
        "IRON_CONDOR": {
            "name": "Iron Condor",
            "description": "Sell both a put spread and call spread. Profit if stock stays in a range between strikes.",
            "risk_level": "Medium",
            "max_loss": "Width of spread - Net credit received",
            "max_gain": "Net credit received (limited)",
            "best_for": ["Expect low movement", "High IV", "Range-bound stocks"],
            "experience": "Intermediate",
            "capital_req": "Medium",
            "typical_win_rate": "60-70%",
            "examples": ["Sell 30D iron condor 5-10% OTM wings"],
            "notes": "Best used on range-bound stocks with high IV.",
            "example_trade": {"dte": 30, "wing_width_pct": 0.05, "qty": 1, "estimated_risk": 500}
        },
        "CREDIT_SPREAD": {
            "name": "Credit Spread (Bull Put or Bear Call)",
            "description": "Sell a spread to collect credit. Bull put spread = bullish, Bear call spread = bearish.",
            "risk_level": "Medium",
            "max_loss": "Width of spread - Net credit",
            "max_gain": "Net credit received (limited)",
            "best_for": ["Directional bias with defined risk", "High IV", "Want better probability than buying options"],
            "experience": "Intermediate",
            "capital_req": "Medium",
            "typical_win_rate": "60-70%"
        },
        "DEBIT_SPREAD": {
            "name": "Debit Spread (Bull Call or Bear Put)",
            "description": "Buy a spread to reduce cost. Bull call spread = bullish, Bear put spread = bearish.",
            "risk_level": "Medium",
            "max_loss": "Net debit paid (limited)",
            "max_gain": "Width of spread - Net debit",
            "best_for": ["Directional bias", "Lower cost than buying single option", "Moderate IV"],
            "experience": "Intermediate",
            "capital_req": "Low-Medium",
            "typical_win_rate": "40-55%"
        },
        "LONG_STRADDLE": {
            "name": "Long Straddle",
            "description": "Buy both a call and put at the same strike. Profit from big move in either direction.",
            "risk_level": "High",
            "max_loss": "Total premium paid for both options",
            "max_gain": "Unlimited (if big move occurs)",
            "best_for": ["Expect big move but unsure of direction", "Low IV before event", "Earnings plays"],
            "experience": "Advanced",
            "capital_req": "Medium-High",
            "typical_win_rate": "35-50%",
            "examples": ["Buy 30D ATM straddle into earnings"],
            "notes": "High cost; works best if large move expected.",
            "example_trade": {"dte": 30, "strike_offset_pct": 0.0, "qty": 1, "estimated_risk": 800}
        },
        "WHEEL_STRATEGY": {
            "name": "The Wheel Strategy",
            "description": "Sell puts until assigned, then sell calls against the stock. Repeat.",
            "risk_level": "Medium",
            "max_loss": "Stock value decline",
            "max_gain": "Premium collected consistently",
            "best_for": ["Generate steady income", "Willing to own stock", "High IV stocks"],
            "experience": "Intermediate",
            "capital_req": "High",
            "typical_win_rate": "70-80%",
            "examples": ["Sell puts until assigned, then sell covered calls"],
            "notes": "Good income strategy but requires capital planning.",
            "example_trade": {"dte": 45, "strike_offset_pct": -0.08, "qty": 1, "estimated_risk": 1000}
        },
        "SHORT_STRANGLE": {
            "name": "Short Strangle",
            "description": "Sell an OTM call and an OTM put to collect premium; profit if the stock stays within the range.",
            "risk_level": "High",
            "max_loss": "Potentially large (if stock gaps large), defined only with additional hedges",
            "max_gain": "Net premium received",
            "best_for": ["Expect low movement", "High IV", "Range-bound markets"],
            "experience": "Advanced",
            "capital_req": "High",
            "typical_win_rate": "50-65%",
            "examples": ["Sell 30D 1.05x call and 0.95x put at same expiry"],
            "notes": "Requires active management and margin; consider hedges or defined-risk modifications."
        },
        "CALENDAR_SPREAD": {
            "name": "Calendar Spread",
            "description": "Buy longer-dated option and sell shorter-dated option at same strike to play for theta on the front leg.",
            "risk_level": "Medium",
            "max_loss": "Premium paid for long leg",
            "max_gain": "Variable (depends on front-month decay and move)",
            "best_for": ["Expect limited near-term movement", "Low to moderate IV", "Earnings or event timing plays"],
            "experience": "Intermediate",
            "capital_req": "Low-Medium",
            "typical_win_rate": "40-60%",
            "examples": ["Buy 60D call, sell 30D call at same strike"],
            "notes": "Works best when front-month premium decays faster than back-month."
        },
        "PUT_DIAGONAL": {
            "name": "Put Diagonal Spread",
            "description": "Buy longer-dated put and sell shorter-dated put at different strikes to create a defined-risk bearish income trade.",
            "risk_level": "Medium-High",
            "max_loss": "Net debit paid",
            "max_gain": "Difference between strikes minus net debit",
            "best_for": ["Mildly bearish outlook", "Want defined risk", "Use when IV term-structure favors front-month"],
            "experience": "Intermediate",
            "capital_req": "Low-Medium",
            "typical_win_rate": "40-55%",
            "examples": ["Buy 90D put, sell 30D put at higher strike"],
            "notes": "Can be adjusted into defined-risk hedges if market moves quickly."
        }
    }
    
    @classmethod
    def get_recommendations(cls, analysis: StockAnalysis, 
                          user_experience: str, risk_tolerance: str,
                          capital_available: float, outlook: str) -> List[StrategyRecommendation]:
        """Generate personalized strategy recommendations"""
        
        recommendations = []
        
        # Score each strategy
        for strategy_key, strategy_info in cls.STRATEGIES.items():
            score = 0
            reasoning_parts = []
            
            # Experience level filter
            if user_experience == "Beginner" and strategy_info["experience"] == "Advanced":
                continue
            if user_experience == "Beginner" and strategy_info["experience"] == "Intermediate":
                score -= 20
            
            # Risk tolerance filter
            if risk_tolerance == "Conservative" and strategy_info["risk_level"] in ["High", "Medium-High"]:
                score -= 30
            if risk_tolerance == "Aggressive" and strategy_info["risk_level"] == "Low":
                score -= 10
            
            # Capital requirements
            capital_req = strategy_info.get("capital_req", "Medium")
            if capital_available < 5000 and capital_req == "High":
                score -= 40
                reasoning_parts.append("⚠️ May require more capital than available")
            
            # IV considerations
            if analysis.iv_rank > 60:
                if "High IV" in strategy_info["best_for"]:
                    score += 30
                    reasoning_parts.append(f"✅ High IV Rank ({analysis.iv_rank}%) - premium selling favorable")
                if strategy_key in ["SELL_PUT", "SELL_CALL", "IRON_CONDOR", "CREDIT_SPREAD"]:
                    score += 25
            elif analysis.iv_rank < 40:
                if strategy_key in ["BUY_CALL", "BUY_PUT", "DEBIT_SPREAD"]:
                    score += 25
                    reasoning_parts.append(f"✅ Low IV Rank ({analysis.iv_rank}%) - option buying favorable")
            
            # Market outlook alignment
            if outlook == "Bullish":
                if strategy_key in ["SELL_PUT", "BUY_CALL", "CREDIT_SPREAD"]:
                    score += 25
                    reasoning_parts.append("✅ Aligns with bullish outlook")
            elif outlook == "Bearish":
                if strategy_key in ["BUY_PUT", "CREDIT_SPREAD"]:
                    score += 25
                    reasoning_parts.append("✅ Aligns with bearish outlook")
            elif outlook == "Neutral":
                if strategy_key in ["IRON_CONDOR", "SELL_CALL", "WHEEL_STRATEGY"]:
                    score += 25
                    reasoning_parts.append("✅ Good for neutral/range-bound markets")
            
            # Technical indicators
            if analysis.rsi < 30 and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 15
                reasoning_parts.append(f"✅ RSI oversold ({analysis.rsi}) - potential bounce")
            elif analysis.rsi > 70 and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 15
                reasoning_parts.append(f"✅ RSI overbought ({analysis.rsi}) - potential pullback")
            
            if analysis.macd_signal == "BULLISH" and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 10
                reasoning_parts.append("✅ MACD bullish crossover")
            elif analysis.macd_signal == "BEARISH" and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 10
                reasoning_parts.append("✅ MACD bearish crossover")
            
            # Trend alignment
            if "UPTREND" in analysis.trend and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 15
                reasoning_parts.append(f"✅ Stock in {analysis.trend}")
            elif "DOWNTREND" in analysis.trend and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 15
                reasoning_parts.append(f"✅ Stock in {analysis.trend}")
            
            # Sentiment
            if analysis.sentiment_score > 0.3 and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 10
                reasoning_parts.append("✅ Positive news sentiment")
            elif analysis.sentiment_score < -0.3 and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 10
                reasoning_parts.append("✅ Negative news sentiment")
            
            # Earnings risk
            if analysis.earnings_days_away is not None and analysis.earnings_days_away <= 7:
                if strategy_key in ["IRON_CONDOR", "LONG_STRADDLE"]:
                    score += 15
                    reasoning_parts.append(f"✅ Earnings in {analysis.earnings_days_away} days - volatility play")
                else:
                    score -= 25
                    reasoning_parts.append(f"⚠️ Earnings in {analysis.earnings_days_away} days - high risk")
            
            # Beginner bonus
            if user_experience == "Beginner" and strategy_info["experience"] == "Beginner-Friendly":
                score += 15
                reasoning_parts.append("✅ Beginner-friendly")
            
            # Win rate for conservative traders
            if risk_tolerance == "Conservative":
                win_rate = int(strategy_info.get("typical_win_rate", "50%").split("-")[0].replace("%", ""))
                if win_rate >= 60:
                    score += 10
                    reasoning_parts.append(f"✅ High win rate (~{strategy_info['typical_win_rate']})")
            
            confidence = max(0, min(1, (score + 50) / 100))
            
            if confidence > 0.3:
                recommendations.append(StrategyRecommendation(
                    strategy_name=strategy_info["name"],
                    action=strategy_key,
                    confidence=confidence,
                    reasoning="\n".join(reasoning_parts) if reasoning_parts else strategy_info["description"],
                    risk_level=strategy_info["risk_level"],
                    max_loss=strategy_info["max_loss"],
                    max_gain=strategy_info["max_gain"],
                    best_conditions=strategy_info["best_for"],
                    experience_level=strategy_info["experience"],
                    examples=strategy_info.get("examples"),
                    notes=strategy_info.get("notes"),
                    example_trade=strategy_info.get("example_trade")
                ))
        
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:5]
    
    @classmethod
    def get_custom_strategies(cls, user_experience: str = "Intermediate") -> List[CustomStrategy]:
        """
        Get available custom/advanced strategies filtered by user experience level.
        
        Args:
            user_experience: User's experience level (Beginner, Intermediate, Advanced, Professional)
        
        Returns:
            List of CustomStrategy objects suitable for the user
        """
        return get_strategies_by_experience(user_experience)
    
    @classmethod
    def convert_custom_to_recommendation(cls, custom_strategy: CustomStrategy, confidence: float = 0.7) -> StrategyRecommendation:
        """
        Convert a CustomStrategy to a StrategyRecommendation for UI compatibility.
        
        Args:
            custom_strategy: The custom strategy to convert
            confidence: Confidence score (0-1)
        
        Returns:
            StrategyRecommendation object
        """
        # Build reasoning from strategy details
        reasoning_parts = [
            f"📊 Source: {custom_strategy.source}",
            f"🎯 Philosophy: {custom_strategy.philosophy[:200]}...",
            f"📈 Key Metrics: {', '.join([f'{k}: {v}' for k, v in list(custom_strategy.key_metrics.items())[:3]])}",
            f"⚠️ Risk Level: {custom_strategy.risk_level}",
            f"💰 Capital Required: {custom_strategy.capital_requirement}"
        ]
        
        if custom_strategy.typical_win_rate:
            reasoning_parts.append(f"✅ Win Rate: {custom_strategy.typical_win_rate}")
        
        # Add warnings
        if custom_strategy.warnings:
            reasoning_parts.append(f"\n⚠️ WARNINGS:\n" + "\n".join(custom_strategy.warnings[:3]))
        
        # Build best conditions from required conditions
        best_conditions = [
            f"Market: {cond.value}" for cond in custom_strategy.required_conditions
        ]
        best_conditions.extend([f"Product: {prod}" for prod in custom_strategy.suitable_products[:3]])
        
        # Build examples from setup rules
        examples = [rule.condition for rule in custom_strategy.setup_rules[:3]]
        
        return StrategyRecommendation(
            strategy_name=custom_strategy.name,
            action=custom_strategy.strategy_id,
            confidence=confidence,
            reasoning="\n".join(reasoning_parts),
            risk_level=custom_strategy.risk_level,
            max_loss="See strategy details",
            max_gain="See strategy details",
            best_conditions=best_conditions,
            experience_level=custom_strategy.experience_level,
            examples=examples,
            notes=custom_strategy.notes,
            example_trade=None  # Custom strategies have complex setup, not simple example trades
        )

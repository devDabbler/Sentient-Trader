"""
Option strategy templates that can be manually added and saved.
These templates integrate with the existing Strategy Advisor logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime


@dataclass
class OptionStrategyTemplate:
    """Template for an option trading strategy"""
    # Identity
    strategy_id: str
    name: str
    description: str
    
    # Classification
    strategy_type: str  # "SINGLE_LEG", "SPREAD", "MULTI_LEG", "COMPLEX"
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL", "VOLATILITY"
    
    # Risk Profile
    risk_level: str  # "Low", "Medium", "High", "Very High"
    max_loss: str
    max_gain: str
    
    # Requirements
    experience_level: str  # "Beginner", "Intermediate", "Advanced", "Professional"
    capital_requirement: str  # "Low", "Medium", "High", "Very High"
    
    # Market Conditions
    best_for: List[str]  # Conditions when strategy works best
    ideal_iv_rank: str  # "Low (<30)", "Medium (30-60)", "High (>60)", "Any"
    ideal_market_outlook: List[str]  # "Bullish", "Bearish", "Neutral"
    
    # Trade Details
    typical_dte: str  # e.g., "30-45 days"
    typical_win_rate: Optional[str] = None
    profit_target: Optional[str] = None
    stop_loss: Optional[str] = None
    
    # Option Alpha Specific
    option_alpha_compatible: bool = True
    option_alpha_action: str = ""  # e.g., "SELL_PUT", "BUY_CALL", "IRON_CONDOR"
    
    # Additional Details
    setup_steps: List[str] = field(default_factory=list)
    management_rules: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Metadata
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    source: Optional[str] = None  # Where you learned this strategy
    tags: List[str] = field(default_factory=list)


class OptionStrategyTemplateManager:
    """Manages saving, loading, and retrieving option strategy templates"""
    
    def __init__(self, templates_file: str = "data/option_strategy_templates.json"):
        self.templates_file = templates_file
        self.templates: Dict[str, OptionStrategyTemplate] = {}
        self._ensure_data_dir()
        self.load_templates()
    
    def _ensure_data_dir(self):
        """Ensure the data directory exists"""
        os.makedirs(os.path.dirname(self.templates_file), exist_ok=True)
    
    def load_templates(self):
        """Load templates from JSON file"""
        if os.path.exists(self.templates_file):
            try:
                with open(self.templates_file, 'r') as f:
                    data = json.load(f)
                    for strategy_id, template_data in data.items():
                        self.templates[strategy_id] = OptionStrategyTemplate(**template_data)
            except Exception as e:
                print(f"Error loading templates: {e}")
    
    def save_templates(self):
        """Save templates to JSON file"""
        try:
            data = {
                strategy_id: {
                    k: v for k, v in template.__dict__.items()
                }
                for strategy_id, template in self.templates.items()
            }
            with open(self.templates_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving templates: {e}")
            return False
    
    def add_template(self, template: OptionStrategyTemplate) -> bool:
        """Add a new template"""
        self.templates[template.strategy_id] = template
        return self.save_templates()
    
    def update_template(self, strategy_id: str, template: OptionStrategyTemplate) -> bool:
        """Update an existing template"""
        if strategy_id in self.templates:
            self.templates[strategy_id] = template
            return self.save_templates()
        return False
    
    def delete_template(self, strategy_id: str) -> bool:
        """Delete a template"""
        if strategy_id in self.templates:
            del self.templates[strategy_id]
            return self.save_templates()
        return False
    
    def get_template(self, strategy_id: str) -> Optional[OptionStrategyTemplate]:
        """Get a specific template"""
        return self.templates.get(strategy_id)
    
    def get_all_templates(self) -> List[OptionStrategyTemplate]:
        """Get all templates"""
        return list(self.templates.values())
    
    def get_templates_by_experience(self, experience_level: str) -> List[OptionStrategyTemplate]:
        """Get templates suitable for experience level"""
        experience_hierarchy = {
            "Beginner": ["Beginner"],
            "Intermediate": ["Beginner", "Intermediate"],
            "Advanced": ["Beginner", "Intermediate", "Advanced"],
            "Professional": ["Beginner", "Intermediate", "Advanced", "Professional"]
        }
        allowed_levels = experience_hierarchy.get(experience_level, ["Beginner"])
        return [t for t in self.templates.values() if t.experience_level in allowed_levels]
    
    def get_templates_by_direction(self, direction: str) -> List[OptionStrategyTemplate]:
        """Get templates by market direction"""
        return [t for t in self.templates.values() if direction.upper() in t.direction.upper()]
    
    def get_templates_by_risk(self, max_risk_level: str) -> List[OptionStrategyTemplate]:
        """Get templates within risk tolerance"""
        risk_hierarchy = ["Low", "Medium", "High", "Very High"]
        max_index = risk_hierarchy.index(max_risk_level) if max_risk_level in risk_hierarchy else 0
        allowed_risks = risk_hierarchy[:max_index + 1]
        return [t for t in self.templates.values() if t.risk_level in allowed_risks]
    
    def get_option_alpha_compatible(self) -> List[OptionStrategyTemplate]:
        """Get templates compatible with Option Alpha"""
        return [t for t in self.templates.values() if t.option_alpha_compatible]
    
    def search_templates(self, query: str) -> List[OptionStrategyTemplate]:
        """Search templates by name, description, or tags"""
        query_lower = query.lower()
        results = []
        for template in self.templates.values():
            if (query_lower in template.name.lower() or 
                query_lower in template.description.lower() or
                any(query_lower in tag.lower() for tag in template.tags)):
                results.append(template)
        return results


# Initialize global template manager
template_manager = OptionStrategyTemplateManager()


# ============================================================================
# Pre-loaded Example Templates
# ============================================================================

# Example 1: Poor Man's Covered Call
POOR_MANS_COVERED_CALL = OptionStrategyTemplate(
    strategy_id="poor_mans_covered_call",
    name="Poor Man's Covered Call (PMCC)",
    description="Buy a deep ITM LEAPS call and sell shorter-term OTM calls against it. Capital-efficient alternative to covered calls.",
    strategy_type="SPREAD",
    direction="BULLISH",
    risk_level="Medium",
    max_loss="Net debit paid (LEAPS cost - credits collected)",
    max_gain="Limited to short call strike - long call strike - net debit",
    experience_level="Intermediate",
    capital_requirement="Medium",
    best_for=[
        "Bullish long-term outlook",
        "Want covered call benefits without buying 100 shares",
        "Lower capital requirement than traditional covered call",
        "Stable to rising stock price"
    ],
    ideal_iv_rank="Medium (30-60)",
    ideal_market_outlook=["Bullish"],
    typical_dte="Long leg: 12-24 months (LEAPS), Short leg: 30-45 days",
    typical_win_rate="60-70%",
    profit_target="50-75% of short call credit",
    stop_loss="Close if stock drops significantly below long call strike",
    option_alpha_compatible=True,
    option_alpha_action="DIAGONAL_SPREAD",
    setup_steps=[
        "Buy deep ITM call (Delta 0.70-0.80) with 12-24 months to expiration",
        "Sell OTM call (Delta 0.20-0.30) with 30-45 days to expiration",
        "Ensure long call strike is lower than short call strike",
        "Net debit should be significantly less than buying 100 shares"
    ],
    management_rules=[
        "Roll short call when it reaches 50-75% profit or at 21 DTE",
        "If stock drops significantly, consider closing entire position",
        "If stock rises past short call, can roll up and out for credit",
        "Monitor long call value - should maintain extrinsic value"
    ],
    examples=[
        "Stock at $100: Buy $80 call (24 months) for $25, Sell $110 call (45 days) for $2",
        "Net cost: $2,300 vs $10,000 for 100 shares",
        "Collect $200/month selling calls against LEAPS"
    ],
    warnings=[
        "⚠️ Long call loses value if stock drops significantly",
        "⚠️ Requires active management of short calls",
        "⚠️ Early assignment risk on short calls (especially before dividends)",
        "⚠️ Less profitable than covered calls in strong bull markets"
    ],
    notes="Great strategy for bullish traders with limited capital. Mimics covered call mechanics with 70-80% less capital.",
    source="Option Alpha Community",
    tags=["diagonal", "leaps", "income", "bullish", "capital-efficient"]
)

# Example 2: Jade Lizard
JADE_LIZARD = OptionStrategyTemplate(
    strategy_id="jade_lizard",
    name="Jade Lizard",
    description="Sell an OTM call spread and an OTM put. No upside risk, profit if stock stays flat or rises moderately.",
    strategy_type="MULTI_LEG",
    direction="NEUTRAL",
    risk_level="Medium-High",
    max_loss="Put strike - net credit (downside only)",
    max_gain="Net credit received",
    experience_level="Advanced",
    capital_requirement="High",
    best_for=[
        "High IV environment",
        "Neutral to slightly bullish bias",
        "Want to eliminate upside risk",
        "Comfortable with downside risk"
    ],
    ideal_iv_rank="High (>60)",
    ideal_market_outlook=["Neutral", "Bullish"],
    typical_dte="30-45 days",
    typical_win_rate="65-75%",
    profit_target="50% of credit received",
    stop_loss="2x credit received or stock breaks put strike",
    option_alpha_compatible=True,
    option_alpha_action="JADE_LIZARD",
    setup_steps=[
        "Sell OTM put (Delta ~0.30)",
        "Sell OTM call (Delta ~0.30)",
        "Buy further OTM call to cap upside risk",
        "Ensure total credit > width of call spread (eliminates upside risk)"
    ],
    management_rules=[
        "Close at 50% profit to reduce risk",
        "If stock drops toward put, consider rolling put down and out",
        "If stock rises, call spread provides protection",
        "Monitor for early assignment on short options"
    ],
    examples=[
        "Stock at $100: Sell $95 put for $2, Sell $110 call for $2, Buy $115 call for $0.50",
        "Net credit: $3.50 ($350 per contract)",
        "No upside risk since credit ($3.50) > call spread width ($5)"
    ],
    warnings=[
        "⚠️ Significant downside risk if stock drops",
        "⚠️ Requires margin for naked put component",
        "⚠️ Best in high IV - credit must exceed call spread width",
        "⚠️ Complex position management"
    ],
    notes="Advanced neutral strategy that eliminates upside risk. Named after the jade lizard which has no natural predators from above.",
    source="tastytrade",
    tags=["neutral", "high-iv", "income", "defined-risk-upside", "advanced"]
)

# ============================================================================
# Comprehensive Strategy Templates from Options Trading Strategies Guide
# ============================================================================

# BEGINNER STRATEGIES (Low Risk)

# 1. Covered Call
COVERED_CALL = OptionStrategyTemplate(
    strategy_id="covered_call",
    name="Covered Call",
    description="Own 100 shares of stock and sell a call option against it to collect premium. Stock may be called away if price rises.",
    strategy_type="SINGLE_LEG",
    direction="BULLISH",
    risk_level="Low",
    max_loss="Stock price decline (partially offset by premium)",
    max_gain="Premium + (Strike - Stock Price)",
    experience_level="Beginner",
    capital_requirement="High",
    best_for=[
        "Stock owners looking for income",
        "Neutral to slightly bullish outlook",
        "Want to generate income from existing holdings",
        "Willing to sell stock at higher price"
    ],
    ideal_iv_rank="Medium (30-60)",
    ideal_market_outlook=["Bullish", "Neutral"],
    typical_dte="30-45 days",
    typical_win_rate="65-75%",
    profit_target="50-75% of premium collected",
    stop_loss="Close if stock drops significantly",
    option_alpha_compatible=True,
    option_alpha_action="SELL_CALL",
    setup_steps=[
        "Own 100 shares of stock",
        "Sell 1 call option at strike price above current price",
        "Collect premium immediately",
        "Choose strike 5-10% above current price"
    ],
    management_rules=[
        "Roll up and out if stock rises past strike",
        "Close at 50-75% profit",
        "Monitor for early assignment before dividends",
        "Consider closing if stock drops significantly"
    ],
    examples=[
        "Stock: XYZ at $50, Sell 1 call at $55 strike, 30-45 days out",
        "Premium collected: $1.50 per share ($150 total)",
        "Max profit: $150 premium + $500 if called away = $650"
    ],
    warnings=[
        "⚠️ Stock may be called away if price rises above strike",
        "⚠️ Limited upside potential",
        "⚠️ Early assignment risk before dividends"
    ],
    notes="Great income strategy for stock owners. Provides downside protection through premium collected.",
    source="Options Trading Strategies Guide",
    tags=["income", "bullish", "covered", "beginner", "low-risk"]
)

# 2. Cash-Secured Put
CASH_SECURED_PUT = OptionStrategyTemplate(
    strategy_id="cash_secured_put",
    name="Cash-Secured Put",
    description="Sell a put option with cash to secure the obligation. Collect premium and potentially buy stock at lower price.",
    strategy_type="SINGLE_LEG",
    direction="BULLISH",
    risk_level="Low to Moderate",
    max_loss="Strike price - premium (if stock goes to $0)",
    max_gain="Premium collected",
    experience_level="Beginner",
    capital_requirement="High",
    best_for=[
        "Wanting to own stock at lower price",
        "Bullish outlook",
        "Willing to own stock at strike price",
        "Income generation with defined risk"
    ],
    ideal_iv_rank="Medium (30-60)",
    ideal_market_outlook=["Bullish"],
    typical_dte="30-45 days",
    typical_win_rate="65-75%",
    profit_target="50-75% of premium collected",
    stop_loss="Close if stock drops significantly below strike",
    option_alpha_compatible=True,
    option_alpha_action="SELL_PUT",
    setup_steps=[
        "Have cash equal to 100 shares × strike price",
        "Sell 1 put option at desired entry price",
        "Collect premium immediately",
        "Choose strike 5-10% below current price"
    ],
    management_rules=[
        "Roll down and out if stock approaches strike",
        "Close at 50-75% profit",
        "Be prepared to buy stock if assigned",
        "Monitor stock fundamentals"
    ],
    examples=[
        "Stock: ABC at $100, Sell 1 put at $95 strike, 30-45 days out",
        "Premium collected: $2.00 per share ($200 total)",
        "Cash reserved: $9,500"
    ],
    warnings=[
        "⚠️ Obligated to buy stock if price drops below strike",
        "⚠️ Requires significant capital",
        "⚠️ Assignment risk if stock drops"
    ],
    notes="Excellent strategy for buying stocks you want at a discount. Part of The Wheel strategy.",
    source="Options Trading Strategies Guide",
    tags=["income", "bullish", "cash-secured", "beginner", "wheel"]
)

# 3. Long Call (Stock Replacement)
LONG_CALL = OptionStrategyTemplate(
    strategy_id="long_call",
    name="Long Call (Stock Replacement)",
    description="Buy a call option, typically deep in-the-money, as a capital-efficient alternative to owning stock.",
    strategy_type="SINGLE_LEG",
    direction="BULLISH",
    risk_level="Moderate",
    max_loss="Premium paid",
    max_gain="Unlimited (stock price - strike - premium)",
    experience_level="Beginner",
    capital_requirement="Low",
    best_for=[
        "Bullish outlook with defined risk",
        "Limited capital",
        "Want leverage without margin",
        "Defined risk directional play"
    ],
    ideal_iv_rank="Low (<30)",
    ideal_market_outlook=["Bullish"],
    typical_dte="60-90 days",
    typical_win_rate="40-60%",
    profit_target="100% of premium paid",
    stop_loss="50% of premium paid",
    option_alpha_compatible=True,
    option_alpha_action="BUY_CALL",
    setup_steps=[
        "Buy 1 call option, typically 60-90 days out",
        "Use deep in-the-money calls for stock replacement",
        "Risk only premium paid",
        "Choose delta 0.70-0.80 for stock replacement"
    ],
    management_rules=[
        "Close at 100% profit or 50% loss",
        "Consider rolling if approaching expiration",
        "Monitor time decay",
        "Don't hold through expiration"
    ],
    examples=[
        "Stock: DEF at $50, Buy 1 call at $45 strike, 90 days out",
        "Premium paid: $7.00 per share ($700 total)",
        "Max profit: Unlimited if stock rises"
    ],
    warnings=[
        "⚠️ Time decay works against you",
        "⚠️ Can lose 100% of premium",
        "⚠️ Requires stock to move significantly"
    ],
    notes="Capital-efficient way to get bullish exposure. Use deep ITM calls for stock replacement.",
    source="Options Trading Strategies Guide",
    tags=["directional", "bullish", "leverage", "beginner", "defined-risk"]
)

# 4. Long Put (Portfolio Protection)
LONG_PUT = OptionStrategyTemplate(
    strategy_id="long_put",
    name="Long Put (Portfolio Protection)",
    description="Buy a put option to protect existing stock positions or profit from stock decline.",
    strategy_type="SINGLE_LEG",
    direction="BEARISH",
    risk_level="Low (as insurance)",
    max_gain="Put strike - stock decline - premium",
    max_loss="Premium paid",
    experience_level="Beginner",
    capital_requirement="Low",
    best_for=[
        "Protecting existing stock positions",
        "Bearish outlook",
        "Portfolio insurance",
        "Defined risk bearish play"
    ],
    ideal_iv_rank="Low (<30)",
    ideal_market_outlook=["Bearish"],
    typical_dte="60-90 days",
    typical_win_rate="40-60%",
    profit_target="100% of premium paid",
    stop_loss="50% of premium paid",
    option_alpha_compatible=True,
    option_alpha_action="BUY_PUT",
    setup_steps=[
        "Own stock (for protection) OR",
        "Buy 1 put option at or below current price",
        "Acts as insurance policy",
        "Choose delta 0.30-0.50 for protection"
    ],
    management_rules=[
        "Close at 100% profit or 50% loss",
        "Consider rolling if approaching expiration",
        "Monitor time decay",
        "Don't hold through expiration"
    ],
    examples=[
        "Own 100 shares of GHI at $60, Buy 1 put at $55 strike, 60-90 days out",
        "Premium paid: $2.50 per share ($250 total)",
        "Protects against losses below $55"
    ],
    warnings=[
        "⚠️ Time decay works against you",
        "⚠️ Can lose 100% of premium",
        "⚠️ Requires stock to move significantly"
    ],
    notes="Excellent for portfolio protection. Acts as insurance against stock decline.",
    source="Options Trading Strategies Guide",
    tags=["protection", "bearish", "insurance", "beginner", "defined-risk"]
)

# CONSERVATIVE INCOME STRATEGIES

# 5. The Wheel Strategy
WHEEL_STRATEGY = OptionStrategyTemplate(
    strategy_id="wheel_strategy",
    name="The Wheel Strategy",
    description="Three-step income strategy: 1) Sell cash-secured put, 2) If assigned, own stock, 3) Sell covered calls on owned stock. Repeat.",
    strategy_type="MULTI_LEG",
    direction="BULLISH",
    risk_level="Low to Moderate",
    max_loss="Stock price decline (partially offset by premiums)",
    max_gain="Steady, consistent income",
    experience_level="Intermediate",
    capital_requirement="High",
    best_for=[
        "Income generation with stocks you like",
        "Sideways to slightly bullish markets",
        "Willing to own and manage stock positions",
        "Consistent monthly income"
    ],
    ideal_iv_rank="Medium (30-60)",
    ideal_market_outlook=["Bullish", "Neutral"],
    typical_dte="30-45 days per cycle",
    typical_win_rate="70-80%",
    profit_target="2-4% monthly return on capital",
    stop_loss="Close if stock fundamentals deteriorate",
    option_alpha_compatible=True,
    option_alpha_action="WHEEL_STRATEGY",
    setup_steps=[
        "Step 1: Sell cash-secured put on stock you want to own",
        "Step 2: If assigned, own 100 shares of stock",
        "Step 3: Sell covered calls on owned stock",
        "Step 4: If called away, repeat from Step 1"
    ],
    management_rules=[
        "Choose stocks you're willing to own long-term",
        "Roll puts down and out if tested",
        "Roll calls up and out if tested",
        "Take assignment and manage accordingly"
    ],
    examples=[
        "Week 1-2: Sell put on JKL at $50, collect $150",
        "If assigned: Own 100 shares at $50",
        "Week 3-4: Sell call at $52, collect $100",
        "If called away: Sell stock at $52, profit $200 + $250 premium"
    ],
    warnings=[
        "⚠️ Requires significant capital",
        "⚠️ Must be willing to own stock",
        "⚠️ Requires active management",
        "⚠️ Can tie up capital for extended periods"
    ],
    notes="Excellent income strategy for patient investors. Combines put selling and covered calls.",
    source="Options Trading Strategies Guide",
    tags=["income", "wheel", "bullish", "intermediate", "consistent"]
)

# INTERMEDIATE STRATEGIES (Moderate Risk)

# 6. Bull Call Spread
BULL_CALL_SPREAD = OptionStrategyTemplate(
    strategy_id="bull_call_spread",
    name="Bull Call Spread",
    description="Buy a call at lower strike and sell a call at higher strike (same expiration). Limited risk and reward.",
    strategy_type="SPREAD",
    direction="BULLISH",
    risk_level="Moderate",
    max_loss="Net debit",
    max_gain="Strike width - net debit",
    experience_level="Intermediate",
    capital_requirement="Low",
    best_for=[
        "Moderately bullish outlook",
        "Want defined risk and reward",
        "Lower cost than long call",
        "Expect moderate rise in stock price"
    ],
    ideal_iv_rank="Low (<30)",
    ideal_market_outlook=["Bullish"],
    typical_dte="30-60 days",
    typical_win_rate="60-70%",
    profit_target="50% of max gain",
    stop_loss="50% of net debit",
    option_alpha_compatible=True,
    option_alpha_action="BULL_CALL_SPREAD",
    setup_steps=[
        "Buy 1 call at lower strike",
        "Sell 1 call at higher strike (same expiration)",
        "Net debit position",
        "Choose strikes 5-10% apart"
    ],
    management_rules=[
        "Close at 50% of max gain",
        "Close at 50% loss",
        "Consider rolling if approaching expiration",
        "Monitor both legs"
    ],
    examples=[
        "Stock: PQR at $50, Buy $50 call for $3, Sell $55 call for $1",
        "Net cost: $200",
        "Max profit: $300 (if stock at $55+)"
    ],
    warnings=[
        "⚠️ Limited upside potential",
        "⚠️ Requires stock to move up",
        "⚠️ Time decay affects both legs"
    ],
    notes="Great for moderate bullish plays with defined risk. Lower cost than long calls.",
    source="Options Trading Strategies Guide",
    tags=["spread", "bullish", "defined-risk", "intermediate", "moderate"]
)

# 7. Bear Put Spread
BEAR_PUT_SPREAD = OptionStrategyTemplate(
    strategy_id="bear_put_spread",
    name="Bear Put Spread",
    description="Buy a put at higher strike and sell a put at lower strike (same expiration). Limited risk and reward.",
    strategy_type="SPREAD",
    direction="BEARISH",
    risk_level="Moderate",
    max_loss="Net debit",
    max_gain="Strike width - net debit",
    experience_level="Intermediate",
    capital_requirement="Low",
    best_for=[
        "Moderately bearish outlook",
        "Want defined risk and reward",
        "Lower cost than long put",
        "Expect moderate decline in stock price"
    ],
    ideal_iv_rank="Low (<30)",
    ideal_market_outlook=["Bearish"],
    typical_dte="30-60 days",
    typical_win_rate="60-70%",
    profit_target="50% of max gain",
    stop_loss="50% of net debit",
    option_alpha_compatible=True,
    option_alpha_action="BEAR_PUT_SPREAD",
    setup_steps=[
        "Buy 1 put at higher strike",
        "Sell 1 put at lower strike (same expiration)",
        "Net debit position",
        "Choose strikes 5-10% apart"
    ],
    management_rules=[
        "Close at 50% of max gain",
        "Close at 50% loss",
        "Consider rolling if approaching expiration",
        "Monitor both legs"
    ],
    examples=[
        "Stock: STU at $50, Buy $50 put for $3, Sell $45 put for $1",
        "Net cost: $200",
        "Max profit: $300 (if stock at $45-)"
    ],
    warnings=[
        "⚠️ Limited downside potential",
        "⚠️ Requires stock to move down",
        "⚠️ Time decay affects both legs"
    ],
    notes="Great for moderate bearish plays with defined risk. Lower cost than long puts.",
    source="Options Trading Strategies Guide",
    tags=["spread", "bearish", "defined-risk", "intermediate", "moderate"]
)

# 8. Iron Condor
IRON_CONDOR = OptionStrategyTemplate(
    strategy_id="iron_condor",
    name="Iron Condor",
    description="Sell OTM call spread above stock and OTM put spread below stock. Collect net credit. Profit if stock stays between strikes.",
    strategy_type="MULTI_LEG",
    direction="NEUTRAL",
    risk_level="Moderate",
    max_loss="Strike width - net credit",
    max_gain="Net credit received",
    experience_level="Intermediate",
    capital_requirement="Medium",
    best_for=[
        "Neutral market, expect low volatility",
        "High IV environment",
        "Want defined risk",
        "Expect stock to stay in range"
    ],
    ideal_iv_rank="High (>60)",
    ideal_market_outlook=["Neutral"],
    typical_dte="30-45 days",
    typical_win_rate="65-75%",
    profit_target="50% of credit received",
    stop_loss="2x credit received",
    option_alpha_compatible=True,
    option_alpha_action="IRON_CONDOR",
    setup_steps=[
        "Sell OTM call spread (above stock)",
        "Sell OTM put spread (below stock)",
        "Collect net credit",
        "Choose strikes 10-15% from current price"
    ],
    management_rules=[
        "Close at 50% profit",
        "Close at 2x credit loss",
        "Roll untested side if tested",
        "Monitor both spreads"
    ],
    examples=[
        "Stock: VWX at $100, Sell $110/$115 call spread, Sell $90/$85 put spread",
        "Total credit: $200",
        "Max profit: $200 if stock stays between $90-$110"
    ],
    warnings=[
        "⚠️ Requires stock to stay in range",
        "⚠️ High IV environment needed",
        "⚠️ Complex position management"
    ],
    notes="Excellent neutral strategy for high IV environments. Defined risk and reward.",
    source="Options Trading Strategies Guide",
    tags=["neutral", "high-iv", "income", "defined-risk", "intermediate"]
)

# 9. Calendar Spread (Time Spread)
CALENDAR_SPREAD = OptionStrategyTemplate(
    strategy_id="calendar_spread",
    name="Calendar Spread (Time Spread)",
    description="Sell near-term option and buy longer-term option at same strike. Profit from time decay differential.",
    strategy_type="SPREAD",
    direction="NEUTRAL",
    risk_level="Moderate",
    max_loss="Net debit",
    max_gain="Variable, when short expires worthless",
    experience_level="Intermediate",
    capital_requirement="Low",
    best_for=[
        "Expect stock to stay near strike",
        "Profit from time decay",
        "Low volatility expected",
        "Neutral outlook"
    ],
    ideal_iv_rank="Low (<30)",
    ideal_market_outlook=["Neutral"],
    typical_dte="Short: 30 days, Long: 60-90 days",
    typical_win_rate="60-70%",
    profit_target="50% of net debit",
    stop_loss="50% of net debit",
    option_alpha_compatible=True,
    option_alpha_action="CALENDAR_SPREAD",
    setup_steps=[
        "Sell near-term option",
        "Buy longer-term option (same strike)",
        "Net debit position",
        "Choose ATM or slightly OTM strikes"
    ],
    management_rules=[
        "Close at 50% profit",
        "Close at 50% loss",
        "Roll short leg if needed",
        "Monitor time decay differential"
    ],
    examples=[
        "Stock: YZA at $50, Sell $50 call 30 days for $2, Buy $50 call 90 days for $4",
        "Net cost: $200",
        "Profit if short expires worthless"
    ],
    warnings=[
        "⚠️ Requires stock to stay near strike",
        "⚠️ Time decay works against long leg",
        "⚠️ Complex Greeks management"
    ],
    notes="Great for time decay plays. Requires stock to stay near strike at short expiration.",
    source="Options Trading Strategies Guide",
    tags=["time-decay", "neutral", "calendar", "intermediate", "moderate"]
)

# ADVANCED NEUTRAL STRATEGIES

# 10. Short Straddle
SHORT_STRADDLE = OptionStrategyTemplate(
    strategy_id="short_straddle",
    name="Short Straddle",
    description="Sell 1 call and 1 put at-the-money (same expiration). Collect premium. Profit if stock stays flat.",
    strategy_type="MULTI_LEG",
    direction="NEUTRAL",
    risk_level="High",
    max_loss="Unlimited (on call side), substantial (on put side)",
    max_gain="Total premium collected",
    experience_level="Advanced",
    capital_requirement="High",
    best_for=[
        "Advanced traders expecting no movement",
        "High IV before earnings",
        "Expect stock to stay flat",
        "High premium collection"
    ],
    ideal_iv_rank="High (>60)",
    ideal_market_outlook=["Neutral"],
    typical_dte="30-45 days",
    typical_win_rate="60-70%",
    profit_target="50% of premium collected",
    stop_loss="2x premium collected",
    option_alpha_compatible=True,
    option_alpha_action="SHORT_STRADDLE",
    setup_steps=[
        "Sell 1 call at-the-money",
        "Sell 1 put at-the-money (same expiration)",
        "Collect premium",
        "Choose ATM strikes"
    ],
    management_rules=[
        "Close at 50% profit",
        "Close at 2x premium loss",
        "Roll out if tested",
        "Monitor both sides closely"
    ],
    examples=[
        "Stock: BCD at $100, Sell $100 call for $4, Sell $100 put for $4",
        "Total credit: $800",
        "Max profit: $800 if stock stays at $100"
    ],
    warnings=[
        "⚠️ Unlimited upside risk",
        "⚠️ Substantial downside risk",
        "⚠️ Requires high IV",
        "⚠️ Very risky strategy"
    ],
    notes="High-risk strategy for experienced traders. Requires stock to stay exactly flat.",
    source="Options Trading Strategies Guide",
    tags=["neutral", "high-iv", "advanced", "high-risk", "straddle"]
)

# 11. Short Strangle
SHORT_STRANGLE = OptionStrategyTemplate(
    strategy_id="short_strangle",
    name="Short Strangle",
    description="Sell 1 OTM call and 1 OTM put (same expiration). Collect premium. Profit if stock stays between strikes.",
    strategy_type="MULTI_LEG",
    direction="NEUTRAL",
    risk_level="High",
    max_loss="Unlimited (on call side), substantial (on put side)",
    max_gain="Total premium collected",
    experience_level="Advanced",
    capital_requirement="High",
    best_for=[
        "Expect stock to stay within wide range",
        "High IV environment",
        "Want wider profit zone than straddle",
        "Advanced neutral strategy"
    ],
    ideal_iv_rank="High (>60)",
    ideal_market_outlook=["Neutral"],
    typical_dte="30-45 days",
    typical_win_rate="65-75%",
    profit_target="50% of premium collected",
    stop_loss="2x premium collected",
    option_alpha_compatible=True,
    option_alpha_action="SHORT_STRANGLE",
    setup_steps=[
        "Sell 1 OTM call",
        "Sell 1 OTM put (same expiration)",
        "Collect premium",
        "Choose strikes 5-10% from current price"
    ],
    management_rules=[
        "Close at 50% profit",
        "Close at 2x premium loss",
        "Roll out if tested",
        "Monitor both sides"
    ],
    examples=[
        "Stock: EFG at $100, Sell $110 call for $2, Sell $90 put for $2",
        "Total credit: $400",
        "Max profit: $400 if stock stays between $90-$110"
    ],
    warnings=[
        "⚠️ Unlimited upside risk",
        "⚠️ Substantial downside risk",
        "⚠️ Requires high IV",
        "⚠️ Very risky strategy"
    ],
    notes="Wider profit zone than straddle but still high risk. Better for range-bound markets.",
    source="Options Trading Strategies Guide",
    tags=["neutral", "high-iv", "advanced", "high-risk", "strangle"]
)

# 12. Butterfly Spread
BUTTERFLY_SPREAD = OptionStrategyTemplate(
    strategy_id="butterfly_spread",
    name="Butterfly Spread",
    description="Buy 1 call at lower strike, sell 2 calls at middle strike, buy 1 call at higher strike. Profit if stock at middle strike.",
    strategy_type="MULTI_LEG",
    direction="NEUTRAL",
    risk_level="Low to Moderate",
    max_loss="Net debit",
    max_gain="Middle strike - lower strike - net debit",
    experience_level="Advanced",
    capital_requirement="Low",
    best_for=[
        "Expect stock to be at middle strike at expiration",
        "Low volatility expected",
        "Want defined risk",
        "Precise directional play"
    ],
    ideal_iv_rank="Low (<30)",
    ideal_market_outlook=["Neutral"],
    typical_dte="30-60 days",
    typical_win_rate="40-60%",
    profit_target="100% of net debit",
    stop_loss="50% of net debit",
    option_alpha_compatible=True,
    option_alpha_action="BUTTERFLY_SPREAD",
    setup_steps=[
        "Buy 1 call at lower strike",
        "Sell 2 calls at middle strike",
        "Buy 1 call at higher strike",
        "Net debit (small)"
    ],
    management_rules=[
        "Close at 100% profit",
        "Close at 50% loss",
        "Monitor for early assignment",
        "Don't hold through expiration"
    ],
    examples=[
        "Stock: HIJ at $50, Buy $45 call for $7, Sell 2x $50 calls for $8, Buy $55 call for $2",
        "Net cost: $100",
        "Max profit: $400 if stock at $50"
    ],
    warnings=[
        "⚠️ Requires precise stock movement",
        "⚠️ Low probability of max profit",
        "⚠️ Early assignment risk on short calls"
    ],
    notes="Precise strategy requiring stock to be exactly at middle strike. Low cost, high reward potential.",
    source="Options Trading Strategies Guide",
    tags=["neutral", "butterfly", "precise", "advanced", "defined-risk"]
)

# 13. Iron Butterfly
IRON_BUTTERFLY = OptionStrategyTemplate(
    strategy_id="iron_butterfly",
    name="Iron Butterfly",
    description="Sell call and put at-the-money, buy call and put further OTM. Net credit. Profit if stock stays at middle strike.",
    strategy_type="MULTI_LEG",
    direction="NEUTRAL",
    risk_level="Moderate",
    max_loss="Strike width - net credit",
    max_gain="Net credit received",
    experience_level="Advanced",
    capital_requirement="Medium",
    best_for=[
        "Expect minimal stock movement",
        "High IV environment",
        "Want defined risk",
        "Precise neutral play"
    ],
    ideal_iv_rank="High (>60)",
    ideal_market_outlook=["Neutral"],
    typical_dte="30-45 days",
    typical_win_rate="60-70%",
    profit_target="50% of credit received",
    stop_loss="2x credit received",
    option_alpha_compatible=True,
    option_alpha_action="IRON_BUTTERFLY",
    setup_steps=[
        "Sell call at-the-money",
        "Sell put at-the-money",
        "Buy call further OTM",
        "Buy put further OTM"
    ],
    management_rules=[
        "Close at 50% profit",
        "Close at 2x credit loss",
        "Monitor for early assignment",
        "Roll if tested"
    ],
    examples=[
        "Stock: KLM at $50, Sell $50 call for $3, Sell $50 put for $3, Buy $55 call for $1, Buy $45 put for $1",
        "Net credit: $400",
        "Max profit: $400 if stock stays at $50"
    ],
    warnings=[
        "⚠️ Requires stock to stay exactly at middle strike",
        "⚠️ High IV environment needed",
        "⚠️ Early assignment risk"
    ],
    notes="Combines straddle and butterfly. Requires precise stock movement for max profit.",
    source="Options Trading Strategies Guide",
    tags=["neutral", "high-iv", "butterfly", "advanced", "defined-risk"]
)

# 14. Jade Lizard (already defined above, keeping for completeness)

# Add all comprehensive strategy templates to manager on initialization
comprehensive_templates = [
    COVERED_CALL, CASH_SECURED_PUT, LONG_CALL, LONG_PUT, WHEEL_STRATEGY,
    BULL_CALL_SPREAD, BEAR_PUT_SPREAD, IRON_CONDOR, CALENDAR_SPREAD,
    SHORT_STRADDLE, SHORT_STRANGLE, BUTTERFLY_SPREAD, IRON_BUTTERFLY
]

for template in comprehensive_templates:
    if not template_manager.get_template(template.strategy_id):
        template_manager.add_template(template)

# Add example templates to manager on initialization
if not template_manager.get_template("poor_mans_covered_call"):
    template_manager.add_template(POOR_MANS_COVERED_CALL)
if not template_manager.get_template("jade_lizard"):
    template_manager.add_template(JADE_LIZARD)

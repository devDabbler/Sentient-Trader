"""
Advanced option strategies with detailed playbooks and AI validation support.
Includes professional strategies from various sources for comprehensive analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class StrategyType(Enum):
    """Types of advanced strategies"""
    FUTURES_OPTIONS_SELLING = "futures_options_selling"
    LEAPS_RECOVERY = "leaps_recovery"


class MarketCondition(Enum):
    """Market conditions for strategy execution"""
    HIGH_VOLATILITY = "high_volatility"
    BLOOD_IN_STREETS = "blood_in_streets"
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    PRE_CATALYST = "pre_catalyst"


@dataclass
class StrategyParameter:
    """Individual parameter for a strategy"""
    name: str
    value: Any
    description: str
    required: bool = True
    validation_rule: Optional[str] = None


@dataclass
class TradeSetupRule:
    """Rules for setting up a trade"""
    condition: str
    action: str
    priority: int
    notes: Optional[str] = None


@dataclass
class RiskManagementRule:
    """Risk management rules for a strategy"""
    rule_type: str  # "stop_loss", "profit_target", "position_sizing", "exit_trigger"
    value: Any
    description: str
    mandatory: bool = True


@dataclass
class CustomStrategy:
    """
    A complete advanced strategy with all details for AI validation.
    """
    # Identity
    strategy_id: str
    name: str
    source: str  # Reddit username or post link
    strategy_type: StrategyType
    
    # Core Description
    description: str
    philosophy: str
    key_metrics: Dict[str, Any]  # Performance metrics from source
    
    # Market Requirements
    suitable_products: List[str]  # Tickers, futures, ETFs
    required_conditions: List[MarketCondition]
    unsuitable_conditions: List[MarketCondition]
    
    # Trade Mechanics
    parameters: List[StrategyParameter]
    setup_rules: List[TradeSetupRule]
    risk_management: List[RiskManagementRule]
    
    # Experience & Risk
    experience_level: str  # "Beginner", "Intermediate", "Advanced", "Professional"
    risk_level: str  # "Low", "Medium", "High", "Very High"
    capital_requirement: str
    typical_win_rate: Optional[str] = None
    
    # AI Validation Criteria
    validation_checklist: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    
    # Additional Context
    notes: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# STRATEGY 1: Professional Futures Options Selling ("The Chicken Approach")
# ============================================================================

FUTURES_OPTIONS_SELLING_STRATEGY = CustomStrategy(
    strategy_id="futures_chicken",
    name="Professional Futures Options Selling (The Chicken Approach)",
    source="Professional Portfolio Manager - $8.7M AUM",
    strategy_type=StrategyType.FUTURES_OPTIONS_SELLING,
    
    description=(
        "Sell strangles or puts on major commodity futures (Gold, Silver, Copper, Crude Oil, "
        "Natural Gas, Wheat, Soybeans) with a focus on avoiding uncertainty and only entering "
        "positions where odds are overwhelmingly favorable. Exit proactively before major unpredictable events."
    ),
    
    philosophy=(
        "The 'Chicken' Approach: Only trade when odds are too obviously in your favor. "
        "Avoid all uncertainty by exiting before major unpredictable news events (tariffs, geopolitical conflicts, "
        "major policy shifts). Focus on collecting premium from non-critical volatility spikes."
    ),
    
    key_metrics={
        "ytd_return": "27.64%",
        "winning_trades": "12/12 (100%)",
        "sharpe_ratio": 2.47,
        "portfolio_size": "$8.7M",
        "strategy_focus": "Selling options on futures"
    },
    
    suitable_products=[
        "GC (Gold Futures)",
        "SI (Silver Futures)",
        "HG (Copper Futures)",
        "CL (Crude Oil Futures)",
        "NG (Natural Gas Futures)",
        "ZW (Wheat Futures)",
        "ZS (Soybeans Futures)"
    ],
    
    required_conditions=[
        MarketCondition.HIGH_VOLATILITY,
        MarketCondition.SIDEWAYS
    ],
    
    unsuitable_conditions=[
        MarketCondition.PRE_CATALYST  # Exit before major events
    ],
    
    parameters=[
        StrategyParameter(
            name="DTE",
            value="45-120 days",
            description="Days to expiration. Aim for ~45 DTE for optimal theta decay, go longer if needed for desired strikes.",
            required=True,
            validation_rule="45 <= dte <= 120"
        ),
        StrategyParameter(
            name="profit_target",
            value="25-50%",
            description="Exit at 50% of credit collected to manage risk and recycle capital",
            required=True,
            validation_rule="0.25 <= target <= 0.50"
        ),
        StrategyParameter(
            name="stop_loss",
            value="50%",
            description="Very tight stop loss at 50% of credit collected",
            required=True,
            validation_rule="stop_loss == 0.50"
        ),
        StrategyParameter(
            name="max_margin_per_position",
            value="30%",
            description="Maximum 30% of capital margin per position",
            required=True,
            validation_rule="margin <= 0.30"
        ),
        StrategyParameter(
            name="position_type",
            value="Strangle or Put",
            description="Sell strangles (neutral) or puts (bullish bias)",
            required=True
        ),
        StrategyParameter(
            name="correlation_rule",
            value="Never trade correlated products simultaneously",
            description="Avoid positions that move together during single market events",
            required=True
        )
    ],
    
    setup_rules=[
        TradeSetupRule(
            condition="Step 1: Check CVOL (Commodity Volatility)",
            action="Ensure 3-month volatility is relatively high compared to average. High CVOL = higher premiums.",
            priority=1,
            notes="High CVOL is a prerequisite"
        ),
        TradeSetupRule(
            condition="Step 2: Analyze Chart Technicals",
            action="Determine if price is in Uptrend, Downtrend, or Sidetrend. This determines neutral vs directional bias.",
            priority=2,
            notes="Determines strangle vs put/call selection"
        ),
        TradeSetupRule(
            condition="Step 3: Fundamental Analysis (MOST CRITICAL)",
            action="Determine core price drivers. Is recent news Critical or Non-Critical?",
            priority=3,
            notes="Critical news (tariffs, war, mine closures) = DO NOT TRADE. Non-critical news = opportunity."
        ),
        TradeSetupRule(
            condition="Step 4: Position Setup - Decision Matrix",
            action=(
                "IF (Fundamental reason to go UP) AND (Recent news is CRITICAL): AVOID/EXIT\n"
                "IF (Fundamental reason to go UP) AND (Recent news is NON-CRITICAL): Sell Short Puts (Bullish)\n"
                "IF (NO fundamental reason up/down) AND (Recent news is NON-CRITICAL): Short Strangle or Short Puts (Neutral)"
            ),
            priority=4,
            notes="Every other combination is lower probability - avoid"
        ),
        TradeSetupRule(
            condition="Strike Selection",
            action="Select wide strikes (low delta) far OTM to allow wide price movement range",
            priority=5,
            notes="Use historical trading ranges (e.g., within range for 10 years)"
        )
    ],
    
    risk_management=[
        RiskManagementRule(
            rule_type="profit_target",
            value="50%",
            description="Close position immediately at 50% of max profit (credit collected)",
            mandatory=True
        ),
        RiskManagementRule(
            rule_type="stop_loss",
            value="50%",
            description="Close position at 50% loss relative to premium collected",
            mandatory=True
        ),
        RiskManagementRule(
            rule_type="exit_trigger",
            value="Preemptive Exit (The Chicken Rule)",
            description="Exit immediately if any major unpredictable event surfaces after trade is opened",
            mandatory=True
        ),
        RiskManagementRule(
            rule_type="position_sizing",
            value="30% max margin",
            description="Ensure margin used for any single trade does not exceed 30% of total capital",
            mandatory=True
        ),
        RiskManagementRule(
            rule_type="management",
            value="Never rolls positions",
            description="Prefer to close and re-evaluate rather than adjusting (rolling)",
            mandatory=True
        )
    ],
    
    experience_level="Professional",
    risk_level="Medium",
    capital_requirement="Very High ($100k+ recommended for futures)",
    typical_win_rate="100% (12/12 trades YTD)",
    
    validation_checklist=[
        "Is CVOL elevated compared to historical average?",
        "Have you analyzed the chart for trend direction?",
        "Have you identified fundamental price drivers?",
        "Is the recent news NON-CRITICAL (not tariffs, war, policy shifts)?",
        "Are strikes wide enough (low delta, far OTM)?",
        "Is margin usage under 30% for this position?",
        "Are you avoiding correlated products?",
        "Do you have a plan to exit before any major scheduled events?",
        "Is DTE between 45-120 days?",
        "Have you set profit target at 50% and stop loss at 50%?"
    ],
    
    red_flags=[
        "Trading before/during major geopolitical events",
        "Trading correlated products simultaneously",
        "Using more than 30% margin on single position",
        "Ignoring fundamental analysis",
        "Trading when CVOL is low",
        "Rolling positions instead of closing",
        "Not having clear exit plan before major news"
    ],
    
    warnings=[
        "⚠️ Requires futures trading approval and significant capital",
        "⚠️ Futures options have different margin requirements than equity options",
        "⚠️ This strategy requires deep product-specific knowledge of commodities",
        "⚠️ Must monitor geopolitical and fundamental news constantly",
        "⚠️ Professional-level strategy - not suitable for beginners"
    ],
    
    notes=(
        "This is a professional money manager's strategy with exceptional results. "
        "The key differentiator is the 'Chicken' approach - proactively avoiding uncertainty "
        "rather than trying to predict it. Requires discipline to exit positions before major events."
    )
)


# ============================================================================
# STRATEGY 2: LEAPS Recovery Strategy ("Blood in the Streets")
# ============================================================================

LEAPS_RECOVERY_STRATEGY = CustomStrategy(
    strategy_id="leaps_recovery",
    name="LEAPS Recovery Strategy (Blood in the Streets)",
    source="Professional Trader - $30k to $548k in 7mo",
    strategy_type=StrategyType.LEAPS_RECOVERY,
    
    description=(
        "Use Long-term Equity Anticipation Securities (LEAPS) Call Options on S&P 500 (SPY) "
        "to leverage a capital-efficient bet on market recovery following a major downturn. "
        "Enter during peak fear, hold through recovery, optionally roll up to compound gains."
    ),
    
    philosophy=(
        "Buy when there's 'blood in the streets' - during maximum panic when sentiment is crushed "
        "and recovery appears impossible. Use LEAPS (1.5-2+ years out) to minimize theta decay "
        "and give the market ample time to recover. Exit into FOMO when positive catalysts converge."
    ),
    
    key_metrics={
        "initial_investment": "$30,000",
        "final_value": "$548,000",
        "return": "1,727% (17.27x)",
        "timeframe": "7 months",
        "instrument": "SPY LEAPS Calls",
        "entry_date": "April 8 (tariff-induced crash)",
        "expiration": "March 31, 2026"
    },
    
    suitable_products=[
        "SPY (S&P 500 ETF)",
        "QQQ (Nasdaq-100 ETF)",
        "IWM (Russell 2000 ETF)",
        "DIA (Dow Jones ETF)",
        "Other major index ETFs"
    ],
    
    required_conditions=[
        MarketCondition.BLOOD_IN_STREETS,
        MarketCondition.HIGH_VOLATILITY
    ],
    
    unsuitable_conditions=[
        MarketCondition.UPTREND  # Don't enter during uptrends
    ],
    
    parameters=[
        StrategyParameter(
            name="entry_timing",
            value="Major non-catastrophic market sell-off",
            description="Wait for blood in the streets - policy shock, tariff panic, short-term economic scare",
            required=True,
            validation_rule="Sentiment must be crushed, VIX elevated"
        ),
        StrategyParameter(
            name="expiration",
            value="1.5 to 2+ years (LEAPS)",
            description="Long expiration minimizes theta decay and gives market time to recover",
            required=True,
            validation_rule="dte >= 540 days (18 months)"
        ),
        StrategyParameter(
            name="strike_selection",
            value="ITM or Near-ATM (Delta 0.70-0.85)",
            description="In-the-money calls for maximum delta exposure with least time decay risk",
            required=True,
            validation_rule="0.70 <= delta <= 0.85"
        ),
        StrategyParameter(
            name="position_sizing",
            value="Significant capital allocation",
            description="This is a leveraged bet - size appropriately for risk tolerance",
            required=True,
            validation_rule="Don't risk more than you can afford to lose"
        ),
        StrategyParameter(
            name="roll_up_strategy",
            value="Optional",
            description="As position becomes profitable, can sell and buy higher strike LEAPS to compound gains",
            required=False
        ),
        StrategyParameter(
            name="exit_timing",
            value="Catalytic FOMO moment",
            description="Exit when positive news converges (Fed cuts, political clarity, strong earnings)",
            required=True
        )
    ],
    
    setup_rules=[
        TradeSetupRule(
            condition="Step 1: Identify the Setup - Wait for Blood in the Streets",
            action=(
                "Look for major, NON-CATASTROPHIC market-wide sell-off:\n"
                "- Policy/tariff shock\n"
                "- Short-term economic data scare\n"
                "- Non-systemic panic\n"
                "Sentiment must be crushed, making recovery appear impossible to average investor."
            ),
            priority=1,
            notes="Entry during maximum panic ensures cheap options due to high IV and low spot price"
        ),
        TradeSetupRule(
            condition="Step 2: Select the Instrument - Choose LEAPS on SPY",
            action="Select SPY Call options with expiration at least 1.5 to 2+ years out",
            priority=2,
            notes="Long expiration (LEAPS) minimizes theta decay effects"
        ),
        TradeSetupRule(
            condition="Step 3: Execute Entry - Buy ITM Calls",
            action=(
                "Invest capital into LEAPS position:\n"
                "- Choose strike price that is ITM or Near-ATM\n"
                "- Target Delta of 0.70 to 0.85\n"
                "- This gives maximum exposure with least time decay risk"
            ),
            priority=3,
            notes="ITM options move closer to $1 for every $1 the index moves"
        ),
        TradeSetupRule(
            condition="Step 4: Manage Position - The Roll-Up (Optional)",
            action=(
                "As market recovers and LEAPS become significantly profitable:\n"
                "- Sell existing position\n"
                "- Buy new LEAPS with higher strike price\n"
                "- Lock in portion of profit while re-leveraging remaining capital"
            ),
            priority=4,
            notes="This compounds gains but adds complexity and risk"
        ),
        TradeSetupRule(
            condition="Step 5: Plan Exit - Target Catalytic FOMO",
            action=(
                "Look to exit when long-term thesis is about to be confirmed:\n"
                "- Confluence of positive news (Fed cuts, political clarity, strong earnings)\n"
                "- Selling into FOMO moment captures maximum extrinsic value"
            ),
            priority=5,
            notes="Exit before momentum potentially fades"
        )
    ],
    
    risk_management=[
        RiskManagementRule(
            rule_type="position_sizing",
            value="High risk allocation",
            description="This is an aggressive, high-risk strategy. Only allocate capital you can afford to lose entirely.",
            mandatory=True
        ),
        RiskManagementRule(
            rule_type="stop_loss",
            value="None specified",
            description="Strategy relies on long-term recovery thesis. No specific stop loss, but monitor thesis validity.",
            mandatory=False
        ),
        RiskManagementRule(
            rule_type="exit_trigger",
            value="Thesis invalidation or FOMO peak",
            description="Exit if recovery thesis is invalidated OR when positive catalysts converge (FOMO moment)",
            mandatory=True
        ),
        RiskManagementRule(
            rule_type="time_decay",
            value="Managed by LEAPS selection",
            description="Using LEAPS (1.5-2+ years) minimizes theta decay impact",
            mandatory=True
        )
    ],
    
    experience_level="Advanced",
    risk_level="Very High",
    capital_requirement="Medium to High (depends on position size)",
    typical_win_rate="N/A (single directional bet, not systematic)",
    
    validation_checklist=[
        "Is the market experiencing a major sell-off with crushed sentiment?",
        "Is the sell-off NON-CATASTROPHIC (not systemic collapse)?",
        "Are you using LEAPS with at least 1.5-2 years to expiration?",
        "Are you buying ITM or Near-ATM calls (Delta 0.70-0.85)?",
        "Do you have a clear thesis for market recovery?",
        "Can you afford to lose the entire capital allocated?",
        "Have you identified potential exit catalysts (Fed cuts, political clarity, etc.)?",
        "Are you prepared to hold through volatility?",
        "Do you understand the leverage and risk involved?",
        "Have you considered the tax implications of the trade?"
    ],
    
    red_flags=[
        "Entering during market uptrend (not blood in streets)",
        "Using short-dated options instead of LEAPS",
        "Buying OTM calls with low delta",
        "Allocating more capital than you can afford to lose",
        "No clear recovery thesis",
        "Panic selling during normal volatility",
        "Not having exit plan for FOMO moment",
        "Ignoring broader economic indicators"
    ],
    
    warnings=[
        "⚠️ EXTREMELY HIGH RISK - Can lose 100% of capital invested",
        "⚠️ Requires accurate market timing and recovery thesis",
        "⚠️ Massive leverage means massive potential losses",
        "⚠️ Not suitable for beginners or conservative investors",
        "⚠️ Requires emotional discipline to hold through volatility",
        "⚠️ Tax implications: short-term capital gains if held < 1 year",
        "⚠️ One trade example does not guarantee future results"
    ],
    
    notes=(
        "This strategy produced exceptional returns (17x in 7 months) but represents "
        "an extremely aggressive, leveraged bet. It requires perfect timing, strong conviction, "
        "and the ability to withstand significant volatility. The massive returns are possible "
        "precisely because of the massive risk involved. This is NOT a systematic strategy "
        "but rather a tactical, opportunistic trade during market dislocations."
    )
)


# ============================================================================
# Strategy Registry
# ============================================================================

CUSTOM_STRATEGIES = {
    "futures_chicken": FUTURES_OPTIONS_SELLING_STRATEGY,
    "leaps_recovery": LEAPS_RECOVERY_STRATEGY
}


def get_custom_strategy(strategy_id: str) -> Optional[CustomStrategy]:
    """Get a custom strategy by ID"""
    return CUSTOM_STRATEGIES.get(strategy_id)


def get_all_custom_strategies() -> List[CustomStrategy]:
    """Get all available custom strategies"""
    return list(CUSTOM_STRATEGIES.values())


def get_strategies_by_type(strategy_type: StrategyType) -> List[CustomStrategy]:
    """Get strategies filtered by type"""
    return [s for s in CUSTOM_STRATEGIES.values() if s.strategy_type == strategy_type]


def get_strategies_by_experience(experience_level: str) -> List[CustomStrategy]:
    """Get strategies suitable for a given experience level"""
    experience_hierarchy = {
        "Beginner": ["Beginner"],
        "Intermediate": ["Beginner", "Intermediate"],
        "Advanced": ["Beginner", "Intermediate", "Advanced"],
        "Professional": ["Beginner", "Intermediate", "Advanced", "Professional"]
    }
    
    allowed_levels = experience_hierarchy.get(experience_level, ["Beginner"])
    return [s for s in CUSTOM_STRATEGIES.values() if s.experience_level in allowed_levels]

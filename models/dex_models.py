"""
Data models for DEX Launch Hunter system

Tracks new token launches, risk scores, smart money activity, and launch signals.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class Chain(Enum):
    """Supported blockchain networks"""
    ETH = "ethereum"
    BSC = "bsc"
    SOLANA = "solana"
    BASE = "base"
    ARBITRUM = "arbitrum"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"


class RiskLevel(Enum):
    """Token risk assessment levels"""
    EXTREME = "EXTREME"  # Likely rug/scam
    HIGH = "HIGH"  # Multiple red flags
    MEDIUM = "MEDIUM"  # Some concerns
    LOW = "LOW"  # Relatively safe
    SAFE = "SAFE"  # Verified/audited


class LaunchStage(Enum):
    """Token launch lifecycle stages"""
    PRESALE = "PRESALE"  # Pre-launch presale
    JUST_LAUNCHED = "JUST_LAUNCHED"  # < 1 hour old
    EARLY = "EARLY"  # 1-24 hours old
    ESTABLISHING = "ESTABLISHING"  # 1-7 days old
    MATURE = "MATURE"  # > 7 days old


@dataclass
class HolderDistribution:
    """Token holder distribution analysis"""
    top_holders: List[Dict] = field(default_factory=list)
    top1_pct: float = 0.0
    top5_pct: float = 0.0
    top10_pct: float = 0.0
    top20_pct: float = 0.0
    is_centralized: bool = False
    lp_holder_rank: Optional[int] = None  # LP wallet rank in top holders
    deployer_balance_pct: float = 0.0
    total_holders: int = 0
    unique_owners: int = 0
    risk_flags: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)


@dataclass
class ContractSafety:
    """Contract security analysis"""
    is_renounced: bool = False
    is_honeypot: bool = False
    buy_tax: float = 0.0
    sell_tax: float = 0.0
    lp_locked: bool = False
    lp_lock_duration_days: Optional[int] = None
    is_mintable: bool = True
    is_proxy: bool = False
    has_blacklist: bool = False
    owner_can_change_tax: bool = False
    hidden_owner: bool = False
    safety_score: float = 0.0  # 0-100
    safety_checks_passed: int = 0
    safety_checks_total: int = 10
    
    # Solana-specific flags (from on-chain inspection)
    solana_mint_authority_revoked: bool = False  # True if mint authority is null
    solana_freeze_authority_revoked: bool = False  # True if freeze authority is null
    solana_lp_owner_type: Optional[str] = None  # 'burn', 'locker', 'EOA_unknown', 'mixed'
    solana_risk_flags: List[str] = field(default_factory=list)  # Detailed risk flags from on-chain checks
    solana_holder_distribution: Optional[HolderDistribution] = None  # Holder concentration analysis
    solana_metadata_immutable: Optional[bool] = None  # True if metadata update authority revoked
    solana_metadata_update_authority: Optional[str] = None  # Update authority address (None = immutable)


@dataclass
class SmartMoneyActivity:
    """Whale/smart money wallet activity"""
    wallet_address: str
    wallet_name: Optional[str] = None
    wallet_score: float = 0.0  # Historical success rate
    action: str = "BUY"  # BUY, SELL, ADD_LIQUIDITY
    amount_usd: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    transaction_hash: str = ""
    is_dev_wallet: bool = False
    is_known_whale: bool = False


@dataclass
class SocialSignal:
    """Social media/community signals"""
    source: str  # twitter, reddit, telegram, discord
    mention_count: int = 0
    sentiment_score: float = 0.0  # -1 to 1
    engagement_score: float = 0.0  # 0-100
    top_mentions: List[str] = field(default_factory=list)
    influencer_mentions: List[str] = field(default_factory=list)
    trending_rank: Optional[int] = None


@dataclass
class DexPair:
    """DEX trading pair information"""
    pair_address: str
    dex_name: str  # Uniswap, PancakeSwap, Raydium, etc.
    chain: Chain
    base_token_symbol: str
    base_token_address: str
    quote_token_symbol: str = "USDT"
    liquidity_usd: float = 0.0
    volume_24h: float = 0.0
    price_usd: float = 0.0
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    price_change_24h: float = 0.0
    txn_count_5m: int = 0
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    buys_5m: int = 0
    sells_5m: int = 0
    created_at: Optional[datetime] = None
    pair_age_hours: float = 0.0
    url: str = ""


@dataclass
class TokenLaunch:
    """Complete token launch data"""
    # Basic Info
    symbol: str
    name: str
    contract_address: str
    chain: Chain
    
    # Market Data
    price_usd: float = 0.0
    market_cap: float = 0.0
    fully_diluted_valuation: float = 0.0
    liquidity_usd: float = 0.0
    volume_24h: float = 0.0
    
    # Price Action
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    ath_price: Optional[float] = None
    atl_price: Optional[float] = None
    
    # Supply
    circulating_supply: float = 0.0
    total_supply: float = 0.0
    max_supply: Optional[float] = None
    
    # Launch Info
    launch_stage: LaunchStage = LaunchStage.JUST_LAUNCHED
    launched_at: Optional[datetime] = None
    age_hours: float = 0.0
    detected_at: datetime = field(default_factory=datetime.now)
    
    # Launch Timing Indicators (NEW)
    launch_timing: str = "UNKNOWN"  # ULTRA_FRESH, FRESH, EARLY, LATE, MISSED_PUMP
    minutes_since_launch: float = 0.0
    time_to_pump: Optional[str] = None  # "PRIME", "HEATING", "COOLING", "DUMPED"
    missed_pump_likely: bool = False
    breakout_potential: float = 0.0  # 0-100, super fresh coins prime to breakout
    
    # DEX Pairs
    pairs: List[DexPair] = field(default_factory=list)
    primary_dex: Optional[str] = None
    
    # Security
    contract_safety: Optional[ContractSafety] = None
    risk_level: RiskLevel = RiskLevel.HIGH
    holder_distribution: Optional[HolderDistribution] = None  # Holder concentration (for Solana)
    
    # Price Validation (cross-source verification)
    price_consistency_score: Optional[float] = None  # 0-100, higher = more consistent across sources
    price_validation_warnings: List[str] = field(default_factory=list)  # Warnings from price validation
    
    # Activity
    smart_money: List[SmartMoneyActivity] = field(default_factory=list)
    social_signals: Optional[SocialSignal] = None
    
    # Scoring
    pump_potential_score: float = 0.0  # 0-100, higher = more likely to pump
    velocity_score: float = 0.0  # Price momentum score
    volume_spike_score: float = 0.0  # 0-100, volume surge detection (NEW!)
    social_buzz_score: float = 0.0  # Social attention score
    whale_activity_score: float = 0.0  # Smart money interest score
    composite_score: float = 0.0  # Overall launch quality score
    
    # Enhanced Timing Score (NEW)
    timing_advantage_score: float = 0.0  # 0-100, how early you are (100=very early)
    
    # Entry Recommendation (NEW - December 2025)
    # Dict with: recommendation, reason, confidence, age_status, initial_trading_passed, pump_forming
    entry_recommendation: Optional[Dict] = None
    
    # Profitability Analysis (NEW - December 2025)
    breakeven_gain_needed: float = 0.0  # % gain needed to break even after slippage/fees
    real_profit_potential: bool = True  # Whether trade is potentially profitable
    liquidity_tier: str = "UNKNOWN"  # MICRO, LOW, MEDIUM, GOOD, HIGH
    total_round_trip_cost_pct: float = 0.0  # Total slippage + fees + price impact
    estimated_real_profit_pct: float = 0.0  # Real profit if +50% displayed gain
    
    # Alerts
    alert_reasons: List[str] = field(default_factory=list)
    alert_priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    
    # Metadata
    tags: List[str] = field(default_factory=list)  # meme, gaming, defi, etc.
    description: Optional[str] = None
    website: Optional[str] = None
    twitter: Optional[str] = None
    telegram: Optional[str] = None
    
    # Tracking
    is_tracked: bool = False
    added_to_watchlist: bool = False
    notified: bool = False


@dataclass
class WatchedWallet:
    """Smart money wallet being tracked"""
    address: str
    name: str
    description: str = ""
    chain: Chain = Chain.ETH
    success_rate: float = 0.0  # % of profitable trades
    avg_multiple: float = 0.0  # Average gain multiple
    total_trades: int = 0
    profitable_trades: int = 0
    last_activity: Optional[datetime] = None
    is_active: bool = True
    tags: List[str] = field(default_factory=list)  # whale, dev, influencer, etc.
    alert_on_buy: bool = True
    alert_on_sell: bool = False
    min_transaction_usd: float = 1000.0  # Only alert above this threshold


@dataclass
class VerificationChecklist:
    """Manual verification checklist for risky new launches"""
    # What to check
    check_liquidity: bool = False
    check_holders: bool = False
    check_contract: bool = False
    check_socials: bool = False
    check_volume_trend: bool = False
    check_dev_activity: bool = False
    
    # Research links
    dexscreener_url: str = ""
    etherscan_url: str = ""
    telegram_search_url: str = ""
    twitter_search_url: str = ""
    holders_url: str = ""
    
    # User notes
    notes: str = ""
    verified_at: Optional[datetime] = None


@dataclass
class LaunchAlert:
    """Alert for new token launch"""
    token: TokenLaunch
    alert_type: str  # NEW_LAUNCH, PUMP_DETECTED, WHALE_BUY, SOCIAL_BUZZ
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    reasoning: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    sent_to_discord: bool = False
    acknowledged: bool = False
    
    # Verification checklist (NEW)
    checklist: Optional[VerificationChecklist] = None


@dataclass
class HunterConfig:
    """Configuration for DEX Launch Hunter"""
    # Chains to monitor
    enabled_chains: List[Chain] = field(default_factory=lambda: [Chain.ETH, Chain.BSC, Chain.SOLANA])
    
    # Filters
    min_liquidity_usd: float = 1000.0  # Lower to catch early launches
    max_liquidity_usd: float = 5000000.0  # Higher ceiling for trending tokens
    min_volume_24h: float = 500.0  # Lower to catch fresh launches
    max_buy_tax: float = 10.0
    max_sell_tax: float = 10.0
    require_lp_locked: bool = False  # Many new tokens don't have this yet
    min_lp_lock_days: int = 30
    
    # Age filters
    max_age_hours: float = 168.0  # 7 days - catch recent launches, not just ultra-new
    alert_within_minutes: float = 30.0  # Alert if < 30 min old
    
    # Risk tolerance
    max_risk_level: RiskLevel = RiskLevel.HIGH  # Accept higher risk for early launches
    min_safety_score: float = 30.0  # Lower threshold - many new tokens lack data
    
    # Social filters
    min_social_buzz: float = 10.0  # Lower - new tokens may not have buzz yet
    min_mention_count: int = 3  # Lower - catch early mentions
    
    # Smart money
    track_smart_wallets: bool = True
    min_whale_buy_usd: float = 1000.0  # Lower to catch more whale activity
    
    # Scoring thresholds
    min_pump_potential: float = 30.0  # Lower to catch more opportunities
    min_composite_score: float = 20.0  # Much lower - let users decide quality
    
    # Monitoring
    scan_interval_seconds: int = 15  # Check every 15 seconds (fast mode for catching pumps)
    enable_discord_alerts: bool = True
    enable_telegram_alerts: bool = False
    
    # Safety
    auto_blacklist_honeypots: bool = True
    auto_blacklist_high_tax: bool = True  # Tax > 15%
    verify_contract_before_alert: bool = True
    
    # Lenient Mode (RECOMMENDED for new token discovery)
    # When True: Allows tokens with mint/freeze authority (common for new Solana launches)
    # These tokens are HIGHER RISK but most new meme coins keep these initially
    lenient_solana_mode: bool = True  # Default ON - most new tokens have these authorities
    
    # Discovery Settings
    discovery_mode: str = "aggressive"  # "conservative", "balanced", "aggressive"
    # aggressive = lower filters, find more tokens (higher risk)
    # balanced = moderate filters (default)
    # conservative = strict filters, fewer but safer tokens
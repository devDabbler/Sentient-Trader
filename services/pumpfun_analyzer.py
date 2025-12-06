"""
Pump.fun Token Analyzer - Direct API analysis for bonding curve tokens

This module provides pump.fun-specific token analysis BEFORE tokens graduate
to Raydium/DexScreener. It bridges the gap between bonding curve detection
and AI analysis by using pump.fun's own API data.

Features:
- Fetch token data directly from pump.fun API (not DexScreener)
- Analyze holder count, trading activity, creator history
- Check bonding curve progress and momentum
- Score tokens based on pump.fun-specific metrics
- Discord integration for alerts and commands

Usage:
    from services.pumpfun_analyzer import PumpfunAnalyzer
    
    analyzer = PumpfunAnalyzer()
    analysis = await analyzer.analyze_token(mint_address)
    score = analysis.score  # 0-100

Author: Sentient Trader
Created: December 2025
"""

import asyncio
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import httpx
from loguru import logger

# Discord integration
try:
    from src.integrations.discord_channels import AlertCategory, get_discord_webhook
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("Discord channels not available")


class TrustTier(Enum):
    """
    AI Trust Tier for pump.fun tokens - from most trustworthy to most risky.
    
    This helps traders quickly assess token validity by tiering based on:
    - Token age (older = more trusted IF still active)
    - Buy/sell patterns (more buys, diverse buyers = better)
    - Creator history (serial ruggers = avoid)
    - Holder distribution (whale concentration = risky)
    - Social presence (verified socials = better)
    """
    TIER_1_TRUSTED = "🟢 TRUSTED"           # Rare - verified project, good history
    TIER_2_PROMISING = "🔵 PROMISING"       # Good signals, worth a gamble
    TIER_3_NEUTRAL = "🟡 NEUTRAL"           # Mixed signals, proceed with caution
    TIER_4_RISKY = "🟠 RISKY"               # Red flags present, small bets only
    TIER_5_RUG_LIKELY = "🔴 RUG LIKELY"     # High probability rug pull


class PumpfunRisk(Enum):
    """Risk level for pump.fun token"""
    EXTREME = "EXTREME"      # Don't touch - likely rug
    HIGH = "HIGH"            # Very risky - max $10
    MEDIUM = "MEDIUM"        # Risky - max $25
    MODERATE = "MODERATE"    # Standard gambling - max $50
    LOW = "LOW"              # Decent setup - still gambling


@dataclass
class PumpfunTokenAnalysis:
    """Analysis result for a pump.fun bonding curve token"""
    # Identity
    mint: str
    symbol: str
    name: str
    
    # Bonding curve metrics
    progress_pct: float = 0.0           # 0-100%
    market_cap_sol: float = 0.0
    market_cap_usd: float = 0.0
    virtual_sol_reserves: float = 0.0
    virtual_token_reserves: float = 0.0
    
    # Trading activity
    total_trades: int = 0
    buy_count: int = 0
    sell_count: int = 0
    volume_sol: float = 0.0
    trades_1m: int = 0                  # Last minute
    trades_5m: int = 0                  # Last 5 minutes
    
    # Holder analysis
    holder_count: int = 0
    top_holder_pct: float = 0.0         # Top holder's percentage
    creator_holding_pct: float = 0.0    # Creator's remaining holding
    unique_traders: int = 0
    
    # Creator analysis
    creator_address: str = ""
    creator_token_count: int = 0        # How many tokens creator has made
    creator_rug_count: int = 0          # How many rugs
    creator_graduation_count: int = 0   # How many graduations
    
    # Social presence
    has_twitter: bool = False
    has_telegram: bool = False
    has_website: bool = False
    twitter_url: str = ""
    telegram_url: str = ""
    website_url: str = ""
    
    # Momentum signals
    buy_pressure: float = 0.5           # 0-1, ratio of buys
    velocity_score: float = 0.0         # Trading velocity
    momentum_score: float = 0.0         # Overall momentum
    
    # Risk assessment
    risk_level: PumpfunRisk = PumpfunRisk.HIGH
    risk_factors: List[str] = field(default_factory=list)
    green_flags: List[str] = field(default_factory=list)
    
    # Final scores
    score: float = 0.0                  # Overall score 0-100
    pump_potential: float = 0.0         # Chance of pumping 0-100
    rug_risk: float = 0.0               # Chance of rug 0-100
    
    # Recommendation
    recommendation: str = "SKIP"        # SKIP, WATCH, GAMBLE_SMALL, GAMBLE
    max_bet_usd: float = 0.0            # Suggested max bet
    reasoning: str = ""
    
    # Trust tier (NEW - AI-based validity assessment)
    trust_tier: TrustTier = TrustTier.TIER_4_RISKY
    trust_reasoning: str = ""           # Why this tier was assigned
    age_minutes: float = 0.0            # How old the token is
    age_category: str = "UNKNOWN"       # FRESH (<5m), EARLY (<30m), SETTLING (<2h), MATURE (>2h)
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    data_age_seconds: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        data = asdict(self)
        data['analyzed_at'] = self.analyzed_at.isoformat()
        data['risk_level'] = self.risk_level.value
        data['trust_tier'] = self.trust_tier.value
        return data


class PumpfunAnalyzer:
    """
    Analyze pump.fun bonding curve tokens directly.
    
    This provides real analysis for tokens BEFORE they graduate to DEX.
    Uses pump.fun's own API endpoints instead of DexScreener.
    """
    
    # pump.fun API endpoints (unofficial but widely used)
    PUMPFUN_API = "https://frontend-api.pump.fun"
    PUMPPORTAL_API = "https://pumpportal.fun/api"
    
    # SOL price cache
    _sol_price_usd: float = 0.0
    _sol_price_updated: datetime = datetime.min
    
    # Analysis cache
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    CACHE_FILE = DATA_DIR / "pumpfun_analysis_cache.json"
    
    def __init__(self, max_bet_default: float = 25.0):
        """
        Initialize Pump.fun Analyzer
        
        Args:
            max_bet_default: Default max bet for GAMBLE recommendation
        """
        self.max_bet_default = max_bet_default
        self._analysis_cache: Dict[str, PumpfunTokenAnalysis] = {}
        self._cache_ttl_seconds = 60  # Cache for 1 minute
        
        # Discord webhook for alerts
        self.discord_webhook_url = self._get_discord_webhook()
        
        # Ensure data directory exists
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎰 Pump.fun Analyzer initialized")
        logger.info(f"   Default max bet: ${max_bet_default}")
        logger.info(f"   Discord: {'✅ Enabled' if self.discord_webhook_url else '❌ Disabled'}")
    
    def _get_discord_webhook(self) -> Optional[str]:
        """Get Discord webhook URL - routes to PUMPFUN_ALERTS channel"""
        if DISCORD_AVAILABLE:
            webhook = get_discord_webhook(AlertCategory.PUMPFUN_ALERTS)
            if webhook:
                return webhook
            # Fallback to DEX pump alerts
            webhook = get_discord_webhook(AlertCategory.DEX_PUMP_ALERTS)
            if webhook:
                return webhook
        # NO fallback to general DISCORD_WEBHOOK_URL - prevents duplicate alerts to general channel
        # If no pumpfun-specific webhook is configured, alerts are disabled
        logger.warning("No PUMPFUN_ALERTS webhook configured - pumpfun analyzer alerts disabled")
        return None
    
    async def analyze_token(
        self, 
        mint: str, 
        force_refresh: bool = False
    ) -> Optional[PumpfunTokenAnalysis]:
        """
        Analyze a pump.fun token by mint address
        
        Args:
            mint: Token mint address (Solana pubkey)
            force_refresh: Skip cache and fetch fresh data
            
        Returns:
            PumpfunTokenAnalysis or None if failed
        """
        # Check cache first
        if not force_refresh and mint in self._analysis_cache:
            cached = self._analysis_cache[mint]
            age = (datetime.now() - cached.analyzed_at).total_seconds()
            if age < self._cache_ttl_seconds:
                logger.debug(f"Using cached analysis for {cached.symbol} ({age:.0f}s old)")
                return cached
        
        try:
            # Fetch token data from pump.fun API
            token_data = await self._fetch_token_data(mint)
            if not token_data:
                logger.warning(f"Could not fetch token data for {mint}")
                return None
            
            # Fetch trades/activity data
            trades_data = await self._fetch_token_trades(mint)
            
            # Fetch holder data if available
            holder_data = await self._fetch_holder_data(mint)
            
            # Build analysis
            analysis = await self._build_analysis(
                mint=mint,
                token_data=token_data,
                trades_data=trades_data,
                holder_data=holder_data
            )
            
            # Cache result
            self._analysis_cache[mint] = analysis
            
            logger.info(f"📊 Analyzed {analysis.symbol}: Score={analysis.score:.0f} | Risk={analysis.risk_level.value}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {mint}: {e}")
            return None
    
    async def _fetch_token_data(self, mint: str) -> Optional[dict]:
        """Fetch token data from pump.fun API"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Try pump.fun frontend API first
                url = f"{self.PUMPFUN_API}/coins/{mint}"
                response = await client.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.debug(f"Got token data from pump.fun API")
                    return data
                
                # Fallback to PumpPortal API
                url = f"{self.PUMPPORTAL_API}/data/token/{mint}"
                response = await client.get(url)
                
                if response.status_code == 200:
                    logger.debug(f"Got token data from PumpPortal API")
                    return response.json()
                
                logger.warning(f"Token API returned {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching token data: {e}")
            return None
    
    async def _fetch_token_trades(self, mint: str, limit: int = 100) -> List[dict]:
        """Fetch recent trades for token"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                url = f"{self.PUMPFUN_API}/trades/latest"
                params = {"mint": mint, "limit": limit}
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    return response.json()
                return []
                
        except Exception as e:
            logger.debug(f"Error fetching trades: {e}")
            return []
    
    async def _fetch_holder_data(self, mint: str) -> Optional[dict]:
        """Fetch holder distribution data"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                url = f"{self.PUMPFUN_API}/coins/{mint}/holders"
                response = await client.get(url)
                
                if response.status_code == 200:
                    return response.json()
                return None
                
        except Exception as e:
            logger.debug(f"Error fetching holders: {e}")
            return None
    
    async def _get_sol_price(self) -> float:
        """Get current SOL price in USD"""
        # Use cached price if recent
        if (datetime.now() - self._sol_price_updated).total_seconds() < 300:  # 5 min cache
            return self._sol_price_usd
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Use CoinGecko simple price
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {"ids": "solana", "vs_currencies": "usd"}
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    self._sol_price_usd = data.get("solana", {}).get("usd", 200.0)
                    self._sol_price_updated = datetime.now()
                    return self._sol_price_usd
        except Exception as e:
            logger.debug(f"Error fetching SOL price: {e}")
        
        # Default fallback
        return self._sol_price_usd if self._sol_price_usd > 0 else 200.0
    
    async def _build_analysis(
        self,
        mint: str,
        token_data: dict,
        trades_data: List[dict],
        holder_data: Optional[dict]
    ) -> PumpfunTokenAnalysis:
        """Build comprehensive analysis from raw data"""
        
        sol_price = await self._get_sol_price()
        
        # Extract basic info
        symbol = token_data.get("symbol", "???")
        name = token_data.get("name", "Unknown")
        creator = token_data.get("creator", token_data.get("traderPublicKey", ""))
        
        # Bonding curve metrics
        virtual_sol = float(token_data.get("virtual_sol_reserves", token_data.get("vSolInBondingCurve", 0)))
        virtual_tokens = float(token_data.get("virtual_token_reserves", token_data.get("vTokensInBondingCurve", 0)))
        market_cap_sol = float(token_data.get("market_cap", token_data.get("marketCapSol", 0)))
        
        # pump.fun graduates at ~85 SOL in bonding curve
        progress_pct = min((virtual_sol / 85.0) * 100, 100) if virtual_sol > 0 else 0
        
        # Analyze trades
        total_trades = len(trades_data)
        buy_count = sum(1 for t in trades_data if t.get("is_buy", t.get("txType") == "buy"))
        sell_count = total_trades - buy_count
        volume_sol = sum(float(t.get("sol_amount", t.get("solAmount", 0))) for t in trades_data)
        
        # Recent activity (last 1 and 5 minutes)
        now = datetime.now()
        trades_1m = 0
        trades_5m = 0
        for trade in trades_data:
            trade_time = trade.get("timestamp")
            if trade_time:
                try:
                    if isinstance(trade_time, (int, float)):
                        trade_dt = datetime.fromtimestamp(trade_time / 1000 if trade_time > 1e10 else trade_time)
                    else:
                        trade_dt = datetime.fromisoformat(str(trade_time).replace('Z', '+00:00'))
                    
                    age = (now - trade_dt.replace(tzinfo=None)).total_seconds()
                    if age < 60:
                        trades_1m += 1
                    if age < 300:
                        trades_5m += 1
                except:
                    pass
        
        # Holder analysis
        holder_count = 0
        top_holder_pct = 0.0
        if holder_data:
            if isinstance(holder_data, list):
                holder_count = len(holder_data)
                if holder_count > 0 and 'percentage' in holder_data[0]:
                    top_holder_pct = float(holder_data[0].get('percentage', 0))
            elif isinstance(holder_data, dict):
                holder_count = holder_data.get('count', 0)
        
        # Social presence
        twitter_url = token_data.get("twitter", "")
        telegram_url = token_data.get("telegram", "")
        website_url = token_data.get("website", "")
        
        # Calculate scores
        buy_pressure = buy_count / total_trades if total_trades > 0 else 0.5
        velocity_score = min((trades_5m / 50) * 100, 100)  # 50 trades in 5m = 100%
        
        # Risk factors
        risk_factors = []
        green_flags = []
        
        # Analyze risk
        if progress_pct < 10:
            risk_factors.append("Very early (<10% progress)")
        elif progress_pct > 80:
            green_flags.append(f"Near graduation ({progress_pct:.0f}%)")
        
        if holder_count < 10:
            risk_factors.append(f"Very few holders ({holder_count})")
        elif holder_count > 50:
            green_flags.append(f"Good holder count ({holder_count})")
        
        if top_holder_pct > 30:
            risk_factors.append(f"Top holder owns {top_holder_pct:.0f}%")
        
        if total_trades < 10:
            risk_factors.append(f"Low activity ({total_trades} trades)")
        elif total_trades > 100:
            green_flags.append(f"High activity ({total_trades} trades)")
        
        if buy_pressure < 0.4:
            risk_factors.append(f"Sell pressure ({buy_pressure:.0%} buys)")
        elif buy_pressure > 0.6:
            green_flags.append(f"Buy pressure ({buy_pressure:.0%} buys)")
        
        if not twitter_url and not telegram_url:
            risk_factors.append("No social links")
        else:
            if twitter_url:
                green_flags.append("Has Twitter")
            if telegram_url:
                green_flags.append("Has Telegram")
        
        # Calculate overall score
        base_score = 50
        
        # Progress bonus (tokens near graduation are more proven)
        base_score += (progress_pct / 100) * 20
        
        # Activity bonus
        if total_trades > 50:
            base_score += 10
        elif total_trades > 20:
            base_score += 5
        
        # Buy pressure
        base_score += (buy_pressure - 0.5) * 20
        
        # Holder distribution
        if holder_count > 30:
            base_score += 5
        if top_holder_pct < 20:
            base_score += 5
        
        # Social presence
        if twitter_url:
            base_score += 5
        if telegram_url:
            base_score += 3
        
        # Velocity (active trading = good)
        base_score += velocity_score * 0.1
        
        # Apply risk penalty
        base_score -= len(risk_factors) * 3
        
        # Clamp score
        score = max(0, min(100, base_score))
        
        # Pump potential vs rug risk
        pump_potential = min(100, score * 1.2)  # More optimistic
        rug_risk = max(0, 100 - score * 0.8)    # Risk estimate
        
        # Determine risk level
        if score < 20 or len(risk_factors) > 4:
            risk_level = PumpfunRisk.EXTREME
        elif score < 35 or len(risk_factors) > 3:
            risk_level = PumpfunRisk.HIGH
        elif score < 50:
            risk_level = PumpfunRisk.MEDIUM
        elif score < 70:
            risk_level = PumpfunRisk.MODERATE
        else:
            risk_level = PumpfunRisk.LOW
        
        # Recommendation and max bet
        if risk_level == PumpfunRisk.EXTREME:
            recommendation = "SKIP"
            max_bet = 0
            reasoning = f"Too risky: {', '.join(risk_factors[:3])}"
        elif risk_level == PumpfunRisk.HIGH:
            recommendation = "WATCH" if progress_pct > 50 else "SKIP"
            max_bet = 10
            reasoning = f"High risk gamble. {'Watching progress.' if progress_pct > 50 else 'Skip unless YOLO.'}"
        elif risk_level == PumpfunRisk.MEDIUM:
            recommendation = "GAMBLE_SMALL"
            max_bet = 25
            reasoning = f"Medium risk. {green_flags[0] if green_flags else 'Some potential.'}"
        elif risk_level == PumpfunRisk.MODERATE:
            recommendation = "GAMBLE"
            max_bet = 50
            reasoning = f"Decent setup. {', '.join(green_flags[:2])}"
        else:
            recommendation = "GAMBLE"
            max_bet = self.max_bet_default
            reasoning = f"Good signals: {', '.join(green_flags[:3])}"
        
        # Calculate Trust Tier (NEW - AI-based validity assessment)
        trust_tier, trust_reasoning, age_minutes, age_category = self._calculate_trust_tier(
            token_data=token_data,
            progress_pct=progress_pct,
            total_trades=total_trades,
            buy_count=buy_count,
            sell_count=sell_count,
            holder_count=holder_count,
            top_holder_pct=top_holder_pct,
            twitter_url=twitter_url,
            telegram_url=telegram_url,
            website_url=website_url,
            buy_pressure=buy_pressure,
            risk_factors=risk_factors,
            green_flags=green_flags,
            score=score
        )
        
        # Build analysis object
        analysis = PumpfunTokenAnalysis(
            mint=mint,
            symbol=symbol,
            name=name,
            progress_pct=progress_pct,
            market_cap_sol=market_cap_sol,
            market_cap_usd=market_cap_sol * sol_price,
            virtual_sol_reserves=virtual_sol,
            virtual_token_reserves=virtual_tokens,
            total_trades=total_trades,
            buy_count=buy_count,
            sell_count=sell_count,
            volume_sol=volume_sol,
            trades_1m=trades_1m,
            trades_5m=trades_5m,
            holder_count=holder_count,
            top_holder_pct=top_holder_pct,
            creator_address=creator,
            has_twitter=bool(twitter_url),
            has_telegram=bool(telegram_url),
            has_website=bool(website_url),
            twitter_url=twitter_url,
            telegram_url=telegram_url,
            website_url=website_url,
            buy_pressure=buy_pressure,
            velocity_score=velocity_score,
            momentum_score=(buy_pressure * 50) + (velocity_score * 0.5),
            risk_level=risk_level,
            risk_factors=risk_factors,
            green_flags=green_flags,
            score=score,
            pump_potential=pump_potential,
            rug_risk=rug_risk,
            recommendation=recommendation,
            max_bet_usd=max_bet,
            reasoning=reasoning,
            trust_tier=trust_tier,
            trust_reasoning=trust_reasoning,
            age_minutes=age_minutes,
            age_category=age_category,
            analyzed_at=datetime.now(),
        )
        
        return analysis
    
    def _calculate_trust_tier(
        self,
        token_data: dict,
        progress_pct: float,
        total_trades: int,
        buy_count: int,
        sell_count: int,
        holder_count: int,
        top_holder_pct: float,
        twitter_url: str,
        telegram_url: str,
        website_url: str,
        buy_pressure: float,
        risk_factors: List[str],
        green_flags: List[str],
        score: float
    ) -> tuple:
        """
        Calculate AI Trust Tier based on multiple validity criteria.
        
        This tiers tokens from most trustworthy to most likely rug pull,
        analyzing buys, age, creator history, holder patterns, and socials.
        
        Returns:
            (TrustTier, reasoning_str, age_minutes, age_category)
        """
        trust_score = 0
        trust_reasons = []
        
        # 1. TOKEN AGE ANALYSIS
        created_timestamp = token_data.get("created_timestamp", token_data.get("createdAt", 0))
        if created_timestamp:
            try:
                if isinstance(created_timestamp, (int, float)):
                    # Handle milliseconds
                    if created_timestamp > 1e12:
                        created_timestamp = created_timestamp / 1000
                    created_dt = datetime.fromtimestamp(created_timestamp)
                else:
                    created_dt = datetime.fromisoformat(str(created_timestamp).replace('Z', '+00:00'))
                age_minutes = (datetime.now() - created_dt.replace(tzinfo=None)).total_seconds() / 60
            except:
                age_minutes = 0
        else:
            age_minutes = 0
        
        # Age categories and scoring
        if age_minutes < 5:
            age_category = "ULTRA_FRESH"
            # Ultra fresh is HIGH RISK but also HIGH REWARD potential
            trust_score -= 10
            trust_reasons.append(f"⚡ Ultra fresh ({age_minutes:.0f}m) - highest risk/reward")
        elif age_minutes < 15:
            age_category = "FRESH"
            trust_score -= 5
            trust_reasons.append(f"🆕 Fresh token ({age_minutes:.0f}m)")
        elif age_minutes < 30:
            age_category = "EARLY"
            trust_score += 5
            trust_reasons.append(f"⏰ Early entry window ({age_minutes:.0f}m)")
        elif age_minutes < 120:
            age_category = "SETTLING"
            trust_score += 10
            trust_reasons.append(f"📊 Settling period ({age_minutes:.0f}m) - survived initial dump")
        else:
            age_category = "MATURE"
            trust_score += 15
            trust_reasons.append(f"✅ Mature token ({age_minutes/60:.1f}h) - proven survivor")
        
        # 2. BUY PATTERN ANALYSIS
        if total_trades > 0:
            buy_ratio = buy_count / total_trades
            if buy_ratio > 0.7:
                trust_score += 15
                trust_reasons.append(f"🟢 Strong buy pressure ({buy_ratio:.0%})")
            elif buy_ratio > 0.55:
                trust_score += 8
                trust_reasons.append(f"🔵 Healthy buy ratio ({buy_ratio:.0%})")
            elif buy_ratio < 0.35:
                trust_score -= 15
                trust_reasons.append(f"🔴 Heavy selling ({buy_ratio:.0%} buys)")
            
            # Trade count bonus
            if total_trades > 200:
                trust_score += 12
                trust_reasons.append(f"📈 Very active ({total_trades} trades)")
            elif total_trades > 50:
                trust_score += 6
                trust_reasons.append(f"📊 Active trading ({total_trades} trades)")
            elif total_trades < 10:
                trust_score -= 8
                trust_reasons.append(f"⚠️ Low activity ({total_trades} trades)")
        
        # 3. HOLDER DISTRIBUTION ANALYSIS
        if holder_count > 100:
            trust_score += 15
            trust_reasons.append(f"👥 Wide distribution ({holder_count} holders)")
        elif holder_count > 50:
            trust_score += 10
            trust_reasons.append(f"👥 Good holder base ({holder_count})")
        elif holder_count > 20:
            trust_score += 5
        elif holder_count < 10:
            trust_score -= 10
            trust_reasons.append(f"⚠️ Few holders ({holder_count})")
        
        # Whale concentration
        if top_holder_pct > 50:
            trust_score -= 20
            trust_reasons.append(f"🐋 WHALE ALERT: Top holder owns {top_holder_pct:.0f}%")
        elif top_holder_pct > 30:
            trust_score -= 10
            trust_reasons.append(f"🐋 Concentrated: Top holder {top_holder_pct:.0f}%")
        elif top_holder_pct < 15:
            trust_score += 8
            trust_reasons.append(f"✅ Decentralized (top: {top_holder_pct:.0f}%)")
        
        # 4. SOCIAL PRESENCE ANALYSIS
        social_count = sum([bool(twitter_url), bool(telegram_url), bool(website_url)])
        if social_count >= 3:
            trust_score += 15
            trust_reasons.append("🌐 Full social presence (Twitter, TG, Website)")
        elif social_count == 2:
            trust_score += 8
            trust_reasons.append("📱 Good socials (2 platforms)")
        elif social_count == 1:
            trust_score += 3
        else:
            trust_score -= 5
            trust_reasons.append("❌ No social links")
        
        # 5. BONDING CURVE PROGRESS
        if progress_pct > 80:
            trust_score += 15
            trust_reasons.append(f"🎓 Near graduation ({progress_pct:.0f}%)")
        elif progress_pct > 50:
            trust_score += 8
            trust_reasons.append(f"📈 Good progress ({progress_pct:.0f}%)")
        elif progress_pct < 10:
            trust_score -= 5
            trust_reasons.append(f"📉 Early stage ({progress_pct:.0f}%)")
        
        # 6. CREATOR HISTORY (if available)
        creator_token_count = token_data.get("creator_token_count", 0)
        if creator_token_count > 10:
            trust_score -= 15
            trust_reasons.append(f"⚠️ Serial creator ({creator_token_count} tokens)")
        elif creator_token_count > 5:
            trust_score -= 8
            trust_reasons.append(f"📋 Multi-token creator ({creator_token_count})")
        
        # 7. COMBINE WITH EXISTING SCORE
        trust_score += (score - 50) * 0.3  # Weight from existing analysis
        
        # Determine Trust Tier
        if trust_score >= 40:
            tier = TrustTier.TIER_1_TRUSTED
        elif trust_score >= 20:
            tier = TrustTier.TIER_2_PROMISING
        elif trust_score >= 0:
            tier = TrustTier.TIER_3_NEUTRAL
        elif trust_score >= -20:
            tier = TrustTier.TIER_4_RISKY
        else:
            tier = TrustTier.TIER_5_RUG_LIKELY
        
        # Build reasoning string
        reasoning = " | ".join(trust_reasons[:4]) if trust_reasons else "Insufficient data"
        
        return tier, reasoning, age_minutes, age_category

    async def send_creation_alert(self, token) -> bool:
        """
        Send FAST creation alert for new token - no analysis, just detection.
        Use this for immediate alerts when PUMPFUN_ALERT_ON_CREATION=true.
        
        Args:
            token: BondingToken from bonding_curve_monitor
            
        Returns:
            True if alert sent successfully
        """
        if not self.discord_webhook_url:
            logger.warning(f"No webhook URL configured, cannot send creation alert for {token.symbol}")
            return False
        
        logger.info(f"📤 Sending creation alert for {token.symbol} to Discord...")
        
        try:
            # Quick creation alert - minimal API calls for speed
            embed = {
                "title": f"🆕 NEW TOKEN DETECTED: {token.symbol}",
                "description": f"**{token.name}**\n\n🚀 Just launched on pump.fun!",
                "color": 0x00FFFF,  # Cyan for new token
                "fields": [
                    {
                        "name": "📊 Early Stats",
                        "value": f"Trades: {token.total_trades}\nVolume: {token.total_volume_sol:.2f} SOL",
                        "inline": True
                    },
                    {
                        "name": "📈 Bonding",
                        "value": f"Progress: {token.progress_pct:.1f}%\nMcap: ~${token.market_cap_usd:,.0f}",
                        "inline": True
                    },
                    {
                        "name": "💡 Action",
                        "value": "Reply `ANALYZE` for full analysis\nor jump in early!",
                        "inline": True
                    },
                ],
                "footer": {
                    "text": f"⚡ FAST ALERT | pump.fun | {datetime.now().strftime('%H:%M:%S')}"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add trade link
            embed["fields"].append({
                "name": "🔗 Quick Links",
                "value": f"[pump.fun](https://pump.fun/{token.mint}) | [DexScreener](https://dexscreener.com/solana/{token.mint})",
                "inline": False
            })
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                payload = {"embeds": [embed]}
                response = await client.post(self.discord_webhook_url, json=payload)
                
                if response.status_code in [200, 204]:
                    logger.info(f"⚡ Sent creation alert for {token.symbol}")
                    return True
                else:
                    logger.warning(f"Discord returned {response.status_code}: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending creation alert: {e}")
            return False
    
    async def send_graduation_alert(self, migration) -> bool:
        """
        Send FAST graduation alert - token hit 100% and is now on DEX.
        
        Args:
            migration: MigrationEvent from bonding_curve_monitor
            
        Returns:
            True if alert sent successfully
        """
        if not self.discord_webhook_url:
            return False
        
        try:
            embed = {
                "title": f"🎓 GRADUATED: {migration.symbol}",
                "description": f"**Token hit 100% bonding curve!**\n\nNow tradeable on Raydium/DEX - liquidity is live!",
                "color": 0xFFD700,  # Gold for graduation
                "fields": [
                    {
                        "name": "🏆 Status",
                        "value": "✅ Graduated\n✅ On DEX\n✅ Liquidity Added",
                        "inline": True
                    },
                    {
                        "name": "💰 Initial Liquidity",
                        "value": f"~{migration.initial_liquidity_sol:.1f} SOL" if migration.initial_liquidity_sol > 0 else "Check DEX",
                        "inline": True
                    },
                    {
                        "name": "⏰ Timing",
                        "value": "🔥 Early entry moment!\nVolume usually spikes after grad",
                        "inline": True
                    },
                ],
                "footer": {
                    "text": f"🎓 GRADUATION ALERT | {datetime.now().strftime('%H:%M:%S')}"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add trade links
            embed["fields"].append({
                "name": "🔗 Trade Now",
                "value": f"[DexScreener](https://dexscreener.com/solana/{migration.mint}) | [Raydium](https://raydium.io/swap/?inputCurrency=sol&outputCurrency={migration.mint})",
                "inline": False
            })
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                payload = {"embeds": [embed]}
                response = await client.post(self.discord_webhook_url, json=payload)
                
                if response.status_code in [200, 204]:
                    logger.info(f"🎓 Sent graduation alert for {migration.symbol}")
                    return True
                else:
                    logger.warning(f"Discord returned {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending graduation alert: {e}")
            return False

    async def send_analysis_alert(self, analysis: PumpfunTokenAnalysis):
        """Send analysis result to Discord"""
        if not self.discord_webhook_url:
            return
        
        try:
            # Color based on recommendation
            color_map = {
                "SKIP": 0xFF0000,         # Red
                "WATCH": 0xFFAA00,        # Orange
                "GAMBLE_SMALL": 0xFFFF00, # Yellow
                "GAMBLE": 0x00FF00,       # Green
            }
            color = color_map.get(analysis.recommendation, 0x808080)
            
            # Emoji for risk
            risk_emoji = {
                PumpfunRisk.EXTREME: "☠️",
                PumpfunRisk.HIGH: "🔴",
                PumpfunRisk.MEDIUM: "🟡",
                PumpfunRisk.MODERATE: "🟢",
                PumpfunRisk.LOW: "💎",
            }
            
            # Build embed
            embed = {
                "title": f"🎰 PUMP.FUN ANALYSIS: {analysis.symbol}",
                "description": f"**{analysis.name}**\n\n{analysis.reasoning}",
                "color": color,
                "fields": [
                    {
                        "name": "📊 Score",
                        "value": f"**{analysis.score:.0f}/100**\n{risk_emoji.get(analysis.risk_level, '❓')} {analysis.risk_level.value}",
                        "inline": True
                    },
                    {
                        "name": "📈 Bonding Curve",
                        "value": f"Progress: **{analysis.progress_pct:.1f}%**\nMcap: ${analysis.market_cap_usd:,.0f}",
                        "inline": True
                    },
                    {
                        "name": "💰 Recommendation",
                        "value": f"**{analysis.recommendation}**\nMax: ${analysis.max_bet_usd:.0f}",
                        "inline": True
                    },
                    {
                        "name": "📊 Activity",
                        "value": f"Trades: {analysis.total_trades}\nBuys: {analysis.buy_count} ({analysis.buy_pressure:.0%})\n5m: {analysis.trades_5m} trades",
                        "inline": True
                    },
                    {
                        "name": "👥 Holders",
                        "value": f"Count: {analysis.holder_count}\nTop: {analysis.top_holder_pct:.1f}%",
                        "inline": True
                    },
                    {
                        "name": "🔗 Links",
                        "value": f"{'✅ Twitter' if analysis.has_twitter else '❌ Twitter'}\n{'✅ Telegram' if analysis.has_telegram else '❌ Telegram'}",
                        "inline": True
                    },
                ],
                "footer": {
                    "text": f"pump.fun | Reply: BUY $XX or PASS | {datetime.now().strftime('%H:%M:%S')}"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Add Trust Tier prominently
            embed["fields"].insert(0, {
                "name": "🎯 AI TRUST TIER",
                "value": f"**{analysis.trust_tier.value}**\n⏰ Age: {analysis.age_category} ({analysis.age_minutes:.0f}m)",
                "inline": False
            })
            
            # Add trust reasoning
            if analysis.trust_reasoning:
                embed["fields"].insert(1, {
                    "name": "📋 Trust Analysis",
                    "value": analysis.trust_reasoning,
                    "inline": False
                })
            
            # Add risk factors if any
            if analysis.risk_factors:
                embed["fields"].append({
                    "name": "⚠️ Risk Factors",
                    "value": "\n".join(f"• {r}" for r in analysis.risk_factors[:5]),
                    "inline": False
                })
            
            # Add green flags if any
            if analysis.green_flags:
                embed["fields"].append({
                    "name": "✅ Green Flags",
                    "value": "\n".join(f"• {g}" for g in analysis.green_flags[:5]),
                    "inline": False
                })
            
            # Add direct links
            embed["fields"].append({
                "name": "🔗 Trade Links",
                "value": f"[pump.fun](https://pump.fun/{analysis.mint}) | [DexScreener](https://dexscreener.com/solana/{analysis.mint})",
                "inline": False
            })
            
            # Send to Discord
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {"embeds": [embed]}
                response = await client.post(self.discord_webhook_url, json=payload)
                
                if response.status_code not in [200, 204]:
                    logger.warning(f"Discord returned {response.status_code}")
                else:
                    logger.info(f"📨 Sent analysis alert for {analysis.symbol}")
                    
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
    
    def format_analysis_text(self, analysis: PumpfunTokenAnalysis) -> str:
        """Format analysis as text for display"""
        lines = [
            f"🎰 **{analysis.symbol}** ({analysis.name})",
            f"",
            f"🎯 **TRUST TIER:** {analysis.trust_tier.value}",
            f"⏰ **Age:** {analysis.age_category} ({analysis.age_minutes:.0f} min)",
            f"📋 **Trust Analysis:** {analysis.trust_reasoning}",
            f"",
            f"**Score:** {analysis.score:.0f}/100 | **Risk:** {analysis.risk_level.value}",
            f"**Recommendation:** {analysis.recommendation} (Max: ${analysis.max_bet_usd:.0f})",
            f"",
            f"📈 **Bonding Curve:** {analysis.progress_pct:.1f}% | Mcap: ${analysis.market_cap_usd:,.0f}",
            f"📊 **Activity:** {analysis.total_trades} trades | {analysis.buy_pressure:.0%} buys",
            f"👥 **Holders:** {analysis.holder_count} | Top: {analysis.top_holder_pct:.1f}%",
            f"",
            f"**Reasoning:** {analysis.reasoning}",
        ]
        
        if analysis.risk_factors:
            lines.append(f"")
            lines.append(f"⚠️ **Risks:** {', '.join(analysis.risk_factors[:3])}")
        
        if analysis.green_flags:
            lines.append(f"✅ **Signals:** {', '.join(analysis.green_flags[:3])}")
        
        lines.append(f"")
        lines.append(f"🔗 https://pump.fun/{analysis.mint}")
        
        return "\n".join(lines)


# Singleton instance
_pumpfun_analyzer: Optional[PumpfunAnalyzer] = None


def get_pumpfun_analyzer(max_bet_default: float = 25.0) -> PumpfunAnalyzer:
    """Get or create singleton PumpfunAnalyzer instance"""
    global _pumpfun_analyzer
    if _pumpfun_analyzer is None:
        _pumpfun_analyzer = PumpfunAnalyzer(max_bet_default=max_bet_default)
    return _pumpfun_analyzer

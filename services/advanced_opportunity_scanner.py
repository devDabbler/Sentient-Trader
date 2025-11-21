"""
Advanced Opportunity Scanner

Scans for top stocks and options with advanced filtering capabilities.
Designed to catch buzzing stocks and obscure plays before they rocket.
"""

from loguru import logger
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.penny_stock_constants import PENNY_THRESHOLDS, get_price_tier_bonus

from .top_trades_scanner import TopTradesScanner, TopTrade
from .ai_confidence_scanner import AIConfidenceScanner, AIConfidenceTrade
from .social_sentiment_analyzer import SocialSentimentAnalyzer
from analyzers.comprehensive import ComprehensiveAnalyzer
from models.analysis import StockAnalysis
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import time



class ScanType(Enum):
    """Type of scan to perform"""
    OPTIONS = "options"
    STOCKS = "stocks"
    PENNY_STOCKS = "penny_stocks"
    BREAKOUTS = "breakouts"
    MOMENTUM = "momentum"
    BUZZING = "buzzing"
    HOTTEST_STOCKS = "hottest_stocks"
    ALL = "all"


@dataclass
class ScanFilters:
    """Filters for opportunity scanning"""
    # Price filters
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    
    # Volume filters
    min_volume: Optional[int] = None
    min_volume_ratio: Optional[float] = None  # Ratio to average volume
    
    # Momentum filters
    min_change_pct: Optional[float] = None
    max_change_pct: Optional[float] = None
    
    # Score filters
    min_score: Optional[float] = 50.0
    min_confidence_score: Optional[float] = None
    min_ai_rating: Optional[float] = None
    
    # Market cap filters (in millions)
    min_market_cap: Optional[float] = None
    max_market_cap: Optional[float] = None
    
    # Technical filters
    trend_filter: Optional[List[str]] = None  # ["UPTREND", "STRONG UPTREND", etc.]
    require_power_zone: bool = False
    require_ema_reclaim: bool = False
    require_timeframe_alignment: bool = False
    
    # RSI filters
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    
    # IV filters (for options)
    min_iv_rank: Optional[float] = None
    max_iv_rank: Optional[float] = None
    
    # Sector filters
    sectors: Optional[List[str]] = None
    exclude_sectors: Optional[List[str]] = None
    
    # Risk filter
    max_risk_level: Optional[str] = None  # "L", "M", "M-H", "H"
    
    # Catalyst/News filters
    require_recent_catalyst: bool = False
    min_sentiment_score: Optional[float] = None
    
    # Breakout detection
    detect_consolidation_breakout: bool = False
    detect_volume_surge: bool = False
    detect_ma_breakthrough: bool = False
    
    # Entropy filters (market noise detection)
    max_entropy: Optional[float] = None  # Maximum acceptable entropy (filter out noisy stocks)
    min_entropy: Optional[float] = None  # Minimum entropy (if you want volatile/noisy stocks)
    require_low_entropy: bool = False  # Only show stocks with entropy < 50 (structured markets)


@dataclass
class OpportunityResult:
    """Result from opportunity scan"""
    ticker: str
    scan_type: ScanType
    score: float
    price: float
    change_pct: float
    volume: int
    volume_ratio: float
    reason: str
    confidence: str
    risk_level: str
    
    # Enhanced fields
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    trend: Optional[str] = None
    rsi: Optional[float] = None
    iv_rank: Optional[float] = None
    
    # AI fields (optional)
    ai_confidence: Optional[str] = None
    ai_rating: Optional[float] = None
    ai_reasoning: Optional[str] = None
    
    # Breakout indicators
    is_breakout: bool = False
    breakout_signals: List[str] = field(default_factory=list)
    
    # Buzzing indicators
    is_buzzing: bool = False
    buzz_score: float = 0.0
    buzz_reasons: List[str] = field(default_factory=list)
    
    # Reverse split tracking (for penny stocks)
    reverse_splits: List[Dict] = field(default_factory=list)  # [{date, ratio, pre_split_price, post_split_price}]
    has_recent_reverse_split: bool = False
    reverse_split_warning: Optional[str] = None
    
    # Reverse merger candidate indicators
    is_merger_candidate: bool = False
    merger_score: float = 0.0
    merger_signals: List[str] = field(default_factory=list)
    
    # Entropy metrics (optional)
    entropy: Optional[float] = None
    entropy_state: Optional[str] = None
    
    # Full analysis (optional)
    full_analysis: Optional[StockAnalysis] = None


class AdvancedOpportunityScanner:
    """
    Advanced scanner for finding top trading opportunities with filters.
    Combines multiple scanning strategies and AI analysis.
    """
    
    # Extended universe for catching obscure plays
    EXTENDED_UNIVERSE = [
        # Large cap tech (established)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'DIS',
        'INTC', 'ADBE', 'CRM', 'ORCL', 'CSCO', 'AVGO', 'QCOM', 'TXN', 'UBER', 'LYFT',
        
        # Growth/momentum stocks (removed delisted: SQ)
        'PLTR', 'SOFI', 'HOOD', 'COIN', 'RBLX', 'SNAP', 'ABNB', 'DASH', 'SHOP',
        'SNOW', 'NET', 'CRWD', 'ZS', 'DDOG', 'MDB', 'U', 'PATH', 'GTLB', 'S',
        
        # Meme/Reddit stocks (removed delisted: WISH)
        'GME', 'AMC', 'BB', 'NOK', 'BBBY', 'CLOV', 'SKLZ',
        
        # EV/Clean energy (high volatility) (removed delisted: FSR, RIDE, NKLA, NOVA)
        'NIO', 'LCID', 'RIVN', 'PLUG', 'FCEL', 'BE', 'QS', 'BLNK', 'CHPT',
        'GOEV', 'WKHS', 'HYLN', 'EVGO', 'MAXN', 'RUN', 'SEDG', 'ENPH',
        
        # Biotech/Pharma (catalyst-driven) (removed delisted: FREQ, TPTX, BLUE)
        'MRNA', 'BNTX', 'NVAX', 'VXRT', 'OCGN', 'BNGO', 'SAVA', 'SNDL',
        'NVCR', 'ZLAB', 'CASI', 'CRBP', 'DVAX', 'RGNX', 'AKBA', 'ARDX',
        'VKTX', 'ALNY', 'SGMO', 'CRSP', 'EDIT', 'NTLA',
        
        # Crypto-related (high beta)
        'MARA', 'RIOT', 'BITF', 'HUT', 'CLSK', 'ARBK', 'MSTR', 'SI', 'BTBT',
        'CAN', 'SOS', 'EBON', 'FTFT', 'GREE', 'BTCM', 'WULF',
        
        # AI/Tech emerging (removed delisted: VLDR)
        'SOUN', 'BBAI', 'AI', 'KSCP', 'VRAR', 'VUZI', 'KOPN', 'AEHR',
        'WOLF', 'MVIS', 'LAZR', 'LIDR', 'OUST', 'INVZ',
        
        # Cannabis (news-driven) (removed delisted: HEXO)
        'TLRY', 'CGC', 'ACB', 'OGI', 'CRON', 'SNDL',
        
        # Penny stocks/speculative (high risk/reward) (removed delisted: ZOM, IDEX, CLVS, MULN, FFIE, GFAI, ATAI, HOLO, IMPP)
        'GSAT', 'TXMD', 'SIRI', 'SENS',
        
        # Finance/FinTech
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA', 'AXP', 'PYPL',
        
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'OXY', 'MPC', 'PSX', 'VLO',
        
        # SPACs and special situations
        'DWAC', 'PHUN', 'BENE', 'IPOF', 'IPOD',
        
        # ETFs for reference
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV', 'ARKK', 'SQQQ', 'TQQQ'
    ]
    
    def __init__(self, use_ai: bool = True, base_scanner=None, ai_scanner=None, social_analyzer=None):
        """Initialize the scanner with optional pre-created dependencies for performance"""
        self.base_scanner = base_scanner if base_scanner is not None else TopTradesScanner()
        self.ai_scanner = ai_scanner if ai_scanner is not None else (AIConfidenceScanner(use_llm=use_ai) if use_ai else None)
        self.use_ai = use_ai
        self.social_analyzer = social_analyzer if social_analyzer is not None else SocialSentimentAnalyzer()
    
    def scan_opportunities(
        self,
        scan_type: ScanType = ScanType.ALL,
        top_n: int = 20,
        trading_style: str = "SWING_TRADE",
        filters: Optional[ScanFilters] = None,
        custom_tickers: Optional[List[str]] = None,
        use_extended_universe: bool = True,
        use_parallel: bool = True
    ) -> List[OpportunityResult]:
        """
        Scan for top opportunities with advanced filtering
        
        Args:
            scan_type: Type of scan to perform
            top_n: Number of top results to return
            filters: Filter criteria
            custom_tickers: Optional custom ticker list (overrides universe)
            use_extended_universe: Use extended ticker universe for more coverage
            use_parallel: Use parallel processing for 4-8x speedup (default: True)
        
        Returns:
            List of OpportunityResult objects
        """
        filters = filters or ScanFilters()
        
        # Store trading style for use in _analyze_opportunity
        self._current_trading_style = trading_style
        
        logger.info(f"Starting {scan_type.value} scan for top {top_n} opportunities with {trading_style} style")
        logger.info(f"Filters: min_score={filters.min_score}, min_price={filters.min_price}, max_price={filters.max_price}")
        
        # Determine ticker universe
        if custom_tickers:
            universe = custom_tickers
        elif use_extended_universe:
            universe = self.EXTENDED_UNIVERSE
        else:
            if scan_type in [ScanType.PENNY_STOCKS, ScanType.BREAKOUTS]:
                universe = self.base_scanner.PENNY_STOCK_UNIVERSE
            else:
                universe = self.base_scanner.OPTIONS_UNIVERSE
        
        logger.info(f"Scanning {len(universe)} tickers...")
        
        if use_parallel:
            return self._scan_opportunities_parallel(universe, scan_type, filters, top_n)
        else:
            return self._scan_opportunities_sequential(universe, scan_type, filters, top_n)
    
    def _scan_opportunities_sequential(
        self,
        universe: List[str],
        scan_type: ScanType,
        filters: ScanFilters,
        top_n: int
    ) -> List[OpportunityResult]:
        """Sequential opportunity scanning (fallback)"""
        logger.info(f"📊 Using sequential processing for {len(universe)} tickers")
        
        opportunities = []
        
        for ticker in universe:
            try:
                result = self._analyze_opportunity(ticker, scan_type, filters)
                if result:
                    opportunities.append(result)
            except Exception as e:
                logger.debug(f"Error analyzing {ticker}: {e}")
                continue
        
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Found {len(opportunities)} opportunities after filtering")
        
        # Return top N
        return opportunities[:top_n]
    
    def _scan_opportunities_parallel(
        self,
        universe: List[str],
        scan_type: ScanType,
        filters: ScanFilters,
        top_n: int,
        max_workers: int = 8
    ) -> List[OpportunityResult]:
        """Parallel opportunity scanning using ThreadPoolExecutor (4-8x faster)"""
        logger.info(f"🚀 Using parallel processing ({max_workers} workers) for {len(universe))} tickers")
        
        opportunities = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_ticker = {
                executor.submit(self._analyze_opportunity, ticker, scan_type, filters): ticker
                for ticker in universe
            }
            
            # Process results as they complete
            completed = 0
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                completed += 1
                
                try:
                    result = future.result(timeout=10)
                    if result:
                        opportunities.append(result)
                except Exception as e:
                    logger.debug(f"Error analyzing {ticker}: {e}")
                    continue
                
                # Log progress every 10 tickers
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{len(universe))} tickers analyzed")
        
        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Found {len(opportunities)} opportunities after filtering in {elapsed_time:.2f}s")
        
        # Return top N
        return opportunities[:top_n]
    
    def scan_buzzing_stocks(
        self,
        top_n: int = 20,
        trading_style: str = "SWING_TRADE",
        lookback_days: int = 5,
        min_buzz_score: float = 15.0,  # Lowered to 15 to capture social buzz without tech requirements
        max_tickers_to_scan: Optional[int] = None  # Limit number of tickers to scan for faster results
    ) -> List[OpportunityResult]:
        """
        Scan for buzzing/trending stocks showing unusual activity.
        
        Now includes:
        - Technical buzz (volume, volatility, price action)
        - Social sentiment (Reddit, news)
        - Combined scoring for comprehensive buzz detection
        
        Args:
            top_n: Number of results to return
            lookback_days: Days to look back for buzz detection
            min_buzz_score: Minimum buzz score to include (default 30.0 for broader results)
            max_tickers_to_scan: Maximum number of tickers to scan (None = scan all). 
                                Use lower values (e.g., 50) for faster scans.
        
        Returns:
            List of buzzing opportunities
        """
        # Store trading style for use in analysis
        self._current_trading_style = trading_style
        
        # Limit ticker universe if requested for faster scans
        ticker_universe = self.EXTENDED_UNIVERSE
        if max_tickers_to_scan:
            ticker_universe = ticker_universe[:max_tickers_to_scan]
            logger.info(f"🔥 Scanning {len(ticker_universe)} tickers (limited) for buzzing stocks (top {top_n}, min score={min_buzz_score}, style={trading_style})")
        else:
            logger.info(f"🔥 Scanning {len(ticker_universe)} tickers for buzzing stocks (top {top_n}, min score={min_buzz_score}, style={trading_style})")
        
        opportunities = []
        
        for ticker in ticker_universe:
            try:
                # 1. Technical buzz detection (price, volume, volatility)
                buzz_result = self._detect_buzz(ticker, lookback_days)
                
                # 2. Social sentiment analysis (Reddit, news, forums) - ASYNC
                social_result = self._get_social_sentiment_sync(ticker)
                
                # 3. Combine scores
                technical_score = buzz_result['buzz_score'] if buzz_result else 0
                social_score = social_result.get('social_score', 0)
                
                # Use the MAXIMUM of technical and social scores
                # This ensures stocks with strong social buzz OR technical signals get through
                combined_buzz_score = max(technical_score, social_score)
                
                # Only include if meets threshold
                if combined_buzz_score >= min_buzz_score:
                    # Get full analysis
                    analysis = ComprehensiveAnalyzer.analyze_stock(ticker, "SWING_TRADE")
                    
                    if not analysis:
                        continue
                    
                    # Detect reverse splits
                    reverse_split_info = self._detect_reverse_splits(ticker, lookback_years=3)
                    
                    # Detect merger candidates (buzzing stocks might be merger targets)
                    merger_info = self._detect_reverse_merger_candidate(ticker, analysis)
                    
                    # Build comprehensive buzz reasons
                    buzz_reasons = []
                    
                    # Technical buzz reasons
                    if buzz_result:
                        buzz_reasons.extend(buzz_result['reasons'])
                    
                    # Social buzz reasons
                    if social_result['reddit_mentions'] > 0:
                        buzz_reasons.append(f"📱 {social_result['reddit_mentions']} Reddit mentions ({social_result['sentiment']})")
                    if social_result.get('twitter_mentions', 0) > 0:
                        buzz_reasons.append(f"🐦 {social_result['twitter_mentions']} Twitter mentions")
                    if social_result['news_mentions'] > 5:
                        buzz_reasons.append(f"📰 {social_result['news_mentions']} news articles")
                    if social_result['trending_score'] > 60:
                        buzz_reasons.append(f"🔥 Trending (score: {social_result['trending_score']:.0f})")
                    
                    # Boost final score if merger candidate
                    final_buzz_score = combined_buzz_score
                    if merger_info['is_merger_candidate']:
                        final_buzz_score = min(100, final_buzz_score + 10)
                        buzz_reasons.insert(0, "🔄 Reverse merger candidate")
                    
                    result = OpportunityResult(
                        ticker=ticker,
                        scan_type=ScanType.BUZZING,
                        score=final_buzz_score,
                        price=analysis.price,
                        change_pct=analysis.change_pct,
                        volume=analysis.volume,
                        volume_ratio=analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0,
                        reason=" | ".join(buzz_reasons) if buzz_reasons else "Buzzing activity detected",
                        confidence="VERY HIGH" if final_buzz_score >= 75 else "HIGH" if final_buzz_score >= 50 else "MEDIUM",
                        risk_level="M-H",
                        trend=analysis.trend,
                        rsi=analysis.rsi,
                        is_buzzing=True,
                        buzz_score=final_buzz_score,
                        buzz_reasons=buzz_reasons,
                        reverse_splits=reverse_split_info['reverse_splits'],
                        has_recent_reverse_split=reverse_split_info['has_recent_reverse_split'],
                        reverse_split_warning=reverse_split_info['warning'],
                        is_merger_candidate=merger_info['is_merger_candidate'],
                        merger_score=merger_info['merger_score'],
                        merger_signals=merger_info['signals'],
                        entropy=analysis.entropy,
                        entropy_state=analysis.entropy_state,
                        full_analysis=analysis
                    )
                    
                    opportunities.append(result)
                    
            except Exception as e:
                logger.debug(f"Error detecting buzz for {ticker}: {e}")
                continue
        
        # Sort by buzz score (highest first)
        opportunities.sort(key=lambda x: x.buzz_score, reverse=True)
        
        logger.info(f"🔥 Found {len(opportunities)} buzzing stocks with social sentiment")
        if opportunities:
            logger.info("   Top 3: {}({o.buzz_score:.0f})' for o in opportunities[:3]])}", str(', '.join([f'{o.ticker))
        
        return opportunities[:top_n]
    
    def _analyze_opportunity(
        self,
        ticker: str,
        scan_type: ScanType,
        filters: ScanFilters
    ) -> Optional[OpportunityResult]:
        """Analyze a single ticker for opportunities"""
        try:
            # Get comprehensive analysis (use trading_style from instance if available)
            style = getattr(self, '_current_trading_style', "SWING_TRADE")
            analysis = ComprehensiveAnalyzer.analyze_stock(ticker, style)
            
            if not analysis:
                return None
            
            # Calculate score based on scan type
            score = self._calculate_score(analysis, scan_type, filters)

            # Apply filters first (early exit)
            if not self._passes_filters(analysis, filters, score, scan_type):
                return None
            
            # Get market cap and sector info
            stock = yf.Ticker(ticker)
            info = stock.info
            market_cap = info.get('marketCap', 0) / 1_000_000 if info.get('marketCap') else None
            sector = info.get('sector', 'Unknown')
            
            # Get volume ratio
            volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
            
            # Detect breakouts
            breakout_info = self._detect_breakout(analysis)
            
            # Detect reverse splits (especially for penny stocks)
            reverse_split_info = self._detect_reverse_splits(ticker, lookback_years=3)
            
            # Detect reverse merger candidates
            merger_info = self._detect_reverse_merger_candidate(ticker, analysis)
            
            # Build reason
            reason_parts = []
            if abs(analysis.change_pct) > 3:
                reason_parts.append(f"Price move {analysis.change_pct:+.1f}%")
            if volume_ratio > 2:
                reason_parts.append(f"Volume {volume_ratio:.1f}x avg")
            if analysis.ema_reclaim:
                reason_parts.append("EMA Reclaim")
            if analysis.ema_power_zone:
                reason_parts.append("Power Zone")
            if breakout_info['is_breakout']:
                reason_parts.append("Breakout detected")
            if merger_info['is_merger_candidate']:
                reason_parts.append("🔄 Merger candidate")
            
            reason = " | ".join(reason_parts) if reason_parts else f"{analysis.trend}"
            
            # Determine confidence
            if score >= 85:
                confidence = "VERY HIGH"
                risk = "M"
            elif score >= 70:
                confidence = "HIGH"
                risk = "M"
            elif score >= 55:
                confidence = "MEDIUM"
                risk = "M-H"
            else:
                confidence = "LOW"
                risk = "H"
            
            result = OpportunityResult(
                ticker=ticker,
                scan_type=scan_type,
                score=score,
                price=analysis.price,
                change_pct=analysis.change_pct,
                volume=analysis.volume,
                volume_ratio=volume_ratio,
                reason=reason,
                confidence=confidence,
                risk_level=risk,
                market_cap=market_cap,
                sector=sector,
                trend=analysis.trend,
                rsi=analysis.rsi,
                iv_rank=analysis.iv_rank,
                is_breakout=breakout_info['is_breakout'],
                breakout_signals=breakout_info['signals'],
                reverse_splits=reverse_split_info['reverse_splits'],
                has_recent_reverse_split=reverse_split_info['has_recent_reverse_split'],
                reverse_split_warning=reverse_split_info['warning'],
                is_merger_candidate=merger_info['is_merger_candidate'],
                merger_score=merger_info['merger_score'],
                merger_signals=merger_info['signals'],
                entropy=analysis.entropy,
                entropy_state=analysis.entropy_state,
                full_analysis=analysis
            )
            
            return result
            
        except Exception as e:
            logger.debug(f"Error analyzing {ticker}: {e}")
            return None
    
    def _passes_filters(self, analysis: StockAnalysis, filters: ScanFilters, score: float, scan_type: ScanType = None) -> bool:
        """Check if analysis passes all filters"""
        
        # Score filter (primary)
        if filters.min_score is not None and score < filters.min_score:
            return False

        # Penny stock filter - enforce max price for penny stock scans
        if scan_type == ScanType.PENNY_STOCKS:
            if analysis.price >= PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE:
                return False

        # Price filters
        if filters.min_price is not None and analysis.price < filters.min_price:
            return False
        if filters.max_price is not None and analysis.price > filters.max_price:
            return False
        
        # Volume filters
        if filters.min_volume is not None and analysis.volume < filters.min_volume:
            return False
        
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        if filters.min_volume_ratio is not None and volume_ratio < filters.min_volume_ratio:
            return False
        
        # Momentum filters
        if filters.min_change_pct is not None and analysis.change_pct < filters.min_change_pct:
            return False
        if filters.max_change_pct is not None and analysis.change_pct > filters.max_change_pct:
            return False
        
        # Score filters
        if filters.min_confidence_score is not None and analysis.confidence_score < filters.min_confidence_score:
            return False
        
        # Trend filter
        if filters.trend_filter and analysis.trend not in filters.trend_filter:
            return False
        
        # Technical filters
        if filters.require_power_zone and not analysis.ema_power_zone:
            return False
        if filters.require_ema_reclaim and not analysis.ema_reclaim:
            return False
        if filters.require_timeframe_alignment:
            if not analysis.timeframe_alignment or not analysis.timeframe_alignment.get('aligned'):
                return False
        
        # RSI filters
        if filters.min_rsi is not None and analysis.rsi < filters.min_rsi:
            return False
        if filters.max_rsi is not None and analysis.rsi > filters.max_rsi:
            return False
        
        # IV filters
        if filters.min_iv_rank is not None and (analysis.iv_rank is None or analysis.iv_rank < filters.min_iv_rank):
            return False
        if filters.max_iv_rank is not None and (analysis.iv_rank is None or analysis.iv_rank > filters.max_iv_rank):
            return False
        
        # Catalyst filter
        if filters.require_recent_catalyst and (not analysis.catalysts or len(analysis.catalysts) == 0):
            return False
        
        # Sentiment filter
        if filters.min_sentiment_score is not None and analysis.sentiment_score < filters.min_sentiment_score:
            return False
        
        # Entropy filters (market noise detection)
        if analysis.entropy is not None:
            if filters.max_entropy is not None and analysis.entropy > filters.max_entropy:
                return False
            if filters.min_entropy is not None and analysis.entropy < filters.min_entropy:
                return False
            if filters.require_low_entropy and analysis.entropy >= 50:
                return False  # Only pass structured markets (entropy < 50)
        
        return True
    
    def _calculate_score(
        self,
        analysis: StockAnalysis,
        scan_type: ScanType,
        filters: ScanFilters
    ) -> float:
        """Calculate opportunity score based on scan type"""
        
        base_score = analysis.confidence_score
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        
        # Adjust based on scan type
        if scan_type == ScanType.OPTIONS:
            # Boost for good IV characteristics
            if analysis.iv_rank:
                if analysis.iv_rank > 60 or analysis.iv_rank < 40:
                    base_score += 10
        
        elif scan_type == ScanType.MOMENTUM:
            # Boost for strong price action
            if abs(analysis.change_pct) > 5:
                base_score += 15
            elif abs(analysis.change_pct) > 3:
                base_score += 10
        
        elif scan_type == ScanType.BREAKOUTS:
            # Boost for breakout signals
            if analysis.ema_reclaim:
                base_score += 15
            if analysis.ema_power_zone:
                base_score += 10
        
        elif scan_type == ScanType.HOTTEST_STOCKS:
            # Set a base score for stocks meeting hot criteria
            score = 0
            if analysis.change_pct > 10:
                score += 40
            if volume_ratio > 5:
                score += 30
            if analysis.trend == "STRONG UPTREND":
                score += 30
            base_score = score

        elif scan_type == ScanType.PENNY_STOCKS:
            # Boost for low price with volume (using centralized bonus calculation)
            price_bonus = get_price_tier_bonus(analysis.price)
            if price_bonus > 0 or analysis.price < PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE:
                base_score += price_bonus
        
        # Volume boost
        if volume_ratio > 2.5:
            base_score += 10
        elif volume_ratio > 1.5:
            base_score += 5
        
        # Timeframe alignment boost
        if analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned'):
            base_score += 5
        
        # Sector RS boost
        if analysis.sector_rs and analysis.sector_rs.get('rs_score', 0) > 60:
            base_score += 5
        
        return min(100, base_score)
    
    def _detect_breakout(self, analysis: StockAnalysis) -> Dict:
        """Detect breakout signals"""
        signals = []
        is_breakout = False
        
        if analysis.ema_reclaim:
            signals.append("EMA Reclaim")
            is_breakout = True
        
        if analysis.ema_power_zone:
            signals.append("Power Zone Active")
        
        if analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned'):
            signals.append("Timeframe Aligned")
            is_breakout = True
        
        # Volume confirmation
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        if volume_ratio > 1.5:
            signals.append(f"Volume Surge {volume_ratio:.1f}x")
        
        # Momentum check
        if "UPTREND" in analysis.trend and analysis.rsi < 70:
            signals.append("Uptrend Momentum")
        
        return {
            'is_breakout': is_breakout,
            'signals': signals
        }
    
    def _get_social_sentiment_sync(self, ticker: str) -> Dict:
        """
        Get social sentiment in synchronous context.
        Properly handles async/await and Windows subprocess issues.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Social sentiment dict (or empty result on error)
        """
        try:
            # In Streamlit/nest_asyncio environment, reuse the existing event loop
            # This prevents "cannot schedule new futures after shutdown" errors
            # caused by creating new event loops with asyncio.run()
            try:
                loop = asyncio.get_event_loop()
                # Check if loop is closed or None
                if loop is None or loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # No event loop in current thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the async function using the existing/new loop
            social_result = loop.run_until_complete(
                self.social_analyzer.analyze_social_buzz(ticker)
            )
            return social_result
            
        except (NotImplementedError, ImportError) as e:
            # Crawl4ai not available or Windows subprocess issue
            # This is expected on first run - only log once
            if not hasattr(self, '_crawl4ai_warning_shown'):
                logger.warning(f"Social sentiment disabled: {e.__class__.__name__}")
                logger.info("Continuing with technical analysis only (no social data)")
                self._crawl4ai_warning_shown = True
            
            # Return empty result - technical buzz will still work
            return {
                'social_score': 0,
                'reddit_mentions': 0,
                'twitter_mentions': 0,
                'stocktwits_mentions': 0,
                'news_mentions': 0,
                'sentiment': 'NEUTRAL',
                'trending_score': 0
            }
        except Exception as e:
            # Other errors - log but continue
            logger.debug(f"Social sentiment failed for {ticker}: {e}")
            return {
                'social_score': 0,
                'reddit_mentions': 0,
                'twitter_mentions': 0,
                'stocktwits_mentions': 0,
                'news_mentions': 0,
                'sentiment': 'NEUTRAL',
                'trending_score': 0
            }
    
    def _detect_buzz(self, ticker: str, lookback_days: int = 5) -> Optional[Dict]:
        """
        Detect if a stock is buzzing based on unusual activity
        
        Returns:
            Dict with buzz_score and reasons, or None
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{lookback_days}d")
            
            if hist.empty or len(hist) < 3:
                return None
            
            buzz_score = 0
            reasons = []
            
            # 1. Unusual volume (40 points)
            recent_avg_volume = hist['Volume'][-3:].mean()
            overall_avg_volume = hist['Volume'].mean()
            
            if recent_avg_volume > overall_avg_volume * 3:
                buzz_score += 40
                reasons.append(f"Volume surge {recent_avg_volume/overall_avg_volume:.1f}x")
            elif recent_avg_volume > overall_avg_volume * 2:
                buzz_score += 30
                reasons.append(f"High volume {recent_avg_volume/overall_avg_volume:.1f}x")
            elif recent_avg_volume > overall_avg_volume * 1.5:
                buzz_score += 20
                reasons.append("Elevated volume")
            
            # 2. Price volatility (30 points)
            recent_volatility = hist['Close'][-3:].std() / hist['Close'][-3:].mean()
            overall_volatility = hist['Close'].std() / hist['Close'].mean()
            
            if recent_volatility > overall_volatility * 2:
                buzz_score += 30
                reasons.append("High volatility spike")
            elif recent_volatility > overall_volatility * 1.5:
                buzz_score += 20
                reasons.append("Increased volatility")
            
            # 3. Consecutive moves (20 points)
            daily_changes = hist['Close'].pct_change()[-3:]
            if len(daily_changes) >= 3:
                if all(daily_changes > 0.02):  # 3 days up >2%
                    buzz_score += 20
                    reasons.append("Consecutive gains")
                elif all(daily_changes < -0.02):  # 3 days down >2%
                    buzz_score += 15
                    reasons.append("Consecutive drops")
            
            # 4. Gap moves (10 points)
            if len(hist) >= 2:
                latest_open = hist['Open'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]
                gap_pct = abs((latest_open - prev_close) / prev_close * 100)
                
                if gap_pct > 5:
                    buzz_score += 10
                    reasons.append(f"Gap move {gap_pct:.1f}%")
                elif gap_pct > 3:
                    buzz_score += 5
                    reasons.append(f"Gap {gap_pct:.1f}%")
            
            if buzz_score > 0:
                return {
                    'buzz_score': buzz_score,
                    'reasons': reasons
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error detecting buzz for {ticker}: {e}")
            return None
    
    def _detect_reverse_splits(self, ticker: str, lookback_years: int = 3) -> Dict:
        """
        Detect reverse stock splits in the company's history
        
        Args:
            ticker: Stock ticker
            lookback_years: Years to look back for reverse splits
        
        Returns:
            Dict with reverse split data
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get stock actions (splits, dividends, etc.)
            actions = stock.actions
            
            if actions is None or actions.empty:
                return {
                    'reverse_splits': [],
                    'has_recent_reverse_split': False,
                    'warning': None
                }
            
            # Filter for reverse splits (ratio < 1)
            reverse_splits = []
            cutoff_date = datetime.now() - timedelta(days=lookback_years * 365)
            
            if 'Stock Splits' in actions.columns:
                splits = actions['Stock Splits']
                splits = splits[splits != 0]  # Filter out zeros
                
                for date, ratio in splits.items():
                    if ratio < 1:  # Reverse split (e.g., 1:10 = 0.1)
                        # Get price before and after split
                        split_date = pd.Timestamp(date)
                        
                        # Calculate reverse split ratio
                        reverse_ratio = 1 / ratio  # e.g., 0.1 -> 10:1
                        
                        reverse_splits.append({
                            'date': split_date.strftime('%Y-%m-%d'),
                            'ratio': reverse_ratio,
                            'ratio_str': f"1:{int(reverse_ratio))}"
                        })
            
            # Check if recent reverse split (within last year)
            recent_cutoff = datetime.now() - timedelta(days=365)
            recent_splits = [s for s in reverse_splits if datetime.strptime(s['date'], '%Y-%m-%d') > recent_cutoff]
            
            # Generate warning
            warning = None
            if len(reverse_splits) > 0:
                total_splits = len(reverse_splits)
                if total_splits >= 3:
                    warning = f"⚠️ {total_splits} reverse splits in {lookback_years}y - HIGH RISK"
                elif total_splits >= 2:
                    warning = f"⚠️ {total_splits} reverse splits in {lookback_years}y - CAUTION"
                elif recent_splits:
                    warning = f"⚠️ Recent reverse split {recent_splits[0]['ratio_str']}"
                else:
                    warning = f"Previous reverse split detected"
            
            return {
                'reverse_splits': reverse_splits,
                'has_recent_reverse_split': len(recent_splits) > 0,
                'warning': warning
            }
            
        except Exception as e:
            logger.debug(f"Error detecting reverse splits for {ticker}: {e}")
            return {
                'reverse_splits': [],
                'has_recent_reverse_split': False,
                'warning': None
            }
    
    def _detect_reverse_merger_candidate(self, ticker: str, analysis: StockAnalysis) -> Dict:
        """
        Detect if stock is a potential reverse merger candidate
        Based on speculation, sentiment, and corporate indicators
        
        Args:
            ticker: Stock ticker
            analysis: Stock analysis
        
        Returns:
            Dict with merger candidate score and signals
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            merger_score = 0
            signals = []
            
            # 1. Shell company indicators (40 points)
            # Very low market cap
            market_cap = info.get('marketCap', 0)
            if market_cap and market_cap < 50_000_000:  # < $50M
                merger_score += 20
                signals.append(f"Micro-cap ${market_cap/1e6:.1f}M")
            elif market_cap and market_cap < 100_000_000:  # < $100M
                merger_score += 10
                signals.append(f"Small-cap ${market_cap/1e6:.1f}M")
            
            # Low volume (shell companies often have low liquidity)
            if analysis.avg_volume < 100_000:
                merger_score += 10
                signals.append("Low avg volume")
            
            # Minimal revenue (shell characteristic)
            revenue = info.get('totalRevenue', 0)
            if revenue == 0 or revenue is None:
                merger_score += 10
                signals.append("No/minimal revenue")
            
            # 2. Recent unusual activity (30 points)
            volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
            if volume_ratio > 5:  # Massive volume spike
                merger_score += 20
                signals.append(f"Massive volume {volume_ratio:.1f}x")
            elif volume_ratio > 3:
                merger_score += 10
                signals.append(f"Unusual volume {volume_ratio:.1f}x")
            
            # Significant price movement
            if abs(analysis.change_pct) > 20:
                merger_score += 10
                signals.append(f"Large price move {analysis.change_pct:+.1f}%")
            
            # 3. Sentiment & News indicators (30 points)
            # High sentiment with shell characteristics suggests speculation
            if analysis.sentiment_score > 70 and market_cap and market_cap < 100_000_000:
                merger_score += 15
                signals.append("High speculation sentiment")
            
            # Recent catalysts (potential merger news)
            if analysis.catalysts and len(analysis.catalysts) > 0:
                catalyst_text = ' '.join([c.get('headline', '').lower() for c in analysis.catalysts[:3]])
                merger_keywords = ['merger', 'acquisition', 'reverse merger', 'spac', 'combination', 
                                 'transaction', 'deal', 'takeover', 'agreement']
                
                if any(keyword in catalyst_text for keyword in merger_keywords):
                    merger_score += 15
                    signals.append("Merger-related news detected")
            
            # News volume spike
            if analysis.recent_news and len(analysis.recent_news) > 5:  # More than 5 recent news items
                merger_score += 5
                signals.append("High news activity")
            
            is_merger_candidate = merger_score >= 50
            
            return {
                'is_merger_candidate': is_merger_candidate,
                'merger_score': min(100, merger_score),
                'signals': signals
            }
            
        except Exception as e:
            logger.debug(f"Error detecting merger candidate for {ticker}: {e}")
            return {
                'is_merger_candidate': False,
                'merger_score': 0,
                'signals': []
            }
    
    async def cleanup(self):
        """Cleanup resources (social analyzer, crawlers, etc.)"""
        try:
            if hasattr(self, 'social_analyzer') and self.social_analyzer:
                await self.social_analyzer.close()
                logger.info("✅ Scanner cleanup completed")
        except Exception as e:
            logger.debug(f"Error during scanner cleanup: {e}")
    
    def get_scan_summary(self, opportunities: List[OpportunityResult]) -> Dict:
        """Generate summary statistics for scan results"""
        if not opportunities:
            return {
                'total': 0,
                'avg_score': 0,
                'high_confidence': 0,
                'breakouts': 0,
                'buzzing': 0,
                'avg_price': 0,
                'avg_volume_ratio': 0
            }
        
        return {
            'total': len(opportunities),
            'avg_score': round(sum(o.score for o in opportunities) / len(opportunities), 1),
            'high_confidence': len([o for o in opportunities if o.confidence in ['HIGH', 'VERY HIGH']]),
            'breakouts': len([o for o in opportunities if o.is_breakout]),
            'buzzing': len([o for o in opportunities if o.is_buzzing]),
            'reverse_split_stocks': len([o for o in opportunities if o.has_recent_reverse_split]),
            'merger_candidates': len([o for o in opportunities if o.is_merger_candidate]),
            'avg_price': round(sum(o.price for o in opportunities) / len(opportunities), 2),
            'avg_volume_ratio': round(sum(o.volume_ratio for o in opportunities) / len(opportunities), 1),
            'top_ticker': opportunities[0].ticker if opportunities else 'N/A',
            'top_score': opportunities[0].score if opportunities else 0
        }

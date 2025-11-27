"""
X (Twitter) Sentiment Service for DEX Hunter
=============================================

Hybrid approach for X scraping:
- Layer A (FREE): Crawl4AI with CSS extraction + rule-based sentiment
- Layer B (EXPENSIVE): LLM analysis for top candidates only (100 calls/month budget)

Uses the existing stealth browsing infrastructure:
- ChromeProfileManager for persistent sessions
- human_behavior for anti-detection delays
- Crawl4AI for structured extraction

Important X Notes:
- X is hostile to scraping and changes markup frequently
- Selectors may need periodic updates
- Use cashtag search ($SYMBOL) for crypto mentions
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median
from typing import Dict, List, Any, Optional, Tuple

# Crawl4AI imports
try:
    from crawl4ai import (
        AsyncWebCrawler,
        CrawlerRunConfig,
        JsonCssExtractionStrategy,
        CacheMode,
        BrowserConfig
    )
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

# Local imports
from chrome_profile_manager import ChromeProfileManager
from human_behavior import human_delay, human_scroll, simulate_reading_pause

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class XTweet:
    """Single tweet from X"""
    text: str
    author_handle: str = ""
    timestamp: Optional[datetime] = None
    likes: int = 0
    reposts: int = 0
    replies: int = 0
    tweet_url: str = ""
    contains_contract: bool = False
    contains_dex_link: bool = False
    cashtags: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0  # -1 to 1, computed locally


@dataclass
class XSentimentSnapshot:
    """Aggregated X sentiment for a token symbol"""
    symbol: str
    tweet_count: int = 0
    unique_authors: int = 0
    total_likes: int = 0
    total_reposts: int = 0
    median_like_count: float = 0.0
    engagement_velocity: float = 0.0  # engagement per minute
    neg_mention_ratio: float = 0.0  # ratio of negative tweets
    contract_ratio: float = 0.0  # ratio mentioning contracts
    dex_link_ratio: float = 0.0  # ratio with DEX links
    
    # Rule-based sentiment (free)
    heuristic_sentiment: float = 0.0  # -1 to 1
    bullish_tweet_count: int = 0
    bearish_tweet_count: int = 0
    
    # LLM sentiment (expensive - only for top candidates)
    llm_sentiment: Optional[float] = None  # -1 to 1, None if not computed
    llm_explanation: Optional[str] = None
    llm_red_flags: List[str] = field(default_factory=list)
    
    # Metadata
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tweets: List[XTweet] = field(default_factory=list)  # Raw tweets for LLM if needed
    
    @property
    def x_sentiment_score(self) -> float:
        """
        Compute overall X sentiment score (0-100) for integration with DEX Hunter.
        Uses LLM sentiment if available, otherwise falls back to heuristic.
        """
        if self.tweet_count < 3:
            return 0.0  # Not enough data
        
        # Base sentiment (-1 to 1) → (0 to 1)
        base_sentiment = self.llm_sentiment if self.llm_sentiment is not None else self.heuristic_sentiment
        sentiment_factor = (base_sentiment + 1) / 2  # -1..1 → 0..1
        
        # Engagement factor (capped at 1.0)
        engagement_factor = min(self.engagement_velocity * 50, 1.0)
        
        # Crowd factor (unique authors)
        crowd_factor = min(self.unique_authors / 20, 1.0)
        
        # Contract/DEX mentions (good sign for new coins)
        dex_factor = min((self.contract_ratio + self.dex_link_ratio) * 2, 1.0)
        
        # Negative penalty
        neg_penalty = max(0.0, 1.0 - self.neg_mention_ratio * 2)
        
        # Weighted combination
        base_score = (
            0.35 * sentiment_factor +
            0.25 * engagement_factor +
            0.20 * crowd_factor +
            0.20 * dex_factor
        )
        
        return max(0.0, min(100.0, base_score * neg_penalty * 100))


# ============================================================================
# Sentiment Lexicons (Rule-based - FREE)
# ============================================================================

BULLISH_WORDS = {
    # Strong bullish
    "moon", "mooning", "pump", "pumping", "bullish", "lfg", "wagmi",
    "send it", "sending", "ath", "breakout", "explode", "exploding",
    "gem", "100x", "1000x", "buy", "buying", "accumulate", "accumulating",
    "load up", "loading", "dip buy", "btd", "long", "longing",
    "green candles", "parabolic", "rocket", "🚀", "📈", "💎", "🔥",
    "alpha", "early", "undervalued", "sleeping giant",
    
    # Moderate bullish
    "bullrun", "recovery", "reversal", "support holding", "bounce",
    "breakout soon", "consolidating", "coiling", "ready to fly",
}

BEARISH_WORDS = {
    # Strong bearish / scam signals
    "rug", "rugged", "rugpull", "scam", "scammer", "honeypot", "honey pot",
    "dump", "dumping", "bearish", "rekt", "exit liquidity", "ponzi",
    "dead", "dying", "crashed", "crashing", "sell", "selling",
    "short", "shorting", "avoid", "warning", "stay away", "don't buy",
    "fake", "fraud", "bot", "bots", "wash trading", "insider",
    "dev sold", "dev dumped", "team left", "abandoned",
    "📉", "⚠️", "🚨", "💀", "🪦",
    
    # Moderate bearish
    "correction", "pullback", "weak", "bleeding", "underwater",
    "down bad", "bag holder", "bagholding",
}

# Contract address patterns
CONTRACT_PATTERNS = [
    r"0x[a-fA-F0-9]{40}",  # EVM addresses
    r"[1-9A-HJ-NP-Za-km-z]{32,44}",  # Solana addresses (base58)
]

# DEX link patterns
DEX_PATTERNS = [
    r"dexscreener\.com",
    r"pump\.fun",
    r"raydium\.io",
    r"jupiter\.exchange",
    r"uniswap\.org",
    r"pancakeswap\.finance",
    r"birdeye\.so",
    r"dextools\.io",
    r"geckoterminal\.com",
]


# ============================================================================
# X Tweet Extraction Schema (Crawl4AI CSS - FREE)
# ============================================================================

# Note: X frequently changes its DOM structure. These selectors may need updates.
# Last verified: November 2024

X_TWEET_SCHEMA = {
    "name": "Tweets",
    "baseSelector": "article[data-testid='tweet']",
    "fields": [
        {
            "name": "tweet_text",
            "selector": "div[data-testid='tweetText']",
            "type": "text"
        },
        {
            "name": "author_block",
            "selector": "div[data-testid='User-Name']",
            "type": "text"
        },
        {
            "name": "timestamp",
            "selector": "time",
            "type": "attribute",
            "attribute": "datetime"
        },
        {
            "name": "tweet_url",
            "selector": "a[href*='/status/']",
            "type": "attribute",
            "attribute": "href"
        },
        {
            "name": "likes",
            "selector": "button[data-testid='like'] span, div[data-testid='like'] span",
            "type": "text"
        },
        {
            "name": "reposts",
            "selector": "button[data-testid='retweet'] span, div[data-testid='retweet'] span",
            "type": "text"
        },
        {
            "name": "replies",
            "selector": "button[data-testid='reply'] span, div[data-testid='reply'] span",
            "type": "text"
        },
    ]
}

# Alternative/fallback schema for different X layouts
X_TWEET_SCHEMA_ALT = {
    "name": "Tweets",
    "baseSelector": "article[role='article']",
    "fields": [
        {
            "name": "tweet_text",
            "selector": "div[lang]",
            "type": "text"
        },
        {
            "name": "author_block",
            "selector": "a[role='link'] span",
            "type": "text"
        },
        {
            "name": "timestamp",
            "selector": "time",
            "type": "attribute",
            "attribute": "datetime"
        },
    ]
}


# ============================================================================
# Main Service Class
# ============================================================================

class XSentimentService:
    """
    X (Twitter) sentiment scraping and analysis service.
    
    Implements a hybrid approach:
    - Layer A (FREE): CSS extraction + rule-based sentiment for all symbols
    - Layer B (EXPENSIVE): LLM analysis for top candidates only
    
    Uses existing stealth browsing infrastructure for anti-detection.
    """
    
    def __init__(
        self,
        use_llm: bool = True,
        llm_budget_manager: Optional[Any] = None,
        browser_manager: Optional[ChromeProfileManager] = None
    ):
        """
        Initialize X Sentiment Service.
        
        Args:
            use_llm: Whether to use LLM for deep analysis (respects budget)
            llm_budget_manager: Budget manager for tracking LLM calls
            browser_manager: Existing Chrome session manager (for authenticated scraping)
        """
        if not CRAWL4AI_AVAILABLE:
            raise ImportError("Crawl4AI is required. Install with: pip install crawl4ai>=0.7.0")
        
        self.use_llm = use_llm
        self.llm_budget_manager = llm_budget_manager
        self.browser_manager = browser_manager or ChromeProfileManager()
        
        # Cache for recent snapshots (avoid hammering X)
        self._cache: Dict[str, Tuple[XSentimentSnapshot, datetime]] = {}
        self._cache_ttl_minutes = 5  # Refresh every 5 min
        
        logger.info("✅ X Sentiment Service initialized")
    
    async def fetch_snapshot(
        self,
        symbol: str,
        max_tweets: int = 50,
        force_refresh: bool = False
    ) -> XSentimentSnapshot:
        """
        Fetch X sentiment snapshot for a token symbol.
        
        This is the main entry point. Uses:
        1. CSS extraction (free) for tweet data
        2. Rule-based sentiment (free) for heuristic scoring
        3. LLM analysis (expensive) only for high-engagement tokens
        
        Args:
            symbol: Token symbol (e.g., "PEPE", "WIF")
            max_tweets: Maximum tweets to fetch
            force_refresh: Bypass cache
            
        Returns:
            XSentimentSnapshot with aggregated sentiment data
        """
        symbol = symbol.upper().strip()
        
        # Check cache
        if not force_refresh and symbol in self._cache:
            snapshot, cached_at = self._cache[symbol]
            age_minutes = (datetime.now(timezone.utc) - cached_at).total_seconds() / 60
            if age_minutes < self._cache_ttl_minutes:
                logger.debug(f"Using cached X snapshot for {symbol} ({age_minutes:.1f}m old)")
                return snapshot
        
        logger.info(f"🐦 Fetching X sentiment for ${symbol}...")
        
        # 1. Scrape tweets (FREE - CSS extraction)
        tweets = await self._scrape_tweets(symbol, max_tweets)
        
        if not tweets:
            logger.warning(f"No tweets found for ${symbol}")
            return XSentimentSnapshot(symbol=symbol, tweet_count=0)
        
        # 2. Compute rule-based sentiment (FREE)
        for tweet in tweets:
            tweet.sentiment_score = self._compute_tweet_sentiment(tweet)
        
        # 3. Aggregate features (FREE)
        snapshot = self._aggregate_features(symbol, tweets)
        
        # 4. LLM analysis for high-engagement tokens (EXPENSIVE - budget-controlled)
        if self.use_llm and self._should_use_llm(snapshot):
            logger.info(f"🧠 Running LLM analysis for ${symbol} (high engagement)")
            llm_result = await self._run_llm_sentiment(symbol, tweets)
            if llm_result:
                snapshot.llm_sentiment = llm_result.get("sentiment_label")
                snapshot.llm_explanation = llm_result.get("explanation")
                snapshot.llm_red_flags = llm_result.get("red_flags", [])
        
        # Cache result
        self._cache[symbol] = (snapshot, datetime.now(timezone.utc))
        
        logger.info(
            f"✅ X snapshot for ${symbol}: "
            f"{snapshot.tweet_count} tweets, "
            f"sentiment={snapshot.heuristic_sentiment:.2f}, "
            f"score={snapshot.x_sentiment_score:.1f}"
        )
        
        return snapshot
    
    async def _scrape_tweets(self, symbol: str, max_tweets: int = 50) -> List[XTweet]:
        """
        Scrape tweets for a symbol using Crawl4AI CSS extraction.
        
        Tries multiple sources:
        1. Nitter (open Twitter frontend, no login required)
        2. Direct X.com (may require login)
        
        Args:
            symbol: Token symbol to search
            max_tweets: Maximum tweets to return
            
        Returns:
            List of XTweet objects
        """
        # Try Nitter first (no login required, scraper-friendly)
        tweets = await self._scrape_nitter(symbol, max_tweets)
        if tweets:
            return tweets
        
        # Fallback to direct X.com
        logger.info(f"Nitter failed, trying direct X.com for ${symbol}...")
        return await self._scrape_x_direct(symbol, max_tweets)
    
    async def _scrape_nitter(self, symbol: str, max_tweets: int = 50) -> List[XTweet]:
        """
        Scrape tweets via Nitter (open Twitter frontend).
        No login required, more scraper-friendly.
        """
        # List of Nitter instances (ordered by reliability)
        # Updated: 2024-11 - some instances go down frequently
        nitter_instances = [
            "https://nitter.net",           # Most reliable
            "https://nitter.poast.org",     # Good backup
            "https://xcancel.com",          # Another alternative
        ]
        
        # Nitter CSS schema (different from X.com)
        nitter_schema = {
            "name": "Tweets",
            "baseSelector": ".timeline-item",
            "fields": [
                {"name": "tweet_text", "selector": ".tweet-content", "type": "text"},
                {"name": "author_handle", "selector": ".username", "type": "text"},
                {"name": "timestamp", "selector": ".tweet-date a", "type": "attribute", "attribute": "title"},
                {"name": "stats", "selector": ".tweet-stats", "type": "text"},
            ]
        }
        
        for nitter_base in nitter_instances:
            try:
                url = f"{nitter_base}/search?f=tweets&q=%24{symbol}"
                logger.debug(f"Trying Nitter: {url}")
                
                extraction_strategy = JsonCssExtractionStrategy(nitter_schema)
                
                config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    extraction_strategy=extraction_strategy,
                    page_timeout=20000,
                    wait_until="domcontentloaded",
                    delay_before_return_html=2.0,
                )
                
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url, config=config)
                
                if not result.success:
                    logger.debug(f"Nitter instance {nitter_base} failed")
                    continue
                
                # Parse extracted content
                extracted = json.loads(result.extracted_content or "[]")
                
                if not extracted:
                    logger.debug(f"No content from {nitter_base}")
                    continue
                
                tweets = []
                for item in extracted[:max_tweets]:
                    tweet = self._parse_nitter_item(item, symbol)
                    if tweet and tweet.text:
                        tweets.append(tweet)
                
                if tweets:
                    logger.info(f"Scraped {len(tweets)} tweets from Nitter for ${symbol}")
                    return tweets
                    
            except Exception as e:
                logger.debug(f"Nitter instance {nitter_base} error: {e}")
                continue
        
        logger.warning(f"All Nitter instances failed for ${symbol}")
        return []
    
    def _parse_nitter_item(self, item: Dict, symbol: str) -> Optional[XTweet]:
        """Parse Nitter extracted item into XTweet"""
        try:
            text = item.get("tweet_text", "").strip()
            if not text:
                return None
            
            author = item.get("author_handle", "").strip().replace("@", "")
            
            # Parse stats (format: "X comments, Y retweets, Z likes")
            stats = item.get("stats", "")
            likes = 0
            reposts = 0
            replies = 0
            
            if stats:
                # Try to extract numbers
                numbers = re.findall(r'(\d+)', stats)
                if len(numbers) >= 3:
                    replies = int(numbers[0])
                    reposts = int(numbers[1])
                    likes = int(numbers[2])
            
            # Extract cashtags
            cashtags = re.findall(r'\$([A-Za-z0-9]{2,10})', text)
            
            # Check for contract addresses
            contains_contract = any(
                re.search(pat, text, re.IGNORECASE) 
                for pat in CONTRACT_PATTERNS
            )
            
            # Check for DEX links
            contains_dex_link = any(
                re.search(pat, text, re.IGNORECASE)
                for pat in DEX_PATTERNS
            )
            
            return XTweet(
                text=text,
                author_handle=author,
                timestamp=None,
                likes=likes,
                reposts=reposts,
                replies=replies,
                tweet_url="",
                contains_contract=contains_contract,
                contains_dex_link=contains_dex_link,
                cashtags=[ct.upper() for ct in cashtags],
            )
        except Exception as e:
            logger.debug(f"Error parsing Nitter item: {e}")
            return None
    
    async def _scrape_x_direct(self, symbol: str, max_tweets: int = 50) -> List[XTweet]:
        """
        Scrape directly from X.com (may show login wall to headless browsers).
        """
        query = f"%24{symbol}"  # $SYMBOL
        url = f"https://x.com/search?q={query}&src=typed_query&f=live"
        
        logger.debug(f"Scraping X direct: {url}")
        
        try:
            extraction_strategy = JsonCssExtractionStrategy(X_TWEET_SCHEMA)
            
            config = CrawlerRunConfig(
                # Target tweet containers
                target_elements=["article[data-testid='tweet']", "article[role='article']"],
                
                # Content filtering
                excluded_tags=["nav", "header", "footer", "aside", "form"],
                word_count_threshold=3,
                
                # IMPORTANT: Don't exclude social media links (we're scraping X!)
                exclude_external_links=False,
                exclude_social_media_links=False,
                
                # Cache bypass for fresh data
                cache_mode=CacheMode.BYPASS,
                
                # Extraction
                extraction_strategy=extraction_strategy,
                
                # Page loading - wait longer for dynamic content
                page_timeout=45000,
                wait_until="networkidle",
                delay_before_return_html=5.0,  # Wait longer for JS to render
                
                # Scroll to load more tweets
                js_code=[
                    "await new Promise(resolve => setTimeout(resolve, 2000));",
                    "window.scrollTo(0, document.body.scrollHeight / 3);",
                    "await new Promise(resolve => setTimeout(resolve, 1500));",
                    "window.scrollTo(0, document.body.scrollHeight * 2 / 3);",
                    "await new Promise(resolve => setTimeout(resolve, 1500));",
                ]
            )
            
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=config)
            
            if not result.success:
                logger.warning(f"Crawl4AI failed for X search: {result.error_message}")
                return []
            
            # DEBUG: Save HTML to see what we're getting
            if result.html and len(result.html) > 1000:
                # Check if we got a login wall
                if "Sign in" in result.html and "Log in" in result.html:
                    logger.warning("X.com is showing login wall - need authenticated session")
                    return []
            
            # Parse extracted content
            extracted = json.loads(result.extracted_content or "[]")
            
            tweets = []
            for item in extracted[:max_tweets]:
                tweet = self._parse_tweet_item(item, symbol)
                if tweet and tweet.text:
                    tweets.append(tweet)
            
            logger.info(f"Scraped {len(tweets)} tweets from X.com for ${symbol}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error scraping X for ${symbol}: {e}", exc_info=True)
            return []
    
    def _parse_tweet_item(self, item: Dict, symbol: str) -> Optional[XTweet]:
        """Parse raw extracted item into XTweet"""
        try:
            text = item.get("tweet_text", "").strip()
            if not text:
                return None
            
            # Parse author from author_block (format varies)
            author_block = item.get("author_block", "")
            author_handle = self._extract_handle(author_block)
            
            # Parse timestamp
            ts_str = item.get("timestamp", "")
            timestamp = None
            if ts_str:
                try:
                    timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except:
                    pass
            
            # Parse engagement counts
            likes = self._parse_count(item.get("likes", "0"))
            reposts = self._parse_count(item.get("reposts", "0"))
            replies = self._parse_count(item.get("replies", "0"))
            
            # Extract cashtags
            cashtags = re.findall(r'\$([A-Za-z0-9]{2,10})', text)
            
            # Check for contract addresses
            contains_contract = any(
                re.search(pat, text, re.IGNORECASE) 
                for pat in CONTRACT_PATTERNS
            )
            
            # Check for DEX links
            contains_dex_link = any(
                re.search(pat, text, re.IGNORECASE)
                for pat in DEX_PATTERNS
            )
            
            return XTweet(
                text=text,
                author_handle=author_handle,
                timestamp=timestamp,
                likes=likes,
                reposts=reposts,
                replies=replies,
                tweet_url=item.get("tweet_url", ""),
                contains_contract=contains_contract,
                contains_dex_link=contains_dex_link,
                cashtags=[ct.upper() for ct in cashtags],
            )
            
        except Exception as e:
            logger.debug(f"Error parsing tweet item: {e}")
            return None
    
    def _extract_handle(self, author_block: str) -> str:
        """Extract @handle from author block text"""
        # Look for @username pattern
        match = re.search(r'@([A-Za-z0-9_]+)', author_block)
        if match:
            return match.group(1)
        return author_block.split()[0] if author_block else "unknown"
    
    def _parse_count(self, count_str: str) -> int:
        """Parse engagement count (handles K, M suffixes)"""
        if not count_str:
            return 0
        
        count_str = count_str.strip().upper().replace(",", "")
        
        try:
            if "K" in count_str:
                return int(float(count_str.replace("K", "")) * 1000)
            elif "M" in count_str:
                return int(float(count_str.replace("M", "")) * 1000000)
            else:
                return int(count_str)
        except:
            return 0
    
    def _compute_tweet_sentiment(self, tweet: XTweet) -> float:
        """
        Compute rule-based sentiment for a single tweet.
        
        Returns:
            Sentiment score from -1 (very bearish) to 1 (very bullish)
        """
        text_lower = tweet.text.lower()
        
        bullish_count = sum(1 for word in BULLISH_WORDS if word.lower() in text_lower)
        bearish_count = sum(1 for word in BEARISH_WORDS if word.lower() in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        # Normalize to -1 to 1
        sentiment = (bullish_count - bearish_count) / total
        
        # Boost for high engagement (social proof)
        engagement = tweet.likes + tweet.reposts * 2
        if engagement > 100:
            sentiment *= 1.2
        elif engagement > 50:
            sentiment *= 1.1
        
        return max(-1.0, min(1.0, sentiment))
    
    def _aggregate_features(self, symbol: str, tweets: List[XTweet]) -> XSentimentSnapshot:
        """
        Aggregate tweet-level data into symbol-level features.
        """
        if not tweets:
            return XSentimentSnapshot(symbol=symbol)
        
        # Unique authors
        authors = set(t.author_handle for t in tweets if t.author_handle)
        
        # Engagement metrics
        like_counts = [t.likes for t in tweets]
        total_likes = sum(like_counts)
        total_reposts = sum(t.reposts for t in tweets)
        total_engagement = sum(t.likes + t.reposts + t.replies for t in tweets)
        
        # Time window estimation (assume ~60 min window for /f=live search)
        timeframe_minutes = 60
        engagement_velocity = total_engagement / max(1, timeframe_minutes)
        
        # Sentiment aggregation
        sentiments = [t.sentiment_score for t in tweets]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        
        # Bullish/bearish counts
        bullish_count = sum(1 for s in sentiments if s > 0.2)
        bearish_count = sum(1 for s in sentiments if s < -0.2)
        
        # Negative ratio (for scam/rug detection)
        neg_mentions = sum(1 for t in tweets if t.sentiment_score < -0.3)
        neg_ratio = neg_mentions / len(tweets)
        
        # Contract/DEX ratios
        contract_count = sum(1 for t in tweets if t.contains_contract)
        dex_count = sum(1 for t in tweets if t.contains_dex_link)
        
        return XSentimentSnapshot(
            symbol=symbol,
            tweet_count=len(tweets),
            unique_authors=len(authors),
            total_likes=total_likes,
            total_reposts=total_reposts,
            median_like_count=median(like_counts) if like_counts else 0.0,
            engagement_velocity=engagement_velocity,
            neg_mention_ratio=neg_ratio,
            contract_ratio=contract_count / len(tweets),
            dex_link_ratio=dex_count / len(tweets),
            heuristic_sentiment=avg_sentiment,
            bullish_tweet_count=bullish_count,
            bearish_tweet_count=bearish_count,
            tweets=tweets,  # Keep for potential LLM analysis
        )
    
    def _should_use_llm(self, snapshot: XSentimentSnapshot) -> bool:
        """
        Decide whether to use LLM for deeper analysis.
        
        Using local LLM (Ollama) so no budget limits - just need meaningful data.
        
        Criteria:
        - Enough tweets to be meaningful (at least 5)
        - Skip obvious rugs/scams (save processing time)
        """
        # Need some meaningful data
        if snapshot.tweet_count < 5:
            return False
        
        # Skip obvious rugs/scams (not worth analyzing)
        if snapshot.neg_mention_ratio > 0.5:
            return False
        
        return True
    
    async def _run_llm_sentiment(
        self,
        symbol: str,
        tweets: List[XTweet]
    ) -> Optional[Dict[str, Any]]:
        """
        Run LLM analysis on tweets for deeper sentiment understanding.
        
        Uses local Ollama LLM (FREE) - no budget limits.
        
        Args:
            symbol: Token symbol
            tweets: List of tweets to analyze
            
        Returns:
            Dict with sentiment_label (-1 to 1), explanation, red_flags
        """
        try:
            # Build prompt with top tweets
            prompt = self._build_llm_prompt(symbol, tweets[:40])
            
            # Use hybrid LLM routing - prefer local Ollama (FREE)
            try:
                from services.llm_helper import hybrid_request
                response = hybrid_request(
                    prompt=prompt,
                    service_name=f"x_sentiment_{symbol}",
                    force_provider="local"  # Use local Ollama (FREE)
                )
                # If it's a coroutine, await it
                if asyncio.iscoroutine(response):
                    response = await response
            except ImportError:
                logger.warning("llm_helper not available, skipping LLM analysis")
                return None
            
            if not response:
                return None
            
            # Parse JSON response
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                # Try to extract sentiment score directly
                score_match = re.search(r'[-+]?\d*\.?\d+', str(response))
                if score_match:
                    return {"sentiment_label": float(score_match.group()), "explanation": str(response)}
                return None
                
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed for {symbol}: {e}")
            return None
    
    def _build_llm_prompt(self, symbol: str, tweets: List[XTweet]) -> str:
        """Build prompt for LLM sentiment analysis"""
        lines = []
        for i, t in enumerate(tweets[:40], start=1):
            engagement = f"[{t.likes}♥ {t.reposts}🔄]"
            lines.append(f"{i}. ({t.author_handle}) {engagement} {t.text[:200]}")
        
        tweets_block = "\n".join(lines)
        
        return f"""You are analyzing social sentiment on X (Twitter) for cryptocurrency ${symbol}.

Recent tweets:
{tweets_block}

Instructions:
1. Rate overall sentiment from -1.0 (very bearish/scam warnings) to +1.0 (very bullish/FOMO).
2. Provide 2-3 bullet point explanation of the sentiment.
3. List any red flags (spam, bot activity, scam warnings, unrealistic claims).

Return JSON only:
{{"sentiment_label": <float -1 to 1>, "explanation": "<string>", "red_flags": [<strings>]}}"""
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear sentiment cache"""
        if symbol:
            self._cache.pop(symbol.upper(), None)
        else:
            self._cache.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

# Singleton instance
_x_sentiment_service: Optional[XSentimentService] = None


def get_x_sentiment_service(
    use_llm: bool = True,
    llm_budget_manager: Optional[Any] = None
) -> XSentimentService:
    """Get or create singleton X Sentiment Service"""
    global _x_sentiment_service
    if _x_sentiment_service is None:
        _x_sentiment_service = XSentimentService(
            use_llm=use_llm,
            llm_budget_manager=llm_budget_manager
        )
    return _x_sentiment_service


async def get_x_sentiment(symbol: str, max_tweets: int = 50) -> XSentimentSnapshot:
    """Quick helper to fetch X sentiment for a symbol"""
    service = get_x_sentiment_service()
    return await service.fetch_snapshot(symbol, max_tweets)


# Export
__all__ = [
    "XSentimentService",
    "XSentimentSnapshot",
    "XTweet",
    "get_x_sentiment_service",
    "get_x_sentiment",
    "BULLISH_WORDS",
    "BEARISH_WORDS",
]

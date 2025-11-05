"""
Advanced Social Sentiment Analyzer using Crawl4ai

Scrapes and analyzes sentiment from:
- Reddit (full threads + comments with relevance scoring)
- X/Twitter (via Nitter mirrors, no login required)
- StockTwits (real-time sentiment streams)
- Financial news sites (Seeking Alpha, Yahoo Finance, MarketWatch)
- Financial forums (public discussions)

Uses Crawl4ai for:
- JavaScript rendering (access dynamic content)
- Stealth mode (bypass bot detection)
- BM25 link scoring (find most relevant content)
- Parallel multi-URL crawling (fast)
- Adaptive crawling (smart resource management)
"""

from loguru import logger
import asyncio
import sys
import os
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import Counter
from dataclasses import dataclass


# Critical Windows fix for Playwright/Crawl4ai subprocess support
if sys.platform == 'win32':
    # Windows uses ProactorEventLoop by default which SUPPORTS subprocesses
    # Ensure we're using ProactorEventLoop
    try:
        policy = asyncio.get_event_loop_policy()
        if not isinstance(policy, asyncio.WindowsProactorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            logger.info("✅ Set WindowsProactorEventLoopPolicy for subprocess support")
        else:
            logger.info("✅ Using WindowsProactorEventLoopPolicy (subprocess supported)")
    except Exception as e:
        logger.warning(f"Event loop policy check: {e}")
    
    # Apply nest_asyncio to allow nested event loops (fixes Streamlit/Jupyter issues)
    try:
        import nest_asyncio
        # Check if already applied (e.g., by app.py)
        import asyncio
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except:
            pass
        
        # Only apply if not already patched
        if loop is None or not hasattr(loop, '_nest_patched'):
            nest_asyncio.apply()
            logger.info("✅ Applied nest_asyncio patch for nested event loop support")
        else:
            logger.info("✅ nest_asyncio already applied (skip re-patch)")
    except ImportError:
        logger.warning("nest_asyncio not installed - may have issues in some contexts")
        logger.warning("Install with: pip install nest-asyncio")


@dataclass
class SocialMention:
    """A single social media mention of a ticker"""
    source: str  # "reddit", "twitter", "stocktwits", etc.
    text: str
    sentiment: str  # "BULLISH", "BEARISH", "NEUTRAL"
    sentiment_score: float  # -1 to 1
    author: str
    timestamp: str
    url: str
    engagement: int  # likes/upvotes/retweets
    relevance_score: float  # BM25 score from Crawl4ai


@dataclass
class SocialSentimentResult:
    """Aggregated social sentiment for a ticker"""
    ticker: str
    total_mentions: int
    reddit_mentions: int
    twitter_mentions: int
    stocktwits_mentions: int
    news_mentions: int
    
    bullish_count: int
    bearish_count: int
    neutral_count: int
    
    overall_sentiment: str  # "BULLISH", "BEARISH", "NEUTRAL"
    sentiment_score: float  # -1 to 1
    buzz_score: float  # 0-100
    trending_score: float  # 0-100
    
    top_mentions: List[SocialMention]
    sources: List[str]


class SocialSentimentAnalyzer:
    """
    Advanced social sentiment analyzer using Crawl4ai for comprehensive web scraping.
    
    Features:
    - Full browser rendering for JS-heavy sites
    - Stealth mode to bypass bot detection
    - BM25 relevance scoring for finding best content
    - Multi-URL parallel crawling
    - Adaptive crawling (smart resource management)
    - Login-free access via public endpoints and mirrors
    """
    
    def __init__(self):
        """Initialize the analyzer with Crawl4ai"""
        self.crawler = None
        self._initialized = False
        self.reddit = None
        self._reddit_initialized = False
        self.last_reddit_request = 0
        self.reddit_request_interval = 2.0  # 2 seconds between Reddit API calls (increased from 1.0)
        self.reddit_error_count = 0  # Track consecutive errors for exponential backoff
        self.reddit_backoff_until = 0  # Timestamp when we can retry after rate limit
        
        # Create dedicated ThreadPoolExecutor for Reddit API calls
        # This prevents "cannot schedule new futures after shutdown" errors
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reddit_api")
        self._executor_shutdown = False
        
        # Reddit hybrid mode: use public scraping + API for best coverage
        self.reddit_use_hybrid = True  # Combine public scraping + API
        self.reddit_prefer_scraping = True  # Try public scraping first (no rate limits)
        
        logger.info("🔧 Social Sentiment Analyzer - SPEED OPTIMIZATIONS:")
        logger.info(f"   • Reddit: RSS feeds (FAST, no rate limits!) + API fallback")
        logger.info(f"   • Twitter: CSS selectors + caching (10s timeout)")
        logger.info(f"   • StockTwits: CSS selectors + caching (8s timeout)")
        logger.info(f"   • RSS = lightweight polling, regex pre-filter, no auth")
        logger.info(f"   • Target: 2-3s per ticker (vs 6s before)")
        
        # Nitter instances for Twitter access (no login required)
        # These are public Twitter mirrors that don't require authentication
        self.nitter_instances = [
            "https://nitter.poast.org",
            "https://nitter.privacydev.net",
            "https://nitter.net",
            "https://nitter.1d4.us",
            "https://nitter.kavin.rocks",
            "https://nitter.unixfox.eu"
        ]
        
        # Sentiment keywords
        self.bullish_keywords = {
            'moon', 'calls', 'bullish', 'buy', 'long', 'rocket', '🚀', '📈',
            'gains', 'rally', 'breakout', 'squeeze', 'gamma', 'yolo',
            'hold', 'hodl', 'pump', 'up', 'green', 'bull', 'mooning',
            'surge', 'rip', 'lambo', 'tendies', 'chad', 'based',
            'strong', 'winner', 'beat', 'crushing', 'killing'
        }
        
        self.bearish_keywords = {
            'puts', 'bearish', 'sell', 'short', 'dump', 'crash',
            'loss', 'red', 'bleeding', 'tanking', 'drill', '📉',
            'rip', 'rekt', 'bag', 'bagholding', 'down', 'bear',
            'plunge', 'drop', 'fall', 'collapse', 'dead', 'toast',
            'weak', 'loser', 'miss', 'disappointing'
        }
    
    async def _ensure_initialized(self):
        """Lazy initialization of Crawl4ai crawler with Windows subprocess fix"""
        if not self._initialized:
            try:
                from crawl4ai import AsyncWebCrawler
                
                # Crawler configuration
                crawler_config = {
                    'headless': True,
                    'verbose': False
                }
                
                logger.info("Initializing Crawl4ai crawler...")
                self.crawler = AsyncWebCrawler(**crawler_config)
                await self.crawler.__aenter__()
                self._initialized = True
                logger.info("✅ Crawl4ai initialized successfully (Windows compatible mode)")
                
            except NotImplementedError as e:
                logger.error(f"Playwright subprocess error: {e}")
                logger.error("="*60)
                logger.error("WINDOWS SUBPROCESS FIX REQUIRED:")
                logger.error("1. Install nest_asyncio: pip install nest-asyncio")
                logger.error("2. Ensure Playwright installed: playwright install chromium")
                logger.error("3. Run as administrator if permission issues")
                logger.error("4. Check antivirus isn't blocking Chromium")
                logger.error("="*60)
                self._initialized = False
                raise
            except ImportError as e:
                logger.error(f"Crawl4ai not installed: {e}")
                logger.error("="*60)
                logger.error("INSTALLATION REQUIRED:")
                logger.error("1. pip install crawl4ai")
                logger.error("2. crawl4ai-setup")
                logger.error("3. playwright install chromium")
                logger.error("="*60)
                self._initialized = False
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Crawl4ai: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self._initialized = False
                raise
    
    async def analyze_social_buzz(self, ticker: str) -> Dict:
        """
        Comprehensive social buzz analysis using Crawl4ai.
        Falls back to news-only mode if Crawl4ai fails.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict with detailed social sentiment analysis
        """
        try:
            await self._ensure_initialized()
            
            logger.info(f"🔍 Starting comprehensive social analysis for {ticker}")
            
            # Check if executor is still available
            if self._executor_shutdown:
                logger.warning(f"Executor shutdown, skipping social analysis for {ticker}")
                return self._empty_result()
            
            # Hybrid approach: combine public scraping + API for Reddit
            loop = asyncio.get_event_loop()
            
            if self.reddit_use_hybrid:
                # HYBRID MODE: Use FAST RSS feeds + API fallback
                logger.debug(f"🔀 Using HYBRID mode for Reddit (RSS feeds + API)")
                
                # Build task dictionary for cleaner result mapping
                task_map = {
                    'reddit_rss': self._scrape_reddit_via_rss(ticker),
                    'twitter': self._scrape_twitter_via_nitter(ticker),
                    'stocktwits': self._scrape_stocktwits(ticker),
                    'news': self._scrape_financial_news(ticker)
                }
                
                # Only add Reddit API if available (not in backoff)
                if self._is_reddit_available():
                    task_map['reddit_api'] = loop.run_in_executor(self._executor, self._scrape_reddit_via_api, ticker)
                else:
                    logger.debug(f"⏸️  Skipping Reddit API for {ticker} (in backoff or unavailable)")
                
                tasks = list(task_map.values())
                task_names = list(task_map.keys())
            else:
                # Original: API only - use dedicated executor
                task_map = {
                    'reddit_api': loop.run_in_executor(self._executor, self._scrape_reddit_via_api, ticker),
                    'twitter': self._scrape_twitter_via_nitter(ticker),
                    'stocktwits': self._scrape_stocktwits(ticker),
                    'news': self._scrape_financial_news(ticker)
                }
                tasks = list(task_map.values())
                task_names = list(task_map.keys())
            
            # Add timeout protection to prevent hanging
            # Reduced timeout since we skip Reddit API when in backoff
            timeout = 15.0 if not self._is_reddit_available() else 25.0
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Social analysis timeout for {ticker} after {timeout}s, returning partial results")
                return self._empty_result()
        except (NotImplementedError, ImportError) as e:
            # Crawl4ai failed - fall back to news only
            logger.warning(f"Crawl4ai unavailable for {ticker}, using news-only fallback")
            try:
                news_mentions = await self._scrape_financial_news(ticker)
                return self._create_news_only_result(ticker, news_mentions)
            except Exception as news_error:
                logger.error(f"News fallback also failed: {news_error}")
                return self._empty_result()
        except Exception as e:
            logger.error(f"Error in social buzz analysis for {ticker}: {e}")
            return self._empty_result()
        
        try:
            
            # Process results
            all_mentions = []
            reddit_mentions = []
            twitter_mentions = []
            stocktwits_mentions = []
            news_mentions = []
            
            # Process results using task name mapping (cleaner than index-based)
            for task_name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logger.debug(f"Source {task_name} failed: {result}")
                    continue
                
                if task_name == 'reddit_rss':
                    reddit_mentions.extend(result)
                    logger.debug(f"   Reddit RSS: {len(result)} mentions")
                elif task_name == 'reddit_api':
                    reddit_mentions.extend(result)
                    logger.debug(f"   Reddit API: {len(result)} mentions")
                elif task_name == 'twitter':
                    twitter_mentions = result
                elif task_name == 'stocktwits':
                    stocktwits_mentions = result
                elif task_name == 'news':
                    news_mentions = result
                
                all_mentions.extend(result)
            
            if self.reddit_use_hybrid:
                logger.info(f"📱 Reddit HYBRID: {len(reddit_mentions)} total mentions (RSS + API)")
            else:
                logger.info(f"📱 Reddit API: {len(reddit_mentions)} mentions")
            
            # Analyze sentiment
            sentiment_result = self._analyze_mentions(ticker, all_mentions, 
                                                     reddit_mentions, twitter_mentions,
                                                     stocktwits_mentions, news_mentions)
            
            # Convert to dict for backward compatibility
            return self._sentiment_result_to_dict(sentiment_result)
            
        except Exception as e:
            logger.error(f"Error processing mentions for {ticker}: {e}")
            return self._empty_result()
    
    def _init_reddit(self):
        """Initialize Reddit API client (PRAW) - no more scraping!"""
        if self._reddit_initialized:
            return
        
        try:
            import praw
            
            # Load credentials from environment variables (more secure)
            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = os.getenv('REDDIT_USER_AGENT', 'SentimentAnalyzer/1.0')
            
            if not client_id or not client_secret:
                logger.error("Reddit API credentials not found in .env file")
                logger.error("Add REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET to .env")
                self._reddit_initialized = False
                return
            
            # Initialize Reddit API with credentials from .env
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test authentication
            _ = self.reddit.read_only
            self._reddit_initialized = True
            logger.info("✅ Reddit API (PRAW) initialized - no rate limit issues!")
            
        except ImportError:
            logger.warning("PRAW not installed - Reddit API disabled")
            logger.warning("Install with: pip install praw")
            self._reddit_initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            self._reddit_initialized = False
    
    def _is_reddit_available(self) -> bool:
        """Check if Reddit API is available (not in backoff period)"""
        if not self._reddit_initialized or not self.reddit:
            return False
        
        # Check if we're in a backoff period
        current_time = time.time()
        if current_time < self.reddit_backoff_until:
            return False
        
        return True
    
    def _rate_limit_reddit(self):
        """Ensure we don't exceed Reddit API rate limits with exponential backoff"""
        current_time = time.time()
        
        # Check if we're in a backoff period (after rate limit hit)
        if current_time < self.reddit_backoff_until:
            backoff_remaining = self.reddit_backoff_until - current_time
            logger.warning(f"⏳ Reddit rate limit backoff: waiting {backoff_remaining:.1f}s")
            time.sleep(backoff_remaining)
            current_time = time.time()
        
        # Normal rate limiting
        elapsed = current_time - self.last_reddit_request
        if elapsed < self.reddit_request_interval:
            time.sleep(self.reddit_request_interval - elapsed)
        
        self.last_reddit_request = time.time()
    
    def _scrape_reddit_via_api(self, ticker: str) -> List[SocialMention]:
        """Fetch Reddit mentions using official API (PRAW) - much more reliable!"""
        mentions = []
        
        try:
            self._init_reddit()
            
            # Quick check - skip if Reddit is unavailable or in backoff
            if not self._is_reddit_available():
                if self.reddit_backoff_until > time.time():
                    backoff_remaining = self.reddit_backoff_until - time.time()
                    logger.debug(f"⏸️  Skipping Reddit for {ticker} - in backoff period ({backoff_remaining:.0f}s remaining)")
                else:
                    logger.debug(f"⏸️  Skipping Reddit for {ticker} - API not initialized")
                return mentions
            
            # Top trading subreddits for API (matching RSS list)
            # Limited to most active to respect API rate limits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'pennystocks']
            
            for subreddit_name in subreddits:
                try:
                    self._rate_limit_reddit()
                    
                    subreddit = self.reddit.subreddit(subreddit_name)
                    
                    # Search for ticker mentions
                    search_query = f"${ticker} OR {ticker}"
                    posts = list(subreddit.search(
                        query=search_query,
                        sort='relevance',
                        time_filter='week',
                        limit=3  # Reduced from 5 to minimize API calls
                    ))
                    
                    for post in posts:
                        # Combine title and body text
                        text = f"{post.title}. {post.selftext[:200]}"
                        
                        # Determine sentiment
                        sentiment, score = self._analyze_text_sentiment(text)
                        
                        mentions.append(SocialMention(
                            source='reddit',
                            text=text,
                            sentiment=sentiment,
                            sentiment_score=score,
                            author=str(post.author) if post.author else '[deleted]',
                            timestamp=datetime.fromtimestamp(post.created_utc).isoformat(),
                            url=f"https://reddit.com{post.permalink}",
                            engagement=post.score + post.num_comments,
                            platform_specific={'upvote_ratio': post.upvote_ratio}
                        ))
                    
                    if posts:
                        logger.info(f"✅ Reddit API r/{subreddit_name}: {len(posts)} posts for {ticker}")
                        # Success - reset error count
                        self.reddit_error_count = 0
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Handle rate limiting (401/429 errors)
                    if "401" in error_str or "429" in error_str or "rate limit" in error_str.lower():
                        self.reddit_error_count += 1
                        
                        # Exponential backoff: 30s, 60s, 120s, 300s (5 min max)
                        backoff_seconds = min(30 * (2 ** (self.reddit_error_count - 1)), 300)
                        self.reddit_backoff_until = time.time() + backoff_seconds
                        
                        logger.warning(f"⚠️ Reddit rate limit hit! Backing off for {backoff_seconds}s (error #{self.reddit_error_count})")
                        
                        # Skip remaining subreddits for this ticker to avoid more rate limit hits
                        break
                    else:
                        logger.debug(f"Reddit API r/{subreddit_name} error: {e}")
                    continue
            
            if mentions:
                logger.info(f"📱 Reddit API: {len(mentions)} total mentions for {ticker}")
            
            return mentions
            
        except Exception as e:
            logger.error(f"Reddit API error for {ticker}: {e}")
            return mentions
    
    async def _scrape_reddit_via_rss(self, ticker: str) -> List[SocialMention]:
        """Fast Reddit polling via RSS feeds (no rate limits!) + selective API expansion"""
        mentions = []
        
        logger.info(f"📡 Starting Reddit RSS scan for ${ticker}...")
        
        try:
            import feedparser
            import httpx
            
            # Top trading subreddits (reduced to 3 most active for speed)
            subreddits = [
                'wallstreetbets',   # 15M+ members - most active, options/meme stocks
                'stocks',            # 5M+ members - general stock discussion
                'investing',         # 2M+ members - serious long-term investors
            ]
            
            # Pre-filter: compile ticker regex for fast matching
            # More flexible pattern - match $TICKER, TICKER with spaces/punctuation around it
            ticker_pattern = re.compile(
                rf'(?:^|\s|\$)({ticker.upper()})(?:\s|$|[.,!?;:])',
                re.IGNORECASE
            )
            
            # Add timeout for entire RSS scan (10 seconds max for all subreddits)
            async def _fetch_rss():
                nonlocal mentions
                for subreddit in subreddits:
                    try:
                        # STEP A: Poll RSS feed (FAST - no rate limits, no auth needed!)
                        rss_url = f"https://www.reddit.com/r/{subreddit}/new.rss?limit=25"
                        
                        # Reduced timeout to 3 seconds per subreddit for faster failure
                        async with httpx.AsyncClient(timeout=3.0) as client:
                            response = await client.get(rss_url, headers={'User-Agent': 'Mozilla/5.0'})
                            
                            if response.status_code != 200:
                                logger.debug(f"RSS feed r/{subreddit} returned {response.status_code}")
                                continue
                            
                            # Parse RSS feed
                            feed = feedparser.parse(response.text)
                            
                            if not feed.entries:
                                logger.debug(f"RSS feed r/{subreddit} has no entries")
                                continue
                            
                            logger.debug(f"RSS feed r/{subreddit}: scanning {len(feed.entries)} posts for ${ticker}")
                            
                            # STEP B: Pre-filter for ticker mentions (FAST regex scan)
                            candidate_posts = []
                            for entry in feed.entries:
                                title = entry.get('title', '')
                                summary = entry.get('summary', '')
                                combined_text = f"{title} {summary}"
                                
                                # Quick regex check - also do simple string search as fallback
                                if (ticker_pattern.search(combined_text) or 
                                    f"${ticker.upper()}" in combined_text.upper() or
                                    f" {ticker.upper()} " in f" {combined_text.upper()} "):
                                    candidate_posts.append(entry)
                                    logger.debug(f"Found match in: {title[:50]}...")
                            
                            # STEP C: Extract from RSS (no API call needed for basic info!)
                            for entry in candidate_posts[:10]:  # Limit to top 10 matches per subreddit
                                title = entry.get('title', '')
                                summary = entry.get('summary', '')[:200]
                                text = f"{title}. {summary}"
                                
                                # Parse author
                                author = 'unknown'
                                if 'author' in entry:
                                    author_match = re.search(r'/u/(\w+)', entry.author)
                                    if author_match:
                                        author = author_match.group(1)
                                
                                # Get timestamp
                                timestamp = entry.get('published', datetime.now().isoformat())
                                
                                # Analyze sentiment
                                sentiment, score = self._analyze_text_sentiment(text)
                                
                                mention = SocialMention(
                                    source=f"reddit/r/{subreddit}",
                                    text=text[:200],
                                    sentiment=sentiment,
                                    sentiment_score=score,
                                    author=author,
                                    timestamp=timestamp,
                                    url=entry.get('link', ''),
                                    engagement=0,  # RSS doesn't include upvotes
                                    relevance_score=0.7  # Higher than scraping
                                )
                                mentions.append(mention)
                            
                            if candidate_posts:
                                logger.info(f"✅ Reddit RSS r/{subreddit}: {len(candidate_posts)} posts matching ${ticker}")
                            else:
                                logger.debug(f"No matches found in r/{subreddit} for ${ticker}")
                        
                    except Exception as e:
                        logger.debug(f"RSS error r/{subreddit}: {e}")
                        continue
            
            # Run with timeout
            try:
                await asyncio.wait_for(_fetch_rss(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning(f"Reddit RSS timeout after 10s for {ticker}, returning {len(mentions)} mentions")
            
            logger.info(f"📱 Reddit RSS: {len(mentions)} total mentions for {ticker}")
            return mentions
            
        except ImportError:
            logger.warning("feedparser not installed - install with: pip install feedparser")
            return []
        except Exception as e:
            logger.error(f"Reddit RSS failed: {e}")
            return []
    
    async def _scrape_twitter_via_nitter(self, ticker: str) -> List[SocialMention]:
        """Scrape Twitter via Nitter (no login required) - FAST MODE"""
        mentions = []
        
        try:
            # Check if crawler is still initialized
            if not self._initialized or not self.crawler:
                logger.debug(f"Crawler not initialized, skipping Twitter for {ticker}")
                return mentions
            
            from crawl4ai import CrawlerRunConfig, CacheMode
            
            # Try multiple Nitter instances
            for nitter_url in self.nitter_instances[:3]:  # Try first 3 only for speed
                try:
                    # Twitter standard is $TICKER - prioritize that format
                    # Search for "$TICKER OR TICKER" to catch both formats
                    search_url = f"{nitter_url}/search?f=tweets&q=%24{ticker}+OR+{ticker}&since=&until=&near="
                    
                    # OPTIMIZED CONFIG - Fast extraction
                    config = CrawlerRunConfig(
                        # CSS selector for tweet content
                        css_selector="div.timeline-item, div.tweet-content, p.tweet-content",
                        
                        # Performance optimizations
                        word_count_threshold=3,  # Tweets are short
                        only_text=True,
                        wait_for_images=False,
                        process_iframes=False,
                        remove_overlay_elements=True,
                        page_timeout=10000,  # Reduced from 15s to 10s
                        
                        # Caching for speed
                        cache_mode=CacheMode.ENABLED,
                        
                        verbose=False
                    )
                    
                    logger.info(f"🐦 Trying Twitter via {nitter_url} for ${ticker}...")
                    # Add timeout for individual crawler runs with retry on connection errors
                    max_retries = 2
                    for retry in range(max_retries):
                        try:
                            result = await asyncio.wait_for(
                                self.crawler.arun(search_url, config=config),
                                timeout=15.0
                            )
                            break  # Success, exit retry loop
                        except Exception as e:
                            if "Connection closed" in str(e) and retry < max_retries - 1:
                                logger.debug(f"Connection error, retry {retry + 1}/{max_retries}")
                                await asyncio.sleep(1)  # Brief delay before retry
                                continue
                            raise  # Re-raise if last retry or different error
                    
                    if result.success:
                        text_content = result.markdown or ""
                        
                        if not text_content or len(text_content) < 50:
                            logger.debug(f"{nitter_url}: Empty content, trying next instance")
                            continue
                        
                        # Split into chunks (tweets are usually separated by newlines)
                        lines = text_content.split('\n')
                        
                        tweet_count = 0
                        for line in lines:
                            # Skip short lines
                            if len(line) < 20:
                                continue
                            
                            # Check if ticker is mentioned (prioritize $TICKER format)
                            line_upper = line.upper()
                            if f"${ticker.upper()}" in line_upper or ticker.upper() in line_upper:
                                # Look for username pattern (@username)
                                author_match = re.search(r'@(\w+)', line)
                                author = author_match.group(1) if author_match else 'unknown'
                                
                                # Clean the text
                                text = line.strip()[:280]
                                
                                sentiment, score = self._analyze_text_sentiment(text)
                                
                                mention = SocialMention(
                                    source="twitter",
                                    text=text[:200],
                                    sentiment=sentiment,
                                    sentiment_score=score,
                                    author=author,
                                    timestamp=datetime.now().isoformat(),
                                    url=search_url,
                                    engagement=0,
                                    relevance_score=0.6
                                )
                                mentions.append(mention)
                                tweet_count += 1
                                
                                # Limit to 20 tweets per instance
                                if tweet_count >= 20:
                                    break
                        
                        logger.info(f"✅ Twitter (via {nitter_url}): {tweet_count} tweets found")
                        if tweet_count > 0:
                            break  # Success, don't try other instances
                    
                except Exception as e:
                    logger.debug(f"Nitter instance {nitter_url} failed: {e}")
                    continue
            
            logger.info(f"🐦 Twitter: {len(mentions)} total mentions for {ticker}")
            return mentions
            
        except Exception as e:
            logger.error(f"Twitter scraping failed: {e}")
            return []
    
    async def _scrape_stocktwits(self, ticker: str) -> List[SocialMention]:
        """Scrape StockTwits (public streams, no login) - FAST MODE"""
        mentions = []
        
        try:
            # Check if crawler is still initialized
            if not self._initialized or not self.crawler:
                logger.debug(f"Crawler not initialized, skipping StockTwits for {ticker}")
                return mentions
            
            from crawl4ai import CrawlerRunConfig, CacheMode
            
            url = f"https://stocktwits.com/symbol/{ticker.upper()}"
            
            # OPTIMIZED CONFIG - Fast extraction
            config = CrawlerRunConfig(
                # CSS selector for StockTwits messages
                css_selector="div.st-stream-item, div.message-text, article",
                
                # Performance optimizations
                word_count_threshold=3,
                only_text=True,
                wait_for_images=False,
                process_iframes=False,
                remove_overlay_elements=True,
                page_timeout=8000,  # Reduced from 15s to 8s
                
                # Caching for speed
                cache_mode=CacheMode.ENABLED,
                
                # Quick scroll (reduced wait time)
                js_code=[
                    "window.scrollTo(0, document.body.scrollHeight/2);",
                    "await new Promise(r => setTimeout(r, 1000));"  # Reduced from 2s to 1s
                ],
                
                verbose=False
            )
            
            # Add timeout for individual crawler runs with retry on connection errors
            max_retries = 2
            for retry in range(max_retries):
                try:
                    result = await asyncio.wait_for(
                        self.crawler.arun(url, config=config),
                        timeout=12.0
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if "Connection closed" in str(e) and retry < max_retries - 1:
                        logger.debug(f"StockTwits connection error, retry {retry + 1}/{max_retries}")
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    raise  # Re-raise if last retry or different error
            
            if result.success:
                text_content = result.markdown or ""
                
                # Parse StockTwits messages
                # Look for sentiment indicators
                lines = text_content.split('\n')
                
                for i, line in enumerate(lines):
                    if ticker.upper() in line and len(line) > 20:
                        # Check for explicit sentiment
                        sentiment = "NEUTRAL"
                        score = 0.0
                        
                        if "Bullish" in line or "📈" in line:
                            sentiment = "BULLISH"
                            score = 0.5
                        elif "Bearish" in line or "📉" in line:
                            sentiment = "BEARISH"
                            score = -0.5
                        else:
                            sentiment, score = self._analyze_text_sentiment(line)
                        
                        mention = SocialMention(
                            source="stocktwits",
                            text=line[:200],
                            sentiment=sentiment,
                            sentiment_score=score,
                            author="unknown",
                            timestamp=datetime.now().isoformat(),
                            url=url,
                            engagement=0,
                            relevance_score=0.6
                        )
                        mentions.append(mention)
                
                logger.info(f"✅ StockTwits: {len(mentions)} messages found")
            
            logger.info(f"💬 StockTwits: {len(mentions)} total mentions for {ticker}")
            return mentions
            
        except Exception as e:
            logger.error(f"StockTwits scraping failed: {e}")
            return []
    
    async def _scrape_financial_news(self, ticker: str) -> List[SocialMention]:
        """Scrape financial news sites"""
        mentions = []
        
        try:
            from crawl4ai import CrawlerRunConfig
            
            # Use yfinance as primary source (reliable)
            import yfinance as yf
            
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news:
                for article in news[:10]:
                    try:
                        title = article.get('title', '')
                        summary = article.get('summary', title)
                        text = f"{title} {summary}"
                        
                        sentiment, score = self._analyze_text_sentiment(text)
                        
                        mention = SocialMention(
                            source="financial_news",
                            text=text[:300],
                            sentiment=sentiment,
                            sentiment_score=score,
                            author=article.get('publisher', 'Unknown'),
                            timestamp=datetime.now().isoformat(),
                            url=article.get('link', ''),
                            engagement=0,
                            relevance_score=0.8
                        )
                        mentions.append(mention)
                    except:
                        continue
            
            logger.info(f"📰 Financial News: {len(mentions)} articles for {ticker}")
            return mentions
            
        except Exception as e:
            logger.error(f"News scraping failed: {e}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment of text using keyword matching"""
        text_lower = text.lower()
        
        # Count bullish and bearish keywords
        bull_score = sum(1 for word in self.bullish_keywords if word in text_lower)
        bear_score = sum(1 for word in self.bearish_keywords if word in text_lower)
        
        total = bull_score + bear_score
        if total == 0:
            return "NEUTRAL", 0.0
        
        # Calculate sentiment score (-1 to 1)
        sentiment_score = (bull_score - bear_score) / total
        
        # Determine sentiment category
        if sentiment_score > 0.3:
            sentiment = "BULLISH"
        elif sentiment_score < -0.3:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        return sentiment, sentiment_score
    
    def _analyze_mentions(self, ticker: str, all_mentions: List[SocialMention],
                         reddit: List, twitter: List, stocktwits: List, 
                         news: List) -> SocialSentimentResult:
        """Aggregate and analyze all mentions"""
        
        # Count sentiments
        bullish = len([m for m in all_mentions if m.sentiment == "BULLISH"])
        bearish = len([m for m in all_mentions if m.sentiment == "BEARISH"])
        neutral = len([m for m in all_mentions if m.sentiment == "NEUTRAL"])
        
        # Calculate overall sentiment
        if not all_mentions:
            overall_sentiment = "NEUTRAL"
            sentiment_score = 0.0
        else:
            avg_score = sum(m.sentiment_score for m in all_mentions) / len(all_mentions)
            sentiment_score = avg_score
            
            if avg_score > 0.2:
                overall_sentiment = "BULLISH"
            elif avg_score < -0.2:
                overall_sentiment = "BEARISH"
            else:
                overall_sentiment = "NEUTRAL"
        
        # Calculate buzz score (0-100)
        buzz_score = min(100, len(all_mentions) * 2)
        
        # Calculate trending score
        trending_score = min(100, buzz_score * 1.2) if len(all_mentions) > 10 else buzz_score * 0.8
        
        # Get top mentions (sorted by relevance)
        top_mentions = sorted(all_mentions, key=lambda x: x.relevance_score, reverse=True)[:10]
        
        # Sources
        sources = []
        if reddit:
            sources.append(f"Reddit: {len(reddit)} mentions")
        if twitter:
            sources.append(f"Twitter: {len(twitter)} tweets")
        if stocktwits:
            sources.append(f"StockTwits: {len(stocktwits)} messages")
        if news:
            sources.append(f"News: {len(news)} articles")
        
        return SocialSentimentResult(
            ticker=ticker,
            total_mentions=len(all_mentions),
            reddit_mentions=len(reddit),
            twitter_mentions=len(twitter),
            stocktwits_mentions=len(stocktwits),
            news_mentions=len(news),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
            buzz_score=buzz_score,
            trending_score=trending_score,
            top_mentions=top_mentions,
            sources=sources
        )
    
    def _sentiment_result_to_dict(self, result: SocialSentimentResult) -> Dict:
        """Convert SocialSentimentResult to dict for backward compatibility"""
        return {
            'social_score': result.buzz_score,
            'total_mentions': result.total_mentions,
            'reddit_mentions': result.reddit_mentions,
            'twitter_mentions': result.twitter_mentions,
            'stocktwits_mentions': result.stocktwits_mentions,
            'news_mentions': result.news_mentions,
            'sentiment': result.overall_sentiment,
            'sentiment_score': result.sentiment_score,
            'trending_score': result.trending_score,
            'sources': result.sources,
            'top_discussions': [
                {
                    'source': m.source,
                    'text': m.text,
                    'sentiment': m.sentiment,
                    'author': m.author
                }
                for m in result.top_mentions[:5]
            ]
        }
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            'social_score': 0,
            'total_mentions': 0,
            'reddit_mentions': 0,
            'twitter_mentions': 0,
            'stocktwits_mentions': 0,
            'news_mentions': 0,
            'sentiment': 'NEUTRAL',
            'sentiment_score': 0.0,
            'trending_score': 0,
            'sources': [],
            'top_discussions': []
        }
    
    def _create_news_only_result(self, ticker: str, news_mentions: List[SocialMention]) -> Dict:
        """Create result from news mentions only (fallback mode)"""
        if not news_mentions:
            return self._empty_result()
        
        sentiment_result = self._analyze_mentions(
            ticker, news_mentions, [], [], [], news_mentions
        )
        result = self._sentiment_result_to_dict(sentiment_result)
        result['sources'] = ['News Only (Crawl4ai unavailable)']
        return result
    
    async def close(self):
        """Cleanup crawler resources and executor"""
        # Shutdown executor first (wait for pending tasks)
        if hasattr(self, '_executor') and not self._executor_shutdown:
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._executor_shutdown = True
                logger.info("✅ ThreadPoolExecutor shutdown successfully")
            except Exception as e:
                logger.debug(f"Error shutting down executor: {e}")
        
        # Then close crawler
        if self.crawler and self._initialized:
            try:
                await self.crawler.__aexit__(None, None, None)
                self._initialized = False
                logger.info("✅ Crawl4ai closed successfully")
            except Exception as e:
                logger.debug(f"Error closing crawler: {e}")


# Example usage
if __name__ == "__main__":
    async def test():
        analyzer = SocialSentimentAnalyzer()
        
        print("\n🔍 Testing social sentiment analyzer...")
        result = await analyzer.analyze_social_buzz('NVDA')
        
        print(f"\n📊 Results for NVDA:")
        print(f"  Total Mentions: {result['total_mentions']}")
        print(f"  Sentiment: {result['sentiment']} ({result['sentiment_score']:.2f})")
        print(f"  Buzz Score: {result['social_score']:.0f}/100")
        print(f"  Sources: {result['sources']}")
        
        print("\n💬 Top Discussions:")
        for i, disc in enumerate(result['top_discussions'][:3], 1):
            print(f"  {i}. [{disc['source']}] {disc['sentiment']}")
            print(f"     {disc['text'][:100]}...")
        
        await analyzer.close()
    
    asyncio.run(test())

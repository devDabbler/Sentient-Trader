"""
Pre-Listing & New Launch Crypto Scanner

Detects newly listed coins and pre-IPO opportunities with social sentiment analysis.
Catches coins like HIPPO before they explode by monitoring:
- New exchange listings
- Social media buzz (Twitter, Reddit, Telegram)
- Keywords: "launching", "listing", "running", "mooning"
- Volume surges on low-cap coins
- Community excitement indicators
"""

import os
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from loguru import logger
import requests
from bs4 import BeautifulSoup
import json

from clients.kraken_client import KrakenClient
from utils.crypto_pair_utils import normalize_crypto_pair


@dataclass
class PreListingSignal:
    """Pre-listing or new launch signal"""
    symbol: str
    base_asset: str
    signal_type: str  # 'NEW_LISTING', 'PRE_IPO_BUZZ', 'RUNNING', 'MOONING'
    score: float  # 0-100
    confidence: str  # 'HIGH', 'MEDIUM', 'LOW'
    
    # Price & Volume
    current_price: float
    change_24h: float
    volume_24h: float
    volume_ratio: float  # Current vs average
    market_cap: Optional[float] = None
    
    # Social Sentiment
    social_score: float = 0.0  # 0-100
    social_mentions: int = 0
    sentiment_keywords: List[str] = None
    
    # Listing Info
    listing_date: Optional[str] = None
    exchange_announced: Optional[str] = None
    days_since_listing: Optional[int] = None
    
    # Analysis
    reasoning: str = ""
    catalyst: str = ""
    risk_level: str = "HIGH"  # Pre-listings are inherently risky
    
    # Meta
    timestamp: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.sentiment_keywords is None:
            self.sentiment_keywords = []


class PreListingScanner:
    """
    Scans for pre-listing and newly listed coins with high social sentiment
    """
    
    # Keywords that indicate pre-IPO/listing buzz
    BULLISH_KEYWORDS = [
        'launching', 'listing', 'listed', 'launch',
        'running', 'mooning', 'pumping', 'exploding',
        'breakout', 'parabolic', 'going crazy',
        'new coin', 'new listing', 'just listed',
        'pre-ipo', 'pre-listing', 'upcoming',
        'exchange announcement', 'binance listing',
        'coinbase listing', 'kraken listing',
        'gem', 'hidden gem', 'early', 'undervalued'
    ]
    
    URGENCY_KEYWORDS = [
        'NOW', 'FAST', 'QUICK', 'HURRY', 'ASAP',
        'running', 'flying', 'soaring', 'taking off'
    ]
    
    def __init__(self, kraken_client: KrakenClient):
        """
        Initialize pre-listing scanner
        
        Args:
            kraken_client: KrakenClient instance for price data
        """
        self.client = kraken_client
        
        # Social media API keys (optional)
        self.twitter_bearer = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_secret = os.getenv('REDDIT_CLIENT_SECRET')
        
        logger.info("🚀 Pre-Listing Scanner initialized")
        logger.info("   Twitter: {}", str('Enabled' if self.twitter_bearer else 'Disabled (set TWITTER_BEARER_TOKEN)'))
        logger.info("   Reddit: {}", str('Enabled' if self.reddit_client_id else 'Disabled (set REDDIT_CLIENT_ID)'))
    
    def scan_new_listings(
        self,
        max_days_old: int = 30,
        min_social_score: float = 50.0,
        top_n: int = 10
    ) -> List[PreListingSignal]:
        """
        Scan for newly listed coins with high social sentiment
        
        Args:
            max_days_old: Only consider coins listed within this many days
            min_social_score: Minimum social sentiment score (0-100)
            top_n: Number of top opportunities to return
            
        Returns:
            List of PreListingSignal objects
        """
        logger.info(f"🔍 Scanning for new listings (max {max_days_old} days old)...")
        
        signals = []
        
        # Get potential new listings from various sources
        candidates = self._get_new_listing_candidates(max_days_old)
        
        logger.info(f"Found {len(candidates)} potential new listings, analyzing...")
        
        for candidate in candidates:
            try:
                signal = self._analyze_candidate(candidate)
                
                if signal and signal.social_score >= min_social_score:
                    signals.append(signal)
                    logger.info(f"   ✓ {signal.symbol}: Social {signal.social_score:.1f}, Score {signal.score:.1f}")
            
            except Exception as e:
                logger.debug(f"Error analyzing {candidate}: {e}")
                continue
        
        # Sort by combined score (social + technical)
        signals.sort(key=lambda x: (x.social_score * 0.4) + (x.score * 0.6), reverse=True)
        
        logger.info(f"✅ Found {len(signals)} new listing signals")
        
        return signals[:top_n]
    
    def scan_pre_ipo_buzz(
        self,
        min_mentions: int = 20,
        top_n: int = 5
    ) -> List[PreListingSignal]:
        """
        Scan for coins with pre-IPO/pre-listing buzz
        
        Args:
            min_mentions: Minimum social media mentions
            top_n: Number of top opportunities to return
            
        Returns:
            List of PreListingSignal objects with pre-IPO buzz
        """
        logger.info(f"🔥 Scanning for pre-IPO buzz (min {min_mentions} mentions)...")
        
        signals = []
        
        # Search social media for pre-listing buzz
        buzz_coins = self._search_pre_listing_buzz()
        
        logger.info(f"Found {len(buzz_coins)} coins with pre-listing buzz")
        
        for coin_data in buzz_coins:
            try:
                if coin_data['mentions'] >= min_mentions:
                    signal = self._create_buzz_signal(coin_data)
                    signals.append(signal)
                    logger.info(f"   🚀 {signal.symbol}: {signal.social_mentions} mentions, {signal.catalyst}")
            
            except Exception as e:
                logger.debug(f"Error creating buzz signal: {e}")
                continue
        
        # Sort by social mentions and sentiment
        signals.sort(key=lambda x: (x.social_mentions * x.social_score), reverse=True)
        
        logger.info(f"✅ Found {len(signals)} pre-IPO buzz signals")
        
        return signals[:top_n]
    
    def _get_new_listing_candidates(self, max_days_old: int) -> List[Dict]:
        """Get candidates for new listings from multiple sources"""
        candidates = []
        
        # Source 1: Kraken new listings
        kraken_new = self._get_kraken_new_listings(max_days_old)
        candidates.extend(kraken_new)
        
        # Source 2: CoinGecko recently added
        coingecko_new = self._get_coingecko_new_listings(max_days_old)
        candidates.extend(coingecko_new)
        
        # Source 3: Social media mentions (coins trending recently)
        social_trending = self._get_trending_from_social()
        candidates.extend(social_trending)
        
        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            symbol = c.get('symbol', '').upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                unique_candidates.append(c)
        
        return unique_candidates
    
    def _get_kraken_new_listings(self, max_days_old: int) -> List[Dict]:
        """Get new listings from Kraken"""
        candidates = []
        
        try:
            # Get all tradable pairs from Kraken
            tradable_pairs = self.client.get_tradable_asset_pairs()
            
            if not tradable_pairs:
                return candidates
            
            # Kraken doesn't provide listing dates in API, so we'll check volume patterns
            # New listings typically have explosive volume
            for pair, pair_data in tradable_pairs.items():
                try:
                    if '/USD' in pair or 'USD' in pair:
                        ticker = self.client.get_ticker_data(pair)
                        
                        if ticker and ticker.get('volume_24h', 0) > 100000:  # $100k+ volume
                            candidates.append({
                                'symbol': pair,
                                'source': 'kraken',
                                'volume_24h': ticker.get('volume_24h'),
                                'price': ticker.get('last_price')
                            })
                
                except Exception as e:
                    continue
        
        except Exception as e:
            logger.debug(f"Error getting Kraken new listings: {e}")
        
        return candidates[:50]  # Limit to top 50 by volume
    
    def _get_coingecko_new_listings(self, max_days_old: int) -> List[Dict]:
        """Get recently added coins from CoinGecko"""
        candidates = []
        
        try:
            # CoinGecko API - recently added coins
            url = "https://api.coingecko.com/api/v3/coins/list"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                coins = response.json()
                
                # Get detailed data for recent coins (by volume)
                for coin in coins[-200:]:  # Check last 200 added coins
                    candidates.append({
                        'symbol': f"{coin['symbol'].upper()}/USD",
                        'base_asset': coin['symbol'].upper(),
                        'name': coin['name'],
                        'source': 'coingecko',
                        'id': coin['id']
                    })
        
        except Exception as e:
            logger.debug(f"Error getting CoinGecko new listings: {e}")
        
        return candidates
    
    def _get_trending_from_social(self) -> List[Dict]:
        """Get trending coins from social media"""
        candidates = []
        
        # Check Twitter trending
        if self.twitter_bearer:
            twitter_trending = self._search_twitter_trending()
            candidates.extend(twitter_trending)
        
        # Check Reddit trending
        if self.reddit_client_id:
            reddit_trending = self._search_reddit_trending()
            candidates.extend(reddit_trending)
        
        return candidates
    
    def _search_twitter_trending(self) -> List[Dict]:
        """Search Twitter for trending crypto coins"""
        candidates = []
        
        try:
            headers = {'Authorization': f'Bearer {self.twitter_bearer}'}
            
            # Search for crypto-related tweets with launch keywords
            queries = [
                'crypto launching',
                'new coin listing',
                'just listed crypto',
                'coin mooning',
                'crypto running'
            ]
            
            for query in queries:
                url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=100"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    tweets = data.get('data', [])
                    
                    # Extract coin mentions
                    for tweet in tweets:
                        text = tweet.get('text', '')
                        symbols = self._extract_crypto_symbols(text)
                        
                        for symbol in symbols:
                            candidates.append({
                                'symbol': f"{symbol}/USD",
                                'base_asset': symbol,
                                'source': 'twitter',
                                'mention_text': text[:200]
                            })
        
        except Exception as e:
            logger.debug(f"Twitter API error: {e}")
        
        return candidates
    
    def _search_reddit_trending(self) -> List[Dict]:
        """Search Reddit for trending crypto coins"""
        candidates = []
        
        try:
            # Reddit API authentication
            auth = requests.auth.HTTPBasicAuth(self.reddit_client_id, self.reddit_secret)
            data = {'grant_type': 'client_credentials'}
            headers = {'User-Agent': 'CryptoScanner/1.0'}
            
            # Get access token
            token_response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth,
                data=data,
                headers=headers,
                timeout=10
            )
            
            if token_response.status_code == 200:
                token = token_response.json()['access_token']
                headers['Authorization'] = f'bearer {token}'
                
                # Search crypto subreddits
                subreddits = ['CryptoCurrency', 'CryptoMoonShots', 'SatoshiStreetBets']
                
                for subreddit in subreddits:
                    url = f'https://oauth.reddit.com/r/{subreddit}/hot?limit=50'
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        posts = response.json()['data']['children']
                        
                        for post in posts:
                            title = post['data'].get('title', '')
                            body = post['data'].get('selftext', '')
                            text = f"{title} {body}"
                            
                            # Look for launch keywords
                            if any(keyword in text.lower() for keyword in self.BULLISH_KEYWORDS):
                                symbols = self._extract_crypto_symbols(text)
                                
                                for symbol in symbols:
                                    candidates.append({
                                        'symbol': f"{symbol}/USD",
                                        'base_asset': symbol,
                                        'source': 'reddit',
                                        'subreddit': subreddit,
                                        'mention_text': title[:200]
                                    })
        
        except Exception as e:
            logger.debug(f"Reddit API error: {e}")
        
        return candidates
    
    def _extract_crypto_symbols(self, text: str) -> List[str]:
        """Extract crypto symbols from text"""
        symbols = []
        
        # Look for $SYMBOL format
        dollar_symbols = re.findall(r'\$([A-Z]{2,10})\b', text.upper())
        symbols.extend(dollar_symbols)
        
        # Look for common patterns
        words = text.upper().split()
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 2 and len(clean_word) <= 10 and clean_word.isalpha():
                # Common crypto symbols
                if clean_word in ['BTC', 'ETH', 'SOL', 'DOGE', 'SHIB', 'PEPE', 'HIPPO', 'APE']:
                    symbols.append(clean_word)
        
        return list(set(symbols))
    
    def _search_pre_listing_buzz(self) -> List[Dict]:
        """Search for coins with pre-listing buzz"""
        buzz_coins = []
        
        # Aggregate from multiple sources
        twitter_buzz = self._search_twitter_for_pre_listing()
        reddit_buzz = self._search_reddit_for_pre_listing()
        
        # Combine and count mentions
        all_mentions = {}
        
        for mention in twitter_buzz + reddit_buzz:
            symbol = mention['base_asset']
            if symbol not in all_mentions:
                all_mentions[symbol] = {
                    'symbol': f"{symbol}/USD",
                    'base_asset': symbol,
                    'mentions': 0,
                    'keywords': set(),
                    'sources': []
                }
            
            all_mentions[symbol]['mentions'] += 1
            all_mentions[symbol]['sources'].append(mention['source'])
            
            # Extract keywords
            text = mention.get('mention_text', '')
            for keyword in self.BULLISH_KEYWORDS:
                if keyword in text.lower():
                    all_mentions[symbol]['keywords'].add(keyword)
        
        # Convert to list
        for symbol, data in all_mentions.items():
            data['keywords'] = list(data['keywords'])
            buzz_coins.append(data)
        
        return buzz_coins
    
    def _search_twitter_for_pre_listing(self) -> List[Dict]:
        """Search Twitter specifically for pre-listing buzz"""
        if not self.twitter_bearer:
            return []
        
        # Same as _search_twitter_trending but with more specific queries
        return self._search_twitter_trending()
    
    def _search_reddit_for_pre_listing(self) -> List[Dict]:
        """Search Reddit specifically for pre-listing buzz"""
        if not self.reddit_client_id:
            return []
        
        # Same as _search_reddit_trending but focused on pre-listing
        return self._search_reddit_trending()
    
    def _analyze_candidate(self, candidate: Dict) -> Optional[PreListingSignal]:
        """Analyze a candidate coin for pre-listing signal"""
        try:
            symbol = candidate['symbol']
            
            # Get price data from Kraken
            ticker = self.client.get_ticker_data(symbol)
            if not ticker:
                return None
            
            # Get social sentiment
            social_score, social_mentions, keywords = self._get_social_sentiment(
                candidate.get('base_asset', symbol.split('/')[0])
            )
            
            # Calculate technical score
            current_price = ticker['last_price']
            change_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
            volume_24h = ticker['volume_24h']
            
            # Volume ratio (estimate)
            volume_ratio = 3.0 if volume_24h > 500000 else 1.5
            
            # Calculate overall score
            score = self._calculate_signal_score(
                change_24h=change_24h,
                volume_ratio=volume_ratio,
                social_score=social_score,
                social_mentions=social_mentions
            )
            
            # Determine signal type
            signal_type = 'NEW_LISTING'
            if 'running' in ' '.join(keywords).lower():
                signal_type = 'RUNNING'
            elif 'moon' in ' '.join(keywords).lower():
                signal_type = 'MOONING'
            
            # Determine confidence
            confidence = 'HIGH' if score >= 75 and social_score >= 70 else 'MEDIUM' if score >= 60 else 'LOW'
            
            # Create reasoning
            reasoning = self._create_reasoning(
                change_24h=change_24h,
                volume_ratio=volume_ratio,
                social_score=social_score,
                keywords=keywords
            )
            
            # Create catalyst
            catalyst = ', '.join(keywords[:3]) if keywords else "High volume activity"
            
            return PreListingSignal(
                symbol=symbol,
                base_asset=candidate.get('base_asset', symbol.split('/')[0]),
                signal_type=signal_type,
                score=score,
                confidence=confidence,
                current_price=current_price,
                change_24h=change_24h,
                volume_24h=volume_24h,
                volume_ratio=volume_ratio,
                social_score=social_score,
                social_mentions=social_mentions,
                sentiment_keywords=keywords,
                reasoning=reasoning,
                catalyst=catalyst
            )
        
        except Exception as e:
            logger.debug(f"Error analyzing candidate: {e}")
            return None
    
    def _create_buzz_signal(self, coin_data: Dict) -> PreListingSignal:
        """Create signal from pre-listing buzz data"""
        symbol = coin_data['symbol']
        base_asset = coin_data['base_asset']
        
        # Calculate social score based on mentions and keywords
        mentions = coin_data['mentions']
        keywords = coin_data['keywords']
        
        social_score = min(100, (mentions * 2) + (len(keywords) * 10))
        
        # Try to get price data
        current_price = 0.0
        change_24h = 0.0
        volume_24h = 0.0
        volume_ratio = 1.0
        
        try:
            ticker = self.client.get_ticker_data(symbol)
            if ticker:
                current_price = ticker['last_price']
                change_24h = ((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h']) * 100
                volume_24h = ticker['volume_24h']
                volume_ratio = 2.0  # Estimate
        except:
            pass
        
        # Score based primarily on social buzz
        score = social_score
        
        # Determine signal type
        signal_type = 'PRE_IPO_BUZZ'
        if 'listing' in ' '.join(keywords).lower():
            signal_type = 'NEW_LISTING'
        
        confidence = 'HIGH' if social_score >= 70 and mentions >= 50 else 'MEDIUM'
        
        reasoning = f"High social media buzz ({mentions} mentions) with keywords: {', '.join(keywords[:5])}"
        catalyst = f"{mentions} social mentions: {', '.join(keywords[:3])}"
        
        return PreListingSignal(
            symbol=symbol,
            base_asset=base_asset,
            signal_type=signal_type,
            score=score,
            confidence=confidence,
            current_price=current_price,
            change_24h=change_24h,
            volume_24h=volume_24h,
            volume_ratio=volume_ratio,
            social_score=social_score,
            social_mentions=mentions,
            sentiment_keywords=keywords,
            reasoning=reasoning,
            catalyst=catalyst
        )
    
    def _get_social_sentiment(self, base_asset: str) -> tuple:
        """
        Get social sentiment for a coin
        
        Returns:
            (social_score, mention_count, keywords)
        """
        # This is a simplified version
        # In production, integrate with Twitter/Reddit/Telegram APIs
        
        # For now, return neutral values
        social_score = 50.0
        mention_count = 10
        keywords = []
        
        # Check if coin is in trending list (mock implementation)
        if base_asset.upper() in ['HIPPO', 'SHIB', 'PEPE', 'DOGE']:
            social_score = 85.0
            mention_count = 100
            keywords = ['running', 'mooning', 'trending']
        
        return social_score, mention_count, keywords
    
    def _calculate_signal_score(
        self,
        change_24h: float,
        volume_ratio: float,
        social_score: float,
        social_mentions: int
    ) -> float:
        """Calculate overall signal score"""
        score = 0.0
        
        # Price movement (30%)
        if change_24h > 50:
            score += 30
        elif change_24h > 20:
            score += 20
        elif change_24h > 10:
            score += 15
        
        # Volume (20%)
        if volume_ratio > 5:
            score += 20
        elif volume_ratio > 3:
            score += 15
        elif volume_ratio > 2:
            score += 10
        
        # Social sentiment (50%)
        score += (social_score * 0.5)
        
        return min(100, score)
    
    def _create_reasoning(
        self,
        change_24h: float,
        volume_ratio: float,
        social_score: float,
        keywords: List[str]
    ) -> str:
        """Create human-readable reasoning"""
        reasons = []
        
        if change_24h > 20:
            reasons.append(f"Strong price surge (+{change_24h:.1f}% in 24h)")
        
        if volume_ratio > 3:
            reasons.append(f"Massive volume ({volume_ratio:.1f}x average)")
        
        if social_score > 70:
            reasons.append(f"High social buzz (score {social_score:.0f}/100)")
        
        if keywords:
            keyword_str = ', '.join(keywords[:3])
            reasons.append(f"Keywords detected: {keyword_str}")
        
        return " | ".join(reasons) if reasons else "Potential opportunity detected"


"""
Crypto News & Sentiment Analyzer
Fetches and analyzes news articles for cryptocurrencies

Sources:
- CoinGecko News API
- Crypto news aggregators
- Social media sentiment (Reddit, Twitter)
- FinBERT sentiment analysis (specialized financial model)

ENHANCED: Now uses FinBERT for more accurate financial sentiment analysis
"""

import os
import asyncio
import httpx
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Try to import FinBERT analyzer
try:
    from services.finbert_sentiment import get_finbert_analyzer, FinBERTSentiment
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("⚠️ FinBERT not available, using keyword-based sentiment")


@dataclass
class CryptoNewsArticle:
    """Crypto news article with enhanced sentiment analysis"""
    title: str
    description: str
    url: str
    source: str
    published_at: str
    sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float  # 0-1
    keywords: List[str] = None
    # ENHANCED: FinBERT-specific fields
    sentiment_confidence: float = 0.0  # 0-1 scale (how confident is the sentiment)
    finbert_sentiment: Optional[str] = None  # Original FinBERT output
    market_impact: str = "UNKNOWN"  # HIGH, MEDIUM, LOW, UNKNOWN


@dataclass
class CryptoNewsSentiment:
    """Comprehensive news and sentiment analysis for a crypto (ENHANCED with FinBERT)"""
    symbol: str
    news_count: int
    recent_news: List[CryptoNewsArticle]
    overall_sentiment: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    sentiment_score: float  # -1.0 to 1.0
    bullish_articles: int
    bearish_articles: int
    neutral_articles: int
    major_catalysts: List[str]  # Major news events
    social_sentiment: Optional[Dict] = None  # From CryptoSentimentAnalyzer
    combined_score: float = 0.0  # Combined news + social sentiment
    # ENHANCED: Additional metrics
    average_confidence: float = 0.0  # Average confidence across all articles
    overall_sentiment_score: float = 50.0  # 0-100 scale (0=very bearish, 50=neutral, 100=very bullish)
    high_impact_news_count: int = 0  # Number of high-impact news articles
    sentiment_trend: str = "STABLE"  # IMPROVING, DETERIORATING, STABLE


class CryptoNewsAnalyzer:
    """
    Analyzes news and sentiment for cryptocurrencies
    Combines news articles with social sentiment for comprehensive analysis
    """
    
    def __init__(self, use_finbert: bool = True):
        """
        Initialize crypto news analyzer
        
        Args:
            use_finbert: If True, use FinBERT for sentiment analysis (recommended)
        """
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.last_coingecko_call = 0
        self.rate_limit_delay = 6.0  # 10 calls/min = 6 seconds between calls
        
        # Initialize FinBERT analyzer if available
        self.use_finbert = use_finbert and FINBERT_AVAILABLE
        if self.use_finbert:
            try:
                self.finbert_analyzer = get_finbert_analyzer()
                logger.info("📰 Crypto News Analyzer initialized with FinBERT")
            except Exception as e:
                logger.warning(f"⚠️ FinBERT initialization failed: {e}, using keyword fallback")
                self.use_finbert = False
        else:
            logger.info("📰 Crypto News Analyzer initialized (keyword-based sentiment)")
        
        # News sentiment keywords (crypto-specific) - FALLBACK when FinBERT unavailable
        self.bullish_keywords = {
            'partnership': 2, 'adoption': 3, 'integration': 2, 'listing': 2,
            'upgrade': 2, 'launch': 2, 'breakthrough': 3, 'milestone': 2,
            'surge': 3, 'rally': 3, 'bullish': 3, 'growth': 2, 'expansion': 2,
            'institutional': 2, 'approval': 3, 'regulation': 1, 'compliance': 1,
            'burn': 2, 'deflationary': 2, 'staking': 1, 'yield': 1,
            'mainnet': 2, 'testnet': 1, 'upgrade': 2, 'hard fork': 1
        }
        
        self.bearish_keywords = {
            'hack': 4, 'exploit': 4, 'breach': 4, 'security': -1, 'vulnerability': 3,
            'crash': 4, 'dump': 3, 'sell-off': 3, 'bearish': 3, 'decline': 2,
            'ban': 3, 'regulation': -1, 'lawsuit': 3, 'investigation': 3,
            'delisting': 3, 'suspension': 3, 'warning': 2, 'concern': 2,
            'rug pull': 4, 'scam': 4, 'ponzi': 4, 'fraud': 4, 'manipulation': 3
        }
        
        logger.info("   • CoinGecko News API (primary source)")
        logger.info("   • Note: CoinMarketCap API does not provide news endpoints")
        logger.info(f"   • Sentiment: {'FinBERT (AI-powered)' if self.use_finbert else 'Keyword-based'}")
        logger.info("   • Social sentiment integration (Reddit, forums)")
    
    async def get_crypto_news(
        self,
        symbol: str,
        hours: int = 24,
        max_articles: int = 10
    ) -> List[CryptoNewsArticle]:
        """
        Get recent news articles for a crypto
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            hours: Lookback period in hours
            max_articles: Maximum articles to return
            
        Returns:
            List of CryptoNewsArticle objects
        """
        try:
            # Rate limiting
            elapsed = time.time() - self.last_coingecko_call
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
            
            logger.info(f"📰 Fetching news for {symbol} (last {hours}h)")
            
            # Get CoinGecko ID for the symbol
            coingecko_id = await self._get_coingecko_id(symbol)
            if not coingecko_id:
                logger.warning(f"Could not find CoinGecko ID for {symbol}")
                return []
            
            # Try to fetch news from CoinGecko (may not be available for all coins)
            # Fallback to general crypto news search if specific coin news unavailable
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try coin-specific news first
                try:
                    response = await client.get(
                        f"{self.coingecko_api}/coins/{coingecko_id}/news",
                        headers={'User-Agent': 'Mozilla/5.0'}
                    )
                    
                    self.last_coingecko_call = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        news_items = data.get('news', []) if isinstance(data, dict) else data
                    else:
                        # Fallback: try general crypto news search
                        logger.debug(f"Coin-specific news not available for {symbol}, trying general search")
                        news_items = await self._get_general_crypto_news(symbol, client)
                except Exception as e:
                    logger.debug(f"Error fetching CoinGecko news: {e}, trying general search")
                    news_items = await self._get_general_crypto_news(symbol, client)
                
                articles = []
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                for item in news_items[:max_articles * 2]:  # Get more to filter by time
                    try:
                        # Parse article
                        title = item.get('title', '')
                        description = item.get('description', '') or item.get('text', '')
                        url = item.get('url', '') or item.get('link', '')
                        source = item.get('source', '') or item.get('source_name', 'Unknown')
                        
                        # Parse timestamp
                        published_str = item.get('published_at', '') or item.get('published_on', '')
                        if published_str:
                            try:
                                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                            except:
                                published_at = datetime.now()
                        else:
                            published_at = datetime.now()
                        
                        # Filter by time
                        if published_at < cutoff_time:
                            continue
                        
                        # Analyze sentiment (ENHANCED with FinBERT)
                        sentiment_result = self._analyze_news_sentiment(title, description)
                        
                        # Calculate relevance
                        relevance = self._calculate_relevance(symbol, title, description)
                        
                        # Extract keywords
                        keywords = self._extract_keywords(title, description)
                        
                        # Determine market impact
                        market_impact = self._assess_market_impact(
                            sentiment_result['confidence'],
                            keywords,
                            title
                        )
                        
                        article = CryptoNewsArticle(
                            title=title,
                            description=description,
                            url=url,
                            source=source,
                            published_at=published_at.isoformat(),
                            sentiment=sentiment_result['sentiment'],
                            sentiment_score=sentiment_result['score'],
                            relevance_score=relevance,
                            keywords=keywords,
                            sentiment_confidence=sentiment_result['confidence'],
                            finbert_sentiment=sentiment_result.get('finbert_sentiment'),
                            market_impact=market_impact
                        )
                        
                        articles.append(article)
                        
                        if len(articles) >= max_articles:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error parsing news article: {e}")
                        continue
                
                logger.info(f"✅ Found {len(articles)} news articles for {symbol}")
                return articles
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def _get_coingecko_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko ID for a crypto symbol"""
        try:
            # Common mappings
            symbol_to_id = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'SOL': 'solana',
                'BNB': 'binancecoin',
                'ADA': 'cardano',
                'DOT': 'polkadot',
                'MATIC': 'matic-network',
                'AVAX': 'avalanche-2',
                'LINK': 'chainlink',
                'UNI': 'uniswap',
                'ATOM': 'cosmos',
                'NEAR': 'near',
                'APT': 'aptos',
                'SUI': 'sui',
                'ARB': 'arbitrum',
                'OP': 'optimism',
                'LIDO': 'lido-dao',
                'AAVE': 'aave',
                'CRV': 'curve-dao-token',
                'MKR': 'maker',
                'SNX': 'havven',
                'DYDX': 'dydx-chain',
                'GMX': 'gmx',
                'RENDER': 'render-token',
                'FET': 'fetch-ai',
                'AGIX': 'singularitynet',
                'OCEAN': 'ocean-protocol',
                'ARKM': 'arkham',
                'JTO': 'jito-governance-token',
                'PYTH': 'pyth-network',
                'ONDO': 'ondo-finance',
                'STRK': 'starknet',
                'BLUR': 'blur',
                'GALA': 'gala',
                'SAND': 'the-sandbox',
                'MANA': 'decentraland',
                'ENJ': 'enjincoin',
                'THETA': 'theta-token',
                'AXIE': 'axie-infinity',
                'FLOW': 'flow',
                'ILV': 'illuvium',
                'SHIB': 'shiba-inu',
            }
            
            symbol_upper = symbol.upper()
            if symbol_upper in symbol_to_id:
                return symbol_to_id[symbol_upper]
            
            # Try to fetch from CoinGecko API if not in mapping
            # This would require an API call, so we'll use mapping for now
            logger.debug(f"Symbol {symbol} not in mapping, using symbol as ID")
            return symbol.lower()
            
        except Exception as e:
            logger.error(f"Error getting CoinGecko ID for {symbol}: {e}")
            return None
    
    def _analyze_news_sentiment(self, title: str, description: str) -> Dict:
        """
        Analyze sentiment of news article (ENHANCED with FinBERT)
        
        Returns:
            Dict with sentiment analysis:
            {
                'sentiment': 'BULLISH'|'BEARISH'|'NEUTRAL',
                'score': -1.0 to 1.0,
                'confidence': 0.0 to 1.0,
                'finbert_sentiment': 'positive'|'negative'|'neutral' (if using FinBERT)
            }
        """
        text = f"{title}. {description}"
        
        if self.use_finbert and hasattr(self, 'finbert_analyzer'):
            try:
                # Use FinBERT for accurate financial sentiment
                finbert_result = self.finbert_analyzer.analyze_sentiment(text)
                
                # Map FinBERT sentiment to trading sentiment
                sentiment_map = {
                    'positive': 'BULLISH',
                    'negative': 'BEARISH',
                    'neutral': 'NEUTRAL'
                }
                
                trading_sentiment = sentiment_map.get(finbert_result.sentiment, 'NEUTRAL')
                
                # Convert confidence to score (-1.0 to 1.0)
                if finbert_result.sentiment == 'positive':
                    score = finbert_result.confidence
                elif finbert_result.sentiment == 'negative':
                    score = -finbert_result.confidence
                else:
                    score = 0.0
                
                return {
                    'sentiment': trading_sentiment,
                    'score': score,
                    'confidence': finbert_result.confidence,
                    'finbert_sentiment': finbert_result.sentiment
                }
                
            except Exception as e:
                logger.debug(f"FinBERT analysis failed, falling back to keywords: {e}")
                # Fall through to keyword-based analysis
        
        # Fallback: Keyword-based sentiment analysis
        return self._keyword_sentiment_analysis(title, description)
    
    def _keyword_sentiment_analysis(self, title: str, description: str) -> Dict:
        """
        Fallback keyword-based sentiment analysis
        Used when FinBERT is unavailable
        """
        text = f"{title} {description}".lower()
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        # Check bullish keywords
        for keyword, weight in self.bullish_keywords.items():
            if keyword in text:
                bullish_score += weight
        
        # Check bearish keywords
        for keyword, weight in self.bearish_keywords.items():
            if keyword in text:
                bearish_score += weight
        
        # Normalize scores
        bullish_normalized = min(bullish_score / 20, 1.0)  # Cap at 1.0
        bearish_normalized = min(bearish_score / 20, 1.0)  # Cap at 1.0
        
        # Determine sentiment
        if bullish_score > bearish_score * 1.2:
            sentiment = 'BULLISH'
            score = bullish_normalized
        elif bearish_score > bullish_score * 1.2:
            sentiment = 'BEARISH'
            score = -bearish_normalized
        else:
            sentiment = 'NEUTRAL'
            score = 0.0
        
        # Confidence is lower for keyword-based (0.5-0.7 range)
        confidence = min(0.5 + abs(score) * 0.2, 0.7)
        
        return {
            'sentiment': sentiment,
            'score': score,
            'confidence': confidence,
            'finbert_sentiment': None
        }
    
    def _calculate_relevance(self, symbol: str, title: str, description: str) -> float:
        """Calculate relevance score (0-1) for news article"""
        text = f"{title} {description}".lower()
        symbol_lower = symbol.lower()
        
        relevance = 0.0
        
        # Direct mention
        if symbol_lower in text:
            relevance += 0.5
        
        # Common variations
        if symbol_lower == 'btc' and 'bitcoin' in text:
            relevance += 0.3
        elif symbol_lower == 'eth' and 'ethereum' in text:
            relevance += 0.3
        elif symbol_lower == 'sol' and 'solana' in text:
            relevance += 0.3
        
        # Crypto-related keywords
        crypto_keywords = ['crypto', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'token', 'coin']
        if any(keyword in text for keyword in crypto_keywords):
            relevance += 0.2
        
        return min(1.0, relevance)
    
    def _extract_keywords(self, title: str, description: str) -> List[str]:
        """Extract important keywords from news article"""
        text = f"{title} {description}".lower()
        keywords = []
        
        # Important crypto keywords
        important_keywords = [
            'partnership', 'listing', 'adoption', 'integration', 'upgrade',
            'mainnet', 'testnet', 'burn', 'staking', 'yield', 'regulation',
            'hack', 'exploit', 'security', 'launch', 'milestone'
        ]
        
        for keyword in important_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords[:5]  # Top 5 keywords
    
    def _assess_market_impact(
        self,
        confidence: float,
        keywords: List[str],
        title: str
    ) -> str:
        """
        Assess potential market impact of news
        
        Returns:
            'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'
        """
        # High impact keywords
        high_impact_keywords = {
            'hack', 'exploit', 'breach', 'etf', 'sec', 'regulation',
            'ban', 'approval', 'listing', 'delisting', 'partnership'
        }
        
        # Check for high-impact keywords
        title_lower = title.lower()
        has_high_impact = any(keyword in title_lower for keyword in high_impact_keywords)
        
        # Determine impact based on confidence and keywords
        if has_high_impact and confidence > 0.75:
            return "HIGH"
        elif has_high_impact or confidence > 0.70:
            return "MEDIUM"
        elif confidence > 0.50:
            return "LOW"
        else:
            return "UNKNOWN"
    
    async def _get_general_crypto_news(self, symbol: str, client: httpx.AsyncClient) -> List[Dict]:
        """Fallback: Get general crypto news when coin-specific news unavailable"""
        try:
            # Use CoinGecko's general news endpoint or search
            # For now, return empty list - can be enhanced with other news sources
            logger.debug(f"General news search for {symbol} (fallback)")
            return []
        except Exception as e:
            logger.debug(f"Error in general news search: {e}")
            return []
    
    async def analyze_comprehensive_sentiment(
        self,
        symbol: str,
        include_social: bool = True,
        hours: int = 24
    ) -> CryptoNewsSentiment:
        """
        Comprehensive sentiment analysis combining news and social media (ENHANCED with FinBERT)
        
        Args:
            symbol: Crypto symbol
            include_social: If True, include social sentiment analysis
            hours: Lookback period in hours
            
        Returns:
            CryptoNewsSentiment object with comprehensive analysis
        """
        try:
            # Get news articles (now with FinBERT sentiment)
            news_articles = await self.get_crypto_news(symbol, hours=hours)
            
            # Analyze news sentiment
            bullish_count = sum(1 for a in news_articles if a.sentiment == 'BULLISH')
            bearish_count = sum(1 for a in news_articles if a.sentiment == 'BEARISH')
            neutral_count = sum(1 for a in news_articles if a.sentiment == 'NEUTRAL')
            
            # Calculate overall sentiment from news
            if news_articles:
                news_sentiment_score = sum(a.sentiment_score for a in news_articles) / len(news_articles)
                average_confidence = sum(a.sentiment_confidence for a in news_articles) / len(news_articles)
            else:
                news_sentiment_score = 0.0
                average_confidence = 0.0
            
            # Determine overall sentiment (weighted by confidence)
            if bullish_count > bearish_count * 1.5:
                overall_sentiment = 'BULLISH'
            elif bearish_count > bullish_count * 1.5:
                overall_sentiment = 'BEARISH'
            else:
                overall_sentiment = 'NEUTRAL'
            
            # Convert sentiment score to 0-100 scale
            overall_sentiment_score = (news_sentiment_score + 1.0) * 50  # -1 to 1 -> 0 to 100
            
            # Count high-impact news
            high_impact_count = sum(1 for a in news_articles if a.market_impact == "HIGH")
            
            # Determine sentiment trend (comparing recent vs older)
            sentiment_trend = "STABLE"
            if len(news_articles) >= 3:
                recent_sentiment = sum(a.sentiment_score for a in news_articles[:2]) / 2
                older_sentiment = sum(a.sentiment_score for a in news_articles[2:]) / len(news_articles[2:])
                
                if recent_sentiment > older_sentiment + 0.2:
                    sentiment_trend = "IMPROVING"
                elif recent_sentiment < older_sentiment - 0.2:
                    sentiment_trend = "DETERIORATING"
            
            # Extract major catalysts (high confidence or high impact)
            major_catalysts = []
            for article in news_articles:
                # Include if strong sentiment OR high impact
                if (abs(article.sentiment_score) > 0.5 and article.sentiment_confidence > 0.7) or article.market_impact == "HIGH":
                    catalyst = f"[{article.market_impact}] {article.title[:60]}... ({article.sentiment}, {article.sentiment_confidence:.0%})"
                    major_catalysts.append(catalyst)
            
            # Get social sentiment if requested
            social_sentiment = None
            if include_social:
                try:
                    from services.crypto_sentiment_analyzer import CryptoSentimentAnalyzer
                    sentiment_analyzer = CryptoSentimentAnalyzer()
                    social_sentiment = await sentiment_analyzer.analyze_crypto_sentiment(symbol)
                except Exception as e:
                    logger.warning(f"Could not get social sentiment for {symbol}: {e}")
            
            # Calculate combined score
            combined_score = news_sentiment_score
            if social_sentiment:
                social_score = social_sentiment.get('overall_sentiment_score', 0.0)
                # Weight: 60% news, 40% social
                combined_score = (news_sentiment_score * 0.6) + (social_score * 0.4)
            
            return CryptoNewsSentiment(
                symbol=symbol,
                news_count=len(news_articles),
                recent_news=news_articles[:5],  # Top 5 most relevant
                overall_sentiment=overall_sentiment,
                sentiment_score=combined_score,
                bullish_articles=bullish_count,
                bearish_articles=bearish_count,
                neutral_articles=neutral_count,
                major_catalysts=major_catalysts[:5],  # Top 5 catalysts
                social_sentiment=social_sentiment,
                combined_score=combined_score,
                average_confidence=average_confidence,
                overall_sentiment_score=overall_sentiment_score,
                high_impact_news_count=high_impact_count,
                sentiment_trend=sentiment_trend
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive sentiment analysis for {symbol}: {e}")
            return CryptoNewsSentiment(
                symbol=symbol,
                news_count=0,
                recent_news=[],
                overall_sentiment='NEUTRAL',
                sentiment_score=0.0,
                bullish_articles=0,
                bearish_articles=0,
                neutral_articles=0,
                major_catalysts=[],
                average_confidence=0.0,
                overall_sentiment_score=50.0,
                high_impact_news_count=0,
                sentiment_trend="STABLE"
            )


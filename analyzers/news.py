"""News and sentiment analysis."""

import logging
from datetime import datetime
from typing import Dict, List, Tuple
import yfinance as yf

# Import from utils module
from utils.caching import get_cached_news

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    """Fetch and analyze news and catalysts"""
    
    @staticmethod
    def get_stock_news(ticker: str, max_articles: int = 5) -> List[Dict]:
        """Fetch recent news for a stock with enhanced error handling"""
        try:
            logger.info(f"Getting news for {ticker} (max {max_articles} articles)")
            
            # Use cached news for better performance
            news = get_cached_news(ticker)
            
            if not news:
                logger.warning(f"No news data returned from cache for {ticker}")
                # Try direct fetch as fallback
                try:
                    stock = yf.Ticker(ticker)
                    news = stock.news
                    logger.info(f"Direct fetch returned {len(news) if news else 0} articles")
                except Exception as direct_error:
                    logger.error(f"Direct fetch also failed for {ticker}: {direct_error}")
                    return []
            
            articles = []
            for idx, item in enumerate(news[:max_articles]):
                try:
                    # Handle both old and new Yahoo Finance API structures
                    content = item.get('content', item)  # New API has nested content
                    
                    # Extract title from nested content or direct item
                    title = content.get('title', item.get('title', 'No title available'))
                    
                    # Extract publisher information
                    provider = content.get('provider', {})
                    publisher = provider.get('displayName', item.get('publisher', 'Unknown Publisher'))
                    
                    # Extract link - try multiple possible locations
                    link = ''
                    if content.get('canonicalUrl', {}).get('url'):
                        link = content['canonicalUrl']['url']
                    elif content.get('clickThroughUrl', {}).get('url'):
                        link = content['clickThroughUrl']['url']
                    elif item.get('link'):
                        link = item['link']
                    
                    # Handle timestamp conversion more safely
                    published_time = 'Unknown'
                    pub_date = content.get('pubDate', content.get('displayTime', item.get('providerPublishTime')))
                    
                    if pub_date:
                        try:
                            # Handle different timestamp formats
                            if isinstance(pub_date, str):
                                # Parse ISO format dates
                                if 'T' in pub_date and 'Z' in pub_date:
                                    dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                                    published_time = dt.strftime('%Y-%m-%d %H:%M')
                                else:
                                    published_time = pub_date
                            elif isinstance(pub_date, (int, float)):
                                # Handle Unix timestamps
                                published_time = datetime.fromtimestamp(pub_date).strftime('%Y-%m-%d %H:%M')
                            else:
                                published_time = 'Recent'
                        except (ValueError, TypeError, OSError) as time_error:
                            logger.warning(f"Could not parse timestamp for article {idx}: {time_error}")
                            published_time = 'Recent'
                    
                    # Extract summary/description
                    summary = content.get('summary', content.get('description', ''))
                    if summary and len(summary) > 200:
                        summary = summary[:200] + '...'
                    
                    article = {
                        'title': title,
                        'publisher': publisher,
                        'link': link,
                        'published': published_time,
                        'type': content.get('contentType', item.get('type', 'NEWS')),
                        'summary': summary
                    }
                    articles.append(article)
                    logger.debug(f"Processed article {idx + 1}: {article['title'][:50]}...")
                    
                except Exception as article_error:
                    logger.error(f"Error processing article {idx} for {ticker}: {article_error}")
                    continue
            
            logger.info(f"Successfully processed {len(articles)} articles for {ticker}")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    @staticmethod
    def analyze_sentiment(news_articles: List[Dict]) -> Tuple[float, List[str]]:
        """Enhanced sentiment analysis from news titles and summaries"""
        # Expanded sentiment word lists with weights
        positive_words = {
            'surge': 2, 'jump': 2, 'gain': 2, 'profit': 2, 'beat': 3, 'upgrade': 2, 'buy': 2, 'strong': 2, 
            'growth': 2, 'bullish': 3, 'rally': 3, 'soar': 3, 'win': 2, 'success': 2, 'breakthrough': 3,
            'exceeds': 2, 'outperforms': 2, 'rises': 2, 'climbs': 2, 'advances': 2, 'boosts': 2, 'increases': 2,
            'positive': 2, 'optimistic': 2, 'confident': 2, 'expansion': 2, 'record': 2, 'milestone': 2
        }
        
        negative_words = {
            'fall': 2, 'drop': 2, 'loss': 3, 'miss': 3, 'downgrade': 2, 'sell': 2, 'weak': 2, 'decline': 2, 
            'bearish': 3, 'crash': 4, 'plunge': 3, 'concern': 2, 'risk': 2, 'lawsuit': 3, 'investigation': 3,
            'disappoints': 2, 'underperforms': 2, 'declines': 2, 'falls': 2, 'drops': 2, 'sinks': 3, 'tumbles': 3,
            'negative': 2, 'pessimistic': 2, 'worried': 2, 'contraction': 2, 'cut': 2, 'reduce': 2, 'warning': 2
        }
        
        sentiment_score = 0
        signals = []
        total_articles = len(news_articles)
        
        if not news_articles:
            return 0.0, ["No news articles to analyze"]
        
        # Filter out None articles
        news_articles = [a for a in news_articles if a is not None]
        
        if not news_articles:
            return 0.0, ["No valid news articles to analyze"]
        
        for article in news_articles:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            combined_text = f"{title} {summary}"
            
            # Calculate positive and negative scores with weights
            pos_score = sum(weight for word, weight in positive_words.items() if word in combined_text)
            neg_score = sum(weight for word, weight in negative_words.items() if word in combined_text)
            
            # Determine sentiment for this article
            if pos_score > neg_score:
                sentiment_score += 1
                sentiment_icon = "✅"
                sentiment_label = "Positive"
            elif neg_score > pos_score:
                sentiment_score -= 1
                sentiment_icon = "⚠️"
                sentiment_label = "Negative"
            else:
                sentiment_icon = "ℹ️"
                sentiment_label = "Neutral"
            
            # Create detailed signal with sentiment strength
            strength = abs(pos_score - neg_score)
            if strength > 3:
                strength_indicator = "🔥 Strong"
            elif strength > 1:
                strength_indicator = "📈 Moderate"
            else:
                strength_indicator = "📊 Weak"
            
            signals.append(f"{sentiment_icon} {sentiment_label} ({strength_indicator}): {article['title'][:60]}...")
        
        # Normalize to -1 to 1 scale
        sentiment_score = sentiment_score / total_articles if total_articles > 0 else 0
        
        logger.info(f"Sentiment analysis complete: {sentiment_score:.3f} from {total_articles} articles")
        
        return sentiment_score, signals
    
    @staticmethod
    def get_catalysts(ticker: str) -> List[Dict]:
        """Identify upcoming catalysts"""
        catalysts = []
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Earnings date
            earnings_date = info.get('earningsDate')
            if earnings_date:
                if isinstance(earnings_date, list) and earnings_date:
                    earnings_date = earnings_date[0]
                
                if isinstance(earnings_date, (int, float)):
                    earnings_dt = datetime.fromtimestamp(earnings_date)
                    days_away = (earnings_dt - datetime.now()).days
                    
                    if days_away >= 0 and days_away <= 60:
                        impact = "HIGH" if days_away <= 7 else "MEDIUM"
                        catalysts.append({
                            'type': 'Earnings Report',
                            'date': earnings_dt.strftime('%Y-%m-%d'),
                            'days_away': days_away,
                            'impact': impact,
                            'description': f'Earnings report in {days_away} days'
                        })
            
            # Ex-dividend date
            ex_div_date = info.get('exDividendDate')
            if ex_div_date:
                if isinstance(ex_div_date, (int, float)):
                    div_dt = datetime.fromtimestamp(ex_div_date)
                    days_away = (div_dt - datetime.now()).days
                    
                    if days_away >= 0 and days_away <= 30:
                        catalysts.append({
                            'type': 'Ex-Dividend',
                            'date': div_dt.strftime('%Y-%m-%d'),
                            'days_away': days_away,
                            'impact': 'LOW',
                            'description': f'Ex-dividend date in {days_away} days'
                        })
            
            # Check for recent analyst upgrades/downgrades in news
            news = NewsAnalyzer.get_stock_news(ticker, max_articles=10)
            # Filter out None articles
            news = [a for a in news if a is not None]
            
            for article in news:
                try:
                    if article and article.get('title'):
                        title_lower = article['title'].lower()
                        if 'upgrade' in title_lower or 'raises price target' in title_lower:
                            catalysts.append({
                                'type': 'Analyst Upgrade',
                                'date': article.get('published', 'Unknown'),
                                'days_away': 0,
                                'impact': 'MEDIUM',
                                'description': article['title'][:80]
                            })
                        elif 'downgrade' in title_lower or 'lowers price target' in title_lower:
                            catalysts.append({
                                'type': 'Analyst Downgrade',
                                'date': article.get('published', 'Unknown'),
                                'days_away': 0,
                                'impact': 'MEDIUM',
                                'description': article['title'][:80]
                            })
                except Exception as e:
                    logger.debug(f"Error processing catalyst article: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error getting catalysts for {ticker}: {e}")
        
        return catalysts

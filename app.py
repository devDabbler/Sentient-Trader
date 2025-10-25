import streamlit as st
import io
import requests
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yfinance as yf
from enum import Enum
import numpy as np
import os
import sys
from io import TextIOWrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import integration clients from new structure
from src.integrations.tradier_client import TradierClient, validate_tradier_connection

# Import service modules from new structure
from services.llm_strategy_analyzer import LLMStrategyAnalyzer, StrategyAnalysis, extract_bot_config_from_screenshot, create_strategy_comparison
from services.penny_stock_analyzer import PennyStockScorer, PennyStockAnalyzer, StockScores
from services.watchlist_manager import WatchlistManager
from services.ticker_manager import TickerManager
from services.top_trades_scanner import TopTradesScanner, TopTrade
from services.ai_confidence_scanner import AIConfidenceScanner, AIConfidenceTrade
from services.alpha_factors import AlphaFactorCalculator
from services.ml_enhanced_scanner import MLEnhancedScanner, MLEnhancedTrade

# Add caching for better performance with new Streamlit features
@st.cache_data(ttl=60)  # Cache for 1 minute for more real-time data
def get_cached_stock_data(ticker: str):
    """Cache stock data to improve performance - refreshes every minute for real-time accuracy"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        info = stock.info
        return hist, info
    except Exception as e:
        logger.error(f"Error fetching cached data for {ticker}: {e}")
        # Return empty DataFrame and empty info to avoid NoneType errors
        return pd.DataFrame(), {}

@st.cache_data(ttl=180)  # Cache for 3 minutes
def get_cached_news(ticker: str):
    """Cache news data to improve performance"""
    try:
        logger.info(f"Fetching news for {ticker}")
        stock = yf.Ticker(ticker)
        news = stock.news
        logger.info(f"Retrieved {len(news) if news else 0} news articles for {ticker}")
        
        if not news:
            logger.warning(f"No news found for {ticker}")
            return []
            
        return news[:10]  # Increased to 10 articles for better coverage
    except Exception as e:
        logger.error(f"Error fetching cached news for {ticker}: {e}")
        return []

# Configure enhanced logging for debugging
def _create_logging_handlers(log_file: str = 'trading_signals.log') -> List[logging.Handler]:
    """Create logging handlers that explicitly use UTF-8 encoding (helps on Windows consoles)."""
    handlers: List[logging.Handler] = []

    try:
        # File handler with UTF-8 encoding
        fh = logging.FileHandler(log_file, encoding='utf-8')
        handlers.append(fh)
    except Exception:
        # Fallback: File handler without explicit encoding
        try:
            handlers.append(logging.FileHandler(log_file))
        except Exception:
            pass

    # Safe console handler which avoids writing to a closed stream
    class SafeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                stream = getattr(self, 'stream', None)

                # If stream is closed, missing, or not writable, fallback to sys.__stdout__
                if stream is None or getattr(stream, 'closed', False) or not hasattr(stream, 'write'):
                    fallback = getattr(sys, '__stdout__', None)
                    if fallback is None:
                        # Nothing to write to; skip emit
                        return
                    # Prefer wrapping fallback.buffer if available
                    fb = getattr(fallback, 'buffer', None)
                    if fb is not None:
                        try:
                            self.stream = TextIOWrapper(fb, encoding='utf-8', errors='replace')
                        except Exception:
                            # Use fallback as-is
                            self.stream = fallback
                    else:
                        self.stream = fallback

                # Ensure UTF-8 wrapper if underlying buffer exists and not already wrapped
                sbuf = getattr(self.stream, 'buffer', None)
                if sbuf is not None and not isinstance(self.stream, TextIOWrapper):
                    try:
                        self.stream = TextIOWrapper(sbuf, encoding='utf-8', errors='replace')
                    except Exception:
                        pass

                super().emit(record)
            except Exception:
                # Swallow to avoid crashing worker threads. Try to write a minimal error to stderr if possible.
                try:
                    err = getattr(sys, '__stderr__', None)
                    if err is not None and hasattr(err, 'write'):
                        eb = getattr(err, 'buffer', None)
                        msg = f"[Logging failure] {record.getMessage()}\n"
                        try:
                            if eb is not None:
                                es = TextIOWrapper(eb, encoding='utf-8', errors='replace')
                                es.write(msg)
                                try:
                                    es.flush()
                                except Exception:
                                    pass
                            else:
                                err.write(msg)
                                try:
                                    err.flush()
                                except Exception:
                                    pass
                        except Exception:
                            # Last resort: ignore
                            pass
                except Exception:
                    pass

    try:
        # prefer wrapping stdout but use SafeStreamHandler to handle closed streams
        sh = SafeStreamHandler(sys.stdout)
        handlers.append(sh)
    except Exception:
        handlers.append(SafeStreamHandler())

    return handlers


# Configure enhanced logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_create_logging_handlers('trading_signals.log')
)
logger = logging.getLogger(__name__)

# Set specific loggers to INFO/WARNING to reduce noise
logging.getLogger('urllib3').setLevel(logging.INFO)
logging.getLogger('requests').setLevel(logging.INFO)
# Reduce verbosity from httpx/httpcore and OpenAI client (they emit lots of debug logs)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.WARNING)
# Suppress excessive yfinance DEBUG logging
logging.getLogger('yfinance').setLevel(logging.WARNING)
# Suppress peewee database DEBUG logging
logging.getLogger('peewee').setLevel(logging.WARNING)

# Compatibility shims for newer Streamlit APIs that may not exist in older versions.
# This avoids AttributeError during script-run and lets the app degrade gracefully.
try:
    # st is imported at module top; ensure we have it
    _st = st
except Exception:
    _st = None

if _st is not None:
    # toggle -> fallback to checkbox
    if not hasattr(st, 'toggle'):
        def _toggle(label, value=False, **kwargs):
            return st.checkbox(label, value)
        setattr(st, 'toggle', _toggle)

    # status -> fallback to a dummy context manager with an update method
    if not hasattr(st, 'status'):
        class _DummyStatus:
            def __init__(self, *a, **k):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def update(self, label=None, state=None):
                # best-effort: show a spinner or simple text
                try:
                    if label:
                        st.write(label)
                except Exception:
                    pass

        def _status(msg, expanded=False):
            return _DummyStatus()

        setattr(st, 'status', _status)

    # data_editor -> fallback to dataframe display and return the passed DataFrame
    if not hasattr(st, 'data_editor'):
        def _data_editor(df, **kwargs):
            try:
                st.dataframe(df)
            except Exception:
                pass
            return df
        setattr(st, 'data_editor', _data_editor)

    # divider -> fallback to markdown horizontal rule
    if not hasattr(st, 'divider'):
        def _divider():
            try:
                st.markdown('---')
            except Exception:
                pass
        setattr(st, 'divider', _divider)

    # fragment decorator -> no-op decorator when missing
    if not hasattr(st, 'fragment'):
        def _fragment(fn):
            return fn
        setattr(st, 'fragment', _fragment)

    # Provide a minimal column_config namespace to avoid attribute errors when building
    # column_config objects; these dummies are ignored by our fallback data_editor.
    if not hasattr(st, 'column_config'):
        class _DummyCol:
            def __init__(self, *a, **k):
                pass

        class _DummyColConfig:
            TextColumn = _DummyCol
            SelectboxColumn = _DummyCol
            NumberColumn = _DummyCol
            DatetimeColumn = _DummyCol

        setattr(st, 'column_config', _DummyColConfig())

class MarketCondition(Enum):
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"

@dataclass
class StrategyRecommendation:
    """Recommendation for a trading strategy"""
    strategy_name: str
    action: str
    confidence: float
    reasoning: str
    risk_level: str
    max_loss: str
    max_gain: str
    best_conditions: List[str]
    experience_level: str
    examples: Optional[List[str]] = None
    notes: Optional[str] = None
    example_trade: Optional[Dict] = None

@dataclass
class StockAnalysis:
    """Complete stock analysis with technicals, news, and catalysts"""
    ticker: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    rsi: float
    macd_signal: str
    trend: str
    support: float
    resistance: float
    iv_rank: float
    iv_percentile: float
    earnings_date: Optional[str]
    earnings_days_away: Optional[int]
    recent_news: List[Dict]
    catalysts: List[Dict]
    sentiment_score: float
    sentiment_signals: List[str]
    confidence_score: float
    recommendation: str
    # Optional advanced indicator context (additive)
    ema8: Optional[float] = None
    ema21: Optional[float] = None
    demarker: Optional[float] = None
    fib_targets: Optional[Dict[str, float]] = None
    ema_power_zone: Optional[bool] = None
    ema_reclaim: Optional[bool] = None

class TechnicalAnalyzer:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            # Ensure numeric dtype
            prices = pd.to_numeric(prices, errors='coerce').dropna()
            # Calculate price changes and force numeric dtype for safe comparisons
            delta = pd.to_numeric(prices.diff(), errors='coerce').fillna(0.0)
            gain = (delta.where(delta > 0.0, 0.0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0.0, 0.0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return round(rsi.iloc[-1], 2)
        except:
            return 50.0
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[str, float]:
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            if current_macd > current_signal:
                return "BULLISH", round(current_macd - current_signal, 2)
            else:
                return "BEARISH", round(current_macd - current_signal, 2)
        except:
            return "NEUTRAL", 0.0
    
    @staticmethod
    def calculate_support_resistance(prices: pd.Series) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        try:
            recent_prices = prices.tail(20)
            support = round(recent_prices.min(), 2)
            resistance = round(recent_prices.max(), 2)
            return support, resistance
        except:
            current = prices.iloc[-1]
            return round(current * 0.95, 2), round(current * 1.05, 2)
    
    @staticmethod
    def calculate_iv_metrics(ticker: str) -> Tuple[float, float]:
        """Calculate IV Rank and IV Percentile (simulated)"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get options data if available
            try:
                options_dates = stock.options
                if options_dates:
                    opt_chain = stock.option_chain(options_dates[0])
                    
                    # Calculate average implied volatility
                    calls_iv = opt_chain.calls['impliedVolatility'].mean()
                    puts_iv = opt_chain.puts['impliedVolatility'].mean()
                    current_iv = (calls_iv + puts_iv) / 2 * 100
                    
                    # Simulate IV rank (in production, use historical IV data)
                    # IV rank = where current IV sits in the range of past year's IV
                    hist = stock.history(period="1y")
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                    
                    # Estimate IV rank (simplified)
                    iv_rank = min(100, max(0, (current_iv / volatility) * 50))
                    iv_percentile = iv_rank  # Simplified
                    
                    return round(iv_rank, 1), round(iv_percentile, 1)
            except:
                pass
            
            # Fallback: estimate from historical volatility
            hist = stock.history(period="1y")
            if not hist.empty:
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                iv_rank = min(100, max(0, volatility))
                return round(iv_rank, 1), round(iv_rank, 1)
            
        except Exception as e:
            logger.error(f"Error calculating IV metrics: {e}")
        
        return 50.0, 50.0

    # --- New advanced indicators ---
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        try:
            return pd.to_numeric(series, errors='coerce').ewm(span=period, adjust=False).mean()
        except Exception:
            return series

    @staticmethod
    def demarker(df: pd.DataFrame, period: int = 14) -> pd.Series:
        try:
            high = pd.to_numeric(df['High'], errors='coerce')
            low = pd.to_numeric(df['Low'], errors='coerce')
            up = (high - high.shift(1)).clip(lower=0.0)
            dn = (low.shift(1) - low).clip(lower=0.0)
            up_sum = up.rolling(window=period, min_periods=period).sum()
            dn_sum = dn.rolling(window=period, min_periods=period).sum()
            denom = (up_sum + dn_sum).replace(0, np.nan)
            dem = (up_sum / denom).clip(0, 1).fillna(0.5)
            return dem
        except Exception:
            return pd.Series([0.5] * len(df), index=df.index)

    @staticmethod
    def detect_ema_power_zone_and_reclaim(
        df: pd.DataFrame,
        ema8: pd.Series,
        ema21: pd.Series,
        vol_lookback: int = 20
    ) -> Dict[str, object]:
        try:
            close = pd.to_numeric(df['Close'], errors='coerce')
            vol = df['Volume'] if 'Volume' in df.columns else None
            power_zone = bool(close.iloc[-1] > ema8.iloc[-1] > ema21.iloc[-1])
            is_reclaim = False
            reasons = []
            if len(close) >= 3 and ema8.notna().any() and ema21.notna().any():
                prior_below = (close.iloc[-2] < ema8.iloc[-2]) or (close.iloc[-2] < ema21.iloc[-2])
                now_above = (close.iloc[-1] > ema8.iloc[-1]) and (close.iloc[-1] > ema21.iloc[-1])
                # Compute EMA slopes using last two values to avoid typing issues
                try:
                    ema8_last = float(ema8.iloc[-1])
                    ema8_prev = float(ema8.iloc[-2])
                    ema8_up = (ema8_last - ema8_prev) > 0.0
                except Exception:
                    ema8_up = False
                try:
                    ema21_last = float(ema21.iloc[-1])
                    ema21_prev = float(ema21.iloc[-2])
                    ema21_up = (ema21_last - ema21_prev) > 0.0
                except Exception:
                    ema21_up = False
                vol_ok = True
                if vol is not None:
                    avg_vol = pd.to_numeric(vol, errors='coerce').rolling(vol_lookback).mean().iloc[-1]
                    vol_ok = (vol.iloc[-1] > 1.1 * avg_vol) if pd.notna(avg_vol) else True
                follow_through = close.iloc[-1] > close.iloc[-2]
                is_reclaim = bool(prior_below and now_above and ema8_up and ema21_up and vol_ok and follow_through)
                if is_reclaim:
                    reasons.append("Close above 8 & 21 with rising EMAs and follow-through")
                    if vol is not None:
                        reasons.append("Volume > 20D average")
            return {"power_zone": power_zone, "is_reclaim": is_reclaim, "reasons": reasons}
        except Exception:
            return {"power_zone": False, "is_reclaim": False, "reasons": []}

    @staticmethod
    def compute_fib_extensions_from_swing(
        df: pd.DataFrame,
        lookback: int = 180,
        pullback_window: int = 30
    ) -> Optional[Dict[str, float]]:
        try:
            if len(df) < 50:
                return None
            highs = pd.to_numeric(df['High'], errors='coerce').tail(lookback)
            lows = pd.to_numeric(df['Low'], errors='coerce').tail(lookback)
            if highs.empty or lows.empty:
                return None
            recent_highs = highs.tail(min(60, len(highs)))
            b_idx = recent_highs.idxmax()
            B = float(highs.loc[b_idx])
            pre = lows.loc[:b_idx]
            if pre.empty:
                return None
            A = float(pre.min())
            post = lows.loc[b_idx:].head(pullback_window)
            if post.empty:
                return None
            C = float(post.min())
            if not (B > A and B > C and C >= A):
                return None
            swing = B - A
            t1 = C + 1.272 * swing
            t2 = C + 1.618 * swing
            t3a = C + 2.000 * swing
            t3b = C + 2.618 * swing
            return {"A": A, "B": B, "C": C, "T1_1272": float(t1), "T2_1618": float(t2), "T3_200": float(t3a), "T3_2618": float(t3b)}
        except Exception:
            return None

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
                                    from datetime import datetime
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
            for article in news:
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
            logger.error(f"Error getting catalysts for {ticker}: {e}")
        
        return catalysts

class ComprehensiveAnalyzer:
    """Combines all analysis into a complete stock evaluation"""
    
    @staticmethod
    def analyze_stock(ticker: str, trading_style: str = "OPTIONS") -> Optional[StockAnalysis]:
        """Perform complete stock analysis"""
        try:
            # Use cached data for better performance
            hist, info = get_cached_stock_data(ticker)
            
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price / prev_close - 1) * 100)
            
            # Technical indicators
            rsi = TechnicalAnalyzer.calculate_rsi(hist['Close'])
            macd_signal, macd_value = TechnicalAnalyzer.calculate_macd(hist['Close'])
            support, resistance = TechnicalAnalyzer.calculate_support_resistance(hist['Close'])
            iv_rank, iv_percentile = TechnicalAnalyzer.calculate_iv_metrics(ticker)

            # New indicators
            ema8_series = TechnicalAnalyzer.ema(hist['Close'], 8)
            ema21_series = TechnicalAnalyzer.ema(hist['Close'], 21)
            dem_series = TechnicalAnalyzer.demarker(hist, period=14)
            ema_ctx = TechnicalAnalyzer.detect_ema_power_zone_and_reclaim(hist, ema8_series, ema21_series)
            fib_targets = TechnicalAnalyzer.compute_fib_extensions_from_swing(hist)
            
            # Volume analysis
            current_volume = int(hist['Volume'].iloc[-1])
            avg_volume = int(hist['Volume'].mean())
            
            # Determine trend
            if change_pct > 2:
                trend = "STRONG UPTREND"
            elif change_pct > 0.5:
                trend = "UPTREND"
            elif change_pct < -2:
                trend = "STRONG DOWNTREND"
            elif change_pct < -0.5:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
            if ema_ctx.get("power_zone") and trend == "SIDEWAYS":
                trend = "UPTREND"
            
            # News and catalysts
            news = NewsAnalyzer.get_stock_news(ticker)
            sentiment_score, sentiment_signals = NewsAnalyzer.analyze_sentiment(news)
            catalysts = NewsAnalyzer.get_catalysts(ticker)
            
            # Earnings information
            earnings_date = None
            earnings_days_away = None
            for catalyst in catalysts:
                if catalyst['type'] == 'Earnings Report':
                    earnings_date = catalyst['date']
                    earnings_days_away = catalyst['days_away']
                    break
            
            # Calculate confidence score
            confidence_score = ComprehensiveAnalyzer._calculate_confidence(
                rsi, macd_signal, iv_rank, sentiment_score, len(catalysts), earnings_days_away
            )
            
            # Generate recommendation based on trading style
            recommendation = ComprehensiveAnalyzer._generate_recommendation(
                rsi, macd_signal, trend, iv_rank, sentiment_score, earnings_days_away, trading_style,
                ema_ctx=ema_ctx, demarker_value=(dem_series.iloc[-1] if not dem_series.empty else None),
                fib_targets=fib_targets
            )
            
            return StockAnalysis(
                ticker=ticker.upper(),
                price=round(current_price, 2),
                change_pct=round(change_pct, 2),
                volume=current_volume,
                avg_volume=avg_volume,
                rsi=rsi,
                macd_signal=macd_signal,
                trend=trend,
                support=support,
                resistance=resistance,
                iv_rank=iv_rank,
                iv_percentile=iv_percentile,
                earnings_date=earnings_date,
                earnings_days_away=earnings_days_away,
                recent_news=news,
                catalysts=catalysts,
                sentiment_score=sentiment_score,
                sentiment_signals=sentiment_signals,
                confidence_score=confidence_score,
                recommendation=recommendation,
                ema8=float(ema8_series.iloc[-1]) if not ema8_series.empty else None,
                ema21=float(ema21_series.iloc[-1]) if not ema21_series.empty else None,
                demarker=float(dem_series.iloc[-1]) if not dem_series.empty else None,
                fib_targets=fib_targets,
                ema_power_zone=bool(ema_ctx.get("power_zone")),
                ema_reclaim=bool(ema_ctx.get("is_reclaim"))
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    @staticmethod
    def _calculate_confidence(rsi: float, macd_signal: str, iv_rank: float, 
                            sentiment: float, catalyst_count: int, 
                            earnings_days: Optional[int]) -> float:
        """Calculate overall confidence score for trading this stock"""
        score = 50  # Base score
        
        # RSI contribution
        if 30 <= rsi <= 70:
            score += 15  # Neutral RSI is good
        elif rsi < 30:
            score += 10  # Oversold can be opportunity
        elif rsi > 70:
            score += 5   # Overbought is risky
        
        # MACD contribution
        if macd_signal in ["BULLISH", "BEARISH"]:
            score += 10
        
        # IV Rank contribution
        if iv_rank > 50:
            score += 15  # High IV is good for selling premium
        elif iv_rank < 30:
            score += 10  # Low IV is good for buying options
        
        # Sentiment contribution
        score += sentiment * 10  # -10 to +10
        
        # Catalyst contribution
        if catalyst_count > 0:
            score += min(catalyst_count * 5, 15)
        
        # Earnings risk
        if earnings_days is not None and earnings_days <= 7:
            score -= 20  # High risk around earnings
        
        return min(100, max(0, score))
    
    @staticmethod
    def _generate_recommendation(rsi: float, macd_signal: str, trend: str,
                                iv_rank: float, sentiment: float,
                                earnings_days: Optional[int], 
                                trading_style: str = "OPTIONS",
                                ema_ctx: Dict[str, object] | None = None,
                                demarker_value: float | None = None,
                                fib_targets: Dict[str, float] | None = None) -> str:
        """Generate trading recommendation based on selected trading style"""
        
        if trading_style == "DAY_TRADE":
            return ComprehensiveAnalyzer._generate_day_trade_recommendation(
                rsi, macd_signal, trend, sentiment, earnings_days, ema_ctx=ema_ctx
            )
        elif trading_style == "SWING_TRADE":
            return ComprehensiveAnalyzer._generate_swing_trade_recommendation(
                rsi, macd_signal, trend, sentiment, earnings_days,
                ema_ctx=ema_ctx, demarker_value=demarker_value, fib_targets=fib_targets
            )
        elif trading_style == "SCALP":
            return ComprehensiveAnalyzer._generate_scalp_recommendation(
                rsi, macd_signal, trend, sentiment
            )
        elif trading_style == "BUY_HOLD":
            return ComprehensiveAnalyzer._generate_buy_hold_recommendation(
                rsi, macd_signal, trend, sentiment, earnings_days
            )
        else:  # OPTIONS
            return ComprehensiveAnalyzer._generate_options_recommendation(
                rsi, macd_signal, trend, iv_rank, sentiment, earnings_days
            )
    
    @staticmethod
    def _generate_day_trade_recommendation(rsi: float, macd_signal: str, trend: str,
                                          sentiment: float, earnings_days: Optional[int],
                                          ema_ctx: Dict[str, object] | None = None) -> str:
        """Generate day trading recommendation for intraday equity trades"""
        recommendations = []
        
        # Trend-based entry
        if "UPTREND" in trend or "STRONG UPTREND" in trend:
            if rsi < 70:
                recommendations.append("📈 BUY on pullbacks to support levels")
                recommendations.append("Target: Resistance levels for quick profit (0.5-2%)")
            else:
                recommendations.append("⚠️ Overbought - Wait for pullback or avoid")
        elif "DOWNTREND" in trend or "STRONG DOWNTREND" in trend:
            if rsi > 30:
                recommendations.append("📉 SHORT on bounces to resistance")
                recommendations.append("Target: Support levels for quick profit (0.5-2%)")
            else:
                recommendations.append("⚠️ Oversold - Wait for bounce or avoid shorting")
        else:  # SIDEWAYS
            recommendations.append("↔️ Range-bound: BUY near support, SELL near resistance")
            recommendations.append("Use tight stops (0.3-0.5%) for range trading")
        
        # RSI signals
        if rsi < 30:
            recommendations.append("🟢 RSI oversold → Look for bounce/reversal entry")
        elif rsi > 70:
            recommendations.append("🔴 RSI overbought → Look for rejection/reversal short")
        
        # MACD confirmation
        if macd_signal == "BULLISH":
            recommendations.append("✅ MACD bullish → Momentum favors longs")
        elif macd_signal == "BEARISH":
            recommendations.append("❌ MACD bearish → Momentum favors shorts")

        # EMA Power Zone filter
        if ema_ctx and ema_ctx.get("power_zone"):
            recommendations.append("✅ 8>21 EMA Power Zone → Favor long setups")
        
        # Risk management
        recommendations.append("🛡️ Stop Loss: 0.5-1% | Take Profit: 1-3% | Exit by market close")
        
        # Earnings warning
        if earnings_days is not None and earnings_days <= 1:
            recommendations.append("⚠️ EARNINGS TODAY/TOMORROW → Avoid day trading (high volatility risk)")
        
        return "\n".join(recommendations) if recommendations else "Insufficient data for day trade recommendation"
    
    @staticmethod
    def _generate_swing_trade_recommendation(rsi: float, macd_signal: str, trend: str,
                                            sentiment: float, earnings_days: Optional[int],
                                            ema_ctx: Dict[str, object] | None = None,
                                            demarker_value: float | None = None,
                                            fib_targets: Dict[str, float] | None = None) -> str:
        """Generate swing trading recommendation for multi-day equity holds"""
        recommendations = []
        
        # Trend-based strategy
        if "UPTREND" in trend or "STRONG UPTREND" in trend:
            recommendations.append("📈 LONG BIAS: Enter on dips, hold for 3-10 days")
            recommendations.append("Entry: Near support or after consolidation breakout")
            recommendations.append("Target: 5-15% gain to resistance levels")
        elif "DOWNTREND" in trend or "STRONG DOWNTREND" in trend:
            recommendations.append("📉 SHORT BIAS: Enter on rallies, hold for 3-10 days")
            recommendations.append("Entry: Near resistance or after breakdown")
            recommendations.append("Target: 5-15% profit to support levels")
        else:
            recommendations.append("↔️ NEUTRAL: Wait for breakout direction before entering")
        
        # EMA power zone and reclaim context
        if ema_ctx:
            if ema_ctx.get("power_zone"):
                recommendations.append("✅ 8>21 EMA and price above both → Power Zone active")
            if ema_ctx.get("is_reclaim"):
                _reasons = ema_ctx.get("reasons", [])
                reasons = "; ".join([str(x) for x in _reasons]) if isinstance(_reasons, list) else ""
                recommendations.append(f"✅ EMA Reclaim confirmed ({reasons})")

        # DeMarker for precision
        if demarker_value is not None:
            if demarker_value <= 0.30 and ("UPTREND" in trend):
                recommendations.append("🟢 DeMarker ≤ 0.30 in uptrend → High-probability pullback entry")
            elif demarker_value >= 0.70 and ("DOWNTREND" in trend):
                recommendations.append("🔴 DeMarker ≥ 0.70 in downtrend → High-probability short entry")

        # RSI for swing entries
        if rsi < 40 and "UPTREND" in trend:
            recommendations.append("🟢 Good swing entry: RSI pullback in uptrend")
        elif rsi > 60 and "DOWNTREND" in trend:
            recommendations.append("🔴 Good short entry: RSI bounce in downtrend")
        
        # MACD trend confirmation
        if macd_signal == "BULLISH":
            recommendations.append("✅ MACD confirms uptrend → Hold longs, avoid shorts")
        elif macd_signal == "BEARISH":
            recommendations.append("❌ MACD confirms downtrend → Hold shorts, avoid longs")
        
        # Sentiment factor
        if sentiment > 0.3:
            recommendations.append("📰 Positive sentiment → Supports bullish swing trades")
        elif sentiment < -0.3:
            recommendations.append("📰 Negative sentiment → Supports bearish swing trades")
        
        # Fibonacci targets
        if fib_targets:
            t1 = fib_targets.get("T1_1272")
            t2 = fib_targets.get("T2_1618")
            t3 = fib_targets.get("T3_2618") or fib_targets.get("T3_200")
            fib_lines = []
            if t1:
                fib_lines.append(f"🎯 T1 (127.2%): ${t1:.2f} → Take 25%")
            if t2:
                fib_lines.append(f"🎯 T2 (161.8%): ${t2:.2f} → Take 50%")
            if t3:
                fib_lines.append(f"🎯 T3 (200-261.8%): ${t3:.2f} → Trail remaining")
            if fib_lines:
                recommendations.append("📐 Fibonacci Targets:")
                recommendations.extend(fib_lines)
                recommendations.append("🧭 Move stop to breakeven after T1, trail below 21 EMA thereafter")
        else:
            recommendations.append("🛡️ Stop Loss: 3-5% | Take Profit: 8-15% | Hold time: 3-10 days")
        
        # Earnings consideration
        if earnings_days is not None and earnings_days <= 7:
            recommendations.append("⚠️ EARNINGS SOON → Close position before earnings or use wider stops")
        
        return "\n".join(recommendations) if recommendations else "Insufficient data for swing trade recommendation"
    
    @staticmethod
    def _generate_scalp_recommendation(rsi: float, macd_signal: str, trend: str, sentiment: float) -> str:
        """Generate scalping recommendation for very short-term trades"""
        recommendations = []
        
        recommendations.append("⚡ SCALPING STRATEGY (seconds to minutes):")
        
        # Momentum-based scalping
        if "STRONG UPTREND" in trend:
            recommendations.append("🚀 Strong momentum UP → Scalp long on dips (0.1-0.5% targets)")
            recommendations.append("Entry: Quick pullbacks | Exit: Immediate resistance")
        elif "STRONG DOWNTREND" in trend:
            recommendations.append("💥 Strong momentum DOWN → Scalp short on bounces (0.1-0.5% targets)")
            recommendations.append("Entry: Quick bounces | Exit: Immediate support")
        else:
            recommendations.append("⚠️ Low momentum → Scalping difficult, wait for clear direction")
        
        # RSI for quick reversals
        if rsi < 25:
            recommendations.append("🟢 Extreme oversold → Quick bounce scalp opportunity")
        elif rsi > 75:
            recommendations.append("🔴 Extreme overbought → Quick rejection scalp opportunity")
        
        # Risk management for scalping
        recommendations.append("🛡️ TIGHT STOPS: 0.1-0.3% | Target: 0.2-0.5% | Hold: Seconds to 5 minutes")
        recommendations.append("⚡ Requires: Level 2 data, fast execution, high volume stocks")
        recommendations.append("⚠️ High risk: Only for experienced traders with proper tools")
        
        return "\n".join(recommendations)
    
    @staticmethod
    def _generate_buy_hold_recommendation(rsi: float, macd_signal: str, trend: str,
                                         sentiment: float, earnings_days: Optional[int]) -> str:
        """Generate buy and hold recommendation for long-term investing"""
        recommendations = []
        
        recommendations.append("📊 LONG-TERM INVESTMENT ANALYSIS:")
        
        # Overall trend assessment
        if "UPTREND" in trend or "STRONG UPTREND" in trend:
            recommendations.append("✅ Positive long-term trend → Good for accumulation")
            if rsi < 50:
                recommendations.append("🟢 STRONG BUY: Uptrend + pullback = ideal entry point")
            else:
                recommendations.append("🟡 BUY: Uptrend continues, consider dollar-cost averaging")
        elif "DOWNTREND" in trend or "STRONG DOWNTREND" in trend:
            recommendations.append("⚠️ Negative trend → Wait for reversal or avoid")
            recommendations.append("🔴 HOLD/AVOID: Downtrend not ideal for new positions")
        else:
            recommendations.append("🟡 NEUTRAL: Consolidating, wait for breakout direction")
        
        # Value assessment using RSI
        if rsi < 30:
            recommendations.append("💰 Potentially undervalued (oversold) → Good accumulation zone")
        elif rsi > 70:
            recommendations.append("💸 Potentially overvalued (overbought) → Consider waiting")
        
        # Sentiment for long-term
        if sentiment > 0.3:
            recommendations.append("📰 Strong positive sentiment → Supports long-term bullish case")
        elif sentiment < -0.3:
            recommendations.append("📰 Negative sentiment → Research fundamental concerns")
        
        # Long-term strategy
        recommendations.append("📈 Strategy: Dollar-cost average over time, ignore short-term noise")
        recommendations.append("🎯 Target: 20%+ annual returns | Hold time: 6+ months to years")
        recommendations.append("💡 Consider: Selling covered calls for income if you accumulate shares")
        
        # Earnings note
        if earnings_days is not None and earnings_days <= 14:
            recommendations.append("📅 Earnings soon → Good time to review fundamentals")
        
        return "\n".join(recommendations)
    
    @staticmethod
    def _generate_options_recommendation(rsi: float, macd_signal: str, trend: str,
                                        iv_rank: float, sentiment: float,
                                        earnings_days: Optional[int]) -> str:
        """Generate options trading recommendation"""
        recommendations = []
        
        # IV-based strategies
        if iv_rank > 60:
            recommendations.append("High IV → Consider SELLING premium (puts, calls, iron condors)")
        elif iv_rank < 40:
            recommendations.append("Low IV → Consider BUYING options (calls, puts, spreads)")
        
        # Trend-based strategies
        if "UPTREND" in trend:
            if iv_rank > 50:
                recommendations.append("Uptrend + High IV → Sell puts or bull put spreads")
            else:
                recommendations.append("Uptrend + Low IV → Buy calls or bull call spreads")
        elif "DOWNTREND" in trend:
            if iv_rank > 50:
                recommendations.append("Downtrend + High IV → Sell calls or bear call spreads")
            else:
                recommendations.append("Downtrend + Low IV → Buy puts or bear put spreads")
        else:
            if iv_rank > 50:
                recommendations.append("Sideways + High IV → Iron condors or strangles")
        
        # RSI-based
        if rsi < 30:
            recommendations.append("RSI oversold → Potential bullish reversal opportunity")
        elif rsi > 70:
            recommendations.append("RSI overbought → Potential bearish reversal opportunity")
        
        # Earnings warning
        if earnings_days is not None and earnings_days <= 7:
            recommendations.append("⚠️ EARNINGS SOON → High risk! Consider waiting or use defined-risk strategies")
        
        # Sentiment
        if sentiment > 0.3:
            recommendations.append("Positive news sentiment → Bullish bias")
        elif sentiment < -0.3:
            recommendations.append("Negative news sentiment → Bearish bias")
        
        return " | ".join(recommendations) if recommendations else "Insufficient data for recommendation"

class StrategyAdvisor:
    """Intelligent strategy recommendation engine"""
    
    STRATEGIES = {
        "SELL_PUT": {
            "name": "Cash-Secured Put (Sell Put)",
            "description": "Sell a put option to collect premium. You're obligated to buy the stock if it drops below the strike.",
            "risk_level": "Medium",
            "max_loss": "Strike price - Premium received",
            "max_gain": "Premium received (limited)",
            "best_for": ["Bullish or neutral outlook", "High IV", "Want to own stock at lower price"],
            "experience": "Beginner-Friendly",
            "capital_req": "High (need cash to secure)",
            "typical_win_rate": "65-75%",
            "examples": ["Sell 1x 30D put at strike - collect premium", "Cash-secured with 100 shares worth allocation"],
            "notes": "Good income strategy if you’re willing to own the stock; monitor assignment risk around earnings.",
            "example_trade": {"dte": 30, "strike_offset_pct": -0.05, "qty": 2, "estimated_risk": 300}
        },
        "SELL_CALL": {
            "name": "Covered Call (Sell Call)",
            "description": "Sell a call option against stock you own to collect premium. Stock may be called away if price rises.",
            "risk_level": "Low-Medium",
            "max_loss": "Unlimited if stock drops (but you own stock)",
            "max_gain": "Premium + (Strike - Stock Purchase Price)",
            "best_for": ["Own the stock", "Neutral to slightly bullish", "Generate income"],
            "experience": "Beginner-Friendly",
            "capital_req": "High (need to own 100 shares)",
            "typical_win_rate": "70-80%"
        },
        "BUY_CALL": {
            "name": "Long Call (Buy Call)",
            "description": "Buy a call option for the right to buy stock at strike price. Bullish directional bet.",
            "risk_level": "Medium-High",
            "max_loss": "Premium paid (limited)",
            "max_gain": "Unlimited",
            "best_for": ["Strong bullish conviction", "Low to medium IV", "Limited capital for directional bet"],
            "experience": "Beginner-Friendly",
            "capital_req": "Low-Medium",
            "typical_win_rate": "30-45%",
            "examples": ["Buy a near-the-money call 30D for directional upside"],
            "notes": "Good for directional bullish bets when you expect a move.",
            "example_trade": {"dte": 30, "strike_offset_pct": 0.02, "qty": 1, "estimated_risk": 150}
        },
        "BUY_PUT": {
            "name": "Long Put (Buy Put)",
            "description": "Buy a put option for the right to sell stock at strike price. Bearish directional bet or hedge.",
            "risk_level": "Medium-High",
            "max_loss": "Premium paid (limited)",
            "max_gain": "Strike price - Premium (large potential)",
            "best_for": ["Bearish conviction", "Portfolio hedge", "Low to medium IV"],
            "experience": "Beginner-Friendly",
            "capital_req": "Low-Medium",
            "typical_win_rate": "30-45%"
        },
        "IRON_CONDOR": {
            "name": "Iron Condor",
            "description": "Sell both a put spread and call spread. Profit if stock stays in a range between strikes.",
            "risk_level": "Medium",
            "max_loss": "Width of spread - Net credit received",
            "max_gain": "Net credit received (limited)",
            "best_for": ["Expect low movement", "High IV", "Range-bound stocks"],
            "experience": "Intermediate",
            "capital_req": "Medium",
            "typical_win_rate": "60-70%",
            "examples": ["Sell 30D iron condor 5-10% OTM wings"],
            "notes": "Best used on range-bound stocks with high IV.",
            "example_trade": {"dte": 30, "wing_width_pct": 0.05, "qty": 1, "estimated_risk": 500}
        },
        "CREDIT_SPREAD": {
            "name": "Credit Spread (Bull Put or Bear Call)",
            "description": "Sell a spread to collect credit. Bull put spread = bullish, Bear call spread = bearish.",
            "risk_level": "Medium",
            "max_loss": "Width of spread - Net credit",
            "max_gain": "Net credit received (limited)",
            "best_for": ["Directional bias with defined risk", "High IV", "Want better probability than buying options"],
            "experience": "Intermediate",
            "capital_req": "Medium",
            "typical_win_rate": "60-70%"
        },
        "DEBIT_SPREAD": {
            "name": "Debit Spread (Bull Call or Bear Put)",
            "description": "Buy a spread to reduce cost. Bull call spread = bullish, Bear put spread = bearish.",
            "risk_level": "Medium",
            "max_loss": "Net debit paid (limited)",
            "max_gain": "Width of spread - Net debit",
            "best_for": ["Directional bias", "Lower cost than buying single option", "Moderate IV"],
            "experience": "Intermediate",
            "capital_req": "Low-Medium",
            "typical_win_rate": "40-55%"
        },
        "LONG_STRADDLE": {
            "name": "Long Straddle",
            "description": "Buy both a call and put at the same strike. Profit from big move in either direction.",
            "risk_level": "High",
            "max_loss": "Total premium paid for both options",
            "max_gain": "Unlimited (if big move occurs)",
            "best_for": ["Expect big move but unsure of direction", "Low IV before event", "Earnings plays"],
            "experience": "Advanced",
            "capital_req": "Medium-High",
            "typical_win_rate": "35-50%",
            "examples": ["Buy 30D ATM straddle into earnings"],
            "notes": "High cost; works best if large move expected.",
            "example_trade": {"dte": 30, "strike_offset_pct": 0.0, "qty": 1, "estimated_risk": 800}
        },
        "WHEEL_STRATEGY": {
            "name": "The Wheel Strategy",
            "description": "Sell puts until assigned, then sell calls against the stock. Repeat.",
            "risk_level": "Medium",
            "max_loss": "Stock value decline",
            "max_gain": "Premium collected consistently",
            "best_for": ["Generate steady income", "Willing to own stock", "High IV stocks"],
            "experience": "Intermediate",
            "capital_req": "High",
            "typical_win_rate": "70-80%",
            "examples": ["Sell puts until assigned, then sell covered calls"],
            "notes": "Good income strategy but requires capital planning.",
            "example_trade": {"dte": 45, "strike_offset_pct": -0.08, "qty": 1, "estimated_risk": 1000}
        },
        "SHORT_STRANGLE": {
            "name": "Short Strangle",
            "description": "Sell an OTM call and an OTM put to collect premium; profit if the stock stays within the range.",
            "risk_level": "High",
            "max_loss": "Potentially large (if stock gaps large), defined only with additional hedges",
            "max_gain": "Net premium received",
            "best_for": ["Expect low movement", "High IV", "Range-bound markets"],
            "experience": "Advanced",
            "capital_req": "High",
            "typical_win_rate": "50-65%",
            "examples": ["Sell 30D 1.05x call and 0.95x put at same expiry"],
            "notes": "Requires active management and margin; consider hedges or defined-risk modifications."
        },
        "CALENDAR_SPREAD": {
            "name": "Calendar Spread",
            "description": "Buy longer-dated option and sell shorter-dated option at same strike to play for theta on the front leg.",
            "risk_level": "Medium",
            "max_loss": "Premium paid for long leg",
            "max_gain": "Variable (depends on front-month decay and move)",
            "best_for": ["Expect limited near-term movement", "Low to moderate IV", "Earnings or event timing plays"],
            "experience": "Intermediate",
            "capital_req": "Low-Medium",
            "typical_win_rate": "40-60%",
            "examples": ["Buy 60D call, sell 30D call at same strike"],
            "notes": "Works best when front-month premium decays faster than back-month."
        },
        "PUT_DIAGONAL": {
            "name": "Put Diagonal Spread",
            "description": "Buy longer-dated put and sell shorter-dated put at different strikes to create a defined-risk bearish income trade.",
            "risk_level": "Medium-High",
            "max_loss": "Net debit paid",
            "max_gain": "Difference between strikes minus net debit",
            "best_for": ["Mildly bearish outlook", "Want defined risk", "Use when IV term-structure favors front-month"],
            "experience": "Intermediate",
            "capital_req": "Low-Medium",
            "typical_win_rate": "40-55%",
            "examples": ["Buy 90D put, sell 30D put at higher strike"],
            "notes": "Can be adjusted into defined-risk hedges if market moves quickly."
        }
    }
    
    @classmethod
    def get_recommendations(cls, analysis: StockAnalysis, 
                          user_experience: str, risk_tolerance: str,
                          capital_available: float, outlook: str) -> List[StrategyRecommendation]:
        """Generate personalized strategy recommendations"""
        
        recommendations = []
        
        # Score each strategy
        for strategy_key, strategy_info in cls.STRATEGIES.items():
            score = 0
            reasoning_parts = []
            
            # Experience level filter
            if user_experience == "Beginner" and strategy_info["experience"] == "Advanced":
                continue
            if user_experience == "Beginner" and strategy_info["experience"] == "Intermediate":
                score -= 20
            
            # Risk tolerance filter
            if risk_tolerance == "Conservative" and strategy_info["risk_level"] in ["High", "Medium-High"]:
                score -= 30
            if risk_tolerance == "Aggressive" and strategy_info["risk_level"] == "Low":
                score -= 10
            
            # Capital requirements
            capital_req = strategy_info.get("capital_req", "Medium")
            if capital_available < 5000 and capital_req == "High":
                score -= 40
                reasoning_parts.append("⚠️ May require more capital than available")
            
            # IV considerations
            if analysis.iv_rank > 60:
                if "High IV" in strategy_info["best_for"]:
                    score += 30
                    reasoning_parts.append(f"✅ High IV Rank ({analysis.iv_rank}%) - premium selling favorable")
                if strategy_key in ["SELL_PUT", "SELL_CALL", "IRON_CONDOR", "CREDIT_SPREAD"]:
                    score += 25
            elif analysis.iv_rank < 40:
                if strategy_key in ["BUY_CALL", "BUY_PUT", "DEBIT_SPREAD"]:
                    score += 25
                    reasoning_parts.append(f"✅ Low IV Rank ({analysis.iv_rank}%) - option buying favorable")
            
            # Market outlook alignment
            if outlook == "Bullish":
                if strategy_key in ["SELL_PUT", "BUY_CALL", "CREDIT_SPREAD"]:
                    score += 25
                    reasoning_parts.append("✅ Aligns with bullish outlook")
            elif outlook == "Bearish":
                if strategy_key in ["BUY_PUT", "CREDIT_SPREAD"]:
                    score += 25
                    reasoning_parts.append("✅ Aligns with bearish outlook")
            elif outlook == "Neutral":
                if strategy_key in ["IRON_CONDOR", "SELL_CALL", "WHEEL_STRATEGY"]:
                    score += 25
                    reasoning_parts.append("✅ Good for neutral/range-bound markets")
            
            # Technical indicators
            if analysis.rsi < 30 and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 15
                reasoning_parts.append(f"✅ RSI oversold ({analysis.rsi}) - potential bounce")
            elif analysis.rsi > 70 and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 15
                reasoning_parts.append(f"✅ RSI overbought ({analysis.rsi}) - potential pullback")
            
            if analysis.macd_signal == "BULLISH" and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 10
                reasoning_parts.append("✅ MACD bullish crossover")
            elif analysis.macd_signal == "BEARISH" and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 10
                reasoning_parts.append("✅ MACD bearish crossover")
            
            # Trend alignment
            if "UPTREND" in analysis.trend and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 15
                reasoning_parts.append(f"✅ Stock in {analysis.trend}")
            elif "DOWNTREND" in analysis.trend and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 15
                reasoning_parts.append(f"✅ Stock in {analysis.trend}")
            
            # Sentiment
            if analysis.sentiment_score > 0.3 and strategy_key in ["BUY_CALL", "SELL_PUT"]:
                score += 10
                reasoning_parts.append("✅ Positive news sentiment")
            elif analysis.sentiment_score < -0.3 and strategy_key in ["BUY_PUT", "SELL_CALL"]:
                score += 10
                reasoning_parts.append("✅ Negative news sentiment")
            
            # Earnings risk
            if analysis.earnings_days_away is not None and analysis.earnings_days_away <= 7:
                if strategy_key in ["IRON_CONDOR", "LONG_STRADDLE"]:
                    score += 15
                    reasoning_parts.append(f"✅ Earnings in {analysis.earnings_days_away} days - volatility play")
                else:
                    score -= 25
                    reasoning_parts.append(f"⚠️ Earnings in {analysis.earnings_days_away} days - high risk")
            
            # Beginner bonus
            if user_experience == "Beginner" and strategy_info["experience"] == "Beginner-Friendly":
                score += 15
                reasoning_parts.append("✅ Beginner-friendly")
            
            # Win rate for conservative traders
            if risk_tolerance == "Conservative":
                win_rate = int(strategy_info.get("typical_win_rate", "50%").split("-")[0].replace("%", ""))
                if win_rate >= 60:
                    score += 10
                    reasoning_parts.append(f"✅ High win rate (~{strategy_info['typical_win_rate']})")
            
            confidence = max(0, min(1, (score + 50) / 100))
            
            if confidence > 0.3:
                recommendations.append(StrategyRecommendation(
                    strategy_name=strategy_info["name"],
                    action=strategy_key,
                    confidence=confidence,
                    reasoning="\n".join(reasoning_parts) if reasoning_parts else strategy_info["description"],
                    risk_level=strategy_info["risk_level"],
                    max_loss=strategy_info["max_loss"],
                    max_gain=strategy_info["max_gain"],
                    best_conditions=strategy_info["best_for"],
                    experience_level=strategy_info["experience"]
                ))
        
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:5]

@dataclass
class TradingConfig:
    """Configuration for trading parameters and guardrails"""
    max_daily_orders: int = 10
    max_position_per_ticker: int = 5
    max_daily_risk: float = 1000.0
    min_dte: int = 7
    max_dte: int = 45
    min_iv_rank: float = 20.0
    max_iv_rank: float = 80.0
    min_volume: int = 100
    max_bid_ask_spread: float = 0.50
    allowed_strategies: Optional[List[str]] = None
    trading_start_hour: int = 9
    trading_end_hour: int = 15
    trading_start_minute: int = 45
    trading_end_minute: int = 45
    
    def __post_init__(self):
        if self.allowed_strategies is None:
            self.allowed_strategies = [
                "SELL_CALL", "SELL_PUT", "BUY_CALL", "BUY_PUT",
                "IRON_CONDOR", "CREDIT_SPREAD", "DEBIT_SPREAD",
                "LONG_STRADDLE", "WHEEL_STRATEGY"
            ]

class SignalValidator:
    """Validates trading signals against guardrails"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.daily_orders = 0
        self.daily_risk = 0.0
        self.ticker_positions = {}
        self.last_reset = datetime.now().date()
    
    def reset_daily_counters(self):
        """Reset counters at start of new trading day"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_orders = 0
            self.daily_risk = 0.0
            self.ticker_positions = {}
            self.last_reset = today
            logger.info("Daily counters reset")
    
    def is_trading_hours(self) -> Tuple[bool, str]:
        """Check if current time is within trading hours"""
        now = datetime.now()
        start_time = now.replace(hour=self.config.trading_start_hour, 
                                 minute=self.config.trading_start_minute, second=0)
        end_time = now.replace(hour=self.config.trading_end_hour, 
                               minute=self.config.trading_end_minute, second=0)
        
        if now < start_time:
            return False, f"Before trading hours (starts at {start_time.strftime('%H:%M')})"
        if now > end_time:
            return False, f"After trading hours (ends at {end_time.strftime('%H:%M')})"
        
        return True, "Within trading hours"
    
    def validate_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Comprehensive signal validation"""
        self.reset_daily_counters()
        
        in_hours, hours_msg = self.is_trading_hours()
        if not in_hours:
            return False, hours_msg
        
        if self.daily_orders >= self.config.max_daily_orders:
            return False, f"Daily order limit reached ({self.config.max_daily_orders})"
        
        estimated_risk = signal.get('estimated_risk', 0)
        if self.daily_risk + estimated_risk > self.config.max_daily_risk:
            return False, f"Daily risk limit would be exceeded (${self.config.max_daily_risk})"
        
        ticker = signal.get('ticker', '').upper()
        if not ticker:
            return False, "Ticker is required"
        
        current_positions = self.ticker_positions.get(ticker, 0)
        if current_positions >= self.config.max_position_per_ticker:
            return False, f"Max positions for {ticker} reached ({self.config.max_position_per_ticker})"
        
        action = signal.get('action', '').upper()
        if action not in self.config.allowed_strategies:
            return False, f"Strategy {action} not in allowed list"
        
        dte = signal.get('dte')
        if dte is not None:
            if dte < self.config.min_dte or dte > self.config.max_dte:
                return False, f"DTE {dte} outside allowed range ({self.config.min_dte}-{self.config.max_dte})"
        
        qty = signal.get('qty', 0)
        if qty <= 0 or qty > 10:
            return False, f"Quantity {qty} must be between 1 and 10"
        
        return True, "Signal validated successfully"
    
    def record_order(self, signal: Dict):
        """Record an order for tracking"""
        self.daily_orders += 1
        self.daily_risk += signal.get('estimated_risk', 0)
        ticker = signal.get('ticker', '').upper()
        self.ticker_positions[ticker] = self.ticker_positions.get(ticker, 0) + 1

class OptionAlphaClient:
    """Client for sending signals to Option Alpha webhook"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('OPTION_ALPHA_WEBHOOK_URL')
        self.timeout = 10
    
    def send_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Send signal to Option Alpha webhook"""
        try:
            if not self.webhook_url:
                return False, "Option Alpha webhook URL not configured"
            
            signal['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Sending signal to Option Alpha: {json.dumps(signal, indent=2)}")
            
            response = requests.post(
                self.webhook_url,
                json=signal,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            
            logger.info(f"Signal sent successfully. Response: {response.status_code}")
            return True, f"Signal sent successfully (Status: {response.status_code})"
            
        except requests.exceptions.Timeout:
            error_msg = "Request timed out"
            logger.error(error_msg)
            return False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

def calculate_dte(expiry_date: str) -> int:
    """Calculate days to expiration"""
    try:
        exp_date = datetime.strptime(expiry_date, '%Y-%m-%d')
        dte = (exp_date - datetime.now()).days
        return max(0, dte)
    except:
        return 0

# Streamlit App
def main():
    # If Streamlit ScriptRunContext isn't present (e.g., calling main() directly),
    # bail out early to avoid noisy warnings and unexpected behavior. Users should
    # run this app with: streamlit run app.py
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            logger.info("Streamlit ScriptRunContext not detected - main() called outside 'streamlit run'. Exiting early.")
            return
    except Exception:
        # Older streamlit versions or bare mode - allow execution but continue.
        pass

    st.set_page_config(
        page_title="Sentient Trader",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced visual appeal with new Streamlit features
    st.markdown("""
    <style>
    /* Modern clean theme for trading platform */
    .stMetric {
        background-color: #FFFFFF;
        border: 2px solid #E5E7EB;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .stMetric > div > div > div {
        color: #1F2937;
    }
    
    /* Custom status indicators */
    .stStatus {
        border-radius: 12px;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
    }
    
    /* Enhanced data editor styling */
    .stDataEditor {
        border-radius: 12px;
        border: 1px solid #E5E7EB;
    }
    
    /* Custom badge colors */
    .stBadge {
        background-color: #F3F4F6;
        color: #374151;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
    }
    
    /* Trading-themed colors for metrics with better contrast */
    .profit-metric {
        color: #059669 !important;
        font-weight: 600;
    }
    
    .loss-metric {
        color: #DC2626 !important;
        font-weight: 600;
    }
    
    .neutral-metric {
        color: #D97706 !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modern clean styling for better readability
    st.markdown("""
    <style>
    /* Enhanced metric text styling for light theme */
    .stMetric {
        color: #1F2937 !important;
    }

    .stMetric label, .stMetric .metric-label, .stMetric .stMetricLabel {
        color: #6B7280 !important;
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    /* Ensure all metric text is visible with proper contrast */
    .stMetric * {
        color: #1F2937 !important;
    }

    .stMetric .stMetricValue {
        color: #111827 !important;
        font-weight: 700;
        font-size: 1.5rem;
    }

    .stMetric .stMetricDelta {
        color: #059669 !important;
        font-weight: 600;
    }

    /* Clean alert styling */
    .stAlert {
        background-color: #FEFEFE !important;
        color: #1F2937 !important;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .stAlert * {
        color: #1F2937 !important;
    }

    /* Success alerts */
    .stAlert[data-baseweb="notification"] {
        background-color: #F0FDF4 !important;
        border-color: #BBF7D0;
    }

    /* Warning alerts */
    .stAlert[data-baseweb="notification"][aria-label="warning"] {
        background-color: #FFFBEB !important;
        border-color: #FED7AA;
    }

    /* Error alerts */
    .stAlert[data-baseweb="notification"][aria-label="error"] {
        background-color: #FEF2F2 !important;
        border-color: #FECACA;
    }

    /* Info alerts */
    .stAlert[data-baseweb="notification"][aria-label="info"] {
        background-color: #EFF6FF !important;
        border-color: #BFDBFE;
    }

    /* Clean expander styling */
    .stExpander .stExpanderHeader {
        color: #1F2937 !important;
        background-color: #F9FAFB !important;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }

    .stExpander .stExpanderHeader:hover {
        background-color: #F3F4F6 !important;
    }

    .stExpander .stExpanderHeader * {
        color: #1F2937 !important;
    }

    /* Clean container styling */
    .stContainer {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border-radius: 12px;
    }

    .stContainer * {
        color: #1F2937 !important;
    }

    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #F8FAFC;
    }

    /* Tab styling improvements */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background-color: #EBF8FF;
        color: #1E40AF;
    }
    </style>
""", unsafe_allow_html=True)
    
    st.title("📈 Sentient Trader Platform")
    st.caption("Real-time analysis, news, catalysts, technical indicators & intelligent strategy recommendations")
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = TradingConfig()
    if 'validator' not in st.session_state:
        st.session_state.validator = SignalValidator(st.session_state.config)
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []
    if 'paper_mode' not in st.session_state:
        st.session_state.paper_mode = True
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Trading Mode")
        paper_mode = st.toggle("Paper Trading Mode", value=st.session_state.paper_mode)
        st.session_state.paper_mode = paper_mode
        
        if paper_mode:
            st.info("🔒 Paper trading: Signals logged only")
        else:
            st.warning("⚠️ LIVE TRADING ENABLED")
        
        st.subheader("Option Alpha Webhook")
        webhook_url = st.text_input(
            "Webhook URL",
            value="https://app.optionalpha.com/api/webhooks/XXXX",
            type="password" if not paper_mode else "default"
        )
        
        st.subheader("Guardrails")
        with st.expander("Risk Limits"):
            max_daily_orders = st.number_input("Max Daily Orders", 1, 50, st.session_state.config.max_daily_orders)
            max_daily_risk = st.number_input("Max Daily Risk ($)", 100, 10000, int(st.session_state.config.max_daily_risk))
            max_position_per_ticker = st.number_input("Max Positions per Ticker", 1, 10, st.session_state.config.max_position_per_ticker)
        
        if st.button("Update Configuration"):
            st.session_state.config = TradingConfig(
                max_daily_orders=max_daily_orders,
                max_daily_risk=float(max_daily_risk),
                max_position_per_ticker=max_position_per_ticker
            )
            st.session_state.validator = SignalValidator(st.session_state.config)
            st.success("Configuration updated!")

        # Prominent Strategy Analyzer summary (quick access)
        st.markdown("---")
        st.subheader("🤖 Strategy Analyzer (Quick View)")
        # Show a compact summary if we have a recent analysis in session state
        analysis = st.session_state.get('strategy_analysis') or st.session_state.get('current_analysis')
        if analysis:
            try:
                bot_name = getattr(analysis, 'bot_name', getattr(analysis, 'ticker', 'Bot'))
                overall = getattr(analysis, 'overall_rating', getattr(analysis, 'rating', 'N/A'))
                risk = getattr(analysis, 'risk_score', None)
                conf = getattr(analysis, 'confidence', None)

                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.write(f"**{bot_name}**")
                    st.write(f"{getattr(analysis, 'summary', '')}")
                with c2:
                    st.metric("Overall", overall)
                with c3:
                    st.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else conf)

                if st.button("🔎 Open Strategy Analyzer", use_container_width=True):
                    # Set a flag so the main tabs can react (we can't switch tabs programmatically reliably)
                    st.session_state.goto_strategy_analyzer = True
                    st.rerun()
            except Exception:
                st.write("Compact summary unavailable")
        else:
            st.info("Run a strategy analysis to see a quick summary here.")
    
    # Main tabs - Reorganized for clarity
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
        "🏠 Dashboard",
        "🔥 Top Options Trades",
        "💰 Top Penny Stocks",
        "⭐ My Tickers",
        "🔍 Stock Intelligence", 
        "🎯 Strategy Advisor", 
        "📊 Generate Signal", 
        "📜 Signal History",
        "📚 Strategy Guide",
        "🏦 Tradier Account",
        "📈 IBKR Trading",
        "⚡ Scalping/Day Trade",
        "🤖 Strategy Analyzer"
    ])
    
    with tab1:
        logger.info(f"🏁 TAB1 RENDERING - Session state: show_quick_trade={st.session_state.get('show_quick_trade', 'NOT SET')}, has_analysis={st.session_state.get('current_analysis') is not None}")
        st.header("🔍 Comprehensive Stock Intelligence")
        st.write("Get real-time analysis including news, catalysts, technical indicators, and IV metrics.")
        st.info("💡 **Works with ALL stocks:** Blue chips, penny stocks (<$5), OTC stocks, and runners. Automatically detects momentum plays!")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_ticker = st.text_input(
                "Enter Ticker Symbol to Analyze", 
                value="SOFI",
                help="Enter any ticker: AAPL, TSLA, penny stocks (SNDL, GNUS), or OTC stocks"
            ).upper()
        
        with col2:
            trading_style_display = st.selectbox(
                "Trading Style",
                options=["📊 Day Trade", "📈 Swing Trade", "⚡ Scalp", "💎 Buy & Hold", "🎯 Options"],
                index=0,
                help="Select your trading style for personalized recommendations"
            )
            # Map display names to internal codes
            style_map = {
                "📊 Day Trade": "DAY_TRADE",
                "📈 Swing Trade": "SWING_TRADE",
                "⚡ Scalp": "SCALP",
                "💎 Buy & Hold": "BUY_HOLD",
                "🎯 Options": "OPTIONS"
            }
            trading_style = style_map[trading_style_display]
        
        with col3:
            st.write("")
            st.write("")
            analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
        
        # Quick examples with style descriptions
        st.caption("**Examples:** AAPL (blue chip) | SNDL (penny stock) | SPY (ETF) | TSLA (volatile) | Any OTC stock")
        
        # Dynamic caption based on selected style
        style_descriptions = {
            "DAY_TRADE": "💡 **Day Trade:** Intraday equity trades, exit by market close (0.5-3% targets)",
            "SWING_TRADE": "💡 **Swing Trade:** Multi-day equity holds, 3-10 day timeframe (5-15% targets)",
            "SCALP": "💡 **Scalp:** Ultra-short term, seconds to minutes (0.1-0.5% targets, high risk)",
            "BUY_HOLD": "💡 **Buy & Hold:** Long-term investing, 6+ months (20%+ annual targets)",
            "OPTIONS": "💡 **Options:** Calls, puts, spreads based on IV and trend analysis"
        }
        st.caption(style_descriptions[trading_style])
        
        # Quick Trade Modal - AT TOP so it's immediately visible when Execute button is clicked
        logger.info(f"🔍 Checking modal display: show_quick_trade={st.session_state.get('show_quick_trade', False)}")
        if st.session_state.get('show_quick_trade', False):
            logger.info("🚀 DISPLAYING QUICK TRADE MODAL AT TOP OF TAB1")
            st.divider()
            st.header("🚀 Execute Trade")
            
            # Get the selected recommendation and analysis
            selected_rec = st.session_state.get('selected_recommendation', None)
            analysis = st.session_state.get('current_analysis', None)
            
            if not analysis:
                logger.error("❌ Modal error: No analysis data in session state")
                st.error("❌ Analysis data not available. Please analyze a stock first.")
                if st.button("Close"):
                    st.session_state.show_quick_trade = False
                    st.rerun()
            else:
                logger.info(f"✅ Modal has analysis data: ticker={analysis.ticker}, price={analysis.price}")
                if selected_rec:
                    logger.info(f"✅ Modal has recommendation: {selected_rec.get('type')} - {selected_rec.get('strategy', 'N/A')}")
                    st.subheader(f"📋 {selected_rec['type']} - {selected_rec.get('strategy', selected_rec.get('action', ''))}")
                else:
                    st.subheader(f"📋 Quick Trade: {st.session_state.get('quick_trade_ticker', 'N/A')}")
                
                # Check if Tradier is connected
                if not st.session_state.tradier_client:
                    st.error("❌ Tradier not connected. Please configure in the 🏦 Tradier Account tab.")
                    if st.button("Close", key="close_no_tradier"):
                        st.session_state.show_quick_trade = False
                        st.rerun()
                else:
                    verdict_action = st.session_state.get('quick_trade_verdict', 'N/A')
                    st.success(f"✅ Tradier Connected | Verdict: **{verdict_action}**")
                    
                    # Show AI recommendation details if available
                    if selected_rec:
                        st.info(f"**AI Reasoning:** {selected_rec.get('reasoning', 'N/A')}")
                        if selected_rec['type'] == 'STOCK':
                            st.caption(f"Stop Loss: ${selected_rec['stop_loss']:.2f} | Target: ${selected_rec['target']:.2f} | Hold: {selected_rec['hold_time']}")
                        else:
                            st.caption(f"Strike: {selected_rec.get('strike_suggestion', 'N/A')} | DTE: {selected_rec.get('dte_suggestion', 'N/A')}")
                    
                    trade_col1, trade_col2 = st.columns(2)
                    
                    with trade_col1:
                        st.write("**Order Configuration:**")
                        
                        # Pre-fill based on recommendation
                        if selected_rec and selected_rec['type'] == 'STOCK':
                            default_symbol = selected_rec['symbol']
                            default_action = selected_rec['action'].lower().replace('_', '_')
                            default_qty = 10
                            default_type = selected_rec['order_type']
                            default_price = selected_rec.get('price', st.session_state.get('quick_trade_price', analysis.price))
                        elif selected_rec and selected_rec['type'] == 'OPTION':
                            default_symbol = selected_rec['symbol']
                            default_action = selected_rec['action']
                            default_qty = selected_rec.get('quantity', 1)
                            default_type = "limit"
                            default_price = st.session_state.get('quick_trade_price', analysis.price)
                        else:
                            default_symbol = st.session_state.get('quick_trade_ticker', analysis.ticker)
                            default_action = "buy"
                            default_qty = 10
                            default_type = "market"
                            default_price = st.session_state.get('quick_trade_price', analysis.price)
                        
                        trade_symbol = st.text_input("Symbol", value=default_symbol, key="modal_trade_symbol")
                        
                        # Determine if this is an options trade
                        is_options_trade = selected_rec and selected_rec['type'] == 'OPTION'
                        
                        if is_options_trade:
                            st.warning("⚠️ **Options Trade:** You'll need to specify the exact option symbol (e.g., AAPL250117C150)")
                            trade_class = st.selectbox("Order Class", ["option", "equity"], index=0, key="modal_trade_class")
                            
                            if trade_class == "option":
                                st.info(f"💡 **Suggested Strike:** {selected_rec.get('strike_suggestion', 'N/A')}")
                                st.info(f"💡 **Suggested Expiration:** {selected_rec.get('dte_suggestion', 'N/A')}")
                                trade_side = st.selectbox("Action", 
                                                        ["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"],
                                                        index=0 if 'buy' in default_action else 1,
                                                        key="modal_trade_side")
                                trade_quantity = st.number_input("Contracts", min_value=1, value=default_qty, step=1, key="modal_trade_qty")
                            else:
                                trade_side = st.selectbox("Action", ["buy", "sell", "sell_short", "buy_to_cover"], key="modal_trade_side2")
                                trade_quantity = st.number_input("Quantity (shares)", min_value=1, value=default_qty, step=1, key="modal_trade_qty2")
                        else:
                            trade_class = "equity"
                            if default_action == "SELL_SHORT":
                                side_index = 2
                            elif default_action == "BUY":
                                side_index = 0
                            else:
                                side_index = 0
                            
                            trade_side = st.selectbox("Action", 
                                                    ["buy", "sell", "sell_short", "buy_to_cover"],
                                                    index=side_index,
                                                    key="modal_trade_side3")
                            trade_quantity = st.number_input("Quantity (shares)", min_value=1, value=default_qty, step=1, key="modal_trade_qty3")
                        
                        # Bracket order eligibility check (for display purposes)
                        can_use_bracket = (trade_class == "equity" and trade_side in ["buy", "sell"])
                        
                        trade_type = st.selectbox("Order Type", 
                                                ["market", "limit"],
                                                index=0 if default_type == "market" else 1,
                                                key="modal_trade_type",
                                                help="💡 Select 'limit' to enable automatic bracket orders with stop-loss & take-profit")
                        
                        if trade_type == "limit":
                            trade_price = st.number_input("Limit Price", 
                                                         min_value=0.01, 
                                                         value=float(default_price) if default_price else float(analysis.price),
                                                         step=0.01,
                                                         format="%.2f",
                                                         key="modal_trade_price")
                            
                            # Show bracket order preview
                            if can_use_bracket:
                                # Get stop/target for preview
                                if selected_rec and selected_rec['type'] == 'STOCK':
                                    preview_stop = selected_rec['stop_loss']
                                    preview_target = selected_rec['target']
                                else:
                                    preview_stop = analysis.support
                                    preview_target = analysis.resistance
                                
                                # Validate and adjust
                                if trade_side == "buy":
                                    if preview_stop >= trade_price:
                                        preview_stop = trade_price * 0.97
                                    if preview_target <= trade_price:
                                        preview_target = trade_price * 1.05
                                else:
                                    if preview_stop <= trade_price:
                                        preview_stop = trade_price * 1.03
                                    if preview_target >= trade_price:
                                        preview_target = trade_price * 0.95
                                
                                st.success(f"🎯 **BRACKET ORDER ACTIVE**")
                                st.info(f"✅ Entry: ${trade_price:.2f} | 🎯 Target: ${preview_target:.2f} | 🛑 Stop: ${preview_stop:.2f}")
                            else:
                                st.info(f"Limit order will execute when price reaches ${trade_price:.2f}")
                        else:
                            trade_price = None
                            st.warning(f"⚠️ Market orders execute immediately - **bracket orders NOT available**")
                            st.info(f"💡 To enable automatic stop-loss & take-profit, change to 'limit' order type")
                    
                    with trade_col2:
                        st.write("**Order Summary:**")
                        
                        # Show bracket order mode indicator
                        will_use_bracket = (
                            trade_class == "equity" and 
                            trade_type == "limit" and 
                            trade_side in ["buy", "sell"] and
                            trade_price is not None
                        )
                        if will_use_bracket:
                            st.success("🎯 **BRACKET MODE**: Auto stop-loss & take-profit enabled")
                        else:
                            st.info("📊 **SIMPLE ORDER MODE**")
                        
                        st.divider()
                        
                        # Calculate estimated cost
                        if is_options_trade:
                            st.warning("Options pricing requires real-time quote - estimate not available")
                            estimated_cost = "TBD"
                        else:
                            if trade_type == "limit" and trade_price:
                                estimated_cost = trade_price * trade_quantity
                            else:
                                estimated_cost = analysis.price * trade_quantity
                            st.metric("Estimated Cost", f"${estimated_cost:,.2f}")
                        
                        st.metric("Verdict", verdict_action)
                        
                        if selected_rec:
                            st.metric("AI Confidence", f"{selected_rec['confidence']:.0f}/100")
                        
                        # Risk warning based on verdict
                        if verdict_action in ["AVOID / WAIT", "CAUTIOUS BUY"]:
                            st.warning("⚠️ Analysis suggests caution with this trade!")
                        elif verdict_action == "STRONG BUY":
                            st.success("✅ Analysis shows strong confidence!")
                        
                        if selected_rec and selected_rec['type'] == 'STOCK':
                            st.caption(f"**Stop Loss:** ${selected_rec['stop_loss']:.2f}")
                            st.caption(f"**Target:** ${selected_rec['target']:.2f}")
                        elif selected_rec and selected_rec['type'] == 'OPTION':
                            st.caption(f"**Max Profit:** {selected_rec.get('max_profit', 'N/A')}")
                            st.caption(f"**Max Risk:** {selected_rec.get('max_risk', 'N/A')}")
                        else:
                            st.caption(f"**Stop Loss Suggestion:** ${analysis.support:.2f}")
                            st.caption(f"**Target Suggestion:** ${analysis.resistance:.2f}")
                    
                    # Place order button
                    st.write("")
                    confirm_col1, confirm_col2 = st.columns(2)
                    
                    with confirm_col1:
                        if st.button("✅ Place Order", type="primary", use_container_width=True, key="modal_place_order"):
                            with st.spinner("Placing order..."):
                                try:
                                    # Determine if we can use bracket orders
                                    # Bracket orders require: equity class, limit entry, buy/sell side, and stop/target prices
                                    use_bracket = (
                                        trade_class == "equity" and 
                                        trade_type == "limit" and 
                                        trade_side in ["buy", "sell"] and
                                        trade_price is not None
                                    )
                                    
                                    if use_bracket:
                                        # Get stop loss and target prices
                                        if selected_rec and selected_rec['type'] == 'STOCK':
                                            stop_loss = selected_rec['stop_loss']
                                            target = selected_rec['target']
                                        else:
                                            # Use technical support/resistance levels
                                            stop_loss = analysis.support
                                            target = analysis.resistance
                                        
                                        # Validate that stop/target make sense for the order direction
                                        if trade_side == "buy":
                                            # For buy orders: stop should be below entry, target above
                                            if stop_loss >= trade_price:
                                                stop_loss = round(trade_price * 0.97, 2)  # Default 3% stop
                                            if target <= trade_price:
                                                target = round(trade_price * 1.05, 2)  # Default 5% target
                                        else:
                                            # For sell orders: stop should be above entry, target below
                                            if stop_loss <= trade_price:
                                                stop_loss = round(trade_price * 1.03, 2)
                                            if target >= trade_price:
                                                target = round(trade_price * 0.95, 2)
                                        logger.info(f"🎯 Placing bracket order: {trade_symbol} {trade_side} {trade_quantity} @ ${trade_price} (SL: ${stop_loss:.2f}, Target: ${target:.2f})")
                                        
                                        success, result = st.session_state.tradier_client.place_bracket_order(
                                            symbol=trade_symbol.upper(),
                                            side=trade_side,
                                            quantity=trade_quantity,
                                            entry_price=trade_price,
                                            take_profit_price=target,
                                            stop_loss_price=stop_loss,
                                            duration='gtc',  # Use GTC for bracket orders
                                            tag=f"AIREC{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                        )
                                    else:
                                        # Fallback to regular order for market orders or options
                                        order_data = {
                                            "class": trade_class,
                                            "symbol": trade_symbol.upper(),
                                            "side": trade_side,
                                            "quantity": str(trade_quantity),
                                            "type": trade_type,
                                            "duration": "day",
                                            "tag": f"AIREC{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                        }
                                        
                                        if trade_type == "limit" and trade_price:
                                            order_data["price"] = str(trade_price)
                                        
                                        # Explain why bracket wasn't used
                                        reason = "market order" if trade_type == "market" else "options trade" if trade_class != "equity" else "non-standard side"
                                        logger.info(f"🚀 Placing REGULAR order ({reason}): {trade_symbol} {trade_side} {trade_quantity} @ {trade_type}")
                                        success, result = st.session_state.tradier_client.place_order(order_data)
                                    
                                    if success:
                                        order_id = result.get('order', {}).get('id', 'Unknown')
                                        if use_bracket:
                                            st.success(f"🎉 Bracket order placed successfully! Order ID: {order_id}")
                                            st.info(f"✅ Entry: ${trade_price} | 🎯 Target: ${target:.2f} | 🛑 Stop: ${stop_loss:.2f}")
                                        else:
                                            st.success(f"🎉 Order placed successfully! Order ID: {order_id}")
                                        st.json(result)
                                        
                                        # Log the trade
                                        logger.info(f"AI recommendation executed: {trade_symbol} {trade_side} {trade_quantity} @ {trade_type}")
                                        
                                        # Clear the modal after successful order
                                        if st.button("Close & Refresh", key="close_success"):
                                            st.session_state.show_quick_trade = False
                                            st.session_state.selected_recommendation = None
                                            st.rerun()
                                    else:
                                        st.error(f"❌ Order failed: {result.get('error', 'Unknown error')}")
                                        st.json(result)
                                except Exception as e:
                                    st.error(f"❌ Error placing order: {str(e)}")
                                    logger.error(f"Quick trade error: {e}", exc_info=True)
                    
                    with confirm_col2:
                        if st.button("❌ Cancel", use_container_width=True, key="modal_cancel"):
                            st.session_state.show_quick_trade = False
                            st.session_state.selected_recommendation = None
                            st.rerun()
            st.divider()
        
        if analyze_btn and search_ticker:
            # Use new st.status for better progress indication
            with st.status(f"🔍 Analyzing {search_ticker}...", expanded=True) as status:
                st.write("📊 Fetching market data...")
                time.sleep(0.5)  # Simulate processing time
                
                st.write("📈 Calculating technical indicators...")
                time.sleep(0.5)
                
                st.write("📰 Analyzing news sentiment...")
                time.sleep(0.5)
                
                st.write("🎯 Identifying catalysts...")
                time.sleep(0.5)
                
                st.write(f"🤖 Generating {trading_style_display} recommendations...")
                analysis = ComprehensiveAnalyzer.analyze_stock(search_ticker, trading_style)
                
                if analysis:
                    status.update(label=f"✅ Analysis complete for {search_ticker}", state="complete")
                else:
                    status.update(label=f"❌ Analysis failed for {search_ticker}", state="error")
                
                if analysis:
                    logger.info(f"💾 Storing analysis in session state: {analysis.ticker} @ ${analysis.price:.2f}")
                    st.session_state.current_analysis = analysis
                    logger.info(f"✅ Analysis stored. Quick trade flag status: {st.session_state.get('show_quick_trade', False)}")
                    
                    # Detect penny stock and runner characteristics
                    is_penny_stock = analysis.price < 5.0
                    is_otc = analysis.ticker.endswith(('.OTC', '.PK', '.QB'))
                    volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
                    is_runner = volume_vs_avg > 200 and analysis.change_pct > 10  # 200%+ volume spike and 10%+ gain
                    
                    # Header metrics
                    st.success(f"✅ Analysis complete for {analysis.ticker}")
                    
                    # Special alerts for penny stocks and runners
                    if is_runner:
                        st.warning(f"🚀 **RUNNER DETECTED!** {volume_vs_avg:+.0f}% volume spike with {analysis.change_pct:+.1f}% price move!")
                    
                    if is_penny_stock:
                        st.info(f"💰 **PENNY STOCK** (${analysis.price:.4f}) - High risk/high reward. Use caution and proper position sizing.")
                    
                    if is_otc:
                        st.warning("⚠️ **OTC STOCK** - Lower liquidity, wider spreads, higher risk. Limited data may be available.")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    
                    with metric_col1:
                        price_display = f"${analysis.price:.4f}" if is_penny_stock else f"${analysis.price:.2f}"
                        st.metric("Price", price_display, f"{analysis.change_pct:+.2f}%")
                    with metric_col2:
                        st.metric("Trend", analysis.trend)
                    with metric_col3:
                        st.metric("Confidence", f"{int(analysis.confidence_score)}%")
                    with metric_col4:
                        st.metric("IV Rank", f"{analysis.iv_rank}%")
                    with metric_col5:
                        volume_indicator = "🔥" if volume_vs_avg > 100 else "📊"
                        st.metric(f"{volume_indicator} Volume", f"{analysis.volume:,}", f"{volume_vs_avg:+.1f}%")
                    
                    # Runner Metrics (if detected)
                    if is_runner or volume_vs_avg > 100:
                        st.subheader("🚀 Runner / Momentum Metrics")
                        
                        runner_col1, runner_col2, runner_col3, runner_col4 = st.columns(4)
                        
                        with runner_col1:
                            st.metric("Volume Spike", f"{volume_vs_avg:+.0f}%")
                            if volume_vs_avg > 300:
                                st.caption("🔥 EXTREME volume!")
                            elif volume_vs_avg > 200:
                                st.caption("🔥 Very high volume")
                            else:
                                st.caption("📈 Elevated volume")
                        
                        with runner_col2:
                            st.metric("Price Change", f"{analysis.change_pct:+.2f}%")
                            if abs(analysis.change_pct) > 20:
                                st.caption("🚀 Major move!")
                            elif abs(analysis.change_pct) > 10:
                                st.caption("📈 Strong move")
                        
                        with runner_col3:
                            # Calculate momentum score
                            momentum_score = min(100, (abs(analysis.change_pct) * 2 + volume_vs_avg / 5))
                            st.metric("Momentum Score", f"{momentum_score:.0f}/100")
                            if momentum_score > 80:
                                st.caption("🔥 HOT!")
                            elif momentum_score > 60:
                                st.caption("🔥 Strong")
                        
                        with runner_col4:
                            # Risk level for runners
                            runner_risk = "EXTREME" if is_penny_stock and volume_vs_avg > 300 else "VERY HIGH" if volume_vs_avg > 200 else "HIGH"
                            st.metric("Runner Risk", runner_risk)
                            st.caption("⚠️ Use stops!")
                        
                        if is_runner:
                            st.warning("""
**Runner Trading Tips:**
- ✅ Use tight stop losses (3-5%)
- ✅ Take profits quickly (don't be greedy)
- ✅ Watch for volume decline (exit signal)
- ✅ Avoid chasing - wait for pullbacks
- ❌ Don't hold overnight (high gap risk)
                            """)
                    
                    # Technical Indicators
                    st.subheader("📊 Technical Indicators")
                    
                    tech_col1, tech_col2, tech_col3 = st.columns(3)
                    
                    with tech_col1:
                        st.metric("RSI (14)", f"{analysis.rsi:.1f}")
                        if analysis.rsi < 30:
                            st.caption("🟢 Oversold - potential buy")
                        elif analysis.rsi > 70:
                            st.caption("🔴 Overbought - potential sell")
                        else:
                            st.caption("🟡 Neutral")
                    
                    with tech_col2:
                        st.metric("MACD Signal", analysis.macd_signal)
                        if analysis.macd_signal == "BULLISH":
                            st.caption("🟢 Bullish momentum")
                        elif analysis.macd_signal == "BEARISH":
                            st.caption("🔴 Bearish momentum")
                        else:
                            st.caption("🟡 Neutral momentum")
                    
                    with tech_col3:
                        st.metric("Support", f"${analysis.support}")
                        st.metric("Resistance", f"${analysis.resistance}")
                    
                    # IV Analysis
                    st.subheader("📈 Implied Volatility Analysis")
                    
                    iv_col1, iv_col2, iv_col3 = st.columns(3)
                    
                    with iv_col1:
                        st.metric("IV Rank", f"{analysis.iv_rank}%")
                        if analysis.iv_rank > 60:
                            st.caption("🔥 High IV - Great for selling premium")
                        elif analysis.iv_rank < 40:
                            st.caption("❄️ Low IV - Good for buying options")
                        else:
                            st.caption("➡️ Moderate IV")
                    
                    with iv_col2:
                        st.metric("IV Percentile", f"{analysis.iv_percentile}%")
                    
                    with iv_col3:
                        if analysis.iv_rank > 50:
                            st.info("💡 Consider: Selling puts, covered calls, iron condors")
                        else:
                            st.info("💡 Consider: Buying calls/puts, debit spreads")
                    
                    # Catalysts
                    st.subheader("📅 Upcoming Catalysts")
                    
                    if analysis.catalysts:
                        for catalyst in analysis.catalysts:
                            impact_color = {
                                'HIGH': '🔴',
                                'MEDIUM': '🟡',
                                'LOW': '🟢'
                            }.get(catalyst['impact'], '⚪')
                            
                            with st.expander(f"{impact_color} {catalyst['type']} - {catalyst['date']} ({catalyst.get('days_away', 'N/A')} days away)"):
                                st.write(f"**Impact Level:** {catalyst['impact']}")
                                st.write(f"**Details:** {catalyst['description']}")
                                
                                if catalyst['type'] == 'Earnings Report' and catalyst.get('days_away', 999) <= 7:
                                    st.warning("⚠️ Earnings within 7 days - expect high volatility!")
                    else:
                        st.info("No major catalysts identified in the next 60 days")
                    
                    # News & Sentiment
                    st.subheader("📰 Recent News & Sentiment")
                    
                    # Add refresh button for news
                    col_refresh, col_info = st.columns([1, 4])
                    with col_refresh:
                        if st.button("🔄 Refresh News", help="Get the latest news and sentiment"):
                            # Clear cache for this ticker
                            get_cached_news.clear()
                            st.rerun()
                    
                    with col_info:
                        if analysis.recent_news:
                            st.success(f"✅ Found {len(analysis.recent_news)} recent news articles")
                        else:
                            st.warning("⚠️ No recent news found - this may indicate low news volume or connectivity issues")
                    
                    sentiment_col1, sentiment_col2 = st.columns([1, 3])
                    
                    with sentiment_col1:
                        sentiment_label = "POSITIVE" if analysis.sentiment_score > 0.2 else "NEGATIVE" if analysis.sentiment_score < -0.2 else "NEUTRAL"
                        sentiment_color = "🟢" if analysis.sentiment_score > 0.2 else "🔴" if analysis.sentiment_score < -0.2 else "🟡"
                        
                        st.metric("News Sentiment", f"{sentiment_color} {sentiment_label}")
                        st.metric("Sentiment Score", f"{analysis.sentiment_score:.2f}")
                        
                        # Show sentiment signals if available
                        if hasattr(analysis, 'sentiment_signals') and analysis.sentiment_signals:
                            with st.expander("📊 Sentiment Analysis Details"):
                                for signal in analysis.sentiment_signals[:3]:  # Show top 3
                                    st.write(signal)
                    
                    with sentiment_col2:
                        if analysis.recent_news:
                            st.write("**Latest News Articles:**")
                            for idx, article in enumerate(analysis.recent_news[:5]):
                                # Create a more informative expander
                                expander_title = f"📰 {article['title'][:70]}..." if len(article['title']) > 70 else f"📰 {article['title']}"
                                
                                with st.expander(expander_title):
                                    col_pub, col_time = st.columns(2)
                                    with col_pub:
                                        st.write(f"**Publisher:** {article['publisher']}")
                                    with col_time:
                                        st.write(f"**Published:** {article['published']}")
                                    
                                    # Show summary if available
                                    if article.get('summary'):
                                        st.write("**Summary:**")
                                        st.write(article['summary'])
                                    
                                    # Link to full article
                                    if article.get('link'):
                                        st.write(f"[📖 Read Full Article]({article['link']})")
                                    
                                    # Show article type
                                    if article.get('type'):
                                        st.caption(f"Type: {article['type']}")
                        else:
                            st.info("📭 No recent news found for this ticker. This could be due to:")
                            st.write("• Low news volume for this stock")
                            st.write("• Temporary connectivity issues")
                            st.write("• Yahoo Finance API limitations")
                            st.write("• Try refreshing the news or check back later")
                    
                    # Penny Stock Risk Assessment (if applicable)
                    if is_penny_stock:
                        st.subheader("⚠️ Penny Stock Risk Assessment")
                        
                        risk_col1, risk_col2 = st.columns(2)
                        
                        with risk_col1:
                            st.warning("""
**Penny Stock Risks:**
- 🔴 High volatility (can swing 20-50%+ daily)
- 🔴 Low liquidity (harder to exit positions)
- 🔴 Wide bid-ask spreads (higher trading costs)
- 🔴 Manipulation risk (pump & dump schemes)
- 🔴 Limited financial data/transparency
- 🔴 Higher bankruptcy risk
                            """)
                        
                        with risk_col2:
                            st.success("""
**Penny Stock Trading Rules:**
- ✅ Never risk more than 1-2% of portfolio
- ✅ Use limit orders (avoid market orders)
- ✅ Set tight stop losses (5-10%)
- ✅ Take profits quickly (don't be greedy)
- ✅ Research company fundamentals
- ✅ Watch for unusual volume spikes
- ✅ Avoid stocks with no news/catalysts
                            """)
                        
                        # Calculate penny stock score
                        penny_score = 0
                        if volume_vs_avg > 100: penny_score += 25
                        if analysis.change_pct > 5: penny_score += 20
                        if analysis.rsi < 70: penny_score += 20
                        if len(analysis.recent_news) > 0: penny_score += 20
                        if analysis.sentiment_score > 0: penny_score += 15
                        
                        st.metric("Penny Stock Opportunity Score", f"{penny_score}/100")
                        
                        if penny_score > 70:
                            st.success("🟢 Strong opportunity - but still use caution!")
                        elif penny_score > 50:
                            st.info("🟡 Moderate opportunity - proceed carefully")
                        else:
                            st.warning("🔴 Weak setup - consider waiting for better entry")
                    
                    # Timeframe-Specific Analysis
                    st.subheader(f"⏰ {trading_style_display} Analysis")
                    
                    # Calculate timeframe-specific metrics
                    if trading_style == "DAY_TRADE":
                        # Day trading focus: quick moves, tight stops
                        timeframe_score = 0
                        reasons = []
                        
                        if volume_vs_avg > 100:
                            timeframe_score += 30
                            reasons.append(f"✅ High volume (+{volume_vs_avg:.0f}%) - good for day trading")
                        else:
                            reasons.append(f"⚠️ Volume only +{volume_vs_avg:.0f}% - may lack intraday momentum")
                        
                        if abs(analysis.change_pct) > 2:
                            timeframe_score += 25
                            reasons.append(f"✅ Strong intraday move ({analysis.change_pct:+.1f}%)")
                        else:
                            reasons.append("⚠️ Low intraday volatility - limited profit potential")
                        
                        if 30 < analysis.rsi < 70:
                            timeframe_score += 20
                            reasons.append("✅ RSI in tradeable range (not overbought/oversold)")
                        
                        if not is_penny_stock:
                            timeframe_score += 15
                            reasons.append("✅ Not a penny stock - better liquidity for day trading")
                        else:
                            reasons.append("⚠️ Penny stock - higher risk, use smaller size")
                        
                        if analysis.trend != "NEUTRAL":
                            timeframe_score += 10
                            reasons.append(f"✅ Clear trend ({analysis.trend}) - easier to trade")
                        
                        st.metric("Day Trading Suitability", f"{timeframe_score}/100")
                        
                        for reason in reasons:
                            st.write(reason)
                        
                        if timeframe_score > 70:
                            st.success("🟢 **EXCELLENT** for day trading - strong setup!")
                        elif timeframe_score > 50:
                            st.info("🟡 **GOOD** for day trading - proceed with caution")
                        else:
                            st.warning("🔴 **POOR** for day trading - consider swing/position trading instead")
                        
                        st.write("**Day Trading Strategy:**")
                        st.write(f"• 🎯 Entry: ${analysis.price:.2f}")
                        st.write(f"• 🛑 Stop: ${analysis.support:.2f} (support level)")
                        st.write(f"• 💰 Target: ${analysis.resistance:.2f} (resistance level)")
                        st.write(f"• ⏰ Hold time: Minutes to hours (close before market close)")
                        st.write(f"• 📊 Watch: Volume, L2 order book, momentum")
                    
                    elif trading_style == "SWING_TRADE":
                        # Swing trading focus: multi-day moves, catalysts
                        timeframe_score = 0
                        reasons = []
                        
                        if len(analysis.catalysts) > 0:
                            timeframe_score += 30
                            reasons.append(f"✅ {len(analysis.catalysts)} upcoming catalyst(s) - potential multi-day move")
                        else:
                            reasons.append("⚠️ No near-term catalysts - may lack swing momentum")
                        
                        if analysis.trend != "NEUTRAL":
                            timeframe_score += 25
                            reasons.append(f"✅ Strong {analysis.trend} trend - good for swing trading")
                        
                        if analysis.sentiment_score > 0.2:
                            timeframe_score += 20
                            reasons.append(f"✅ Positive sentiment ({analysis.sentiment_score:.2f}) - bullish setup")
                        elif analysis.sentiment_score < -0.2:
                            timeframe_score += 15
                            reasons.append(f"✅ Negative sentiment ({analysis.sentiment_score:.2f}) - bearish setup")
                        
                        if len(analysis.recent_news) > 3:
                            timeframe_score += 15
                            reasons.append(f"✅ Active news flow ({len(analysis.recent_news)} articles) - sustained interest")
                        
                        if not is_penny_stock or (is_penny_stock and volume_vs_avg > 200):
                            timeframe_score += 10
                            reasons.append("✅ Sufficient liquidity for swing trading")
                        else:
                            reasons.append("⚠️ Low liquidity - may be hard to exit position")
                        
                        st.metric("Swing Trading Suitability", f"{timeframe_score}/100")
                        
                        for reason in reasons:
                            st.write(reason)
                        
                        if timeframe_score > 70:
                            st.success("🟢 **EXCELLENT** for swing trading - strong multi-day setup!")
                        elif timeframe_score > 50:
                            st.info("🟡 **GOOD** for swing trading - monitor catalysts")
                        else:
                            st.warning("🔴 **POOR** for swing trading - better for day trading or long-term hold")
                        
                        st.write("**Swing Trading Strategy:**")
                        st.write(f"• 🎯 Entry: ${analysis.price:.2f} (current price)")
                        # Dynamic stop using 21 EMA if available
                        stop_val = None
                        if getattr(analysis, 'ema21', None):
                            try:
                                stop_val = float(analysis.ema21) * 0.99
                            except Exception:
                                stop_val = None
                        if stop_val is None:
                            stop_val = analysis.support * 0.95
                        st.write(f"• 🛑 Stop: ${stop_val:.2f} (below 21 EMA or support)")

                        # Fibonacci targets if present; fallback to resistance-based target
                        fib = getattr(analysis, 'fib_targets', None)
                        if isinstance(fib, dict) and fib.get('T1_1272'):
                            st.write("• 💰 Targets:")
                            st.write(f"   - T1 (127.2%): ${fib['T1_1272']:.2f} (take 25%)")
                            if fib.get('T2_1618'):
                                st.write(f"   - T2 (161.8%): ${fib['T2_1618']:.2f} (take 50%)")
                            last_t3 = fib.get('T3_2618') or fib.get('T3_200')
                            if last_t3:
                                st.write(f"   - T3 (200-261.8%): ${last_t3:.2f} (trail remaining)")
                            st.write("• 🧭 Move stop to breakeven after T1, trail below 21 EMA thereafter")
                        else:
                            st.write(f"• 💰 Target: ${analysis.resistance * 1.05:.2f} (5% above resistance)")

                        # Context badges
                        if getattr(analysis, 'ema_power_zone', None):
                            st.write("• ✅ 8>21 EMA Power Zone active")
                        if getattr(analysis, 'ema_reclaim', None):
                            st.write("• ✅ EMA Reclaim confirmed")
                        if getattr(analysis, 'demarker', None) is not None:
                            dem = float(analysis.demarker)
                            zone = "Neutral"
                            if dem <= 0.30:
                                zone = "Oversold"
                            elif dem >= 0.70:
                                zone = "Overbought"
                            st.write(f"• 📈 DeMarker(14): {dem:.2f} ({zone})")

                        st.write(f"• ⏰ Hold time: 2-10 days (watch for catalyst completion)")
                        st.write(f"• 📊 Watch: News, catalyst dates, trend continuation")
                        
                        if analysis.catalysts:
                            st.write("**Key Catalysts to Watch:**")
                            for cat in analysis.catalysts[:3]:
                                st.write(f"  • {cat['type']} on {cat['date']} ({cat.get('days_away', 'N/A')} days)")
                    
                    elif trading_style == "BUY_HOLD":  # Buy & Hold
                        # Position trading focus: fundamentals, long-term trends
                        timeframe_score = 0
                        reasons = []
                        
                        if analysis.trend == "BULLISH":
                            timeframe_score += 30
                            reasons.append("✅ Strong bullish trend - good for long-term hold")
                        elif analysis.trend == "BEARISH":
                            timeframe_score += 20
                            reasons.append("✅ Bearish trend - consider short or inverse position")
                        
                        if len(analysis.catalysts) > 2:
                            timeframe_score += 25
                            reasons.append(f"✅ Multiple catalysts ({len(analysis.catalysts)}) - sustained growth potential")
                        
                        if analysis.sentiment_score > 0.3:
                            timeframe_score += 20
                            reasons.append(f"✅ Very positive sentiment ({analysis.sentiment_score:.2f}) - market confidence")
                        
                        if not is_penny_stock:
                            timeframe_score += 15
                            reasons.append("✅ Established stock - lower bankruptcy risk")
                        else:
                            reasons.append("⚠️ Penny stock - very high risk for long-term hold")
                        
                        if analysis.iv_rank < 50:
                            timeframe_score += 10
                            reasons.append(f"✅ Low IV ({analysis.iv_rank}%) - less volatility risk")
                        
                        st.metric("Position Trading Suitability", f"{timeframe_score}/100")
                        
                        for reason in reasons:
                            st.write(reason)
                        
                        if timeframe_score > 70:
                            st.success("🟢 **EXCELLENT** for position trading - strong long-term hold!")
                        elif timeframe_score > 50:
                            st.info("🟡 **GOOD** for position trading - monitor fundamentals")
                        else:
                            st.warning("🔴 **POOR** for position trading - better for short-term trades")
                        
                        if is_penny_stock:
                            st.error("⚠️ **WARNING:** Penny stocks are extremely risky for long-term holds due to bankruptcy risk!")
                        
                        st.write("**Position Trading Strategy:**")
                        st.write(f"• 🎯 Entry: ${analysis.price:.2f} (current price or pullback)")
                        st.write(f"• 🛑 Stop: ${analysis.price * 0.85:.2f} (15% trailing stop)")
                        st.write(f"• 💰 Target: ${analysis.price * 1.30:.2f} (30%+ gain over time)")
                        st.write(f"• ⏰ Hold time: Weeks to months (review quarterly)")
                        st.write(f"• 📊 Watch: Earnings, fundamentals, sector trends, macro conditions")
                        
                        if not is_penny_stock:
                            st.info("💡 **Position Trading Tip:** Consider selling covered calls or cash-secured puts to generate income while holding.")
                    
                    # AI Recommendation
                    st.subheader(f"🤖 AI Trading Recommendation - {trading_style_display}")
                    
                    recommendation_box = st.container()
                    with recommendation_box:
                        # Add penny stock context to recommendation
                        if is_penny_stock and trading_style in ["BUY_HOLD", "SWING_TRADE"]:
                            st.warning("⚠️ **Penny Stock Alert:** High risk/high reward - use proper position sizing and tight stops")
                        
                        # Display recommendation with proper formatting
                        if trading_style == "OPTIONS":
                            st.info(f"**Options Strategy:**\n\n{analysis.recommendation}")
                        else:
                            # For equity strategies, use markdown for better formatting
                            st.markdown(f"**{trading_style_display} Strategy:**")
                            st.markdown(analysis.recommendation)
                    
                    # ML-Enhanced Confidence Analysis (MOVED UP - More Prominent)
                    st.subheader(f"🧠 ML-Enhanced Confidence Analysis for {trading_style_display}")
                    st.write(f"**Advanced multi-factor analysis** using 50+ alpha factors optimized for **{trading_style_display}** strategy.")
                    
                    # Calculate ML analysis BEFORE showing it
                    ml_analysis_available = False
                    alpha_factors = None
                    ml_prediction_score = 0
                    ml_confidence_level = "UNKNOWN"
                    ml_strategy_notes = []
                    
                    try:
                        alpha_calc = AlphaFactorCalculator()
                        alpha_factors = alpha_calc.calculate_factors(search_ticker)
                        
                        if alpha_factors:
                            ml_analysis_available = True
                            
                            # Extract key factors
                            momentum = alpha_factors.get('return_20d', 0) * 100
                            momentum_5d = alpha_factors.get('return_5d', 0) * 100
                            momentum_1d = alpha_factors.get('return_1d', 0) * 100
                            vol_ratio = alpha_factors.get('volume_5d_ratio', 1)
                            rsi = alpha_factors.get('rsi_14', 50)
                            volatility = alpha_factors.get('volatility_20d', 0) * 100
                            macd = alpha_factors.get('macd', 0)
                            macd_signal = alpha_factors.get('macd_signal', 0)
                            
                            # TRADING STYLE SPECIFIC ML SCORING
                            ml_score = 50  # baseline
                            
                            if trading_style == "DAY_TRADE":
                                # Day Trading: Focus on intraday momentum, volume, and volatility
                                st.caption("🎯 ML optimized for intraday moves and quick profits")
                                
                                # Intraday momentum (35%)
                                if momentum_1d > 2:
                                    ml_score += 20
                                    ml_strategy_notes.append(f"✅ Strong intraday momentum (+{momentum_1d:.1f}%)")
                                elif momentum_1d > 1:
                                    ml_score += 12
                                    ml_strategy_notes.append(f"✅ Good intraday momentum (+{momentum_1d:.1f}%)")
                                elif momentum_1d < -2:
                                    ml_score -= 15
                                    ml_strategy_notes.append(f"⚠️ Negative intraday momentum ({momentum_1d:.1f}%)")
                                
                                # Volume is critical for day trading (30%)
                                if vol_ratio > 2.0:
                                    ml_score += 20
                                    ml_strategy_notes.append(f"✅ Exceptional volume ({vol_ratio:.1f}x avg)")
                                elif vol_ratio > 1.5:
                                    ml_score += 15
                                    ml_strategy_notes.append(f"✅ High volume ({vol_ratio:.1f}x avg)")
                                elif vol_ratio < 0.8:
                                    ml_score -= 15
                                    ml_strategy_notes.append(f"⚠️ Low volume ({vol_ratio:.1f}x avg)")
                                
                                # Volatility is good for day trading (20%)
                                if 2 < volatility < 5:
                                    ml_score += 12
                                    ml_strategy_notes.append(f"✅ Good volatility for day trading ({volatility:.1f}%)")
                                elif volatility > 5:
                                    ml_score += 8
                                    ml_strategy_notes.append(f"⚡ High volatility - use tight stops ({volatility:.1f}%)")
                                elif volatility < 1:
                                    ml_score -= 10
                                    ml_strategy_notes.append(f"⚠️ Low volatility - limited profit potential ({volatility:.1f}%)")
                                
                                # RSI for entry timing (15%)
                                if 30 < rsi < 70:
                                    ml_score += 8
                                    ml_strategy_notes.append(f"✅ RSI in tradeable range ({rsi:.0f})")
                                elif rsi < 30:
                                    ml_score += 5
                                    ml_strategy_notes.append(f"🟢 Oversold - bounce opportunity (RSI {rsi:.0f})")
                                elif rsi > 70:
                                    ml_score -= 5
                                    ml_strategy_notes.append(f"🔴 Overbought - reversal risk (RSI {rsi:.0f})")
                            
                            elif trading_style == "SWING_TRADE":
                                # Swing Trading: Focus on multi-day trends and momentum
                                st.caption("🎯 ML optimized for 3-10 day holds and trend continuation")
                                
                                # Multi-day momentum (35%)
                                if momentum_5d > 5:
                                    ml_score += 20
                                    ml_strategy_notes.append(f"✅ Strong 5-day momentum (+{momentum_5d:.1f}%)")
                                elif momentum_5d > 2:
                                    ml_score += 12
                                    ml_strategy_notes.append(f"✅ Good 5-day momentum (+{momentum_5d:.1f}%)")
                                elif momentum_5d < -5:
                                    ml_score -= 15
                                    ml_strategy_notes.append(f"⚠️ Negative 5-day trend ({momentum_5d:.1f}%)")
                                
                                # Trend consistency (25%)
                                if analysis.trend in ["STRONG UPTREND", "UPTREND"] and momentum > 0:
                                    ml_score += 15
                                    ml_strategy_notes.append(f"✅ Consistent uptrend (20d: +{momentum:.1f}%)")
                                elif analysis.trend in ["STRONG DOWNTREND", "DOWNTREND"] and momentum < 0:
                                    ml_score += 10
                                    ml_strategy_notes.append(f"✅ Consistent downtrend (short opportunity)")
                                
                                # Volume confirmation (20%)
                                if vol_ratio > 1.3:
                                    ml_score += 12
                                    ml_strategy_notes.append(f"✅ Volume supports swing ({vol_ratio:.1f}x avg)")
                                elif vol_ratio < 0.7:
                                    ml_score -= 10
                                    ml_strategy_notes.append(f"⚠️ Weak volume for swing ({vol_ratio:.1f}x avg)")
                                
                                # RSI for swing entries (20%)
                                if analysis.trend in ["UPTREND", "STRONG UPTREND"] and rsi < 50:
                                    ml_score += 12
                                    ml_strategy_notes.append(f"✅ Pullback in uptrend (RSI {rsi:.0f})")
                                elif rsi < 30:
                                    ml_score += 8
                                    ml_strategy_notes.append(f"🟢 Oversold - reversal setup (RSI {rsi:.0f})")
                            
                            elif trading_style == "SCALP":
                                # Scalping: Ultra-short term, high volume, tight spreads
                                st.caption("🎯 ML optimized for seconds-to-minutes holds")
                                
                                # Extreme intraday momentum (40%)
                                if abs(momentum_1d) > 3:
                                    ml_score += 25
                                    ml_strategy_notes.append(f"✅ Extreme momentum for scalping ({momentum_1d:+.1f}%)")
                                elif abs(momentum_1d) > 1.5:
                                    ml_score += 15
                                    ml_strategy_notes.append(f"✅ Good scalp momentum ({momentum_1d:+.1f}%)")
                                else:
                                    ml_score -= 20
                                    ml_strategy_notes.append(f"⚠️ Insufficient momentum for scalping ({momentum_1d:+.1f}%)")
                                
                                # Volume is CRITICAL for scalping (35%)
                                if vol_ratio > 3.0:
                                    ml_score += 25
                                    ml_strategy_notes.append(f"✅ Exceptional liquidity ({vol_ratio:.1f}x avg)")
                                elif vol_ratio > 2.0:
                                    ml_score += 18
                                    ml_strategy_notes.append(f"✅ High liquidity ({vol_ratio:.1f}x avg)")
                                elif vol_ratio < 1.5:
                                    ml_score -= 25
                                    ml_strategy_notes.append(f"❌ Insufficient volume for scalping ({vol_ratio:.1f}x avg)")
                                
                                # High volatility needed (25%)
                                if volatility > 4:
                                    ml_score += 15
                                    ml_strategy_notes.append(f"✅ High volatility for scalps ({volatility:.1f}%)")
                                elif volatility < 2:
                                    ml_score -= 15
                                    ml_strategy_notes.append(f"⚠️ Low volatility - limited scalp range ({volatility:.1f}%)")
                            
                            elif trading_style == "BUY_HOLD":
                                # Buy & Hold: Focus on long-term trends and stability
                                st.caption("🎯 ML optimized for 6+ month holds and fundamental strength")
                                
                                # Long-term trend (40%)
                                if momentum > 15:
                                    ml_score += 25
                                    ml_strategy_notes.append(f"✅ Strong long-term uptrend (+{momentum:.1f}%)")
                                elif momentum > 8:
                                    ml_score += 18
                                    ml_strategy_notes.append(f"✅ Good long-term trend (+{momentum:.1f}%)")
                                elif momentum < -10:
                                    ml_score -= 20
                                    ml_strategy_notes.append(f"⚠️ Long-term downtrend ({momentum:.1f}%)")
                                
                                # Trend stability (30%)
                                if analysis.trend in ["STRONG UPTREND", "UPTREND"]:
                                    ml_score += 18
                                    ml_strategy_notes.append(f"✅ Stable uptrend for long-term hold")
                                elif analysis.trend in ["STRONG DOWNTREND", "DOWNTREND"]:
                                    ml_score -= 15
                                    ml_strategy_notes.append(f"⚠️ Downtrend - not ideal for buy & hold")
                                
                                # Lower volatility preferred (15%)
                                if volatility < 2.5:
                                    ml_score += 10
                                    ml_strategy_notes.append(f"✅ Low volatility - stable hold ({volatility:.1f}%)")
                                elif volatility > 5:
                                    ml_score -= 8
                                    ml_strategy_notes.append(f"⚠️ High volatility - risky for long hold ({volatility:.1f}%)")
                                
                                # RSI for value entry (15%)
                                if rsi < 40:
                                    ml_score += 10
                                    ml_strategy_notes.append(f"✅ Undervalued entry (RSI {rsi:.0f})")
                                elif rsi > 70:
                                    ml_score -= 8
                                    ml_strategy_notes.append(f"⚠️ Overvalued (RSI {rsi:.0f})")
                            
                            else:  # OPTIONS
                                # Options: Focus on IV, trend, and volatility
                                st.caption("🎯 ML optimized for options strategies based on IV and trend")
                                
                                # Trend strength for directional plays (30%)
                                if analysis.trend in ["STRONG UPTREND", "UPTREND"] and momentum > 5:
                                    ml_score += 18
                                    ml_strategy_notes.append(f"✅ Strong trend for calls (+{momentum:.1f}%)")
                                elif analysis.trend in ["STRONG DOWNTREND", "DOWNTREND"] and momentum < -5:
                                    ml_score += 15
                                    ml_strategy_notes.append(f"✅ Strong trend for puts ({momentum:.1f}%)")
                                
                                # IV rank consideration (25%)
                                if analysis.iv_rank > 60:
                                    ml_score += 15
                                    ml_strategy_notes.append(f"✅ High IV ({analysis.iv_rank}%) - sell premium")
                                elif analysis.iv_rank < 40:
                                    ml_score += 12
                                    ml_strategy_notes.append(f"✅ Low IV ({analysis.iv_rank}%) - buy options")
                                
                                # Volatility for options (25%)
                                if 2 < volatility < 5:
                                    ml_score += 12
                                    ml_strategy_notes.append(f"✅ Good volatility for options ({volatility:.1f}%)")
                                elif volatility > 6:
                                    ml_score += 8
                                    ml_strategy_notes.append(f"⚡ High vol - expensive options ({volatility:.1f}%)")
                                
                                # MACD for timing (20%)
                                if macd > macd_signal:
                                    ml_score += 10
                                    ml_strategy_notes.append(f"✅ MACD bullish crossover")
                                elif macd < macd_signal:
                                    ml_score -= 5
                                    ml_strategy_notes.append(f"⚠️ MACD bearish")
                            
                            ml_prediction_score = max(0, min(100, ml_score))
                            
                            # Determine confidence level
                            if ml_prediction_score >= 80:
                                ml_confidence_level = "VERY HIGH"
                            elif ml_prediction_score >= 70:
                                ml_confidence_level = "HIGH"
                            elif ml_prediction_score >= 55:
                                ml_confidence_level = "MEDIUM"
                            else:
                                ml_confidence_level = "LOW"
                    except Exception as e:
                        logger.error(f"Error calculating ML analysis: {e}")
                        ml_analysis_available = False
                    
                    # Display ML Analysis prominently
                    if ml_analysis_available and alpha_factors:
                        # Show ML Score prominently
                        ml_col1, ml_col2, ml_col3 = st.columns([2, 1, 1])
                        
                        with ml_col1:
                            st.metric(
                                "🧠 ML Prediction Score",
                                f"{ml_prediction_score:.0f}/100",
                                help="Machine Learning confidence based on 50+ alpha factors"
                            )
                            if ml_confidence_level == "VERY HIGH":
                                st.success(f"✅ **{ml_confidence_level} CONFIDENCE** - Strong ML signals align with this trade")
                            elif ml_confidence_level == "HIGH":
                                st.info(f"✅ **{ml_confidence_level} CONFIDENCE** - Good ML signals support this trade")
                            elif ml_confidence_level == "MEDIUM":
                                st.warning(f"⚠️ **{ml_confidence_level} CONFIDENCE** - Mixed ML signals, proceed with caution")
                            else:
                                st.error(f"❌ **{ml_confidence_level} CONFIDENCE** - Weak ML signals, high risk")
                        
                        with ml_col2:
                            st.metric("Factors Analyzed", f"{len(alpha_factors)}")
                            st.caption("Alpha factors calculated")
                        
                        with ml_col3:
                            # Agreement between ML and traditional analysis
                            agreement_score = abs(ml_prediction_score - analysis.confidence_score)
                            if agreement_score < 15:
                                st.metric("System Agreement", "✅ Strong")
                                st.caption("ML & Technical align")
                            elif agreement_score < 30:
                                st.metric("System Agreement", "⚠️ Moderate")
                                st.caption("Some divergence")
                            else:
                                st.metric("System Agreement", "❌ Weak")
                                st.caption("Significant divergence")
                        
                        # Display ML Strategy-Specific Insights
                        if ml_strategy_notes:
                            st.write(f"**🎯 ML Insights for {trading_style_display}:**")
                            for note in ml_strategy_notes:
                                st.write(f"• {note}")
                        
                        # Key ML Factors
                        st.write("**Key ML Signals:**")
                        col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
                        
                        with col_ml1:
                            momentum = alpha_factors.get('return_20d', 0) * 100
                            st.metric("20-Day Momentum", f"{momentum:+.1f}%")
                            if momentum > 10:
                                st.caption("🔥 Strong uptrend")
                            elif momentum < -10:
                                st.caption("❄️ Strong downtrend")
                            else:
                                st.caption("➡️ Neutral")
                        
                        with col_ml2:
                            vol_ratio = alpha_factors.get('volume_5d_ratio', 1)
                            st.metric("Volume Signal", f"{vol_ratio:.2f}x")
                            if vol_ratio > 1.5:
                                st.caption("🔥 High activity")
                            elif vol_ratio < 0.7:
                                st.caption("❄️ Low activity")
                            else:
                                st.caption("➡️ Normal")
                        
                        with col_ml3:
                            rsi = alpha_factors.get('rsi_14', 50)
                            st.metric("RSI (14)", f"{rsi:.1f}")
                            if rsi > 70:
                                st.caption("⚠️ Overbought")
                            elif rsi < 30:
                                st.caption("✅ Oversold")
                            else:
                                st.caption("➡️ Neutral")
                        
                        with col_ml4:
                            volatility = alpha_factors.get('volatility_20d', 0) * 100
                            st.metric("20-Day Volatility", f"{volatility:.1f}%")
                            if volatility > 4:
                                st.caption("⚡ High vol")
                            elif volatility < 1.5:
                                st.caption("💤 Low vol")
                            else:
                                st.caption("➡️ Moderate")
                        
                        # Show detailed factors in expander
                        with st.expander("🔬 View All 50+ Alpha Factors (Advanced)"):
                            st.info("These are the same factors used by quantitative hedge funds for algorithmic trading.")
                            
                            # Group factors by category
                            price_factors = {k: v for k, v in alpha_factors.items() if 'return' in k or 'ma' in k or 'price' in k}
                            volume_factors = {k: v for k, v in alpha_factors.items() if 'volume' in k}
                            tech_factors = {k: v for k, v in alpha_factors.items() if k in ['rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'bollinger_position']}
                            momentum_factors = {k: v for k, v in alpha_factors.items() if 'momentum' in k or 'rs_' in k}
                            vol_factors = {k: v for k, v in alpha_factors.items() if 'volatility' in k or 'hl_' in k}
                            
                            tab_price, tab_vol, tab_tech, tab_mom, tab_volat = st.tabs(["💰 Price", "📊 Volume", "📈 Technical", "🚀 Momentum", "⚡ Volatility"])
                            
                            with tab_price:
                                for k, v in price_factors.items():
                                    st.write(f"**{k}**: {v:.4f}")
                            
                            with tab_vol:
                                for k, v in volume_factors.items():
                                    st.write(f"**{k}**: {v:.4f}")
                            
                            with tab_tech:
                                for k, v in tech_factors.items():
                                    st.write(f"**{k}**: {v:.4f}")
                            
                            with tab_mom:
                                for k, v in momentum_factors.items():
                                    st.write(f"**{k}**: {v:.4f}")
                            
                            with tab_volat:
                                for k, v in vol_factors.items():
                                    st.write(f"**{k}**: {v:.4f}")
                    else:
                        st.warning("⚠️ ML analysis unavailable for this ticker. Using traditional technical analysis only.")
                        ml_prediction_score = analysis.confidence_score
                        ml_confidence_level = "N/A"
                    
                    st.divider()
                    
                    # COMPREHENSIVE VERDICT - Final Decision Summary
                    st.header("📋 COMPREHENSIVE TRADING VERDICT")
                    st.write(f"**Complete analysis summary for {analysis.ticker} using {trading_style_display} approach**")
                    
                    # Calculate overall verdict score
                    verdict_score = 0
                    verdict_factors = []
                    
                    # Technical Analysis Score (30%)
                    tech_score = analysis.confidence_score
                    verdict_score += tech_score * 0.30
                    verdict_factors.append(("Technical Analysis", tech_score, 30))
                    
                    # ML Analysis Score (30%)
                    if ml_analysis_available:
                        verdict_score += ml_prediction_score * 0.30
                        verdict_factors.append(("ML Prediction", ml_prediction_score, 30))
                    else:
                        verdict_score += tech_score * 0.30  # Fallback to technical
                        verdict_factors.append(("ML Prediction", tech_score, 30))
                    
                    # Sentiment Score (20%)
                    sentiment_score_normalized = (analysis.sentiment_score + 1) * 50  # Convert -1 to 1 range to 0-100
                    verdict_score += sentiment_score_normalized * 0.20
                    verdict_factors.append(("News Sentiment", sentiment_score_normalized, 20))
                    
                    # Catalyst Score (20%)
                    catalyst_score = min(100, len(analysis.catalysts) * 25)  # 25 points per catalyst, max 100
                    verdict_score += catalyst_score * 0.20
                    verdict_factors.append(("Catalysts", catalyst_score, 20))
                    
                    verdict_score = round(verdict_score, 1)
                    
                    # Determine final recommendation
                    if verdict_score >= 75:
                        verdict_color = "success"
                        verdict_emoji = "🟢"
                        verdict_action = "STRONG BUY"
                        verdict_message = "Excellent opportunity with strong signals across all analysis methods."
                        position_size = "Standard to Large (2-5% of portfolio)"
                    elif verdict_score >= 60:
                        verdict_color = "info"
                        verdict_emoji = "🟢"
                        verdict_action = "BUY"
                        verdict_message = "Good opportunity with positive signals. Proceed with confidence."
                        position_size = "Standard (1-3% of portfolio)"
                    elif verdict_score >= 45:
                        verdict_color = "warning"
                        verdict_emoji = "🟡"
                        verdict_action = "CAUTIOUS BUY"
                        verdict_message = "Mixed signals. Consider smaller position or wait for better setup."
                        position_size = "Small (0.5-1.5% of portfolio)"
                    else:
                        verdict_color = "error"
                        verdict_emoji = "🔴"
                        verdict_action = "AVOID / WAIT"
                        verdict_message = "Weak signals across multiple analysis methods. High risk."
                        position_size = "None - Skip this trade"
                    
                    # Display Verdict
                    if verdict_color == "success":
                        st.success(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                    elif verdict_color == "info":
                        st.info(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                    elif verdict_color == "warning":
                        st.warning(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                    else:
                        st.error(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                    
                    st.write(verdict_message)
                    
                    # Verdict Details
                    verdict_col1, verdict_col2 = st.columns(2)
                    
                    with verdict_col1:
                        st.write("**📊 Score Breakdown:**")
                        for factor_name, factor_score, weight in verdict_factors:
                            score_bar = "█" * int(factor_score / 10) + "░" * (10 - int(factor_score / 10))
                            st.write(f"• **{factor_name}** ({weight}%): {factor_score:.0f}/100 {score_bar}")
                        
                        st.write("")
                        st.metric("Overall Verdict Score", f"{verdict_score:.0f}/100")
                    
                    with verdict_col2:
                        st.write("**✅ Action Plan:**")
                        st.write(f"• **Recommended Action:** {verdict_action}")
                        st.write(f"• **Position Size:** {position_size}")
                        st.write(f"• **Entry Price:** ${analysis.price:.2f}")
                        st.write(f"• **Stop Loss:** ${analysis.support:.2f} ({((analysis.support/analysis.price - 1) * 100):.1f}%)")
                        st.write(f"• **Target:** ${analysis.resistance:.2f} ({((analysis.resistance/analysis.price - 1) * 100):.1f}%)")
                        
                        # Risk/Reward
                        risk = abs(analysis.price - analysis.support)
                        reward = abs(analysis.resistance - analysis.price)
                        rr_ratio = reward / risk if risk > 0 else 0
                        st.write(f"• **Risk/Reward Ratio:** {rr_ratio:.2f}:1")
                        
                        if rr_ratio >= 2:
                            st.caption("✅ Excellent risk/reward")
                        elif rr_ratio >= 1.5:
                            st.caption("✅ Good risk/reward")
                        else:
                            st.caption("⚠️ Suboptimal risk/reward")
                    
                    # Key Considerations
                    st.write("**⚠️ Key Considerations:**")
                    considerations = []
                    
                    if is_penny_stock:
                        considerations.append("🔴 **Penny Stock Risk:** High volatility, use tight stops and small position size")
                    
                    if is_runner:
                        considerations.append("🚀 **Runner Alert:** Extreme momentum - take profits quickly, don't chase")
                    
                    if analysis.earnings_days_away and analysis.earnings_days_away <= 7:
                        considerations.append(f"📅 **Earnings in {analysis.earnings_days_away} days:** Expect high volatility, consider closing before earnings")
                    
                    if analysis.iv_rank > 70:
                        considerations.append("⚡ **Very High IV:** Great for selling premium, expensive for buying options")
                    elif analysis.iv_rank < 30:
                        considerations.append("💤 **Low IV:** Good for buying options, poor for selling premium")
                    
                    if analysis.sentiment_score < -0.3:
                        considerations.append("📰 **Negative Sentiment:** Market pessimism may create headwinds")
                    elif analysis.sentiment_score > 0.3:
                        considerations.append("📰 **Positive Sentiment:** Market optimism supports the trade")
                    
                    if ml_analysis_available:
                        agreement_score = abs(ml_prediction_score - analysis.confidence_score)
                        if agreement_score > 30:
                            considerations.append("⚠️ **ML/Technical Divergence:** Significant disagreement between analysis methods - proceed carefully")
                    
                    if not considerations:
                        considerations.append("✅ No major risk factors identified - standard trading rules apply")
                    
                    for consideration in considerations:
                        st.write(f"• {consideration}")
                    
                    # Final Notes
                    st.divider()
                    st.caption(f"""**Analysis completed at:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} | 
**Trading Style:** {trading_style_display} | 
**Data Source:** Yahoo Finance (Real-time) | 
**ML Factors:** {len(alpha_factors) if alpha_factors else 0} alpha factors analyzed""")
                    
                    # AI-POWERED TRADE RECOMMENDATIONS
                    st.divider()
                    st.header("🤖 AI Trade Recommendations")
                    st.write(f"Based on your **{trading_style_display}** analysis and **{verdict_action}** verdict")
                    
                    # Generate AI-powered trade recommendations
                    trade_recommendations = []
                    
                    # Only recommend if confidence is sufficient
                    if verdict_score >= 45:
                        # STOCK TRADE RECOMMENDATION
                        if trading_style in ["DAY_TRADE", "SWING_TRADE", "BUY_HOLD"]:
                            stock_rec = {
                                "type": "STOCK",
                                "symbol": analysis.ticker,
                                "action": "BUY" if analysis.trend in ["UPTREND", "STRONG UPTREND"] else "SELL_SHORT",
                                "quantity": None,  # Will calculate based on position size
                                "order_type": "limit",  # Always use limit orders to enable bracket orders with stop-loss
                                "price": analysis.price,  # Set entry price for all trades
                                "stop_loss": analysis.support,
                                "target": analysis.resistance,
                                "hold_time": "Intraday" if trading_style == "DAY_TRADE" else "3-10 days" if trading_style == "SWING_TRADE" else "6+ months",
                                "confidence": verdict_score,
                                "reasoning": f"ML Score: {ml_prediction_score:.0f}/100, Trend: {analysis.trend}, RSI: {analysis.rsi:.0f}"
                            }
                            trade_recommendations.append(stock_rec)
                        
                        # OPTIONS TRADE RECOMMENDATIONS
                        if trading_style == "OPTIONS" or verdict_score >= 60:
                            # Determine best options strategy based on analysis
                            if analysis.iv_rank > 60:
                                # High IV - Sell premium
                                if analysis.trend in ["UPTREND", "STRONG UPTREND"]:
                                    options_rec = {
                                        "type": "OPTION",
                                        "strategy": "SELL PUT",
                                        "symbol": analysis.ticker,
                                        "action": "sell_to_open",
                                        "option_type": "put",
                                        "strike_suggestion": f"${analysis.support:.2f} (ATM or slightly OTM)",
                                        "dte_suggestion": "30-45 DTE",
                                        "quantity": 1,
                                        "reasoning": f"High IV ({analysis.iv_rank}%) + Uptrend = Sell puts to collect premium",
                                        "max_profit": "Premium collected",
                                        "max_risk": "Strike - Premium (if assigned)",
                                        "confidence": verdict_score
                                    }
                                else:
                                    options_rec = {
                                        "type": "OPTION",
                                        "strategy": "IRON CONDOR",
                                        "symbol": analysis.ticker,
                                        "action": "multi_leg",
                                        "reasoning": f"High IV ({analysis.iv_rank}%) + Sideways = Iron Condor for range-bound profit",
                                        "strike_suggestion": f"Sell at ${analysis.support:.2f} and ${analysis.resistance:.2f}",
                                        "dte_suggestion": "30-45 DTE",
                                        "confidence": verdict_score - 10
                                    }
                            elif analysis.iv_rank < 40:
                                # Low IV - Buy options
                                if analysis.trend in ["UPTREND", "STRONG UPTREND"]:
                                    options_rec = {
                                        "type": "OPTION",
                                        "strategy": "BUY CALL",
                                        "symbol": analysis.ticker,
                                        "action": "buy_to_open",
                                        "option_type": "call",
                                        "strike_suggestion": f"${analysis.price * 1.02:.2f} (slightly OTM)",
                                        "dte_suggestion": "30-60 DTE",
                                        "quantity": 1,
                                        "reasoning": f"Low IV ({analysis.iv_rank}%) + Uptrend = Buy calls for directional move",
                                        "max_profit": "Unlimited",
                                        "max_risk": "Premium paid",
                                        "confidence": verdict_score
                                    }
                                elif analysis.trend in ["DOWNTREND", "STRONG DOWNTREND"]:
                                    options_rec = {
                                        "type": "OPTION",
                                        "strategy": "BUY PUT",
                                        "symbol": analysis.ticker,
                                        "action": "buy_to_open",
                                        "option_type": "put",
                                        "strike_suggestion": f"${analysis.price * 0.98:.2f} (slightly OTM)",
                                        "dte_suggestion": "30-60 DTE",
                                        "quantity": 1,
                                        "reasoning": f"Low IV ({analysis.iv_rank}%) + Downtrend = Buy puts for directional move",
                                        "max_profit": "Strike - Premium",
                                        "max_risk": "Premium paid",
                                        "confidence": verdict_score
                                    }
                                else:
                                    options_rec = None
                            else:
                                # Medium IV - Spreads
                                if analysis.trend in ["UPTREND", "STRONG UPTREND"]:
                                    options_rec = {
                                        "type": "OPTION",
                                        "strategy": "BULL CALL SPREAD",
                                        "symbol": analysis.ticker,
                                        "action": "multi_leg",
                                        "reasoning": f"Medium IV ({analysis.iv_rank}%) + Uptrend = Bull call spread for defined risk",
                                        "strike_suggestion": f"Buy ${analysis.price:.2f}, Sell ${analysis.resistance:.2f}",
                                        "dte_suggestion": "30-45 DTE",
                                        "confidence": verdict_score - 5
                                    }
                                else:
                                    options_rec = None
                            
                            if options_rec:
                                trade_recommendations.append(options_rec)
                    
                    # Display recommendations
                    if trade_recommendations:
                        for i, rec in enumerate(trade_recommendations, 1):
                            with st.expander(f"{'📈' if rec['type'] == 'STOCK' else '🎯'} Recommendation #{i}: {rec['type']} - {rec.get('strategy', rec.get('action', '').upper())}", expanded=True):
                                rec_col1, rec_col2 = st.columns([2, 1])
                                
                                with rec_col1:
                                    if rec['type'] == 'STOCK':
                                        st.write(f"**Strategy:** {rec['action']} {rec['symbol']} stock")
                                        st.write(f"**Order Type:** {rec['order_type'].upper()}")
                                        if rec['price']:
                                            st.write(f"**Entry Price:** ${rec['price']:.2f}")
                                        st.write(f"**Stop Loss:** ${rec['stop_loss']:.2f} ({((rec['stop_loss']/analysis.price - 1) * 100):.1f}%)")
                                        st.write(f"**Target:** ${rec['target']:.2f} ({((rec['target']/analysis.price - 1) * 100):.1f}%)")
                                        st.write(f"**Hold Time:** {rec['hold_time']}")
                                        st.info(f"💡 **Why:** {rec['reasoning']}")
                                        
                                        # Calculate position size based on verdict
                                        if verdict_score >= 75:
                                            position_pct = "2-5%"
                                            shares_example = "20-50"
                                        elif verdict_score >= 60:
                                            position_pct = "1-3%"
                                            shares_example = "10-30"
                                        else:
                                            position_pct = "0.5-1.5%"
                                            shares_example = "5-15"
                                        
                                        st.caption(f"**Suggested Position Size:** {position_pct} of portfolio (~{shares_example} shares for $10k account)")
                                    
                                    else:  # OPTIONS
                                        st.write(f"**Strategy:** {rec['strategy']}")
                                        st.write(f"**Symbol:** {rec['symbol']}")
                                        st.write(f"**Strike:** {rec.get('strike_suggestion', 'See details')}")
                                        st.write(f"**Expiration:** {rec.get('dte_suggestion', '30-45 DTE')}")
                                        if 'max_profit' in rec:
                                            st.write(f"**Max Profit:** {rec['max_profit']}")
                                        if 'max_risk' in rec:
                                            st.write(f"**Max Risk:** {rec['max_risk']}")
                                        st.info(f"💡 **Why:** {rec['reasoning']}")
                                        st.caption(f"**Contracts:** Start with 1-2 contracts, scale based on experience")
                                
                                with rec_col2:
                                    st.metric("Confidence", f"{rec['confidence']:.0f}/100")
                                    
                                    if rec['confidence'] >= 75:
                                        st.success("✅ HIGH CONFIDENCE")
                                    elif rec['confidence'] >= 60:
                                        st.info("✅ GOOD CONFIDENCE")
                                    else:
                                        st.warning("⚠️ MODERATE")
                                    
                                    # Execute button with callback - capture loop variables with default args
                                    def execute_trade_callback(recommendation=rec, price=analysis.price, verdict=verdict_action, rec_num=i):
                                        logger.info(f"🔥 EXECUTE BUTTON CLICKED for recommendation #{rec_num}")
                                        logger.info(f"📊 Setting session state: symbol={recommendation['symbol']}, price={price}, verdict={verdict}")
                                        st.session_state.selected_recommendation = recommendation
                                        st.session_state.quick_trade_ticker = recommendation['symbol']
                                        st.session_state.quick_trade_price = price
                                        st.session_state.quick_trade_verdict = verdict
                                        st.session_state.show_quick_trade = True
                                        logger.info(f"✅ Session state set: show_quick_trade={st.session_state.show_quick_trade}")
                                    
                                    st.button(
                                        f"🚀 Execute This Trade", 
                                        key=f"execute_{i}", 
                                        use_container_width=True, 
                                        type="primary",
                                        on_click=execute_trade_callback
                                    )
                    else:
                        st.warning("⚠️ No trade recommendations - Verdict score too low. Consider waiting for a better setup.")
                    
                    # Other quick actions
                    st.divider()
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        if st.button("🎯 Get More Strategy Ideas", use_container_width=True):
                            st.session_state.goto_strategy_advisor = True
                            st.rerun()
                    
                    with action_col2:
                        if st.button("📊 View in Strategy Analyzer", use_container_width=True):
                            st.session_state.analyzer_ticker = analysis.ticker
                            st.rerun()
                    
                else:
                    st.error(f"❌ Could not analyze {search_ticker}. Please check the ticker symbol.")
        
        elif st.session_state.current_analysis:
            st.info("💡 Previous analysis is displayed. Enter a new ticker and click Analyze to update.")
    
    with tab2:
        st.header("🔥 Top Options Trades")
        st.write("Discover **high-quality** options trading opportunities with AI-enhanced analysis. Only shows plays rated 5.0/10 or higher.")
        
        # Initialize scanners
        if 'ai_scanner' not in st.session_state:
            st.session_state.ai_scanner = AIConfidenceScanner()
        if 'ml_scanner' not in st.session_state:
            st.session_state.ml_scanner = MLEnhancedScanner()
        
        # ML toggle
        use_ml = st.checkbox("🧠 Enable ML Analysis (Qlib)", value=False, key="use_ml_options", 
                            help="Combine 158 alpha factors from Qlib ML with LLM reasoning for maximum confidence")
        
        # Show ML explanation when enabled
        if use_ml:
            with st.expander("ℹ️ What is ML Analysis? (Click to learn more)", expanded=False):
                st.markdown(st.session_state.ml_scanner.get_ml_summary_explanation())
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            top_n = st.slider("Number of trades to scan", 5, 20, 10)
        with col2:
            min_rating = st.slider("Min AI Rating", 0.0, 10.0, 3.0, 0.5, help="Only show plays with this rating or higher (lower = more results)")
        with col3:
            if use_ml:
                min_score = st.slider("Min Ensemble Score", 0.0, 100.0, 70.0, 5.0, help="Minimum ML+LLM+Quant ensemble score")
            else:
                min_score = st.slider("Min Quant Score", 0.0, 100.0, 20.0, 5.0, help="Minimum quantitative score threshold (lower = more results)")
        with col4:
            st.write("")
            st.write("")
            scan_btn = st.button("🔍 Scan Markets", type="primary", use_container_width=True)
        
        if scan_btn:
            with st.status("🔍 Scanning markets for top options trades...", expanded=True) as status:
                st.write("📊 Analyzing market data (real-time via Yahoo Finance)...")
                if use_ml:
                    st.write("🧠 Running ML analysis with 158 alpha factors...")
                st.write(f"🤖 Running AI confidence analysis (filtering for {min_rating}+ rating)...")
                st.write("⚡ Calculating entry/exit levels and strategies...")
                
                try:
                    if use_ml:
                        # Use ML-enhanced scanner
                        trades = st.session_state.ml_scanner.scan_top_options_with_ml(
                            top_n=top_n,
                            min_ensemble_score=min_score
                        )
                    else:
                        # Use standard AI scanner
                        trades = st.session_state.ai_scanner.scan_top_options_with_ai(
                            top_n=top_n, 
                            min_ai_rating=min_rating,
                            min_score=min_score
                        )
                    
                    if trades:
                        status.update(label=f"✅ Found {len(trades)} quality opportunities!", state="complete")
                        st.session_state.top_options_trades = trades
                        
                        # Display summary
                        avg_rating = sum(t.ai_rating for t in trades) / len(trades) if trades else 0
                        high_conf = len([t for t in trades if t.ai_confidence in ['HIGH', 'VERY HIGH']])
                        st.success(f"✅ Found {len(trades)} **quality** opportunities! Avg AI Rating: {avg_rating:.1f}/10 | High Confidence: {high_conf}")
                        
                        # Real-time indicator
                        st.info(f"📡 **Real-time data** fetched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (market hours may affect accuracy)")
                        
                        # Download button with UTF-8 BOM for Excel compatibility
                        if use_ml and hasattr(trades[0], 'ml_prediction_score'):
                            csv_data = "\ufeffTicker,Ensemble Score,ML Score,AI Rating,AI Confidence,Quant Score,Price,Change %,Volume Ratio,Risk Level,Ensemble Confidence,AI Reasoning,AI Risks,Reason\n"
                            for trade in trades:
                                csv_data += f'"{trade.ticker}",{trade.combined_score:.1f},{trade.ml_prediction_score:.1f},{trade.ai_rating},"{trade.ai_confidence}",{trade.score},${trade.price},{trade.change_pct:+.2f}%,{trade.volume_ratio}x,"{trade.risk_level}","{trade.ensemble_confidence}","{trade.ai_reasoning}","{trade.ai_risks}","{trade.reason}"\n'
                        else:
                            csv_data = "\ufeffTicker,AI Rating,AI Confidence,Quant Score,Price,Change %,Volume Ratio,Risk Level,AI Reasoning,AI Risks,Reason\n"
                            for trade in trades:
                                csv_data += f'"{trade.ticker}",{trade.ai_rating},"{trade.ai_confidence}",{trade.score},${trade.price},{trade.change_pct:+.2f}%,{trade.volume_ratio}x,"{trade.risk_level}","{trade.ai_reasoning}","{trade.ai_risks}","{trade.reason}"\n'
                        
                        st.download_button(
                            label="📥 Download Report (CSV)",
                            data=csv_data.encode('utf-8-sig'),
                            file_name=f"ai_options_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.divider()
                        
                        for i, trade in enumerate(trades, 1):
                            # Color-code based on rating
                            if trade.ai_rating >= 8.0:
                                emoji = "🟢"
                            elif trade.ai_rating >= 6.5:
                                emoji = "🟡"
                            else:
                                emoji = "🟠"
                            
                            # Display header with ML metrics if available
                            if use_ml and hasattr(trade, 'combined_score'):
                                header = f"{emoji} #{i} **{trade.ticker}** - Ensemble: {trade.combined_score:.1f}/100 | ML: {trade.ml_prediction_score:.1f} | AI: {trade.ai_rating:.1f}/10 | {trade.ensemble_confidence}"
                            else:
                                header = f"{emoji} #{i} **{trade.ticker}** - AI: {trade.ai_rating:.1f}/10 | Score: {trade.score:.1f}/100 | {trade.ai_confidence}"
                            
                            with st.expander(header, expanded=(i==1)):
                                # Top metrics row - add ML metric if available
                                if use_ml and hasattr(trade, 'combined_score'):
                                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                                else:
                                    col_a, col_b, col_c, col_d = st.columns(4)
                                
                                with col_a:
                                    st.metric("💵 Current Price", f"${trade.price:.2f}", 
                                             f"{trade.change_pct:+.2f}%", delta_color="normal")
                                
                                with col_b:
                                    st.metric("📊 Volume Activity", f"{trade.volume_ratio:.1f}x",
                                             "Above Average" if trade.volume_ratio > 1.5 else "Normal")
                                
                                with col_c:
                                    st.metric("🎯 AI Rating", f"{trade.ai_rating:.1f}/10", 
                                             trade.ai_confidence)
                                
                                with col_d:
                                    st.metric("⚠️ Risk Level", trade.risk_level,
                                             "Manage Carefully" if trade.risk_level in ['H', 'M-H'] else "Standard")
                                
                                # Add ML metric if available
                                if use_ml and hasattr(trade, 'combined_score'):
                                    with col_e:
                                        st.metric("🧠 ML Score", f"{trade.ml_prediction_score:.1f}/100",
                                                 f"{trade.ml_features_count} factors")
                                
                                st.divider()
                                
                                # Unified Confidence Summary (when ML is enabled)
                                if use_ml and hasattr(trade, 'combined_score'):
                                    with st.expander("📊 **UNIFIED CONFIDENCE ANALYSIS** - See complete breakdown", expanded=True):
                                        st.markdown(st.session_state.ml_scanner.get_unified_confidence_summary(trade))
                                    st.divider()
                                
                                # Trading Strategy Section
                                st.markdown("### 🎯 Suggested Trading Strategies")
                                
                                # Calculate support/resistance based on price
                                support = trade.price * 0.97
                                resistance = trade.price * 1.03
                                
                                if trade.change_pct > 2:
                                    strategy_text = f"""**BULLISH SETUP** 🚀
- **Calls**: Buy slightly OTM calls (strike ~${trade.price * 1.02:.2f}) for momentum play
- **Bull Call Spread**: Buy call at ${trade.price:.2f}, sell call at ${resistance:.2f}
- **Sell Puts**: If confident, sell cash-secured puts at ${support:.2f} for premium
"""
                                elif trade.change_pct < -2:
                                    strategy_text = f"""**BEARISH SETUP** 📉
- **Puts**: Buy slightly OTM puts (strike ~${trade.price * 0.98:.2f}) for downside play
- **Bear Put Spread**: Buy put at ${trade.price:.2f}, sell put at ${support:.2f}
- **Sell Calls**: If confident in decline, sell OTM calls at ${resistance:.2f}
"""
                                else:
                                    strategy_text = f"""**NEUTRAL/RANGE SETUP** ⚖️
- **Iron Condor**: Sell calls at ${resistance:.2f}, sell puts at ${support:.2f}
- **Straddle/Strangle**: Buy both calls & puts if expecting volatility spike
- **Theta Plays**: Sell premium via covered calls or cash-secured puts
"""
                                st.info(strategy_text)
                                
                                # Entry/Exit Levels
                                st.markdown("### 📍 Key Price Levels (Estimated)")
                                col_e1, col_e2, col_e3 = st.columns(3)
                                with col_e1:
                                    st.markdown(f"**🟢 Entry Zone**\n${support:.2f} - ${trade.price:.2f}")
                                with col_e2:
                                    st.markdown(f"**🎯 Target**\n${resistance:.2f} (+{((resistance/trade.price - 1) * 100):.1f}%)")
                                with col_e3:
                                    st.markdown(f"**🛑 Stop Loss**\n${support * 0.98:.2f} (-{((1 - support * 0.98/trade.price) * 100):.1f}%)")
                                
                                st.divider()
                                
                                # Analysis sections
                                col_info1, col_info2 = st.columns(2)
                                
                                with col_info1:
                                    st.markdown("**📊 Quantitative Signals**")
                                    st.write(trade.reason or "N/A")
                                    
                                    if trade.ai_reasoning:
                                        st.markdown("**🤖 AI Analysis**")
                                        st.info(trade.ai_reasoning)
                                
                                with col_info2:
                                    if trade.ai_risks:
                                        st.markdown("**⚠️ Risk Assessment**")
                                        st.warning(trade.ai_risks)
                                    
                                    # Additional context
                                    st.markdown("**ℹ️ Trade Context**")
                                    context = f"Volume: {trade.volume:,} ({trade.volume_ratio:.1f}x avg)\n"
                                    context += f"Momentum: {'Strong' if abs(trade.change_pct) > 3 else 'Moderate' if abs(trade.change_pct) > 1 else 'Weak'}\n"
                                    context += f"Confidence: {trade.confidence} (Quant) / {trade.ai_confidence} (AI)"
                                    st.text(context)
                    else:
                        status.update(label="⚠️ No quality opportunities found", state="error")
                        st.warning(f"No opportunities found meeting minimum criteria (AI Rating ≥ {min_rating}, Score ≥ {min_score}). Try lowering the minimum rating or check if markets are open.")
                        
                except Exception as e:
                    status.update(label="❌ Scan failed", state="error")
                    st.error(f"Error during scan: {str(e)[:100]}")
        
        elif 'top_options_trades' in st.session_state and st.session_state.top_options_trades:
            st.info(f"💡 Showing {len(st.session_state.top_options_trades)} previously scanned trades. Click 'Scan Markets' to refresh with real-time data.")
    
    with tab3:
        st.header("💰 Top Penny Stocks")
        st.write("Find **high-potential** penny stock opportunities with comprehensive AI analysis. Only shows plays rated 5.0/10 or higher.")
        
        # Initialize scanners
        if 'ai_scanner' not in st.session_state:
            st.session_state.ai_scanner = AIConfidenceScanner()
        if 'ml_scanner' not in st.session_state:
            st.session_state.ml_scanner = MLEnhancedScanner()
        
        # ML toggle
        use_ml_penny = st.checkbox("🧠 Enable ML Analysis (Qlib)", value=False, key="use_ml_penny", 
                                   help="Combine 158 alpha factors from Qlib ML with LLM reasoning for maximum confidence")
        
        # Show ML explanation when enabled
        if use_ml_penny:
            with st.expander("ℹ️ What is ML Analysis? (Click to learn more)", expanded=False):
                st.markdown(st.session_state.ml_scanner.get_ml_summary_explanation())
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            top_n_penny = st.slider("Number of penny stocks to scan", 5, 15, 8, key="penny_slider")
        with col2:
            min_penny_rating = st.slider("Min AI Rating", 0.0, 10.0, 5.0, 0.5, key="penny_min_rating", help="Only show plays with this rating or higher")
        with col3:
            if use_ml_penny:
                min_penny_score = st.slider("Min Ensemble Score", 0.0, 100.0, 65.0, 5.0, key="penny_min_score", help="Minimum ML+LLM+Quant ensemble score")
            else:
                min_penny_score = st.slider("Min Composite Score", 0.0, 100.0, 40.0, 5.0, key="penny_min_score", help="Minimum composite score threshold")
        with col4:
            st.write("")
            st.write("")
            scan_penny_btn = st.button("🔍 Scan Penny Stocks", type="primary", use_container_width=True)
        
        if scan_penny_btn:
            with st.status("🔍 Scanning for top penny stocks...", expanded=True) as status:
                st.write("📊 Analyzing penny stock data (real-time via Yahoo Finance)...")
                if use_ml_penny:
                    st.write("🧠 Running ML analysis with 158 alpha factors...")
                st.write(f"🤖 Running AI analysis (filtering for {min_penny_rating}+ rating)...")
                st.write("⚡ Calculating momentum, valuation, and catalyst scores...")
                
                try:
                    if use_ml_penny:
                        # Use ML-enhanced scanner
                        trades = st.session_state.ml_scanner.scan_top_penny_stocks_with_ml(
                            top_n=top_n_penny,
                            min_ensemble_score=min_penny_score
                        )
                    else:
                        # Use standard AI scanner
                        trades = st.session_state.ai_scanner.scan_top_penny_stocks_with_ai(
                            top_n=top_n_penny,
                            min_ai_rating=min_penny_rating,
                            min_score=min_penny_score
                        )
                    
                    if trades:
                        status.update(label=f"✅ Found {len(trades)} quality penny stocks!", state="complete")
                        st.session_state.top_penny_trades = trades
                        
                        # Display summary
                        avg_rating = sum(t.ai_rating for t in trades) / len(trades) if trades else 0
                        high_conf = len([t for t in trades if t.ai_confidence in ['HIGH', 'VERY HIGH']])
                        st.success(f"✅ Found {len(trades)} **quality** penny stock opportunities! Avg AI Rating: {avg_rating:.1f}/10 | High Confidence: {high_conf}")
                        
                        # Real-time indicator
                        st.info(f"📡 **Real-time data** fetched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (market hours may affect accuracy)")
                        
                        # Download button with UTF-8 BOM for Excel compatibility
                        if use_ml_penny and hasattr(trades[0], 'ml_prediction_score'):
                            csv_data = "\ufeffTicker,Ensemble Score,ML Score,AI Rating,AI Confidence,Composite Score,Price,Change %,Volume Ratio,Risk Level,Ensemble Confidence,AI Reasoning,AI Risks,Reason\n"
                            for trade in trades:
                                csv_data += f'"{trade.ticker}",{trade.combined_score:.1f},{trade.ml_prediction_score:.1f},{trade.ai_rating},"{trade.ai_confidence}",{trade.score},${trade.price},{trade.change_pct:+.2f}%,{trade.volume_ratio}x,"{trade.risk_level}","{trade.ensemble_confidence}","{trade.ai_reasoning}","{trade.ai_risks}","{trade.reason}"\n'
                        else:
                            csv_data = "\ufeffTicker,AI Rating,AI Confidence,Composite Score,Price,Change %,Volume Ratio,Risk Level,AI Reasoning,AI Risks,Reason\n"
                            for trade in trades:
                                csv_data += f'"{trade.ticker}",{trade.ai_rating},"{trade.ai_confidence}",{trade.score},${trade.price},{trade.change_pct:+.2f}%,{trade.volume_ratio}x,"{trade.risk_level}","{trade.ai_reasoning}","{trade.ai_risks}","{trade.reason}"\n'
                        
                        st.download_button(
                            label="📥 Download Report (CSV)",
                            data=csv_data.encode('utf-8-sig'),
                            file_name=f"ai_penny_stocks_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        st.divider()
                        
                        for i, trade in enumerate(trades, 1):
                            # Color-code based on rating
                            if trade.ai_rating >= 8.0:
                                emoji = "🟢"
                            elif trade.ai_rating >= 6.5:
                                emoji = "🟡"
                            else:
                                emoji = "🟠"
                            
                            # Display header with ML metrics if available
                            if use_ml_penny and hasattr(trade, 'combined_score'):
                                header = f"{emoji} #{i} **{trade.ticker}** - Ensemble: {trade.combined_score:.1f}/100 | ML: {trade.ml_prediction_score:.1f} | AI: {trade.ai_rating:.1f}/10 | {trade.ensemble_confidence}"
                            else:
                                header = f"{emoji} #{i} **{trade.ticker}** - AI: {trade.ai_rating:.1f}/10 | Composite: {trade.score:.1f}/100 | {trade.ai_confidence}"
                            
                            with st.expander(header, expanded=(i==1)):
                                # Top metrics row - add ML metric if available
                                if use_ml_penny and hasattr(trade, 'combined_score'):
                                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                                else:
                                    col_a, col_b, col_c, col_d = st.columns(4)
                                
                                with col_a:
                                    st.metric("💵 Current Price", f"${trade.price:.2f}", 
                                             f"{trade.change_pct:+.2f}%", delta_color="normal")
                                
                                with col_b:
                                    st.metric("📊 Volume Activity", f"{trade.volume_ratio:.1f}x",
                                             "Above Average" if trade.volume_ratio > 1.5 else "Normal")
                                
                                with col_c:
                                    st.metric("🎯 AI Rating", f"{trade.ai_rating:.1f}/10", 
                                             trade.ai_confidence)
                                
                                with col_d:
                                    st.metric("⚠️ Risk Level", trade.risk_level,
                                             "High Risk" if trade.risk_level in ['H', 'M-H'] else "Moderate")
                                
                                # Add ML metric if available
                                if use_ml_penny and hasattr(trade, 'combined_score'):
                                    with col_e:
                                        st.metric("🧠 ML Score", f"{trade.ml_prediction_score:.1f}/100",
                                                 f"{trade.ml_features_count} factors")
                                
                                st.divider()
                                
                                # Unified Confidence Summary (when ML is enabled)
                                if use_ml_penny and hasattr(trade, 'combined_score'):
                                    with st.expander("📊 **UNIFIED CONFIDENCE ANALYSIS** - See complete breakdown", expanded=True):
                                        st.markdown(st.session_state.ml_scanner.get_unified_confidence_summary(trade))
                                    st.divider()
                                
                                # Trading Strategy Section
                                st.markdown("### 🎯 Penny Stock Trading Strategy")
                                
                                # Calculate support/resistance
                                support = trade.price * 0.95
                                resistance = trade.price * 1.08
                                
                                if trade.change_pct > 5:
                                    strategy_text = f"""**MOMENTUM PLAY** 🚀
- **Entry**: Ideally on pullback to ${support:.2f} - ${trade.price * 0.98:.2f}
- **Target 1**: ${trade.price * 1.05:.2f} (+5%)
- **Target 2**: ${resistance:.2f} (+8%)
- **Stop Loss**: ${support * 0.97:.2f} (tight due to volatility)
- **Position Size**: Small (1-3% of portfolio max for penny stocks)
"""
                                elif trade.change_pct < -3:
                                    strategy_text = f"""**REVERSAL WATCH** ⏮️
- **Entry**: Watch for bounce confirmation above ${trade.price * 1.01:.2f}
- **Target**: ${trade.price * 1.06:.2f} (+6% bounce)
- **Stop Loss**: ${support:.2f} (below recent support)
- **Caution**: Confirm reversal with volume before entry
- **Position Size**: Very small (1-2% max)
"""
                                else:
                                    strategy_text = f"""**ACCUMULATION ZONE** 📈
- **Entry**: Build position between ${support:.2f} - ${trade.price:.2f}
- **Target 1**: ${trade.price * 1.04:.2f} (+4%)
- **Target 2**: ${resistance:.2f} (+8%)
- **Stop Loss**: ${support * 0.96:.2f}
- **Strategy**: Scale in gradually, take profits incrementally
"""
                                st.info(strategy_text)
                                
                                # Entry/Exit Levels
                                st.markdown("### 📍 Key Price Levels")
                                col_e1, col_e2, col_e3 = st.columns(3)
                                with col_e1:
                                    st.markdown(f"**🟢 Buy Zone**\n${support:.2f} - ${trade.price * 0.99:.2f}")
                                with col_e2:
                                    st.markdown(f"**🎯 Profit Target**\n${resistance:.2f} (+{((resistance/trade.price - 1) * 100):.1f}%)")
                                with col_e3:
                                    st.markdown(f"**🛑 Stop Loss**\n${support * 0.96:.2f} (-{((1 - support * 0.96/trade.price) * 100):.1f}%)")
                                
                                st.divider()
                                
                                # Analysis sections
                                col_info1, col_info2 = st.columns(2)
                                
                                with col_info1:
                                    st.markdown("**📊 Composite Analysis**")
                                    st.write(trade.reason or "N/A")
                                    
                                    if trade.ai_reasoning:
                                        st.markdown("**🤖 AI Analysis**")
                                        st.info(trade.ai_reasoning)
                                
                                with col_info2:
                                    if trade.ai_risks:
                                        st.markdown("**⚠️ Risk Assessment**")
                                        st.warning(trade.ai_risks)
                                    
                                    # Additional context
                                    st.markdown("**ℹ️ Trade Context**")
                                    context = f"Volume: {trade.volume:,} ({trade.volume_ratio:.1f}x avg)\n"
                                    context += f"Price Change: {trade.change_pct:+.2f}%\n"
                                    context += f"Confidence: {trade.confidence} (Quant) / {trade.ai_confidence} (AI)\n"
                                    context += f"⚠️ **Remember**: Penny stocks are high risk - use small positions!"
                                    st.text(context)
                    else:
                        status.update(label="⚠️ No quality opportunities found", state="error")
                        st.warning(f"No opportunities found meeting minimum criteria (AI Rating ≥ {min_penny_rating}, Score ≥ {min_penny_score}). Try lowering the minimum rating or check if markets are open.")
                        
                except Exception as e:
                    status.update(label="❌ Scan failed", state="error")
                    st.error(f"Error during scan: {str(e)[:100]}")
        
        elif 'top_penny_trades' in st.session_state and st.session_state.top_penny_trades:
            st.info(f"💡 Showing {len(st.session_state.top_penny_trades)} previously scanned penny stocks. Click 'Scan Penny Stocks' to refresh with real-time data.")
    
    with tab4:
        st.header("⭐ My Tickers")
        st.write("Manage your saved tickers and watchlists.")
        
        # Initialize ticker manager
        if 'ticker_manager' not in st.session_state:
            st.session_state.ticker_manager = TickerManager()
        
        tm = st.session_state.ticker_manager
        
        # Add new ticker
        st.subheader("➕ Add New Ticker")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_ticker = st.text_input("Ticker Symbol").upper()
        with col2:
            new_name = st.text_input("Company Name (optional)")
        with col3:
            new_type = st.selectbox("Type", ["stock", "option", "penny_stock", "crypto"])
        
        new_notes = st.text_area("Notes (optional)")
        
        if st.button("➕ Add Ticker"):
            if new_ticker:
                if tm.add_ticker(new_ticker, name=new_name, ticker_type=new_type, notes=new_notes):
                    st.success(f"✅ Added {new_ticker}!")
                    st.rerun()
                else:
                    st.error("❌ Failed to add ticker. It might already exist.")
            else:
                st.warning("⚠️ Ticker symbol is required.")
        
        st.divider()
        
        # View saved tickers
        st.subheader("📋 Your Saved Tickers")
        all_tickers = tm.get_all_tickers(limit=50)

        if all_tickers:
            for ticker in all_tickers:
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Build expander title with ML score if available
                    ml_score = ticker.get('ml_score')
                    if ml_score is not None:
                        score_emoji = "🟢" if ml_score >= 70 else "🟡" if ml_score >= 50 else "🔴"
                        expander_title = f"{score_emoji} **{ticker['ticker']}** ({ticker.get('type', 'N/A')}) - ML Score: {ml_score:.0f}/100"
                    else:
                        expander_title = f"**{ticker['ticker']}** ({ticker.get('type', 'N/A')})"
                    
                    with st.expander(expander_title):
                        st.write(f"**Name:** {ticker.get('name', 'N/A')}")
                        st.write(f"**Notes:** {ticker.get('notes', 'N/A')}")
                        
                        date_added_str = ticker.get('date_added', 'Unknown')
                        if date_added_str != 'Unknown':
                            try:
                                # Parse string, assume it's UTC, and convert to local time
                                dt_utc = datetime.fromisoformat(date_added_str).replace(tzinfo=timezone.utc)
                                dt_local = dt_utc.astimezone()
                                friendly_date = dt_local.strftime('%B %d, %Y at %I:%M %p')
                            except (ValueError, TypeError):
                                friendly_date = date_added_str # Fallback for invalid formats
                        else:
                            friendly_date = 'Unknown'
                        st.write(f"**Added:** {friendly_date}")

                        st.write(f"**Access Count:** {ticker.get('access_count', 0)}")
                        
                        # Display ML analysis if available
                        if ml_score is not None:
                            st.divider()
                            st.markdown("**📊 Latest ML Analysis**")
                            
                            col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
                            with col_ml1:
                                st.metric("ML Score", f"{ml_score:.0f}/100")
                            with col_ml2:
                                momentum = ticker.get('momentum', 0)
                                st.metric("Momentum", f"{momentum:+.1f}%")
                            with col_ml3:
                                vol_ratio = ticker.get('volume_ratio', 0)
                                st.metric("Vol Ratio", f"{vol_ratio:.2f}x")
                            with col_ml4:
                                rsi = ticker.get('rsi', 0)
                                st.metric("RSI", f"{rsi:.0f}")
                            
                            # Show when last analyzed
                            last_analyzed_str = ticker.get('last_analyzed')
                            if last_analyzed_str:
                                try:
                                    dt_analyzed = datetime.fromisoformat(last_analyzed_str).replace(tzinfo=timezone.utc)
                                    dt_local_analyzed = dt_analyzed.astimezone()
                                    friendly_analyzed = dt_local_analyzed.strftime('%B %d, %Y at %I:%M %p')
                                    st.caption(f"Last analyzed: {friendly_analyzed}")
                                except:
                                    pass

                with col2:
                    if st.button("🗑️ Remove", key=f"remove_{ticker['ticker']}", use_container_width=True):
                        if tm.remove_ticker(ticker['ticker']):
                            st.success(f"🗑️ Removed {ticker['ticker']}!")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to remove {ticker['ticker']}.")
            
            # Show ML analysis if requested
            if 'ml_ticker_to_analyze' in st.session_state:
                ticker_to_analyze = st.session_state.ml_ticker_to_analyze
                st.divider()
                st.subheader(f"🧠 ML-Enhanced Analysis: {ticker_to_analyze}")
                
                with st.spinner(f"Analyzing {ticker_to_analyze} with 50+ alpha factors..."):
                    try:
                        alpha_calc = AlphaFactorCalculator()
                        alpha_factors = alpha_calc.calculate_factors(ticker_to_analyze)
                        
                        if alpha_factors:
                            col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
                            
                            with col_ml1:
                                momentum = alpha_factors.get('return_20d', 0) * 100
                                st.metric("20-Day Return", f"{momentum:+.1f}%")
                            
                            with col_ml2:
                                vol_ratio = alpha_factors.get('volume_5d_ratio', 1)
                                st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
                            
                            with col_ml3:
                                rsi = alpha_factors.get('rsi_14', 50)
                                st.metric("RSI", f"{rsi:.1f}")
                            
                            with col_ml4:
                                volatility = alpha_factors.get('volatility_20d', 0) * 100
                                st.metric("Volatility", f"{volatility:.1f}%")
                            
                            # Calculate ML score
                            ml_score = 50  # baseline
                            if momentum > 5:
                                ml_score += 15
                            elif momentum < -5:
                                ml_score -= 15
                            if vol_ratio > 1.5:
                                ml_score += 10
                            if 30 < rsi < 70:
                                ml_score += 10
                            
                            ml_score = max(0, min(100, ml_score))
                            
                            if ml_score >= 70:
                                st.success(f"✅ **HIGH CONFIDENCE** - ML Score: {ml_score:.1f}/100")
                                st.write("Strong signals across multiple factors. Good opportunity.")
                            elif ml_score >= 50:
                                st.info(f"📊 **MEDIUM CONFIDENCE** - ML Score: {ml_score:.1f}/100")
                                st.write("Mixed signals. Monitor for better entry.")
                            else:
                                st.warning(f"⚠️ **LOW CONFIDENCE** - ML Score: {ml_score:.1f}/100")
                                st.write("Weak signals. Consider waiting or passing.")
                            
                            if st.button("❌ Close Analysis"):
                                del st.session_state.ml_ticker_to_analyze
                                st.rerun()
                        else:
                            st.error(f"Could not calculate alpha factors for {ticker_to_analyze}")
                            if st.button("❌ Close"):
                                del st.session_state.ml_ticker_to_analyze
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                        if st.button("❌ Close"):
                            del st.session_state.ml_ticker_to_analyze
                            st.rerun()
        else:
            st.info("No saved tickers yet. Add some above!")
        
        # Bulk ML Analysis Section
        if all_tickers:
            st.divider()
            st.subheader("🧠 Bulk ML Analysis")
            st.write("Run ML-enhanced analysis on all your saved tickers at once.")
            
            if st.button("🚀 Analyze All My Tickers", type="primary", use_container_width=True):
                log_stream = io.StringIO()
                st_handler = logging.StreamHandler(log_stream)
                st_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                # Add handler ONLY to the logger of the service we want to capture
                alpha_factors_logger = logging.getLogger('services.alpha_factors')
                alpha_factors_logger.addHandler(st_handler)

                results = []
                with st.expander("📊 Live Analysis Logs", expanded=True):
                    log_container = st.empty()
                    with st.status("Analyzing your tickers with ML...", expanded=True) as status:
                        ticker_list = [t['ticker'] for t in all_tickers[:10]]  # Limit to 10
                        for i, ticker_symbol in enumerate(ticker_list):
                            status.update(label=f"Analyzing {ticker_symbol} ({i+1}/{len(ticker_list)})...", state="running")
                            try:
                                alpha_calc = AlphaFactorCalculator()
                                alpha_factors = alpha_calc.calculate_factors(ticker_symbol)
                                if alpha_factors:
                                    momentum = alpha_factors.get('return_20d', 0) * 100
                                    vol_ratio = alpha_factors.get('volume_5d_ratio', 1)
                                    rsi = alpha_factors.get('rsi_14', 50)
                                    ml_score = 50 + (15 if momentum > 5 else -15 if momentum < -5 else 0) + (10 if vol_ratio > 1.5 else 0) + (10 if 30 < rsi < 70 else 0)
                                    ml_score = max(0, min(100, ml_score))
                                    results.append({'ticker': ticker_symbol, 'ml_score': ml_score, 'momentum': momentum, 'volume_ratio': vol_ratio, 'rsi': rsi})
                                    
                                    # Save ML analysis results to ticker
                                    tm.update_ml_analysis(ticker_symbol, ml_score=ml_score, momentum=momentum, volume_ratio=vol_ratio, rsi=rsi)
                            except Exception as e:
                                logging.error(f"⚠️ Error analyzing {ticker_symbol}: {e}")
                            log_container.code(log_stream.getvalue())
                        status.update(label="✅ Analysis complete!", state="complete")
                
                alpha_factors_logger.removeHandler(st_handler)

                if results:
                    results.sort(key=lambda x: x['ml_score'], reverse=True)
                    st.success(f"✅ Analyzed {len(results)} tickers")
                    st.subheader("🏆 Top Opportunities from Your Tickers")
                    for i, result in enumerate(results[:5], 1):
                        score = result['ml_score']
                        emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns([1, 1, 1, 1, 1])
                        with col_r1:
                            st.write(f"{emoji} **{result['ticker']}**")
                        with col_r2:
                            st.write(f"Score: **{score:.0f}**/100")
                        with col_r3:
                            st.write(f"Momentum: {result['momentum']:+.1f}%")
                        with col_r4:
                            st.write(f"Vol: {result['volume_ratio']:.2f}x")
                        with col_r5:
                            st.write(f"RSI: {result['rsi']:.0f}")
                else:
                    st.warning("No results to display after analysis.")
    
    with tab5:
        st.header("🔍 Stock Intelligence")
        st.write("Analyze stocks in-depth with AI-powered insights. Use Dashboard tab for quick analysis.")
        st.info("💡 Tip: Use the Dashboard tab for comprehensive stock intelligence and analysis.")
    
    with tab6:
        st.header("🎯 Intelligent Strategy Advisor")
        st.write("Get personalized strategy recommendations based on comprehensive analysis.")
        
        # Check if we have analysis
        if not st.session_state.current_analysis:
            st.warning("⚠️ Please analyze a stock in the 'Dashboard' tab first!")
        else:
            analysis = st.session_state.current_analysis
            st.success(f"Using analysis for: **{analysis.ticker}** (${analysis.price}, {analysis.change_pct:+.2f}%)")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Your Trading Profile")

                user_experience = st.selectbox(
                    "Experience Level",
                    options=["Beginner", "Intermediate", "Advanced"],
                    key='user_experience_select'
                )

                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    options=["Conservative", "Moderate", "Aggressive"],
                    key='risk_tolerance_select'
                )

                capital_available = st.number_input(
                    "Available Capital ($)",
                    min_value=500,
                    max_value=1000000,
                    value=5000,
                    step=500
                )

            with col2:
                st.subheader("Your Market View")

                outlook = st.selectbox(
                    "Market Outlook for this Stock",
                    options=["Bullish", "Bearish", "Neutral"],
                    key='outlook_select',
                    help="What direction do you expect?"
                )

                st.write("**Current Analysis Summary:**")
                st.write(f"• Trend: {analysis.trend}")
                st.write(f"• RSI: {analysis.rsi} {'(Oversold)' if analysis.rsi < 30 else '(Overbought)' if analysis.rsi > 70 else '(Neutral)'}")
                st.write(f"• MACD: {analysis.macd_signal}")
                st.write(f"• IV Rank: {analysis.iv_rank}%")
                st.write(f"• Sentiment: {('Positive' if analysis.sentiment_score > 0.2 else 'Negative' if analysis.sentiment_score < -0.2 else 'Neutral')}")

            if st.button("🚀 Generate Strategy Recommendations", type="primary", use_container_width=True):
                with st.spinner("Analyzing optimal strategies..."):
                    recommendations = StrategyAdvisor.get_recommendations(
                        analysis=analysis,
                        user_experience=user_experience,
                        risk_tolerance=risk_tolerance,
                        capital_available=capital_available,
                        outlook=outlook
                    )

                    if recommendations:
                        st.subheader(f"📋 Top {len(recommendations)} Recommended Strategies for {analysis.ticker}")

                        # Display recommendations as clean cards
                        for idx, rec in enumerate(recommendations, 1):
                            confidence_pct = int(rec.confidence * 100)
                            badge = "🟢 High" if confidence_pct >= 70 else "🟡 Moderate" if confidence_pct >= 50 else "🟠 Low"

                            with st.container():
                                cols = st.columns([1, 3, 1])
                                with cols[0]:
                                    st.markdown(f"**#{idx}**")
                                    st.write(f"**{badge}**")
                                    st.progress(confidence_pct / 100)

                                with cols[1]:
                                    st.markdown(f"### {rec.strategy_name}")
                                    st.write(f"**Match:** {confidence_pct}% • **Risk:** {rec.risk_level} • **Level:** {rec.experience_level}")
                                    st.write("**Why this strategy?**")
                                    st.write(rec.reasoning)

                                    st.write("**When to use / Best conditions:**")
                                    for condition in rec.best_conditions:
                                        st.caption(f"• {condition}")

                                    # Optional examples and notes if present
                                    if hasattr(rec, 'examples') and rec.examples:
                                        st.write("**Examples:**")
                                        for ex in rec.examples:
                                            st.caption(f"• {ex}")

                                    if hasattr(rec, 'notes') and rec.notes:
                                        st.info(rec.notes)

                                with cols[2]:
                                    st.metric("Confidence", f"{confidence_pct}%")
                                    st.write("")
                                    st.write("**Risk/Reward**")
                                    st.write(f"• Max Loss: {rec.max_loss}")
                                    st.write(f"• Max Gain: {rec.max_gain}")

                                    if st.button(f"Select", key=f"use_strategy_{idx}"):
                                        st.session_state.selected_strategy = rec.action
                                        st.session_state.selected_ticker = analysis.ticker
                                        st.success(f"✅ Strategy selected! Go to 'Generate Signal' tab.")
                                    # Load Example Trade button - populates Generate Signal form with suggested defaults
                                    if st.button(f"Load Example Trade", key=f"load_example_{idx}"):
                                        # Derive suggested values from examples or defaults
                                        suggested_qty = 2
                                        suggested_iv = int(st.session_state.current_analysis.iv_rank if st.session_state.current_analysis else 48)
                                        suggested_dte = 30
                                        suggested_expiry = (datetime.now() + timedelta(days=suggested_dte)).date()
                                        # Use the current analysis price as a basis for suggested strike if available
                                        base_price = getattr(analysis, 'price', None) or (st.session_state.current_analysis.price if st.session_state.current_analysis else 10)
                                        suggested_strike = round(float(base_price) * 1.0, 2)
                                        # Set session state fields used by Generate Signal tab
                                        st.session_state.selected_strategy = rec.action
                                        st.session_state.selected_ticker = analysis.ticker
                                        st.session_state.example_trade = {
                                            'expiry': suggested_expiry,
                                            'strike': suggested_strike,
                                            'qty': suggested_qty,
                                            'iv_rank': suggested_iv,
                                            'estimated_risk': 200.0,
                                            'llm_score': float(rec.confidence)
                                        }
                                        st.success("✅ Example trade loaded. Go to 'Generate Signal' to review and send.")
                                    if st.button(f"Details", key=f"details_strategy_{idx}"):
                                        # Expand a modal-like view by showing an expander with full details
                                        with st.expander(f"Details - {rec.strategy_name}", expanded=True):
                                            st.write(rec.reasoning)
                                            st.write("**Best Conditions:**")
                                            for condition in rec.best_conditions:
                                                st.write(f"• {condition}")
                                            st.write("**Risk/Reward:**")
                                            st.write(f"• Max Loss: {rec.max_loss}")
                                            st.write(f"• Max Gain: {rec.max_gain}")
                                            if hasattr(rec, 'examples') and rec.examples:
                                                st.write("**Examples:**")
                                                for ex in rec.examples:
                                                    st.write(f"• {ex}")
                                            if hasattr(rec, 'notes') and rec.notes:
                                                st.write("**Notes:**")
                                                st.write(rec.notes)
                                    st.write("\n")
                    else:
                        st.warning("No suitable strategies found. Try adjusting your parameters.")
    
    with tab7:
        st.header("📊 Generate Trading Signal")
        
        if 'selected_strategy' in st.session_state:
            st.info(f"💡 Using recommended strategy: **{st.session_state.selected_strategy}** for **{st.session_state.get('selected_ticker', 'N/A')}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ticker_input = st.text_input(
                "Ticker Symbol",
                value=st.session_state.get('selected_ticker', 'SOFI'),
                help="Ticker symbol (e.g., AAPL). If you loaded a recommended strategy this may be prefilled."
            )
            ticker = ticker_input.upper() if isinstance(ticker_input, str) and ticker_input else ''
            
            allowed = list(st.session_state.config.allowed_strategies or [])
            default_idx = 0
            sel = st.session_state.get('selected_strategy')
            try:
                if sel in allowed:
                    default_idx = allowed.index(sel)
            except Exception:
                default_idx = 0

            action = st.selectbox(
                "Strategy",
                options=allowed,
                index=default_idx,
                key='signal_strategy_select',
                help="Choose the option strategy to send. Select from recommended or custom strategies."
            )
            
            # Prefill from example_trade when available
            example = st.session_state.get('example_trade') or {}
            default_expiry = example.get('expiry', (datetime.now() + timedelta(days=30)).date())
            expiry_date = st.date_input(
                "Expiration Date",
                value=default_expiry,
                min_value=datetime.now().date(),
                max_value=(datetime.now() + timedelta(days=365)).date(),
                help="Expiration date for the option contract. Be mindful of DTE (days to expiration)."
            )

            default_strike = example.get('strike', 9.0)
            strike = st.number_input("Strike Price", min_value=0.0, value=float(default_strike), step=0.5, format="%.2f")
            st.caption("Strike price for the option(s). Use analysis support/resistance as a reference.")
        
        with col2:
            qty = st.number_input("Quantity (contracts)", min_value=1, max_value=10, value=int(example.get('qty', 2)))
            st.caption("Number of contracts (1 contract = 100 shares). Keep within your capital limits.")
            # Use example or analysis IV if available
            # Determine a safe numeric default for IV rank (never None)
            _ex_iv = example.get('iv_rank') if isinstance(example.get('iv_rank'), (int, float)) else None
            _curr_iv = None
            if st.session_state.current_analysis and getattr(st.session_state.current_analysis, 'iv_rank', None) is not None:
                try:
                    _curr_iv = float(st.session_state.current_analysis.iv_rank)
                except Exception:
                    _curr_iv = None

            if _ex_iv is not None:
                default_iv = float(_ex_iv)
            elif _curr_iv is not None:
                default_iv = _curr_iv
            else:
                default_iv = 48.0

            iv_rank = st.slider("IV Rank (%)", 0, 100, int(default_iv))
            st.caption("Implied Volatility Rank — helps decide premium selling vs buying strategies.")

            estimated_risk = st.number_input("Estimated Risk ($)", min_value=0.0, value=float(example.get('estimated_risk', 200.0)), step=50.0)
            st.caption("Estimated maximum risk for the trade (approx). Used by guardrails.")

            llm_score = st.slider("AI Confidence", 0.0, 1.0, float(example.get('llm_score', 0.77)), 0.01)
            st.caption("AI confidence score for this signal (0.0 low → 1.0 high). Use as guidance, not final truth.")
        
        note = st.text_area(
            "Signal Note",
            value=f"AI-score={llm_score}; IVR={iv_rank}; Strategy={action}",
            help="Additional context"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Validate", width='stretch'):
                dte = calculate_dte(expiry_date.strftime('%Y-%m-%d'))
                
                signal = {
                    'ticker': ticker,
                    'action': action,
                    'expiry': expiry_date.strftime('%Y-%m-%d'),
                    'dte': dte,
                    'strike': strike,
                    'qty': qty,
                    'iv_rank': iv_rank,
                    'estimated_risk': estimated_risk,
                    'llm_score': llm_score,
                    'note': note
                }
                
                is_valid, message = st.session_state.validator.validate_signal(signal)
                
                if is_valid:
                    st.success(f"✅ {message}")
                    st.json(signal)
                else:
                    st.error(f"❌ {message}")
        
        with col2:
            if st.button("🚀 Send Signal", width='stretch', type="primary"):
                dte = calculate_dte(expiry_date.strftime('%Y-%m-%d'))
                
                # Map action to Option Alpha format
                oa_action_mapping = {
                    'SELL_PUT': 'BPS',  # Bull Put Spread
                    'SELL_CALL': 'BCS', # Bear Call Spread
                    'BUY_PUT': 'PUT',
                    'BUY_CALL': 'CALL'
                }
                
                oa_action = oa_action_mapping.get(action, action)
                
                # Determine market condition based on IV rank
                market_condition = "high_vol" if iv_rank > 60 else "normal" if iv_rank > 30 else "low_vol"
                
                signal = {
                    'symbol': 'SPX',  # Fixed to SPX for your bot
                    'action': oa_action,
                    'expiry': expiry_date.strftime('%Y-%m-%d'),
                    'dte': dte,
                    'strike': strike,
                    'quantity': qty,
                    'iv_rank': iv_rank,
                    'market_condition': market_condition,
                    'estimated_risk': estimated_risk,
                    'llm_score': llm_score,
                    'note': f"AI Analysis: {note}"
                }
                
                is_valid, validation_msg = st.session_state.validator.validate_signal(signal)
                
                if not is_valid:
                    st.error(f"❌ {validation_msg}")
                else:
                    if paper_mode:
                        st.info("📝 Paper mode: Signal logged")
                        success = True
                        message = "Signal logged in paper trading mode"
                    else:
                        client = OptionAlphaClient(webhook_url)
                        success, message = client.send_signal(signal)
                    
                    if success:
                        st.session_state.validator.record_order(signal)
                        st.session_state.signal_history.append({
                            **signal,
                            'timestamp': datetime.now().isoformat(),
                            'status': 'Paper' if paper_mode else 'Live',
                            'result': message
                        })
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        
        with col3:
            if st.button("🔄 Reset", width='stretch'):
                st.session_state.validator.reset_daily_counters()
                st.success("Counters reset!")
        
        st.divider()
        st.subheader("Current Status")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("Daily Orders", f"{st.session_state.validator.daily_orders}/{st.session_state.config.max_daily_orders}")
        with m2:
            st.metric("Daily Risk", f"${st.session_state.validator.daily_risk:.0f}/${st.session_state.config.max_daily_risk:.0f}")
        with m3:
            in_hours, _ = st.session_state.validator.is_trading_hours()
            st.metric("Trading Hours", "✅ Open" if in_hours else "❌ Closed")
        with m4:
            st.metric("Mode", "📝 Paper" if paper_mode else "🔴 Live")
    
    with tab8:
        st.header("📜 Signal History")
        
        if st.session_state.signal_history:
            df = pd.DataFrame(st.session_state.signal_history)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Signals", len(df))
            with col2:
                st.metric("Total Risk", f"${df['estimated_risk'].sum():.2f}")
            with col3:
                st.metric("Avg AI Score", f"{df['llm_score'].mean():.2f}")
            
            # Enhanced interactive data editor with new Streamlit features
            st.subheader("📊 Interactive Signal Management")
            
            # Prepare data for editing (only editable columns)
            editable_df = df[['ticker', 'action', 'strike', 'qty', 'estimated_risk', 'llm_score', 'note']].copy()
            editable_df['timestamp'] = df['timestamp']
            editable_df['status'] = df['status']
            
            # Use st.data_editor for interactive editing
            edited_df = st.data_editor(
                editable_df.sort_values('timestamp', ascending=False),
                width='stretch',
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "action": st.column_config.SelectboxColumn(
                        "Strategy",
                        options=st.session_state.config.allowed_strategies,
                        width="medium"
                    ),
                    "strike": st.column_config.NumberColumn("Strike", format="$%.2f", width="small"),
                    "qty": st.column_config.NumberColumn("Quantity", min_value=1, max_value=10, width="small"),
                    "estimated_risk": st.column_config.NumberColumn("Risk", format="$%.2f", width="small"),
                    "llm_score": st.column_config.NumberColumn("AI Score", min_value=0.0, max_value=1.0, format="%.2f", width="small"),
                    "note": st.column_config.TextColumn("Notes", width="large"),
                    "timestamp": st.column_config.DatetimeColumn("Time", width="medium"),
                    "status": st.column_config.TextColumn("Status", width="small")
                },
                disabled=["timestamp", "status"]  # Don't allow editing of these
            )
            
            # Check if data was modified
            if not edited_df.equals(editable_df.sort_values('timestamp', ascending=False)):
                st.success("✅ Signal data updated! Changes will be saved to session state.")
                # Update session state with edited data
                # Note: In a real app, you'd want to save this to a database
                st.session_state.signal_history = edited_df.to_dict('records')
            
            # Enhanced performance analytics with new Streamlit features
            st.subheader("📊 Performance Analytics")
            
            # Strategy performance analysis
            strategy_performance = df.groupby('action').agg({
                'estimated_risk': ['count', 'sum', 'mean'],
                'llm_score': 'mean'
            }).round(2)
            strategy_performance.columns = ['Signal Count', 'Total Risk', 'Avg Risk', 'Avg AI Score']
            strategy_performance = strategy_performance.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Strategy Performance by Signal Count**")
                # Enhanced bar chart with sorting
                st.bar_chart(
                    strategy_performance.set_index('action')['Signal Count'],
                    sort="desc"  # New sorting feature
                )
            
            with col2:
                st.write("**Strategy Performance by Total Risk**")
                st.bar_chart(
                    strategy_performance.set_index('action')['Total Risk'],
                    sort="desc"  # New sorting feature
                )
            
            # Ticker performance analysis
            ticker_performance = df.groupby('ticker').agg({
                'estimated_risk': ['count', 'sum'],
                'llm_score': 'mean'
            }).round(2)
            ticker_performance.columns = ['Signal Count', 'Total Risk', 'Avg AI Score']
            ticker_performance = ticker_performance.reset_index()
            
            st.write("**Ticker Performance Analysis**")
            st.bar_chart(
                ticker_performance.set_index('ticker')['Signal Count'],
                sort="desc"
            )
            
            # Enhanced export with filtering options
            st.subheader("📥 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                export_format = st.selectbox("Export Format", ["CSV", "JSON", "Excel"], key='export_format_select')
            with col2:
                filter_status = st.multiselect("Filter by Status", ["Paper", "Live"], default=["Paper", "Live"])
            with col3:
                if st.button("📥 Export Data", type="primary"):
                    filtered_df = df[df['status'].isin(filter_status)]
                    
                    if export_format == "CSV":
                        # Add UTF-8 BOM for Excel compatibility
                        csv = '\ufeff' + filtered_df.to_csv(index=False)
                        st.download_button("Download CSV", csv.encode('utf-8-sig'), "signals.csv", "text/csv")
                    elif export_format == "JSON":
                        json_data = filtered_df.to_json(orient='records', indent=2)
                        st.download_button("Download JSON", json_data, "signals.json", "application/json")
                    elif export_format == "Excel":
                        # For Excel export, you'd need openpyxl: pip install openpyxl
                        st.info("Excel export requires openpyxl. Install with: pip install openpyxl")
        else:
            st.info("No signals generated yet")
    
    with tab9:
        st.header("📚 Complete Strategy Guide")
        
        for i, (strategy_key, strategy_info) in enumerate(StrategyAdvisor.STRATEGIES.items()):
            with st.expander(f"{strategy_info['name']} - {strategy_info['experience']} | {strategy_info['risk_level']} Risk"):
                st.write(f"**Description:** {strategy_info['description']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Risk Profile:**")
                    st.write(f"• Risk: {strategy_info['risk_level']}")
                    st.write(f"• Max Loss: {strategy_info['max_loss']}")
                    st.write(f"• Max Gain: {strategy_info['max_gain']}")
                    st.write(f"• Win Rate: {strategy_info['typical_win_rate']}")
                
                with col2:
                    st.write("**Requirements:**")
                    st.write(f"• Experience: {strategy_info['experience']}")
                    st.write(f"• Capital: {strategy_info['capital_req']}")
                    st.write("**Best For:**")
                    for condition in strategy_info['best_for']:
                        st.write(f"• {condition}")

            st.divider()
            st.header("🎓 Options Education & Returns Calculator")
            st.write("Learn option terms, see example usages, and try a small returns calculator for common option structures.")

            # Use external pricing helpers (supports American option pricing via binomial tree)
            from services.options_pricing import black_scholes_price, binomial_american_price, greeks_finite_difference


            def calc_long_option_pnl(premium_paid: float, strike: float, underlying_price: float, contracts: int = 1, side: str = 'Call', use_bs: bool = False, days_to_expiry: int = 30, iv: float = 0.5, rf: float = 0.01):
                """Calculate basic P&L metrics for a long call/put bought for a single-leg option.

                Returns a dict with: position_value, pnl, roi_pct, breakeven
                """
                # position value for a call at expiry (intrinsic only)
                is_call = side.lower().startswith('c')
                intrinsic = max(0.0, underlying_price - strike) if is_call else max(0.0, strike - underlying_price)
                if use_bs:
                    # Convert days to years
                    T = max(days_to_expiry, 0) / 365.0
                    theo = black_scholes_price(underlying_price, strike, T, rf, iv, is_call=is_call)
                    position_value = theo * 100 * contracts
                else:
                    position_value = intrinsic * 100 * contracts

                cost = premium_paid * 100 * contracts
                pnl = position_value - cost
                roi_pct = (pnl / cost * 100) if cost != 0 else 0.0
                breakeven = strike + premium_paid if is_call else strike - premium_paid
                return {
                    'position_value': position_value,
                    'pnl': pnl,
                    'roi_pct': roi_pct,
                    'breakeven': breakeven,
                    'max_loss': -cost,
                    'max_gain': 'Unlimited' if True else None
                }

            def calc_vertical_spread_pnl(premium_paid_long: float, premium_received_short: float, long_strike: float, short_strike: float, underlying_price: float, contracts: int = 1, is_call: bool = True, use_bs: bool = False, days_to_expiry: int = 30, iv_long: float = 0.5, iv_short: float = 0.5, rf: float = 0.01):
                """Calculate P&L for a vertical spread (debit or credit depending on premiums).

                Returns dict with pnl, max_loss, max_gain, roi_pct, breakeven
                """
                # net premium paid (debit positive means paid)
                net_premium = (premium_paid_long - premium_received_short)
                cost = net_premium * 100 * contracts

                # Use Black-Scholes theoretical pricing for legs if requested
                if use_bs:
                    T = max(days_to_expiry, 0) / 365.0
                    price_long = black_scholes_price(underlying_price, long_strike, T, rf, iv_long, is_call=is_call)
                    price_short = black_scholes_price(underlying_price, short_strike, T, rf, iv_short, is_call=is_call)
                    position_value = (price_long - price_short) * 100 * contracts if is_call else (price_short - price_long) * 100 * contracts
                else:
                    # For a call vertical, intrinsic difference
                    if is_call:
                        intrinsic_long = max(0.0, underlying_price - long_strike)
                        intrinsic_short = max(0.0, underlying_price - short_strike)
                    else:
                        intrinsic_long = max(0.0, long_strike - underlying_price)
                        intrinsic_short = max(0.0, short_strike - underlying_price)

                    # Position value at expiry = (intrinsic_long - intrinsic_short) * 100
                    position_value = max(0.0, intrinsic_long - intrinsic_short) * 100 * contracts

                pnl = position_value - cost

                width = abs(short_strike - long_strike) * 100 * contracts
                # Max gain for a debit spread is width - cost (if you paid a net debit), for credit spread it's credit received minus assignment costs
                if net_premium >= 0:
                    # net debit
                    max_gain = max(0.0, width - cost)
                    max_loss = -cost
                else:
                    # net credit
                    max_gain = -cost  # credit received
                    max_loss = -(width + cost)  # worst-case if spread assigned against you

                roi_pct = (pnl / abs(cost) * 100) if cost != 0 else 0.0

                # Breakeven approximations
                if is_call:
                    breakeven = long_strike + net_premium
                else:
                    breakeven = long_strike - net_premium

                return {
                    'position_value': position_value,
                    'pnl': pnl,
                    'max_gain': max_gain,
                    'max_loss': max_loss,
                    'roi_pct': roi_pct,
                    'breakeven': breakeven,
                    'net_premium': net_premium
                }


            # --- UI wiring: Load from selected strategy, persist small calc history ---
            if 'calc_history' not in st.session_state:
                st.session_state.calc_history = []


            edu_col1, edu_col2 = st.columns([1, 1])

            with edu_col1:
                st.subheader("Key Option Terms")
                st.markdown("""
                - Strike: The agreed exercise price for the option.
                - Premium: Price paid to buy the option (per share).
                - DTE: Days to expiration.
                - IV / IV Rank: Implied volatility; higher IV usually means higher option prices.
                - Intrinsic Value: The in-the-money amount (if any).
                - Time Value: Portion of premium attributable to time/volatility.
                - Breakeven: Underlying price at expiry where the trade neither makes nor loses money.
                """, unsafe_allow_html=True)

                st.subheader("When to use")
                st.write("Long calls/puts: directional plays with limited risk (premium). Good when expecting big moves.")
                st.write("Vertical spreads: reduce cost and cap profit; useful for defined-risk directional or neutral trades.")

            with edu_col2:
                st.subheader("Try the Returns Calculator")
                # Use a unique key for this educational calculator to avoid collisions with other widgets
                calc_type = st.selectbox("Structure", ["Long Option", "Vertical Spread"], key=f"tab5_edu_calc_type_selectbox_{i}")

                if calc_type == "Long Option":
                    # Prefill from example_trade if available
                    ex = st.session_state.get('example_trade', {})
                    side = st.selectbox("Side", ["Call", "Put"], index=0, key=f'tab5_longoption_side_select_{i}')
                    premium = st.number_input("Premium (per share, $)", min_value=0.0, value=float(ex.get('premium', 1.50)), step=0.01, format="%.2f", key=f"premium_{i}")
                    strike = st.number_input("Strike ($)", min_value=0.01, value=float(ex.get('strike', 50.0)), step=0.01, format="%.2f", key=f"strike_{i}")
                    underlying = st.number_input("Underlying Price at Expiry ($)", min_value=0.0, value=float(ex.get('underlying', strike + 5)), step=0.01, format="%.2f", key=f"underlying_{i}")
                    contracts = st.number_input("Contracts", min_value=1, max_value=100, value=int(ex.get('qty', 1)), key=f"contracts_{i}")
                    days_to_expiry = st.number_input("Days to Expiry", min_value=0, max_value=365, value=int(ex.get('dte', 30)), key=f"days_to_expiry_{i}")
                    iv = st.number_input("Implied Volatility (annual %, e.g. 50 for 50%)", min_value=0.0, max_value=500.0, value=float(ex.get('iv_pct', 50.0)), step=0.1, key=f"iv_{i}") / 100.0
                    model_choice = st.radio("Pricing model", ["American (binomial)", "European (Black-Scholes)"], index=0, key=f"model_choice_{i}")
                    use_bs = (model_choice == "European (Black-Scholes)")
                    # Allow user to tune binomial steps for American pricing
                    # Use a session_state-backed slider so we can auto-apply recommended steps
                    if 'binomial_steps' not in st.session_state:
                        st.session_state.binomial_steps = 300
                    binomial_steps = st.slider("Binomial steps (American model accuracy)", min_value=10, max_value=5000, value=st.session_state.binomial_steps, step=10, key=f'binomial_steps_{i}', help="More steps -> higher accuracy but slower.")

                    # Convergence score weighting: let user trade off time vs accuracy
                    st.subheader("Convergence tuning")
                    weight_time = st.slider("Weight: time vs accuracy (0 = ignore time, 1 = equal, >1 = favor time)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key=f"weight_time_{i}", help="Higher values increase the importance of time in the combined conv_score.")

                    # Named wrapper for binomial pricer to pass into greeks wrapper (helps tracebacks)
                    def binomial_pricer(S_loc: float, K_loc: float, T_loc: float, r_loc: float, sigma_loc: float, is_call_loc: bool = True):
                        return binomial_american_price(S_loc, K_loc, T_loc, r_loc, sigma_loc, is_call=is_call_loc, steps=binomial_steps)

                    # Sensitivity explorer: quick bumps for IV and underlying to show P&L / delta
                    st.subheader("Sensitivity explorer")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        bump_iv = st.number_input("Bump IV (pts, e.g. 1 = +1%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1, key=f"bump_iv_{i}")
                    with col_b:
                        bump_underlying = st.number_input("Bump underlying ($)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.01, key=f"bump_underlying_{i}")

                    if st.button("Load from selected strategy/example", key=f"load_example_{i}"):
                        # If an example_trade exists, prefill fields (session state will supply next rerun)
                        if 'example_trade' in st.session_state:
                            ex2 = st.session_state.example_trade
                            st.session_state._calc_prefill = {
                                'side': 'Call',
                                'premium': ex2.get('premium', ex2.get('estimated_risk', 1.5) / 100.0),
                                'strike': ex2.get('strike'),
                                'underlying': ex2.get('strike'),
                                'qty': ex2.get('qty', 1),
                                'dte': 30,
                                'iv_pct': ex2.get('iv_rank', 48.0)
                            }
                            # We no longer call experimental_rerun for compatibility.
                            st.success("Example trade loaded into session state. Adjust values if needed then click Calculate.")

                    if st.button("Calculate Long Option P&L", key=f"calculate_long_{i}"):
                        res = calc_long_option_pnl(premium, strike, underlying, int(contracts), side, use_bs, int(days_to_expiry), float(iv), rf=0.01)
                        st.write(f"Position value: ${res['position_value']:.2f}")
                        st.write(f"P&L: ${res['pnl']:.2f}")
                        st.write(f"ROI: {res['roi_pct']:.1f}%")
                        st.write(f"Breakeven: ${res['breakeven']:.2f}")
                        st.write(f"Max Loss: ${res['max_loss']:.2f}")
                        # Show greeks depending on model
                        T = max(int(days_to_expiry), 0) / 365.0
                        is_call_flag = side.lower().startswith('c')
                        if use_bs:
                            # greeks_finite_difference will detect black_scholes_price and return analytic greeks
                            greeks = greeks_finite_difference(black_scholes_price, underlying, strike, T, 0.01, iv, is_call=is_call_flag)
                            model_note = "European Black-Scholes (analytic greeks)"
                        else:
                            # Binomial American greeks via finite diffs around the named binomial_pricer
                            greeks = greeks_finite_difference(binomial_pricer, underlying, strike, T, 0.01, iv, is_call=is_call_flag)
                            model_note = f"American (binomial CRR, steps={binomial_steps}) — supports early exercise"

                        # vega in our API is per 1 vol point (1% = +1). Also show per 0.01 (decimal) for clarity.
                        vega_per_1pt = greeks['vega']
                        vega_per_decimal = greeks['vega'] / 100.0
                        # Absolute dollar vega for the given number of contracts (per 1 vol pt)
                        contracts_int = int(contracts)
                        vega_dollars = vega_per_1pt * 100.0 * contracts_int
                        # percent of cost (cost = premium * 100 * contracts)
                        cost = premium * 100.0 * contracts_int
                        pct_of_cost = (vega_dollars / cost * 100.0) if cost != 0 else None

                        st.write(f"Delta: {greeks['delta']:.3f} • Gamma: {greeks['gamma']:.4f} • Vega (per 1 vol pt): {vega_per_1pt:.2f} • Vega (per 0.01 decimal): {vega_per_decimal:.4f} • Theta/day: {greeks['theta']:.2f}")
                        st.write(f"Vega absolute: ${vega_dollars:.2f} per 1 vol pt for {contracts_int} contract(s)" + (f" • {pct_of_cost:.1f}% of cost" if pct_of_cost is not None else ""))
                        st.info(f"Note: {model_note}. Results are approximations; Black-Scholes assumes European exercise and constant vol. American pricing uses a binomial tree numeric method.")
                        # Save to history
                        st.session_state.calc_history.append({
                            'type': 'long', 'side': side, 'premium': premium, 'strike': strike,
                            'underlying': underlying, 'contracts': int(contracts), 'dte': int(days_to_expiry), 'iv': iv, 'use_bs': bool(use_bs), 'binomial_steps': int(binomial_steps),
                            'result_pnl': res['pnl'], 'result_roi': res['roi_pct'], 'timestamp': datetime.now().isoformat()
                        })

                        # Sensitivity outputs: simple bump calculations
                        if bump_iv != 0.0 or bump_underlying != 0.0:
                            bumped_iv = iv + (bump_iv / 100.0)
                            bumped_under = underlying + bump_underlying
                            # Price with bumps
                            if use_bs:
                                base_price = black_scholes_price(underlying, strike, T, 0.01, iv, is_call=is_call_flag)
                                bumped_price = black_scholes_price(bumped_under, strike, T, 0.01, bumped_iv, is_call=is_call_flag)
                            else:
                                base_price = binomial_american_price(underlying, strike, T, 0.01, iv, is_call=is_call_flag, steps=binomial_steps)
                                bumped_price = binomial_american_price(bumped_under, strike, T, 0.01, bumped_iv, is_call=is_call_flag, steps=binomial_steps)

                            pnl_base = (base_price - premium) * 100 * int(contracts)
                            pnl_bumped = (bumped_price - premium) * 100 * int(contracts)
                            st.subheader("Sensitivity results")
                            st.write(f"Base theoretical price: ${base_price:.2f} — P&L: ${pnl_base:.2f}")
                            st.write(f"Bumped price: ${bumped_price:.2f} — P&L: ${pnl_bumped:.2f}")

                        # Micro-benchmark: convergence test for binomial pricer
                        if not use_bs:
                            st.subheader("Binomial convergence micro-benchmark")

                            # Benchmark caps
                            max_steps_allowed = st.number_input("Max benchmark steps cap", min_value=100, max_value=10000, value=5000, step=100, key=f"max_steps_allowed_{i}")
                            max_runtime_seconds = st.number_input("Max benchmark runtime (s)", min_value=1.0, max_value=120.0, value=10.0, step=1.0, key=f"max_runtime_seconds_{i}")

                            if 'benchmark_status' not in st.session_state:
                                st.session_state.benchmark_status = {'running': False, 'results': None, 'started_at': None}

                            def _run_benchmark_background(underlying_loc, strike_loc, T_loc, iv_loc, is_call_loc, step_list_loc, max_runtime_loc):
                                import time as _time
                                res = []
                                t_start = _time.perf_counter()
                                for s in step_list_loc:
                                    # Check runtime cap
                                    if (_time.perf_counter() - t_start) > max_runtime_loc:
                                        break
                                    # Respect step cap
                                    if s > max_steps_allowed:
                                        break
                                    t0 = _time.perf_counter()
                                    p = binomial_american_price(underlying_loc, strike_loc, T_loc, 0.01, iv_loc, is_call=is_call_loc, steps=s)
                                    t1 = _time.perf_counter()
                                    res.append({'steps': s, 'price': p, 'time_s': t1 - t0})

                                # Save results and mark not running
                                st.session_state.benchmark_status['results'] = res
                                st.session_state.benchmark_status['running'] = False
                                st.session_state.benchmark_status['started_at'] = None

                            # Start benchmark async
                            tol_pct = st.number_input("Tolerance (relative %)", min_value=1e-6, max_value=100.0, value=0.1, step=0.01, key=f"tol_pct_{i}", help="Stop when relative price change between successive runs is below this percentage (e.g. 0.1 = 0.1%).")
                            st.caption("Tolerance is relative to the previous run's price. Lower values require tighter convergence and more computation. For very small option prices, consider increasing tolerance to avoid excessive runs.")

                            if st.button("Run convergence benchmark (background)", key=f"run_benchmark_{i}"):
                                import threading as _threading
                                # Build step list, include selected binomial_steps and some canonical points
                                step_list = [50, 200, 500, 1000, max(2000, binomial_steps)]
                                step_list = sorted(list(dict.fromkeys(step_list)))
                                st.session_state.benchmark_status['running'] = True
                                st.session_state.benchmark_status['results'] = None
                                st.session_state.benchmark_status['started_at'] = __import__('time').time()
                                # initialize incremental results container
                                st.session_state.benchmark_status['results'] = []
                                st.session_state.benchmark_status['running'] = True
                                st.session_state.benchmark_status['started_at'] = __import__('time').time()

                                # wrap the background runner to append interim results
                                def _bg_runner_append(*args, **kwargs):
                                    import time as _time
                                    # call into the original runner but append results as they come
                                    underlying_loc, strike_loc, T_loc, iv_loc, is_call_loc, steps_loc, max_runtime_loc = args
                                    t_start = _time.perf_counter()
                                    for s in steps_loc:
                                        if (_time.perf_counter() - t_start) > max_runtime_loc:
                                            break
                                        if s > max_steps_allowed:
                                            break
                                        t0 = _time.perf_counter()
                                        p = binomial_american_price(underlying_loc, strike_loc, T_loc, 0.01, iv_loc, is_call=is_call_loc, steps=s)
                                        t1 = _time.perf_counter()
                                        # append result incrementally
                                        st.session_state.benchmark_status['results'].append({'steps': s, 'price': p, 'time_s': t1 - t0})
                                    # finished
                                    st.session_state.benchmark_status['running'] = False
                                    st.session_state.benchmark_status['started_at'] = None

                                th = _threading.Thread(target=_bg_runner_append, args=(underlying, strike, T, iv, is_call_flag, step_list, float(max_runtime_seconds)), daemon=True)
                                th.start()

                            # Adaptive benchmark: auto-increase steps until price change < tol or caps reached
                            growth_strategy = st.selectbox("Adaptive growth strategy", ["doubling", "multiply", "additive"], index=0, key=f'tab5_growth_strategy_select_{i}', help="How step counts increase between iterations")
                            growth_param = st.number_input("Growth parameter", min_value=1.1, max_value=10.0, value=2.0, step=0.1, key=f"growth_param_{i}", help="Multiplier for 'multiply' strategy, ignored for doubling; for additive, this is the additive step size")

                            if st.button("Run adaptive convergence (background)", key=f"run_adaptive_{i}"):
                                import threading as _threading
                                def _run_adaptive(underlying_loc, strike_loc, T_loc, iv_loc, is_call_loc, start_steps, tol_loc, max_runtime_loc, strategy, param):
                                    import time as _time
                                    res = []
                                    t_start = _time.perf_counter()
                                    # Start from start_steps and double until tolerance met
                                    s = max(50, start_steps)
                                    last_price = None
                                    while True:
                                        if (_time.perf_counter() - t_start) > max_runtime_loc:
                                            break
                                        if s > max_steps_allowed:
                                            break
                                        t0 = _time.perf_counter()
                                        p = binomial_american_price(underlying_loc, strike_loc, T_loc, 0.01, iv_loc, is_call=is_call_loc, steps=s)
                                        t1 = _time.perf_counter()
                                        res.append({'steps': s, 'price': p, 'time_s': t1 - t0})
                                        if last_price is not None:
                                            # use relative change: |p - last| / |last| < tol_loc
                                            if last_price != 0:
                                                rel = abs(p - last_price) / abs(last_price)
                                            else:
                                                rel = abs(p - last_price)
                                            if rel < tol_loc:
                                                break
                                        last_price = p
                                        # Increase according to chosen strategy
                                        if strategy == 'doubling':
                                            if s < 100:
                                                s *= 2
                                            else:
                                                s = int(s * 1.5)
                                        elif strategy == 'multiply':
                                            s = int(s * float(param))
                                        else:  # additive
                                            s = int(s + float(param))

                                    st.session_state.benchmark_status['results'] = res
                                    st.session_state.benchmark_status['running'] = False
                                    st.session_state.benchmark_status['started_at'] = None

                                st.session_state.benchmark_status['running'] = True
                                st.session_state.benchmark_status['results'] = None
                                st.session_state.benchmark_status['started_at'] = __import__('time').time()
                                # pass tolerance as decimal fraction to the adaptive runner
                                th2 = _threading.Thread(target=_run_adaptive, args=(underlying, strike, T, iv, is_call_flag, binomial_steps, float(tol_pct) / 100.0, float(max_runtime_seconds), growth_strategy, float(growth_param)), daemon=True)
                                th2.start()

                            # Show status
                            auto_refresh = st.checkbox("Auto-refresh progress", value=True)

                            if st.session_state.benchmark_status.get('running'):
                                st.info("Benchmark running in background...")

                            # Show intermediate results table if present
                            results = st.session_state.benchmark_status.get('results') or []
                            if results:
                                import pandas as _pd
                                df_partial = _pd.DataFrame(results)
                                st.write("Intermediate results:")
                                st.table(df_partial)

                            # Auto-refresh logic: try to rerun if available, otherwise show Refresh button
                            if auto_refresh and st.session_state.benchmark_status.get('running'):
                                rerun_fn = getattr(st, 'rerun', None)
                                if callable(rerun_fn):
                                    import time as _time
                                    _time.sleep(0.5)
                                    try:
                                        rerun_fn()
                                    except Exception:
                                        pass
                                else:
                                    if st.button("Refresh progress", key=f"refresh_progress_{i}"):
                                        pass

                            # If benchmark finished, show final results and plots
                            if not st.session_state.benchmark_status.get('running'):
                                results_final = st.session_state.benchmark_status.get('results') or []
                                if results_final:
                                    # Baseline is last price
                                    baseline = results_final[-1]['price']
                                    st.write("Convergence results (price, time):")
                                    for r in results_final:
                                        st.write(f"steps={r['steps']:5d}  price=${r['price']:.4f}  dt={r['time_s']*1000:.1f}ms  delta={r['price']-baseline:+.4f}")

                                    # Plot price vs steps and time vs steps and show tolerance line
                                    try:
                                        import pandas as _pd
                                        df = _pd.DataFrame(results_final)
                                        df = df.set_index('steps')

                                        # Price convergence plot with tolerance line
                                        try:
                                            import matplotlib.pyplot as _plt
                                            fig, ax = _plt.subplots()
                                            ax.plot(df.index, df['price'], marker='o')
                                            # add tolerance line around baseline (use list of colors and accepted linestyle)
                                            # draw tolerance band as relative fraction of baseline
                                            tol_frac = float(tol_pct) / 100.0
                                            ax.hlines([baseline * (1.0 + tol_frac), baseline * (1.0 - tol_frac)], xmin=df.index.min(), xmax=df.index.max(), colors=['r', 'r'], linestyles='dashed')
                                            ax.set_xlabel('steps')
                                            ax.set_ylabel('price')
                                            ax.set_title('Price vs Steps (convergence)')
                                            st.pyplot(fig)
                                        except Exception:
                                            # Fallback to simple Streamlit line chart
                                            st.line_chart(df['price'])

                                        # Time plot
                                        st.line_chart(df['time_s'])

                                        # Combined convergence metric: relative change * time per (unit steps)
                                        try:
                                            dfc = df.copy()
                                            # relative absolute change between successive prices
                                            dfc['rel_change'] = dfc['price'].pct_change().abs().fillna(0.0)
                                            # avoid divide by zero for index cast
                                            idx_vals = _pd.Series(dfc.index.astype(float)).replace(0.0, 1.0).values
                                            dfc['time_per_step'] = dfc['time_s'] / idx_vals
                                            # convergence score: rel_change * time_per_step (lower is better)
                                            # apply user weight: conv_score = rel_change * (time_per_step ** weight_time)
                                            dfc['conv_score'] = dfc['rel_change'] * (dfc['time_per_step'] ** float(weight_time))
                                            # show table with new metrics
                                            st.write("Convergence metrics (lower conv_score is better):")
                                            st.dataframe(dfc[['price', 'rel_change', 'time_per_step', 'conv_score']])
                                            # recommend step minimizing conv_score
                                            best_idx = dfc['conv_score'].idxmin()
                                            best_row = dfc.loc[best_idx]
                                            st.success(f"Recommended steps: {int(best_idx)} (conv_score={best_row['conv_score']:.6g}, price=${best_row['price']:.4f})")
                                            if st.button("Auto-apply recommended steps"):
                                                # set the session_state slider value to recommended steps (clamped to slider bounds)
                                                new_steps = int(best_idx)
                                                new_steps = max(10, min(5000, new_steps))
                                                st.session_state['binomial_steps'] = new_steps
                                                rerun_fn = getattr(st, 'rerun', None)
                                                if callable(rerun_fn):
                                                    try:
                                                        rerun_fn()
                                                    except Exception:
                                                        pass
                                            # plot conv_score
                                            st.line_chart(dfc['conv_score'])
                                        except Exception:
                                            pass
                                    except Exception:
                                        st.write("Benchmark finished but plotting failed.")
                                else:
                                    st.write("Benchmark finished but returned no results (capped). Try increasing caps.")

                else:
                    # Vertical spread inputs
                    is_call = st.radio("Type", ["Call Vertical", "Put Vertical"], key=f"is_call_{i}") == "Call Vertical"
                    ex = st.session_state.get('example_trade', {})
                    long_strike = st.number_input("Long Strike ($)", min_value=0.01, value=float(ex.get('long_strike', 48.0)), step=0.01, format="%.2f", key=f"long_strike_{i}")
                    short_strike = st.number_input("Short Strike ($)", min_value=0.01, value=float(ex.get('short_strike', 52.0)), step=0.01, format="%.2f", key=f"short_strike_{i}")
                    premium_long = st.number_input("Premium paid for long leg ($)", min_value=0.0, value=float(ex.get('premium_long', 2.00)), step=0.01, format="%.2f", key=f"premium_long_{i}")
                    premium_short = st.number_input("Premium received for short leg ($)", min_value=0.0, value=float(ex.get('premium_short', 0.50)), step=0.01, format="%.2f", key=f"premium_short_{i}")
                    underlying_price = st.number_input("Underlying Price at Expiry ($)", min_value=0.0, value=float(ex.get('underlying', (long_strike + short_strike) / 2)), step=0.01, format="%.2f", key=f"underlying_price_{i}")
                    contracts_vs = st.number_input("Contracts", min_value=1, max_value=100, value=int(ex.get('qty', 1)), key=f"contracts_vs_{i}")
                    days_to_expiry = st.number_input("Days to Expiry", min_value=0, max_value=365, value=int(ex.get('dte', 30)), key=f"days_to_expiry_vs_{i}")
                    iv_long = st.number_input("IV Long (%)", min_value=0.0, max_value=500.0, value=float(ex.get('iv_long', 50.0)), step=0.1, key=f"iv_long_{i}") / 100.0
                    iv_short = st.number_input("IV Short (%)", min_value=0.0, max_value=500.0, value=float(ex.get('iv_short', 45.0)), step=0.1, key=f"iv_short_{i}") / 100.0
                    use_bs_vs = st.checkbox("Use Black-Scholes for theoretical leg pricing", value=False, key=f"use_bs_vs_{i}")

                    if st.button("Calculate Vertical Spread P&L", key=f"calculate_vertical_{i}"):
                        res = calc_vertical_spread_pnl(premium_long, premium_short, long_strike, short_strike, underlying_price, int(contracts_vs), is_call=is_call, use_bs=bool(use_bs_vs), days_to_expiry=int(days_to_expiry), iv_long=float(iv_long), iv_short=float(iv_short), rf=0.01)
                        st.write(f"Position value: ${res['position_value']:.2f}")
                        st.write(f"P&L: ${res['pnl']:.2f}")
                        st.write(f"Net Premium (paid): ${res['net_premium']*100:.2f}")
                        st.write(f"Max Gain: ${res['max_gain']:.2f}")
                        st.write(f"Max Loss: ${res['max_loss']:.2f}")
                        st.write(f"ROI: {res['roi_pct']:.1f}%")
                        st.write(f"Breakeven: ${res['breakeven']:.2f}")
                        st.info("Note: This simplifies assignment and ignores early assignment, transaction costs and margin effects.")
                        st.session_state.calc_history.append({
                            'type': 'vertical', 'long_strike': long_strike, 'short_strike': short_strike,
                            'premium_long': premium_long, 'premium_short': premium_short, 'underlying': underlying_price,
                            'contracts': int(contracts_vs), 'dte': int(days_to_expiry), 'iv_long': iv_long, 'iv_short': iv_short,
                            'use_bs': bool(use_bs_vs), 'result_pnl': res['pnl'], 'result_roi': res['roi_pct'], 'timestamp': datetime.now().isoformat()
                        })

                # Calculation history and export
                st.subheader("Calculation History")
                if st.session_state.calc_history:
                    hist_df = pd.DataFrame(st.session_state.calc_history)
                    st.dataframe(hist_df.sort_values('timestamp', ascending=False).reset_index(drop=True), width='stretch')
                    if st.button("Export History CSV", key=f"export_csv_{i}"):
                        # Add UTF-8 BOM for Excel compatibility
                        csv = '\ufeff' + hist_df.to_csv(index=False)
                        st.download_button("Download CSV", csv.encode('utf-8-sig'), "calc_history.csv", "text/csv")
                else:
                    st.info("No calculations yet — run one above to populate history.")
    
    
    with tab10:
        # Initialize Tradier client
        from src.integrations.tradier_client import create_tradier_client_from_env
        if 'tradier_client' not in st.session_state:
            logger.info("Initializing Tradier client from environment")
            try:
                st.session_state.tradier_client = create_tradier_client_from_env()
                logger.info("Tradier client initialized successfully: %s", bool(st.session_state.tradier_client))
            except Exception as e:
                logger.error(f"Failed to initialize Tradier client: {e}", exc_info=True)
                st.session_state.tradier_client = None
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Connection Status")
            
            if st.session_state.tradier_client:
                # Test connection
                if st.button("🔍 Test Connection"):
                    with st.spinner("Testing Tradier connection..."):
                        success, message = validate_tradier_connection()
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
                
                # Connection info
                st.info(f"**Account ID:** {st.session_state.tradier_client.account_id}")
                st.info(f"**API URL:** {st.session_state.tradier_client.api_url}")
                
            else:
                st.error("❌ Tradier client not initialized")
                st.warning("Please check your environment variables:")
                st.code("""
TRADIER_ACCOUNT_ID=your_account_id
TRADIER_ACCESS_TOKEN=your_access_token
TRADIER_API_URL=https://sandbox.tradier.com
                """)
        
        with col2:
            st.subheader("📊 Account Overview")
            
            if st.session_state.tradier_client:
                # Get account summary
                if st.button("🔄 Refresh Account Data"):
                    with st.spinner("Fetching account data..."):
                        success, summary = st.session_state.tradier_client.get_account_summary()
                        
                        if success:
                            st.session_state.account_summary = summary
                            st.success("Account data refreshed!")
                        else:
                            st.error(f"Failed to fetch account data: {summary.get('error', 'Unknown error')}")
                
                # Display account summary if available
                if 'account_summary' in st.session_state:
                    summary = st.session_state.account_summary
                    
                    # Balance information
                    balance = summary.get('balance', {})
                    if 'balances' in balance:
                        bal_data = balance['balances']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Cash", f"${float(bal_data.get('total_cash') or 0):,.2f}")
                        with col2:
                            st.metric("Buying Power", f"${float(bal_data.get('buying_power') or 0):,.2f}")
                        with col3:
                            st.metric("Day Trading", f"${float(bal_data.get('day_trading') or 0):,.2f}")
                        with col4:
                            st.metric("Market Value", f"${float(bal_data.get('market_value') or 0):,.2f}")
                    
                    # Positions
                    st.subheader("📈 Current Positions")
                    positions = summary.get('positions', [])
                    
                    if positions:
                        positions_df = pd.DataFrame(positions)
                        
                        # Display key columns
                        display_cols = ['symbol', 'quantity', 'average_cost', 'market_value', 'gain_loss']
                        if all(col in positions_df.columns for col in display_cols):
                            st.dataframe(
                                positions_df[display_cols],
                                width='stretch',
                                column_config={
                                    "symbol": "Symbol",
                                    "quantity": "Quantity", 
                                    "average_cost": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
                                    "market_value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
                                    "gain_loss": st.column_config.NumberColumn("P&L", format="$%.2f")
                                }
                            )
                        else:
                            st.dataframe(positions_df, width='stretch')
                    else:
                        st.info("No positions found")
                    
                    # Recent orders
                    st.subheader("📋 Recent Orders")
                    orders = summary.get('recent_orders', [])
                    
                    if orders:
                        orders_df = pd.DataFrame(orders)
                        
                        # Display key columns
                        display_cols = ['id', 'symbol', 'side', 'quantity', 'status', 'created_at']
                        available_cols = [col for col in display_cols if col in orders_df.columns]
                        
                        if available_cols:
                            st.dataframe(
                                orders_df[available_cols],
                                width='stretch',
                                column_config={
                                    "id": "Order ID",
                                    "symbol": "Symbol",
                                    "side": "Side",
                                    "quantity": "Quantity",
                                    "status": "Status",
                                    "created_at": "Created"
                                }
                            )
                        else:
                            st.dataframe(orders_df, width='stretch')
                    else:
                        st.info("No orders found")
                
                else:
                    st.info("Click 'Refresh Account Data' to load your account information")
            
        # Order management section
        st.subheader("📝 Order Management")
        
        if st.session_state.tradier_client:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Get Order Status**")
                order_id = st.text_input("Order ID", placeholder="Enter order ID")
                
                if st.button("🔍 Get Order Status") and order_id:
                    with st.spinner("Fetching order status..."):
                        success, order_data = st.session_state.tradier_client.get_order_status(order_id)
                        
                        if success:
                            st.success("Order found!")
                            st.json(order_data)
                        else:
                            st.error(f"Failed to get order: {order_data.get('error', 'Unknown error')}")
            
            with col2:
                st.write("**Cancel Order**")
                cancel_order_id = st.text_input("Order ID to Cancel", placeholder="Enter order ID", key="cancel_order")
                
                if st.button("❌ Cancel Order") and cancel_order_id:
                    with st.spinner("Cancelling order..."):
                        success, result = st.session_state.tradier_client.cancel_order(cancel_order_id)
                        
                        if success:
                            st.success("Order cancelled successfully!")
                            st.json(result)
                        else:
                            st.error(f"Failed to cancel order: {result.get('error', 'Unknown error')}")
        
        # Manual order placement section
        st.subheader("🎯 Manual Order Placement")
        
        if st.session_state.tradier_client:
            with st.expander("Place Custom Order"):
                # Order mode selection
                order_mode = st.radio("Order Mode", ["Simple Order", "Bracket Order (OTOCO)"], horizontal=True, key='tab7_order_mode')
                
                if order_mode == "Bracket Order (OTOCO)":
                    st.info("🎯 Bracket orders automatically set take-profit and stop-loss orders after your entry fills")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        symbol = st.text_input("Symbol", placeholder="AAPL", key='tab7_bracket_symbol')
                        side = st.selectbox("Side", ["buy", "sell"], key='tab7_bracket_side')
                        quantity = st.number_input("Quantity", min_value=1, value=10, key='tab7_bracket_qty')
                    
                    with col2:
                        entry_price = st.number_input("Entry Price", min_value=0.01, value=100.00, step=0.01, format="%.2f", key='tab7_bracket_entry')
                        take_profit = st.number_input("Take Profit Price", min_value=0.01, value=105.00, step=0.01, format="%.2f", key='tab7_bracket_profit')
                        stop_loss = st.number_input("Stop Loss Price", min_value=0.01, value=97.00, step=0.01, format="%.2f", key='tab7_bracket_stop')
                    
                    with col3:
                        duration = st.selectbox("Duration", ["gtc", "day"], key='tab7_bracket_duration')
                        tag = st.text_input("Tag", value=f"BRACKET_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key='tab7_bracket_tag')
                        
                        # Calculate percentages
                        if side == "buy":
                            profit_pct = ((take_profit - entry_price) / entry_price) * 100
                            loss_pct = ((entry_price - stop_loss) / entry_price) * 100
                        else:
                            profit_pct = ((entry_price - take_profit) / entry_price) * 100
                            loss_pct = ((stop_loss - entry_price) / entry_price) * 100
                        
                        st.metric("Profit Target", f"{profit_pct:.1f}%")
                        st.metric("Max Loss", f"{loss_pct:.1f}%")
                    
                    if st.button("🎯 Place Bracket Order", type="primary", key='tab7_bracket_submit'):
                        with st.spinner("Placing bracket order..."):
                            success, result = st.session_state.tradier_client.place_bracket_order(
                                symbol=symbol.upper(),
                                side=side,
                                quantity=quantity,
                                entry_price=entry_price,
                                take_profit_price=take_profit,
                                stop_loss_price=stop_loss,
                                duration=duration,
                                tag=tag
                            )
                            
                            if success:
                                st.success("🎉 Bracket order placed successfully!")
                                st.info(f"✅ Entry: ${entry_price} | 🎯 Target: ${take_profit} | 🛑 Stop: ${stop_loss}")
                                st.json(result)
                            else:
                                st.error(f"Failed to place bracket order: {result.get('error', 'Unknown error')}")
                                st.json(result)
                
                else:
                    # Simple order mode
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        order_class = st.selectbox("Order Class", ["equity", "option", "multileg", "combo"], key='tab7_order_class_select')
                        symbol = st.text_input("Symbol", placeholder="AAPL or AAPL240315C150")
                        side = st.selectbox("Side", ["buy", "sell", "buy_to_cover", "sell_short", "sell_to_open", "sell_to_close", "buy_to_open", "buy_to_close"], key='tab7_order_side_select')
                        quantity = st.number_input("Quantity", min_value=1, value=1)
                    
                    with col2:
                        order_type = st.selectbox("Order Type", ["market", "limit", "stop", "stop_limit", "credit", "debit"], key='tab7_order_type_select')
                        duration = st.selectbox("Duration", ["day", "gtc", "pre", "post"], key='tab7_order_duration_select')
                        price = st.number_input("Price", min_value=0.0, value=0.0, step=0.01, format="%.2f")
                        tag = st.text_input("Tag", value=f"MANUAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    
                    if st.button("📤 Place Order", type="primary"):
                        order_data = {
                            "class": order_class,
                            "symbol": symbol.upper(),
                            "side": side,
                            "quantity": str(quantity),
                            "type": order_type,
                            "duration": duration,
                            "tag": tag
                        }
                        
                        if order_type in ["limit", "stop_limit"] and price > 0:
                            order_data["price"] = str(price)
                        
                        with st.spinner("Placing order..."):
                            success, result = st.session_state.tradier_client.place_order(order_data)
                            
                            if success:
                                st.success("Order placed successfully!")
                                st.json(result)
                            else:
                                st.error(f"Failed to place order: {result.get('error', 'Unknown error')}")
        
        # Configuration section
        st.subheader("⚙️ Configuration")
        
        with st.expander("Environment Variables Status"):
            env_vars = {
                "TRADIER_ACCOUNT_ID": os.getenv('TRADIER_ACCOUNT_ID', 'Not set'),
                "TRADIER_ACCESS_TOKEN": os.getenv('TRADIER_ACCESS_TOKEN', 'Not set'),
                "TRADIER_API_URL": os.getenv('TRADIER_API_URL', 'Not set'),
                "OPTION_ALPHA_WEBHOOK_URL": os.getenv('OPTION_ALPHA_WEBHOOK_URL', 'Not set')
            }
            
            for var, value in env_vars.items():
                if 'TOKEN' in var and value != 'Not set':
                    st.code(f"{var}=***{value[-4:] if len(value) > 4 else '***'}")
                else:
                    st.code(f"{var}={value}")
    
    with tab11:
        st.header("📈 IBKR Day Trading / Scalping")
        st.write("Connect to Interactive Brokers for live day trading and scalping. Real-time positions, orders, and execution.")
        
        # Import IBKR client with comprehensive error handling
        ibkr_available = False
        ibkr_error_message = None
        try:
            logger.info("Attempting to import IBKR client modules...")
            from src.integrations.ibkr_client import IBKRClient, create_ibkr_client_from_env, validate_ibkr_connection, IBKRPosition, IBKROrder
            ibkr_available = True
            logger.info("IBKR client modules imported successfully")
        except ImportError as e:
            ibkr_error_message = f"Missing dependency: {e}. Please install: pip install ib_insync"
            logger.error(f"IBKR ImportError: {e}", exc_info=True)
            st.error(f"⚠️ {ibkr_error_message}")
        except RuntimeError as e:
            ibkr_error_message = f"Event loop error: {e}. This is a known issue with asyncio in Streamlit."
            logger.error(f"IBKR RuntimeError: {e}", exc_info=True)
            st.error(f"⚠️ {ibkr_error_message}")
            st.info("💡 Try restarting the Streamlit app to resolve event loop issues.")
        except Exception as e:
            ibkr_error_message = f"Unexpected error: {e}"
            logger.error(f"IBKR unexpected error: {e}", exc_info=True)
            st.error(f"⚠️ {ibkr_error_message}")
            st.code(str(e))
        
        if ibkr_available:
            # Initialize IBKR client in session state
            if 'ibkr_client' not in st.session_state:
                st.session_state.ibkr_client = None
                st.session_state.ibkr_connected = False
            
            # Connection Section
            st.subheader("🔌 Connection Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                ibkr_host = st.text_input("Host", value="127.0.0.1", help="IB Gateway/TWS host address")
            
            with col2:
                ibkr_port = st.number_input(
                    "Port", 
                    value=7497, 
                    help="7497 for paper trading, 7496 for live TWS, 4002 for IB Gateway paper, 4001 for IB Gateway live"
                )
            
            with col3:
                ibkr_client_id = st.number_input("Client ID", value=1, min_value=1, max_value=32)
            
            col_conn1, col_conn2 = st.columns(2)
            
            with col_conn1:
                if st.button("🔗 Connect to IBKR", type="primary", use_container_width=True):
                    try:
                        with st.status("Connecting to Interactive Brokers...") as status:
                            st.write("Initializing connection...")
                            client = IBKRClient(host=ibkr_host, port=int(ibkr_port), client_id=int(ibkr_client_id))
                            
                            st.write("Connecting to IB Gateway/TWS...")
                            if client.connect(timeout=10):
                                st.session_state.ibkr_client = client
                                st.session_state.ibkr_connected = True
                                
                                st.write("Fetching account information...")
                                account_info = client.get_account_info()
                                
                                if account_info:
                                    status.update(label="✅ Connected to IBKR!", state="complete")
                                    st.success(f"Connected to account: {account_info.account_id}")
                                    st.info(f"💰 Buying Power: ${account_info.buying_power:,.2f} | Net Liquidation: ${account_info.net_liquidation:,.2f}")
                                else:
                                    status.update(label="⚠️ Connected but no account info", state="error")
                            else:
                                st.error("Failed to connect. Make sure IB Gateway or TWS is running with API enabled.")
                                status.update(label="❌ Connection failed", state="error")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            
            with col_conn2:
                if st.button("🔌 Disconnect", use_container_width=True):
                    if st.session_state.ibkr_client:
                        st.session_state.ibkr_client.disconnect()
                        st.session_state.ibkr_client = None
                        st.session_state.ibkr_connected = False
                        st.success("Disconnected from IBKR")
            
            st.divider()
            
            # Show connection status
            if st.session_state.ibkr_connected and st.session_state.ibkr_client:
                if st.session_state.ibkr_client.is_connected():
                    st.success("🟢 Connected to IBKR")
                else:
                    st.warning("🟡 Connection lost - please reconnect")
                    st.session_state.ibkr_connected = False
            else:
                st.info("🔴 Not connected to IBKR")
            
            # Main trading interface (only show if connected)
            if st.session_state.ibkr_connected and st.session_state.ibkr_client:
                client = st.session_state.ibkr_client
                
                # Account Information
                st.subheader("💼 Account Information")
                
                try:
                    account_info = client.get_account_info()
                    
                    if account_info:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Net Liquidation", f"${account_info.net_liquidation:,.2f}")
                        
                        with col2:
                            st.metric("Buying Power", f"${account_info.buying_power:,.2f}")
                        
                        with col3:
                            st.metric("Cash", f"${account_info.total_cash_value:,.2f}")
                        
                        with col4:
                            if account_info.is_pdt:
                                st.metric("Day Trades Left", "Unlimited" if account_info.net_liquidation >= 25000 else str(account_info.day_trades_remaining))
                            else:
                                st.metric("Day Trades Left", str(account_info.day_trades_remaining))
                    
                except Exception as e:
                    st.error(f"Error fetching account info: {e}")
                
                st.divider()
                
                # Current Positions
                st.subheader("📊 Current Positions")
                
                if st.button("🔄 Refresh Positions", use_container_width=True):
                    st.rerun()
                
                try:
                    positions = client.get_positions()
                    
                    if positions:
                        positions_data = []
                        for pos in positions:
                            positions_data.append({
                                'Symbol': pos.symbol,
                                'Quantity': int(pos.position),
                                'Avg Cost': f"${pos.avg_cost:.2f}",
                                'Market Price': f"${pos.market_price:.2f}",
                                'Market Value': f"${pos.market_value:,.2f}",
                                'Unrealized P&L': f"${pos.unrealized_pnl:,.2f}",
                                'Realized P&L': f"${pos.realized_pnl:,.2f}"
                            })
                        
                        positions_df = pd.DataFrame(positions_data)
                        st.dataframe(positions_df, use_container_width=True)
                        
                        # Quick flatten buttons
                        st.write("**Quick Actions:**")
                        cols = st.columns(min(len(positions), 4))
                        for idx, pos in enumerate(positions[:4]):
                            with cols[idx]:
                                if st.button(f"Close {pos.symbol}", key=f"flatten_{pos.symbol}"):
                                    if client.flatten_position(pos.symbol):
                                        st.success(f"✅ Closing {pos.symbol}")
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error(f"Failed to close {pos.symbol}")
                    else:
                        st.info("No open positions")
                
                except Exception as e:
                    st.error(f"Error fetching positions: {e}")
                
                st.divider()
                
                # Open Orders
                st.subheader("📝 Open Orders")
                
                try:
                    open_orders = client.get_open_orders()
                    
                    if open_orders:
                        orders_data = []
                        for order in open_orders:
                            orders_data.append({
                                'Order ID': order.order_id,
                                'Symbol': order.symbol,
                                'Action': order.action,
                                'Type': order.order_type,
                                'Qty': order.quantity,
                                'Limit': f"${order.limit_price:.2f}" if order.limit_price else "N/A",
                                'Stop': f"${order.stop_price:.2f}" if order.stop_price else "N/A",
                                'Status': order.status,
                                'Filled': order.filled,
                                'Remaining': order.remaining
                            })
                        
                        orders_df = pd.DataFrame(orders_data)
                        st.dataframe(orders_df, use_container_width=True)
                        
                        # Cancel orders
                        col_cancel1, col_cancel2 = st.columns(2)
                        
                        with col_cancel1:
                            order_id_to_cancel = st.number_input("Order ID to Cancel", min_value=1, step=1)
                            if st.button("❌ Cancel Order"):
                                if client.cancel_order(int(order_id_to_cancel)):
                                    st.success(f"Order {order_id_to_cancel} cancelled")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Failed to cancel order {order_id_to_cancel}")
                        
                        with col_cancel2:
                            st.write("")
                            st.write("")
                            if st.button("❌❌ Cancel ALL Orders", type="secondary"):
                                cancelled = client.cancel_all_orders()
                                st.success(f"Cancelled {cancelled} orders")
                                time.sleep(1)
                                st.rerun()
                    else:
                        st.info("No open orders")
                
                except Exception as e:
                    st.error(f"Error fetching orders: {e}")
                
                st.divider()
                
                # Place New Order
                st.subheader("🎯 Place Order")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    order_symbol = st.text_input("Symbol", value="", key="order_symbol").upper()
                    order_action = st.selectbox("Action", options=["BUY", "SELL"], key="order_action")
                    order_quantity = st.number_input("Quantity", min_value=1, value=100, step=1, key="order_quantity")
                
                with col2:
                    order_type = st.selectbox(
                        "Order Type", 
                        options=["MARKET", "LIMIT", "STOP"],
                        key="order_type"
                    )
                    
                    if order_type == "LIMIT":
                        order_limit_price = st.number_input("Limit Price", min_value=0.01, value=10.0, step=0.01, key="order_limit")
                    elif order_type == "STOP":
                        order_stop_price = st.number_input("Stop Price", min_value=0.01, value=10.0, step=0.01, key="order_stop")
                
                # Place order button
                if st.button("🚀 Place Order", type="primary", use_container_width=True):
                    if not order_symbol:
                        st.error("Please enter a symbol")
                    else:
                        try:
                            with st.status(f"Placing {order_type} order...") as status:
                                result = None
                                
                                if order_type == "MARKET":
                                    result = client.place_market_order(order_symbol, order_action, int(order_quantity))
                                elif order_type == "LIMIT":
                                    result = client.place_limit_order(order_symbol, order_action, int(order_quantity), float(order_limit_price))
                                elif order_type == "STOP":
                                    result = client.place_stop_order(order_symbol, order_action, int(order_quantity), float(order_stop_price))
                                
                                if result:
                                    status.update(label="✅ Order placed!", state="complete")
                                    st.success(f"Order placed: {order_action} {order_quantity} {order_symbol}")
                                    st.json({
                                        'Order ID': result.order_id,
                                        'Symbol': result.symbol,
                                        'Action': result.action,
                                        'Type': result.order_type,
                                        'Quantity': result.quantity,
                                        'Status': result.status
                                    })
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    status.update(label="❌ Order failed", state="error")
                                    st.error("Failed to place order")
                        
                        except Exception as e:
                            st.error(f"Error placing order: {e}")
                
                st.divider()
                
                # Market Data
                st.subheader("📊 Real-Time Market Data")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    market_symbol = st.text_input("Symbol for Quote", value="SPY", key="market_symbol").upper()
                
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("📈 Get Quote", use_container_width=True):
                        if market_symbol:
                            try:
                                market_data = client.get_market_data(market_symbol)
                                
                                if market_data:
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Last", f"${market_data['last']:.2f}")
                                    
                                    with col2:
                                        st.metric("Bid", f"${market_data['bid']:.2f}", delta=f"{market_data['bid_size']}")
                                    
                                    with col3:
                                        st.metric("Ask", f"${market_data['ask']:.2f}", delta=f"{market_data['ask_size']}")
                                    
                                    with col4:
                                        st.metric("Volume", f"{market_data['volume']:,}")
                                else:
                                    st.error("Failed to fetch market data")
                            
                            except Exception as e:
                                st.error(f"Error fetching market data: {e}")
            
            else:
                st.warning("⚠️ Please connect to IBKR to access trading features")
                st.info("**Setup Instructions:**\n"
                       "1. Download and install IB Gateway or TWS from Interactive Brokers\n"
                       "2. Log in with your IBKR credentials\n"
                       "3. Enable API connections in TWS/Gateway settings\n"
                       "4. Set the port number (7497 for paper, 7496 for live)\n"
                       "5. Click 'Connect to IBKR' above")
    
    with tab12:
        st.header("⚡ Scalping & Day Trading Dashboard")
        st.write("Quick entry/exit interface for stock day trading and scalping. Works with both Tradier and IBKR.")
        st.info("💡 **Perfect for:** Blue chips, penny stocks, runners, and high-momentum plays. Get instant scalping signals!")
        
        # Quick Scalping Analyzer
        with st.expander("⚡ Quick Scalping Analyzer - Instant Signals", expanded=True):
            st.write("Get instant scalping signals for ANY ticker - optimized for 1-5 minute trades.")
            
            col_scalp1, col_scalp2, col_scalp3 = st.columns([2, 1, 1])
            
            with col_scalp1:
                scalp_ticker = st.text_input(
                    "Ticker to Scalp",
                    value="SPY",
                    help="Enter any ticker: SPY, QQQ, AAPL, penny stocks, or runners"
                ).upper()
            
            with col_scalp2:
                scalp_mode = st.selectbox(
                    "Scalping Mode",
                    options=["Standard", "Penny Stock", "Runner/Momentum"],
                    help="Penny Stock = tighter stops, Runner = momentum-based"
                )
            
            with col_scalp3:
                st.write("")
                st.write("")
                scalp_analyze_btn = st.button("⚡ Get Scalp Signal", type="primary", use_container_width=True)
            
            if scalp_analyze_btn and scalp_ticker:
                with st.status(f"⚡ Analyzing {scalp_ticker} for scalping...", expanded=True) as scalp_status:
                    st.write("📊 Fetching real-time data...")
                    
                    try:
                        # Get analysis with scalp trading style
                        analysis = ComprehensiveAnalyzer.analyze_stock(scalp_ticker, "SCALP")
                        
                        if analysis:
                            scalp_status.update(label=f"✅ Scalp analysis complete for {scalp_ticker}", state="complete")
                            
                            # Detect characteristics
                            is_penny = analysis.price < 5.0
                            volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
                            is_runner = volume_vs_avg > 200 and abs(analysis.change_pct) > 10
                            
                            # Auto-adjust mode
                            if scalp_mode == "Standard" and is_penny:
                                st.info("💡 Auto-detected penny stock - using tighter stops")
                                scalp_mode = "Penny Stock"
                            elif scalp_mode == "Standard" and is_runner:
                                st.info("💡 Auto-detected runner - using momentum strategy")
                                scalp_mode = "Runner/Momentum"
                            
                            # Calculate scalping parameters based on mode
                            if scalp_mode == "Penny Stock":
                                stop_pct = 3.0  # Tight 3% stop
                                target_pct = 5.0  # Quick 5% target
                                risk_label = "HIGH"
                            elif scalp_mode == "Runner/Momentum":
                                stop_pct = 2.0  # Very tight 2% stop
                                target_pct = 8.0  # Larger 8% target for runners
                                risk_label = "VERY HIGH"
                            else:  # Standard
                                stop_pct = 0.5  # Standard 0.5% stop
                                target_pct = 1.0  # Standard 1% target
                                risk_label = "MEDIUM"
                            
                            # Calculate levels
                            entry_price = analysis.price
                            stop_loss = entry_price * (1 - stop_pct/100)
                            target_price = entry_price * (1 + target_pct/100)
                            
                            # Determine signal
                            signal = "NEUTRAL"
                            signal_color = "🟡"
                            confidence = 50
                            
                            if analysis.rsi < 40 and analysis.macd_signal == "BULLISH" and volume_vs_avg > 50:
                                signal = "BUY"
                                signal_color = "🟢"
                                confidence = 75
                            elif analysis.rsi > 60 and analysis.macd_signal == "BEARISH":
                                signal = "SELL"
                                signal_color = "🔴"
                                confidence = 70
                            elif analysis.trend == "BULLISH" and analysis.rsi < 60:
                                signal = "BUY"
                                signal_color = "🟢"
                                confidence = 65
                            elif analysis.trend == "BEARISH" and analysis.rsi > 40:
                                signal = "SELL"
                                signal_color = "🔴"
                                confidence = 60
                            
                            # Display signal
                            st.markdown(f"## {signal_color} SCALP SIGNAL: {signal}")
                            
                            # Metrics
                            scalp_col1, scalp_col2, scalp_col3, scalp_col4 = st.columns(4)
                            
                            with scalp_col1:
                                st.metric("Entry Price", f"${entry_price:.4f}" if is_penny else f"${entry_price:.2f}")
                                st.caption(f"Current: ${analysis.price:.4f}" if is_penny else f"${analysis.price:.2f}")
                            
                            with scalp_col2:
                                st.metric("Target", f"${target_price:.4f}" if is_penny else f"${target_price:.2f}")
                                st.caption(f"🎯 +{target_pct:.1f}%")
                            
                            with scalp_col3:
                                st.metric("Stop Loss", f"${stop_loss:.4f}" if is_penny else f"${stop_loss:.2f}")
                                st.caption(f"🛑 -{stop_pct:.1f}%")
                            
                            with scalp_col4:
                                st.metric("Confidence", f"{confidence}%")
                                st.metric("Risk", risk_label)
                            
                            # Additional info
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.write("**Technical Indicators:**")
                                st.write(f"• RSI: {analysis.rsi:.1f} {'🟢 Oversold' if analysis.rsi < 30 else '🔴 Overbought' if analysis.rsi > 70 else '🟡 Neutral'}")
                                st.write(f"• MACD: {analysis.macd_signal}")
                                st.write(f"• Trend: {analysis.trend}")
                                st.write(f"• Volume: {volume_vs_avg:+.0f}% vs avg")
                            
                            with info_col2:
                                st.write("**Scalping Strategy:**")
                                if scalp_mode == "Penny Stock":
                                    st.write("• ⚡ Quick in/out (1-5 min)")
                                    st.write("• 🛑 Tight 3% stop loss")
                                    st.write("• 🎯 5% profit target")
                                    st.write("• ⚠️ High risk - small size!")
                                elif scalp_mode == "Runner/Momentum":
                                    st.write("• 🚀 Ride the momentum")
                                    st.write("• 🛑 Very tight 2% stop")
                                    st.write("• 🎯 8% profit target")
                                    st.write("• ⚠️ Exit on volume drop!")
                                else:
                                    st.write("• ⚡ Standard scalp (1-3 min)")
                                    st.write("• 🛑 0.5% stop loss")
                                    st.write("• 🎯 1% profit target")
                                    st.write("• 📊 Watch L2 order book")
                            
                            # Warning for risky setups
                            if signal == "NEUTRAL":
                                st.warning("⚠️ No clear scalping setup right now. Wait for better entry or try another ticker.")
                            elif confidence < 65:
                                st.info("💡 Moderate confidence - consider reducing position size or waiting for confirmation.")
                            
                            # Quick action buttons
                            action_col1, action_col2 = st.columns(2)
                            with action_col1:
                                if st.button(f"📋 Copy {signal} Order to Form", use_container_width=True):
                                    st.session_state['scalp_prefill_symbol'] = scalp_ticker
                                    st.session_state['scalp_prefill_side'] = signal
                                    st.session_state['scalp_prefill_entry'] = entry_price
                                    st.session_state['scalp_prefill_target'] = target_price
                                    st.session_state['scalp_prefill_stop'] = stop_loss
                                    st.success("✅ Copied to order form below!")
                            
                            with action_col2:
                                if st.button("🔄 Refresh Signal", use_container_width=True):
                                    st.rerun()
                        
                        else:
                            scalp_status.update(label=f"❌ Could not analyze {scalp_ticker}", state="error")
                            st.error(f"Unable to fetch data for {scalp_ticker}. Check ticker symbol.")
                    
                    except Exception as e:
                        scalp_status.update(label="❌ Analysis failed", state="error")
                        st.error(f"Error: {e}")
        
        # AI Autopilot Section
        with st.expander("🤖 AI Trading Autopilot - Get Smart Signals", expanded=False):
            st.write("Let AI analyze technicals, news, sentiment, and social media to recommend the best trades.")
            
            col_ai1, col_ai2, col_ai3 = st.columns(3)
            
            with col_ai1:
                ai_symbols = st.text_input(
                    "Symbols to Analyze (comma-separated)",
                    value="SPY,QQQ,AAPL,TSLA,NVDA",
                    help="Enter stock symbols to get AI recommendations"
                )
            
            with col_ai2:
                ai_risk = st.selectbox(
                    "Risk Tolerance",
                    options=["LOW", "MEDIUM", "HIGH"],
                    index=1,
                    help="Your risk appetite"
                )
            
            with col_ai3:
                ai_provider = st.selectbox(
                    "AI Provider",
                    options=["openrouter", "openai", "anthropic"],
                    index=0,
                    help="OpenRouter is free!"
                )
            
            if st.button("🧠 Generate AI Signals", type="primary", use_container_width=True):
                symbols_list = [s.strip().upper() for s in ai_symbols.split(',') if s.strip()]
                
                if not symbols_list:
                    st.error("Please enter at least one symbol")
                else:
                    with st.status("🤖 AI analyzing market data...", expanded=True) as status:
                        try:
                            # Import AI signal generator
                            from ai_trading_signals import create_ai_signal_generator
                            
                            st.write("Initializing AI engine...")
                            ai_generator = create_ai_signal_generator(provider=ai_provider)
                            
                            # Collect data for each symbol
                            st.write(f"Gathering data for {len(symbols_list)} symbols...")
                            
                            technical_data_dict = {}
                            news_data_dict = {}
                            sentiment_data_dict = {}
                            
                            for symbol in symbols_list:
                                try:
                                    st.write(f"Analyzing {symbol}...")
                                    
                                    # Get comprehensive analysis
                                    # ComprehensiveAnalyzer and NewsAnalyzer are already defined globally
                                    # Using OPTIONS as default for signal generation
                                    analysis = ComprehensiveAnalyzer.analyze_stock(symbol, "OPTIONS")
                                    
                                    if analysis:
                                        technical_data_dict[symbol] = {
                                            'price': analysis.price,
                                            'change_pct': analysis.change_pct,
                                            'rsi': analysis.rsi,
                                            'macd_signal': analysis.macd_signal,
                                            'trend': analysis.trend,
                                            'volume': analysis.volume,
                                            'avg_volume': analysis.avg_volume,
                                            'support': analysis.support,
                                            'resistance': analysis.resistance,
                                            'iv_rank': analysis.iv_rank
                                        }
                                        
                                        news_data_dict[symbol] = analysis.recent_news
                                        
                                        sentiment_data_dict[symbol] = {
                                            'score': analysis.sentiment_score,
                                            'signals': analysis.sentiment_signals
                                        }
                                except Exception as e:
                                    st.write(f"⚠️ Error analyzing {symbol}: {e}")
                                    continue
                            
                            st.write("Running AI analysis...")
                            
                            # Get account balance
                            account_balance = 10000.0  # Default
                            try:
                                if 'tradier_client' in st.session_state and st.session_state.tradier_client:
                                    success, bal_data = st.session_state.tradier_client.get_account_balance()
                                    if success and isinstance(bal_data, dict):
                                        # Tradier returns { 'balances': { 'total_cash': ... } }
                                        b = bal_data.get('balances') or {}
                                        account_balance = float(b.get('total_cash') or 0.0)
                            except Exception:
                                pass
                            
                            # Generate signals
                            signals = ai_generator.batch_analyze(
                                symbols=symbols_list,
                                technical_data_dict=technical_data_dict,
                                news_data_dict=news_data_dict,
                                sentiment_data_dict=sentiment_data_dict,
                                account_balance=account_balance,
                                risk_tolerance=ai_risk
                            )
                            
                            status.update(label=f"✅ AI analysis complete! Found {len(signals)} signals", state="complete")
                            
                            if signals:
                                st.success(f"🎯 AI found {len(signals)} high-confidence trading opportunities!")
                                
                                # Display signals
                                for idx, signal in enumerate(signals, 1):
                                    with st.container():
                                        # Signal header with color
                                        signal_color = "🟢" if signal.signal == "BUY" else "🔴" if signal.signal == "SELL" else "⚪"
                                        
                                        col_sig1, col_sig2, col_sig3 = st.columns([2, 1, 1])
                                        
                                        with col_sig1:
                                            st.markdown(f"### {signal_color} {idx}. {signal.symbol} - {signal.signal}")
                                            st.write(f"**AI Reasoning:** {signal.reasoning}")
                                        
                                        with col_sig2:
                                            st.metric("Confidence", f"{signal.confidence:.0f}%")
                                            st.metric("Risk Level", signal.risk_level)
                                        
                                        with col_sig3:
                                            st.metric("Position Size", f"{signal.position_size} shares")
                                            st.metric("Time Horizon", signal.time_horizon)
                                        
                                        # Trading details
                                        col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                                        
                                        with col_detail1:
                                            if signal.entry_price:
                                                st.write(f"**Entry:** ${signal.entry_price:.2f}")
                                        
                                        with col_detail2:
                                            if signal.target_price:
                                                st.write(f"**Target:** ${signal.target_price:.2f}")
                                                profit_pct = ((signal.target_price - signal.entry_price) / signal.entry_price * 100) if signal.entry_price else 0
                                                st.write(f"📈 {profit_pct:+.1f}%")
                                        
                                        with col_detail3:
                                            if signal.stop_loss:
                                                st.write(f"**Stop:** ${signal.stop_loss:.2f}")
                                                loss_pct = ((signal.stop_loss - signal.entry_price) / signal.entry_price * 100) if signal.entry_price else 0
                                                st.write(f"📉 {loss_pct:.1f}%")
                                        
                                        with col_detail4:
                                            potential_profit = (signal.target_price - signal.entry_price) * signal.position_size if signal.entry_price and signal.target_price else 0
                                            st.write(f"**Potential:** ${potential_profit:,.0f}")
                                        
                                        # AI Scores
                                        st.write("**AI Analysis Scores:**")
                                        score_cols = st.columns(4)
                                        
                                        with score_cols[0]:
                                            st.metric("Technical", f"{signal.technical_score:.0f}/100")
                                        with score_cols[1]:
                                            st.metric("Sentiment", f"{signal.sentiment_score:.0f}/100")
                                        with score_cols[2]:
                                            st.metric("News", f"{signal.news_score:.0f}/100")
                                        with score_cols[3]:
                                            st.metric("Social", f"{signal.social_score:.0f}/100")
                                        
                                        # Quick execute buttons
                                        col_exec1, col_exec2 = st.columns(2)
                                        
                                        with col_exec1:
                                            if st.button(f"✅ Execute {signal.signal} Order", key=f"exec_{signal.symbol}_{idx}", type="primary", use_container_width=True):
                                                st.session_state[f'execute_signal_{signal.symbol}'] = signal
                                                st.success(f"Ready to execute! Go to order entry below to place {signal.signal} order for {signal.symbol}")
                                        
                                        with col_exec2:
                                            if st.button(f"📋 Copy to Order Form", key=f"copy_{signal.symbol}_{idx}", use_container_width=True):
                                                # Pre-fill order form
                                                st.session_state['ai_prefill_symbol'] = signal.symbol
                                                st.session_state['ai_prefill_qty'] = signal.position_size
                                                st.session_state['ai_prefill_side'] = signal.signal
                                                st.success(f"Copied to order form! Scroll down to execute.")
                                        
                                        st.divider()
                            else:
                                st.warning("🤔 AI didn't find any high-confidence signals right now.")
                                st.info("""
**Why no signals?**
- **Market is closed** - After-hours data is less reliable
- **Conservative AI** - The AI is being cautious (good thing!)
- **Current symbols** - Try popular tickers: SPY, QQQ, AAPL, MSFT, NVDA, TSLA
- **Risk tolerance** - Try changing from MEDIUM to LOW for more signals

**Tips:**
- Run during market hours (9:30 AM - 4:00 PM ET) for best results
- Use highly liquid stocks (SPY, QQQ, major tech stocks)
- Check back when market conditions improve
                                """)
                        
                        except Exception as e:
                            status.update(label="❌ AI analysis failed", state="error")
                            st.error(f"Error: {e}")
        
        st.divider()
        
        # Platform selection
        st.subheader("🔌 Select Trading Platform")
        
        col_platform1, col_platform2 = st.columns(2)
        
        with col_platform1:
            scalp_platform = st.radio(
                "Trading Platform",
                options=["Tradier", "IBKR"],
                horizontal=True,
                help="Choose which broker to use for scalping"
            )
        
        with col_platform2:
            auto_refresh = st.toggle("Auto-refresh positions", value=False, help="Automatically refresh every 5 seconds")
        
        st.divider()
        
        # Check connection status
        if scalp_platform == "Tradier":
            # Initialize if not exists
            if 'tradier_client' not in st.session_state:
                st.session_state.tradier_client = None
            
            # Check if client exists and is valid
            if st.session_state.tradier_client is None:
                st.warning("⚠️ Tradier not connected.")
                
                # Try to initialize from environment
                col_init1, col_init2 = st.columns(2)
                
                with col_init1:
                    if st.button("🔗 Connect to Tradier", type="primary", use_container_width=True):
                        try:
                            from src.integrations.tradier_client import create_tradier_client_from_env
                            client = create_tradier_client_from_env()
                            if client:
                                st.session_state.tradier_client = client
                                logger.info("Tradier client connected successfully")
                                st.success("✅ Connected to Tradier!")
                                st.rerun()
                            else:
                                st.error("Failed to initialize Tradier client. Check your .env file.")
                                logger.error("Tradier client initialization returned None")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
                            logger.error(f"Tradier connection error: {e}", exc_info=True)
                
                with col_init2:
                    st.info("Or go to **🏦 Tradier Account** tab to configure.")
                
                st.stop()
            
            tradier_client = st.session_state.tradier_client
            
            # Validate client has required attributes
            if not hasattr(tradier_client, 'get_account_balance'):
                st.error("⚠️ Tradier client is not properly initialized.")
                if st.button("🔄 Reinitialize Client"):
                    st.session_state.tradier_client = None
                    st.rerun()
                st.stop()
            
            # Account summary
            st.subheader("💼 Account Summary")
            try:
                balance = tradier_client.get_account_balance()
                if balance and hasattr(balance, 'total_equity'):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Account Value", f"${balance.total_equity:,.2f}")
                    with col2:
                        st.metric("Cash Available", f"${balance.total_cash:,.2f}")
                    with col3:
                        st.metric("Buying Power", f"${balance.option_buying_power:,.2f}")
                    with col4:
                        day_trade_buying_power = getattr(balance, 'day_trade_buying_power', balance.option_buying_power)
                        st.metric("Day Trade Power", f"${day_trade_buying_power:,.2f}")
                else:
                    st.warning("Unable to fetch account balance. Please check connection.")
            except Exception as e:
                st.error(f"Error fetching account balance: {str(e)}")
                logger.error(f"Account balance error: {e}", exc_info=True)
            
            st.divider()
            
            # Quick order entry
            st.subheader("🎯 Quick Order Entry")
            
            # Check for AI prefill
            ai_symbol = st.session_state.get('ai_prefill_symbol', 'SPY')
            ai_qty = st.session_state.get('ai_prefill_qty', 100)
            ai_side = st.session_state.get('ai_prefill_side', 'BUY')
            
            # Show AI recommendation if available
            if 'ai_prefill_symbol' in st.session_state:
                st.info(f"🤖 AI Recommendation loaded: {ai_side} {ai_qty} shares of {ai_symbol}")
            
            col_entry1, col_entry2, col_entry3 = st.columns([2, 1, 1])
            
            with col_entry1:
                scalp_symbol = st.text_input("Symbol", value=ai_symbol, key="scalp_symbol_tradier").upper()
                
            with col_entry2:
                scalp_quantity = st.number_input("Shares", min_value=1, value=ai_qty, step=1, key="scalp_qty_tradier")
            
            with col_entry3:
                side_index = 0 if ai_side == "BUY" else 1
                scalp_side = st.selectbox("Side", options=["BUY", "SELL"], index=side_index, key="scalp_side_tradier")
            
            col_order1, col_order2, col_order3 = st.columns(3)
            
            with col_order1:
                if st.button("🚀 Market Order", type="primary", use_container_width=True, key="market_tradier"):
                    if scalp_symbol:
                        try:
                            with st.spinner(f"Placing market order: {scalp_side} {scalp_quantity} {scalp_symbol}..."):
                                order = tradier_client.place_equity_order(
                                    symbol=scalp_symbol,
                                    side=scalp_side.lower(),
                                    quantity=scalp_quantity,
                                    order_type='market',
                                    duration='day'
                                )
                                if order:
                                    st.success(f"✅ Order placed! ID: {order.get('id', 'N/A')}")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Order failed")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col_order2:
                limit_price = st.number_input("Limit $", min_value=0.01, value=100.0, step=0.01, key="limit_price_tradier")
                if st.button("📊 Limit Order", use_container_width=True, key="limit_tradier"):
                    if scalp_symbol:
                        try:
                            order = tradier_client.place_equity_order(
                                symbol=scalp_symbol,
                                side=scalp_side.lower(),
                                quantity=scalp_quantity,
                                order_type='limit',
                                duration='day',
                                price=limit_price
                            )
                            if order:
                                st.success(f"✅ Limit order placed at ${limit_price}")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col_order3:
                stop_price = st.number_input("Stop $", min_value=0.01, value=100.0, step=0.01, key="stop_price_tradier")
                if st.button("🛑 Stop Order", use_container_width=True, key="stop_tradier"):
                    if scalp_symbol:
                        try:
                            order = tradier_client.place_equity_order(
                                symbol=scalp_symbol,
                                side=scalp_side.lower(),
                                quantity=scalp_quantity,
                                order_type='stop',
                                duration='day',
                                stop=stop_price
                            )
                            if order:
                                st.success(f"✅ Stop order placed at ${stop_price}")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            st.divider()
            
            # Current positions
            st.subheader("📊 Current Positions")
            
            col_pos1, col_pos2 = st.columns([3, 1])
            
            with col_pos2:
                if st.button("🔄 Refresh", use_container_width=True, key="refresh_pos_tradier"):
                    st.rerun()
            
            try:
                success, positions = tradier_client.get_positions()
                
                if not success:
                    st.warning("⚠️ Unable to fetch positions from Tradier API")
                    positions = []
                
                if positions and isinstance(positions, list) and len(positions) > 0:
                    positions_data = []
                    
                    for pos in positions:
                        if not isinstance(pos, dict):
                            continue
                        
                        # Get current quote
                        try:
                            symbol = pos.get('symbol', '')
                            if not symbol:
                                continue
                            
                            quote = tradier_client.get_quote(symbol)
                            if isinstance(quote, dict):
                                current_price = float(quote.get('last', 0))
                            else:
                                current_price = 0
                        except Exception as e:
                            logger.warning(f"Error getting quote for {symbol}: {e}")
                            current_price = 0
                        
                        try:
                            cost_basis = float(pos.get('cost_basis', 0))
                            quantity = float(pos.get('quantity', 0))
                            avg_price = cost_basis / quantity if quantity != 0 else 0
                            current_value = current_price * quantity
                            pnl = current_value - cost_basis
                            pnl_pct = (pnl / cost_basis * 100) if cost_basis != 0 else 0
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error calculating position metrics: {e}")
                            continue
                        
                        positions_data.append({
                            'Symbol': pos['symbol'],
                            'Qty': int(quantity),
                            'Avg Price': f"${avg_price:.2f}",
                            'Current': f"${current_price:.2f}",
                            'Value': f"${current_value:,.2f}",
                            'P&L': f"${pnl:,.2f}",
                            'P&L %': f"{pnl_pct:+.2f}%",
                            '_pnl_raw': pnl  # Hidden column for styling
                        })
                    
                    # Display positions
                    df_positions = pd.DataFrame(positions_data)
                    
                    # Style the dataframe
                    def color_pnl(val):
                        if 'P&L' in val.name or 'P&L %' in val.name:
                            return ['color: green' if '+' in str(x) or (isinstance(x, (int, float)) and x > 0) else 'color: red' if '-' in str(x) or (isinstance(x, (int, float)) and x < 0) else '' for x in val]
                        return ['' for _ in val]
                    
                    # Remove hidden column before display
                    display_df = df_positions.drop(columns=['_pnl_raw'])
                    st.dataframe(display_df, use_container_width=True, height=300)
                    
                    # Quick close buttons
                    st.write("**Quick Actions:**")
                    cols = st.columns(min(len(positions), 4))
                    
                    for idx, pos in enumerate(positions[:4]):
                        with cols[idx]:
                            symbol = pos['symbol']
                            qty = int(float(pos.get('quantity', 0)))
                            
                            if st.button(f"❌ Close {symbol}", key=f"close_{symbol}_tradier", use_container_width=True):
                                side = 'sell' if qty > 0 else 'buy'
                                try:
                                    order = tradier_client.place_equity_order(
                                        symbol=symbol,
                                        side=side,
                                        quantity=abs(qty),
                                        order_type='market',
                                        duration='day'
                                    )
                                    if order:
                                        st.success(f"✅ Closing {symbol}")
                                        time.sleep(1)
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                else:
                    st.info("No open positions")
            
            except Exception as e:
                st.error(f"Error fetching positions: {e}")
                logger.error(f"Positions error details: {type(positions) if 'positions' in locals() else 'undefined'}", exc_info=True)
            
            st.divider()
            
            # Open orders
            st.subheader("📝 Open Orders")
            
            try:
                success, orders = tradier_client.get_orders()
                
                if not success:
                    st.warning("⚠️ Unable to fetch orders from Tradier API")
                    orders = []
                
                if orders and isinstance(orders, list) and len(orders) > 0:
                    orders_data = []
                    
                    for order in orders:
                        if not isinstance(order, dict):
                            continue
                        
                        if order.get('status') not in ['filled', 'canceled', 'rejected']:
                            orders_data.append({
                                'ID': order.get('id', 'N/A'),
                                'Symbol': order.get('symbol', 'N/A'),
                                'Side': order.get('side', 'N/A').upper(),
                                'Qty': order.get('quantity', 0),
                                'Type': order.get('type', 'N/A').upper(),
                                'Price': f"${order.get('price', 0):.2f}" if order.get('price') else 'N/A',
                                'Status': order.get('status', 'N/A').upper()
                            })
                    
                    if orders_data:
                        df_orders = pd.DataFrame(orders_data)
                        st.dataframe(df_orders, use_container_width=True)
                        
                        # Cancel orders
                        col_cancel1, col_cancel2 = st.columns([2, 1])
                        
                        with col_cancel1:
                            order_id_cancel = st.text_input("Order ID to cancel", key="cancel_order_id_tradier")
                        
                        with col_cancel2:
                            st.write("")
                            st.write("")
                            if st.button("❌ Cancel Order", key="cancel_order_tradier"):
                                if order_id_cancel:
                                    try:
                                        result = tradier_client.cancel_order(int(order_id_cancel))
                                        if result:
                                            st.success(f"Order {order_id_cancel} cancelled")
                                            time.sleep(1)
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                    else:
                        st.info("No pending orders")
                else:
                    st.info("No open orders")
            
            except Exception as e:
                st.error(f"Error fetching orders: {e}")
                logger.error(f"Orders error details: {type(orders) if 'orders' in locals() else 'undefined'}", exc_info=True)
        
        elif scalp_platform == "IBKR":
            # Check IBKR connection
            if not st.session_state.get('ibkr_connected') or not st.session_state.get('ibkr_client'):
                st.warning("⚠️ IBKR not connected. Go to **📈 IBKR Trading** tab to connect.")
                st.stop()
            
            try:
                from src.integrations.ibkr_client import IBKRClient
                ibkr_client = st.session_state.ibkr_client
                
                # Account summary
                st.subheader("💼 Account Summary")
                try:
                    account_info = ibkr_client.get_account_info()
                    if account_info:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Net Liquidation", f"${account_info.net_liquidation:,.2f}")
                        with col2:
                            st.metric("Cash", f"${account_info.total_cash_value:,.2f}")
                        with col3:
                            st.metric("Buying Power", f"${account_info.buying_power:,.2f}")
                        with col4:
                            st.metric("Day Trades Left", "Unlimited" if account_info.is_pdt and account_info.net_liquidation >= 25000 else str(account_info.day_trades_remaining))
                except Exception as e:
                    st.error(f"Error fetching account: {e}")
                
                st.divider()
                
                # Quick order entry (same as Tradier but using IBKR client)
                st.subheader("🎯 Quick Order Entry")
                
                col_entry1, col_entry2, col_entry3 = st.columns([2, 1, 1])
                
                with col_entry1:
                    scalp_symbol_ibkr = st.text_input("Symbol", value="SPY", key="scalp_symbol_ibkr").upper()
                    
                with col_entry2:
                    scalp_quantity_ibkr = st.number_input("Shares", min_value=1, value=100, step=1, key="scalp_qty_ibkr")
                
                with col_entry3:
                    scalp_side_ibkr = st.selectbox("Side", options=["BUY", "SELL"], key="scalp_side_ibkr")
                
                col_order1, col_order2, col_order3 = st.columns(3)
                
                with col_order1:
                    if st.button("🚀 Market Order", type="primary", use_container_width=True, key="market_ibkr"):
                        try:
                            order = ibkr_client.place_market_order(scalp_symbol_ibkr, scalp_side_ibkr, scalp_quantity_ibkr)
                            if order:
                                st.success(f"✅ Order placed! ID: {order.order_id}")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with col_order2:
                    limit_price_ibkr = st.number_input("Limit $", min_value=0.01, value=100.0, step=0.01, key="limit_price_ibkr")
                    if st.button("📊 Limit Order", use_container_width=True, key="limit_ibkr"):
                        try:
                            order = ibkr_client.place_limit_order(scalp_symbol_ibkr, scalp_side_ibkr, scalp_quantity_ibkr, limit_price_ibkr)
                            if order:
                                st.success(f"✅ Limit order placed")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                with col_order3:
                    stop_price_ibkr = st.number_input("Stop $", min_value=0.01, value=100.0, step=0.01, key="stop_price_ibkr")
                    if st.button("🛑 Stop Order", use_container_width=True, key="stop_ibkr"):
                        try:
                            order = ibkr_client.place_stop_order(scalp_symbol_ibkr, scalp_side_ibkr, scalp_quantity_ibkr, stop_price_ibkr)
                            if order:
                                st.success(f"✅ Stop order placed")
                                time.sleep(1)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                
                st.divider()
                
                # Positions (similar to Tradier)
                st.subheader("📊 Current Positions")
                
                col_pos1, col_pos2 = st.columns([3, 1])
                
                with col_pos2:
                    if st.button("🔄 Refresh", use_container_width=True, key="refresh_pos_ibkr"):
                        st.rerun()
                
                try:
                    positions = ibkr_client.get_positions()
                    
                    if positions:
                        positions_data = []
                        for pos in positions:
                            positions_data.append({
                                'Symbol': pos.symbol,
                                'Qty': int(pos.position),
                                'Avg Price': f"${pos.avg_cost:.2f}",
                                'Current': f"${pos.market_price:.2f}",
                                'Value': f"${pos.market_value:,.2f}",
                                'P&L': f"${pos.unrealized_pnl:,.2f}"
                            })
                        
                        df_positions = pd.DataFrame(positions_data)
                        st.dataframe(df_positions, use_container_width=True, height=300)
                        
                        # Quick close
                        st.write("**Quick Actions:**")
                        cols = st.columns(min(len(positions), 4))
                        for idx, pos in enumerate(positions[:4]):
                            with cols[idx]:
                                if st.button(f"❌ Close {pos.symbol}", key=f"close_{pos.symbol}_ibkr", use_container_width=True):
                                    if ibkr_client.flatten_position(pos.symbol):
                                        st.success(f"✅ Closing {pos.symbol}")
                                        time.sleep(1)
                                        st.rerun()
                    else:
                        st.info("No open positions")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                
            except ImportError:
                st.error("IBKR client not available")
        
        # Auto-refresh functionality
        if auto_refresh:
            time.sleep(5)
            st.rerun()
    
    with tab13:
        st.header("🤖 Strategy Analyzer")
        st.write("Analyze Option Alpha bot configs using an LLM provider. Choose provider, model and optionally provide an API key to run analysis.")

        col1, col2 = st.columns(2)

        with col1:
            provider = st.selectbox("LLM Provider", options=["openai", "anthropic", "google", "openrouter"], index=0, key='tab11_llm_provider_select')
            model = st.text_input("Model (leave blank for default)", value="")
            api_key_input = st.text_input("API Key (optional, will override env var)", value="", type="password")
            run_btn = st.button("🔎 Run Analysis", type="primary")

        with col2:
            st.subheader("Sample Bot Configuration")
            sample_config = extract_bot_config_from_screenshot()
            st.json(sample_config)

        if run_btn:
            with st.spinner("Running strategy analysis..."):
                try:
                    analyzer = LLMStrategyAnalyzer(provider=provider, model=(model or None), api_key=(api_key_input or None))
                except Exception as e:
                    st.error(f"Failed to initialize analyzer: {e}")
                else:
                    try:
                        analysis = analyzer.analyze_bot_strategy(sample_config)

                        # Save for quick sidebar summary and future access
                        st.session_state.strategy_analysis = analysis

                        # Prominent top-level card
                        st.markdown("---")
                        st.subheader(f"Analysis for: {getattr(analysis, 'bot_name', 'Bot')}")

                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Overall Rating", getattr(analysis, 'overall_rating', 'N/A'))
                        with m2:
                            rs = getattr(analysis, 'risk_score', None)
                            st.metric("Risk Score", f"{rs:.2f}" if isinstance(rs, (int, float)) else rs)
                        with m3:
                            cf = getattr(analysis, 'confidence', None)
                            st.metric("Confidence", f"{cf:.2f}" if isinstance(cf, (int, float)) else cf)

                        # Summary and quick actions
                        summary_col, actions_col = st.columns([3, 1])
                        with summary_col:
                            summary_text = getattr(analysis, 'summary', '') or ''
                            st.write(summary_text)
                        with actions_col:
                            if st.button("🔝 Back to Top"):
                                st.rerun()
                            # Provide a copyable summary text area
                            if st.button("📋 Copy Summary"):
                                st.text_area("Summary (select and copy)", value=summary_text, height=160)

                        # Collapsible detailed sections for readability
                        with st.expander("Strengths", expanded=True):
                            strengths = getattr(analysis, 'strengths', []) or []
                            if strengths:
                                for s in strengths:
                                    st.write(f"• {s}")
                            else:
                                st.write("No strengths found.")

                        with st.expander("Weaknesses", expanded=True):
                            weaknesses = getattr(analysis, 'weaknesses', []) or []
                            if weaknesses:
                                for w in weaknesses:
                                    st.write(f"• {w}")
                            else:
                                st.write("No weaknesses found.")

                        with st.expander("Recommendations", expanded=True):
                            recommendations = getattr(analysis, 'recommendations', []) or []
                            if recommendations:
                                for r in recommendations:
                                    st.write(f"• {r}")
                            else:
                                st.write("No recommendations returned.")

                        # If user triggered quick access from sidebar, show a note and focus
                        if st.session_state.get('goto_strategy_analyzer'):
                            st.success("Opened Strategy Analyzer — scroll down for details below.")
                            # clear the flag
                            st.session_state.goto_strategy_analyzer = False

                        # Add "Apply Strategy" functionality
                        st.divider()
                        st.subheader("🎯 Apply Favorable Strategy Plays")
                        st.write("Use the analyzed strategy to generate trading signals with recommended parameters.")
                        
                        # Extract recommended strategies from analysis
                        recommendations = getattr(analysis, 'recommendations', []) or []
                        if recommendations:
                            st.write("**AI Recommended Strategies:**")
                            for idx, rec in enumerate(recommendations[:3]):
                                with st.expander(f"Recommendation {idx + 1}"):
                                    st.write(rec)
                        
                        col_apply1, col_apply2 = st.columns(2)
                        
                        with col_apply1:
                            # Prefill ticker from analysis if available
                            apply_ticker = st.text_input(
                                "Ticker Symbol",
                                value=getattr(analysis, 'ticker', st.session_state.get('selected_ticker', 'SPX')),
                                help="Enter the ticker for this strategy"
                            )
                            
                            # Strategy selection from recommendations
                            strategy_options = st.session_state.config.allowed_strategies
                            apply_strategy = st.selectbox(
                                "Select Strategy",
                                options=strategy_options,
                                help="Choose the option strategy from analysis"
                            )
                        
                        with col_apply2:
                            # Risk and confidence from analysis
                            risk_score = getattr(analysis, 'risk_score', 5.0)
                            if isinstance(risk_score, (int, float)):
                                estimated_risk = float(risk_score) * 40  # Convert to dollar amount
                            else:
                                estimated_risk = 200.0
                            
                            confidence = getattr(analysis, 'confidence', 0.75)
                            if isinstance(confidence, (int, float)):
                                ai_confidence = float(confidence)
                            else:
                                ai_confidence = 0.75
                            
                            st.metric("Estimated Risk", f"${estimated_risk:.0f}")
                            st.metric("AI Confidence", f"{ai_confidence:.2f}")
                        
                        if st.button("📊 Load into Signal Generator", type="primary", use_container_width=True):
                            # Store strategy parameters in session state for Signal Generator tab
                            st.session_state.selected_ticker = apply_ticker.upper()
                            st.session_state.selected_strategy = apply_strategy
                            st.session_state.example_trade = {
                                'strike': 50.0,  # Default, user will adjust
                                'qty': 2,
                                'estimated_risk': estimated_risk,
                                'llm_score': ai_confidence,
                                'iv_rank': 50.0,  # Default
                                'expiry': (datetime.now() + timedelta(days=30)).date(),
                                'dte': 30
                            }
                            st.success(f"✅ Strategy loaded! Go to 'Generate Signal' tab to configure and send.")
                            st.info(f"Ticker: {apply_ticker} | Strategy: {apply_strategy} | Risk: ${estimated_risk:.0f} | Confidence: {ai_confidence:.2f}")

                    except Exception as e:
                        st.error(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()

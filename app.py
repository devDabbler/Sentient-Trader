from dotenv import load_dotenv

# This must be the very first thing to run to ensure all modules are found
load_dotenv()

# Initialize Loguru logging (must be before any other imports that use logging)
from utils.logging_config import setup_logging
setup_logging()

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --- Standard and Third-Party Libraries ---
import streamlit as st
import asyncio
import yfinance as yf
import io
import requests
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
import time
from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# --- Application Configuration ---
st.set_page_config(
    page_title="Sentient Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Windows-specific asyncio policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- Local Application Modules ---
from integrations.tradier_client import TradierClient, validate_tradier_connection, validate_all_trading_modes
from integrations.trading_config import get_trading_mode_manager, TradingMode, switch_to_paper_mode, switch_to_production_mode
from services.llm_strategy_analyzer import LLMStrategyAnalyzer, StrategyAnalysis, extract_bot_config_from_screenshot, create_strategy_comparison
from services.watchlist_manager import WatchlistManager
from services.ai_trading_signals import AITradingSignalGenerator, TradingSignal
from services.ticker_manager import TickerManager
from services.top_trades_scanner import TopTradesScanner, TopTrade
from services.ai_confidence_scanner import AIConfidenceScanner, AIConfidenceTrade
from services.alpha_factors import AlphaFactorCalculator
from services.ml_enhanced_scanner import MLEnhancedScanner, MLEnhancedTrade
from services.penny_stock_analyzer import PennyStockScorer, PennyStockAnalyzer, StockScores
from services.unified_penny_stock_analysis import UnifiedPennyStockAnalysis
from services.penny_stock_constants import PENNY_THRESHOLDS, is_penny_stock, PENNY_STOCK_FILTER_PRESETS
from services.advanced_opportunity_scanner import AdvancedOpportunityScanner, ScanType, ScanFilters, OpportunityResult
from analyzers.comprehensive import ComprehensiveAnalyzer, StockAnalysis
from analyzers.trading_styles import TradingStyleAnalyzer
from services.event_detectors.sec_detector import SECDetector
from services.enhanced_catalyst_detector import EnhancedCatalystDetector

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

@st.cache_resource
def get_ticker_manager():
    """Cached TickerManager instance to avoid repeated initialization"""
    return TickerManager()

@st.cache_resource
def get_advanced_scanner():
    """Cached AdvancedOpportunityScanner to avoid repeated initialization"""
    return AdvancedOpportunityScanner(use_ai=True)

@st.cache_resource
def get_ai_scanner():
    """Cached AIConfidenceScanner to avoid repeated initialization"""
    return AIConfidenceScanner()

@st.cache_resource
def get_ml_scanner():
    """Cached MLEnhancedScanner to avoid repeated initialization"""
    return MLEnhancedScanner()

@st.cache_resource
def get_kraken_client(api_key: str, api_secret: str):
    """Cached Kraken client to avoid 4-5 second re-initialization on every rerun"""
    from clients.kraken_client import KrakenClient
    logger.info("🔧 Initializing cached Kraken client...")
    client = KrakenClient(api_key=api_key, api_secret=api_secret)
    success, message = client.validate_connection()
    if success:
        logger.info(f"✅ Cached Kraken client connected: {message}")
    else:
        logger.warning(f"⚠️ Kraken client initialized but connection issue: {message}")
    return client

@st.cache_resource
def get_supabase_client():
    """Cached Supabase client to avoid multiple re-initializations per rerun"""
    from clients.supabase_client import get_supabase_client as _get_client
    logger.info("🔧 Initializing cached Supabase client...")
    client = _get_client()
    logger.info("✅ Cached Supabase client ready")
    return client

@st.cache_resource
def get_crypto_scanner(_kraken_client, _crypto_config):
    """Cached CryptoOpportunityScanner to avoid repeated initialization"""
    from services.crypto_scanner import CryptoOpportunityScanner
    logger.info("🔧 Initializing cached Crypto Scanner...")
    return CryptoOpportunityScanner(_kraken_client, _crypto_config)

@st.cache_resource
def get_ai_crypto_scanner(_kraken_client, _crypto_config):
    """Cached AICryptoScanner to avoid repeated initialization"""
    from services.ai_crypto_scanner import AICryptoScanner
    logger.info("🔧 Initializing cached AI Crypto Scanner...")
    return AICryptoScanner(_kraken_client, _crypto_config)

@st.cache_resource
def get_penny_crypto_scanner(_kraken_client, _crypto_config):
    """Cached PennyCryptoScanner to avoid repeated initialization"""
    from services.penny_crypto_scanner import PennyCryptoScanner
    logger.info("🔧 Initializing cached Penny Crypto Scanner...")
    return PennyCryptoScanner(_kraken_client, _crypto_config)

@st.cache_resource
def get_sub_penny_discovery():
    """Cached SubPennyDiscovery to avoid repeated initialization"""
    from services.sub_penny_discovery import SubPennyDiscovery
    logger.info("🔧 Initializing cached Sub-Penny Discovery Engine...")
    return SubPennyDiscovery()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_news_for_ticker(ticker: str, max_articles: int = 10):
    """Cache news fetching to avoid repeated API calls for the same ticker"""
    from analyzers.news import get_stock_news
    try:
        news_data = get_stock_news(ticker, max_articles)
        logger.info(f"Fetched and cached {len(news_data) if news_data else 0} news articles for {ticker}")
        return news_data
    except Exception as e:
        logger.error(f"Error fetching cached news for {ticker}: {e}")
        return []

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
# Logging is now handled by Loguru (see utils/logging_config.py)
# All logging output goes to logs/sentient_trader.log with automatic rotation

# Log app startup
logger.info("="*80)
logger.info("🚀 Sentient Trader Application Started")
logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"📁 Log file: logs/sentient_trader_app.log (resets on each start)")
logger.info("="*80)

# Optional: Create a timestamped archive of previous logs (disabled by default)
# Uncomment the following lines if you want to preserve historical logs
# try:
#     log_dir = os.path.join(os.path.dirname(__file__), 'logs')
#     archive_name = f"sentient_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
#     # This would save a timestamped copy before reset
# except Exception:
#     pass

# Third-party logger suppression is now handled by Loguru in utils/logging_config.py
# No need to manually configure individual loggers

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
    
    # Additional attributes for compatibility
    @property
    def name(self) -> str:
        """Alias for strategy_name for backward compatibility"""
        return self.strategy_name
    
    @property
    def score(self) -> float:
        """Alias for confidence for backward compatibility"""
        return self.confidence
    
    @property
    def experience(self) -> str:
        """Alias for experience_level for backward compatibility"""
        return self.experience_level
    
    @property
    def best_for(self) -> List[str]:
        """Alias for best_conditions for backward compatibility"""
        return self.best_conditions
    
    @property
    def reasoning_list(self) -> List[str]:
        """Convert reasoning string to list for iteration"""
        if isinstance(self.reasoning, str):
            return [line.strip() for line in self.reasoning.split('\n') if line.strip()]
        return self.reasoning if isinstance(self.reasoning, list) else []
    
    # Default values for missing attributes
    win_rate: Optional[str] = None
    capital_req: Optional[str] = None
    description: Optional[str] = None
    setup_steps: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

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
            # Check if capital requirement is high (handle different formats)
            is_high_capital = capital_req.startswith("High") if isinstance(capital_req, str) else capital_req == "High"
            if capital_available < 2000 and is_high_capital:
                score -= 40
                reasoning_parts.append("⚠️ May require more capital than available")
            elif capital_available < 5000 and is_high_capital:
                score -= 10  # Smaller penalty for moderate capital shortfall
                reasoning_parts.append("⚠️ Consider capital requirements carefully")
            
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
                if strategy_key in ["IRON_CONDOR", "SELL_CALL", "SELL_PUT", "WHEEL_STRATEGY"]:
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
            
            # Convert score to confidence (0-1) with better scaling
            # Raw score can be negative, so we normalize it better
            # Use a more generous scaling to avoid very low scores
            base_score = max(0, score + 30)  # Less harsh penalty for negative scores
            confidence = min(1.0, base_score / 80)  # Scale to 0-1 with 80 as max
            
            # Debug: Log scoring details for first few strategies
            if strategy_key in ["SELL_PUT", "SELL_CALL", "IRON_CONDOR"]:
                print(f"DEBUG {strategy_key}: raw_score={score}, base_score={base_score}, confidence={confidence:.3f}")
                print(f"  - IV_rank: {analysis.iv_rank}, trend: {analysis.trend}, outlook: {outlook}")
                print(f"  - capital_available: {capital_available}, capital_req: {capital_req}")
                print(f"  - reasoning: {reasoning_parts}")
            
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
                    experience_level=strategy_info["experience"],
                    win_rate=strategy_info.get("typical_win_rate"),
                    capital_req=strategy_info.get("capital_req", "Medium"),
                    description=strategy_info["description"],
                    setup_steps=strategy_info.get("examples", []),
                    warnings=strategy_info.get("notes", "").split("; ") if strategy_info.get("notes") else []
                ))
        
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:5]

def _apply_filter_preset(filters: ScanFilters, filter_name: str) -> None:
    """Apply a single filter preset to the filters object"""
    if filter_name == "High Confidence Only (Score ≥70)":
        filters.min_score = 70.0
    elif filter_name == "Ultra-Low Price (<$1)":
        filters.max_price = PENNY_THRESHOLDS.ULTRA_LOW_PRICE
    elif filter_name == "Penny Stocks ($1-$5)":
        filters.min_price = PENNY_THRESHOLDS.ULTRA_LOW_PRICE
        filters.max_price = PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE
    elif filter_name == "Volume Surge (>2x avg)":
        filters.min_volume_ratio = 2.0
    elif filter_name == "Strong Momentum (>5% change)":
        filters.min_change_pct = 5.0
    elif filter_name == "Power Zone Stocks Only":
        filters.require_power_zone = True
    elif filter_name == "EMA Reclaim Setups":
        filters.require_ema_reclaim = True

def _apply_secondary_filter(filters: ScanFilters, filter_name: str) -> None:
    """Apply secondary filters that can be combined with primary filters"""
    if filter_name == "High Confidence Only (Score ≥70)":
        if filters.min_score is None or filters.min_score < 70.0:
            filters.min_score = 70.0
    elif filter_name == "Volume Surge (>2x avg)":
        if filters.min_volume_ratio is None or filters.min_volume_ratio < 2.0:
            filters.min_volume_ratio = 2.0
    elif filter_name == "Strong Momentum (>5% change)":
        if filters.min_change_pct is None or filters.min_change_pct < 5.0:
            filters.min_change_pct = 5.0
    elif filter_name == "Power Zone Stocks Only":
        filters.require_power_zone = True
    elif filter_name == "EMA Reclaim Setups":
        filters.require_ema_reclaim = True
    elif filter_name == "RSI Oversold (<30)":
        filters.max_rsi = 30.0
    elif filter_name == "RSI Overbought (>70)":
        filters.min_rsi = 70.0
    elif filter_name == "High IV Rank (>60)":
        filters.min_iv_rank = 60.0
    elif filter_name == "Low IV Rank (<40)":
        filters.max_iv_rank = 40.0

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
    
    # CRITICAL: Initialize trading_mode to PAPER by default for safety
    # This MUST happen before get_trading_mode_manager() is called
    if 'trading_mode' not in st.session_state:
        st.session_state.trading_mode = TradingMode.PAPER
        logger.info("🔒 Initialized trading_mode to PAPER (safe default)")
    else:
        # Validate trading_mode by comparing values (enum identity can change after Streamlit serialization)
        current_mode_value = getattr(st.session_state.trading_mode, 'value', None)
        if current_mode_value not in ['paper', 'production']:
            # Safety check: if trading_mode is corrupted/invalid, reset to PAPER
            logger.warning(f"⚠️ Invalid trading_mode in session state: {st.session_state.trading_mode}. Resetting to PAPER for safety.")
            st.session_state.trading_mode = TradingMode.PAPER
        elif current_mode_value == 'paper':
            # Ensure it's the correct enum instance (fixes serialization issues)
            st.session_state.trading_mode = TradingMode.PAPER
        elif current_mode_value == 'production':
            # Ensure it's the correct enum instance
            st.session_state.trading_mode = TradingMode.PRODUCTION
    
    # Initialize background update queue for ticker analysis
    if 'update_queue' not in st.session_state:
        from queue import Queue
        st.session_state.update_queue = Queue()
        st.session_state.background_processor_started = False
        st.session_state.background_update_results = {}
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Trading Mode Configuration
        st.subheader("Trading Mode")
        
        # Get trading mode manager
        mode_manager = get_trading_mode_manager()
        available_modes = mode_manager.get_available_modes()
        
        if not available_modes:
            st.error("❌ No trading credentials configured")
            st.caption("Please set up environment variables for paper and/or production trading")
        else:
            # Display current mode info
            mode_info = mode_manager.get_mode_display_info()
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.metric("Current Mode", mode_info["mode"], mode_info["status"])
            with col2:
                if mode_info["status"] == "✅ Ready":
                    st.success("Ready")
                else:
                    st.error("Not Ready")
            
            # Mode switching
            if len(available_modes) > 1:
                st.caption("Switch Trading Mode:")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📝 Paper Mode", disabled=mode_manager.is_paper_mode()):
                        if switch_to_paper_mode():
                            # Update .env file for background trader
                            try:
                                update_env_file_for_paper_trading()
                                st.info("✅ Updated .env file for paper trading")
                            except Exception as e:
                                st.warning(f"⚠️ Updated session state but .env update failed: {e}")
                            
                            st.success("Switched to Paper Mode")
                            # Refresh Tradier client for new mode
                            if 'tradier_client' in st.session_state:
                                try:
                                    st.session_state.tradier_client = create_tradier_client_from_env(trading_mode=TradingMode.PAPER)
                                    st.info("Tradier client refreshed for Paper Mode")
                                except Exception as e:
                                    st.warning(f"Failed to refresh Tradier client: {e}")
                            st.rerun()
                        else:
                            st.error("Failed to switch to Paper Mode")
                
                with col2:
                    # Safety confirmation for live trading
                    if 'confirm_live_switch' not in st.session_state:
                        st.session_state.confirm_live_switch = False
                    
                    if not st.session_state.confirm_live_switch:
                        st.warning("⚠️ **LIVE TRADING USES REAL MONEY!**")
                        if st.button("💰 Switch to Production Mode", disabled=mode_manager.is_production_mode(), key="confirm_live_btn"):
                            st.session_state.confirm_live_switch = True
                            st.rerun()
                    else:
                        st.error("⚠️ **FINAL CONFIRMATION: This will use REAL MONEY**")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ Confirm Live Trading", type="primary", key="final_confirm_live"):
                                if switch_to_production_mode():
                                    # Update .env file for background trader
                                    try:
                                        update_env_file_for_live_trading()
                                        st.info("✅ Updated .env file for live trading")
                                    except Exception as e:
                                        st.warning(f"⚠️ Updated session state but .env update failed: {e}")
                                    
                                    st.success("Switched to Production Mode")
                                    st.session_state.confirm_live_switch = False
                                    # Refresh Tradier client for new mode
                                    if 'tradier_client' in st.session_state:
                                        try:
                                            st.session_state.tradier_client = create_tradier_client_from_env(trading_mode=TradingMode.PRODUCTION)
                                            st.info("Tradier client refreshed for Production Mode")
                                        except Exception as e:
                                            st.warning(f"Failed to refresh Tradier client: {e}")
                                    st.rerun()
                                else:
                                    st.error("Failed to switch to Production Mode")
                                    st.session_state.confirm_live_switch = False
                        with col_cancel:
                            if st.button("❌ Cancel", key="cancel_live_switch"):
                                st.session_state.confirm_live_switch = False
                                st.rerun()
            else:
                current_mode = available_modes[0]
                st.info(f"Only {current_mode.value.title()} mode available")
            
            # Mode details
            with st.expander("Mode Details"):
                st.write(f"**API URL:** {mode_info['api_url']}")
                st.write(f"**Account ID:** {mode_info['account_id']}")
                st.write(f"**Status:** {mode_info['status']}")
                
                # Test connection
                if st.button("Test Connection"):
                    success, message = validate_tradier_connection()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Legacy paper mode toggle for backward compatibility
        st.subheader("Legacy Settings")
        
        # Debug information to help diagnose session state issues
        with st.expander("🔍 Debug Information"):
            st.write(f"**Debug - Session state mode:** `{st.session_state.get('trading_mode', 'NOT SET')}`")
            st.write(f"**Debug - mode_manager.is_paper_mode():** `{mode_manager.is_paper_mode()}`")
            st.write(f"**Debug - mode_manager.get_mode():** `{mode_manager.get_mode()}`")
        
        # Double-check mode_manager is synced with session state before displaying banner
        # Force sync by calling is_paper_mode() which syncs internally
        is_paper = mode_manager.is_paper_mode()
        
        # CRITICAL SAFETY CHECK: If session state says paper but manager says production (or vice versa),
        # force sync and default to paper mode for safety
        session_state_mode = st.session_state.get('trading_mode', TradingMode.PAPER)
        if session_state_mode == TradingMode.PAPER and not is_paper:
            logger.error("🚨 CRITICAL MISMATCH: Session state is PAPER but mode_manager reports PRODUCTION! Forcing PAPER mode for safety.")
            st.session_state.trading_mode = TradingMode.PAPER
            mode_manager.set_mode(TradingMode.PAPER)
            is_paper = True
            st.error("🚨 Safety override: Reset to PAPER mode due to state mismatch")
            st.rerun()
        elif session_state_mode == TradingMode.PRODUCTION and is_paper:
            logger.warning("⚠️ Mismatch: Session state is PRODUCTION but mode_manager reports PAPER. Syncing to session state.")
            mode_manager.set_mode(TradingMode.PRODUCTION)
            is_paper = False
        
        paper_mode = st.toggle("Paper Trading Mode (Legacy)", value=is_paper)
        if paper_mode != is_paper:
            if paper_mode:
                switch_to_paper_mode()
            else:
                switch_to_production_mode()
            st.rerun()
        
        # Final check before displaying banner - always sync one more time
        is_paper_final = mode_manager.is_paper_mode()
        if is_paper_final:
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

                if st.button("🔎 Open Strategy Analyzer", width="stretch"):
                    # Set a flag so the main tabs can react (we can't switch tabs programmatically reliably)
                    st.session_state.goto_strategy_analyzer = True
                    st.rerun()
            except Exception:
                st.write("Compact summary unavailable")
        else:
            st.info("Run a strategy analysis to see a quick summary here.")
    
    # Main tabs - Reorganized for clarity
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15 = st.tabs([
        "🏠 Dashboard",
        "🚀 Advanced Scanner",
        "⭐ My Tickers",
        "🔍 Stock Intelligence", 
        "🎯 Strategy Advisor", 
        "📊 Generate Signal", 
        "📜 Signal History",
        "📚 Strategy Guide",
        "📚 Strategy Templates",
        "🏦 Tradier Account",
        "📈 IBKR Trading",
        "⚡ Scalping/Day Trade",
        "🤖 Strategy Analyzer",
        "🤖 Auto-Trader",
        "₿ Crypto Trading"
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
            analyze_btn = st.button("🔍 Analyze Stock", type="primary", width="stretch")
        
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
                                
                                # Add helpful information about finding options symbols
                                with st.expander("📋 How to find valid options symbols", expanded=False):
                                    st.markdown("""
                                    **Since Tradier sandbox has limited options data, here's how to find valid symbols:**
                                    
                                    1. **Tradier Web Platform**: 
                                       - Go to [sandbox.tradier.com](https://sandbox.tradier.com)
                                       - Search for SOFI options
                                       - Copy the exact symbol format
                                    
                                    2. **Yahoo Finance**:
                                       - Search "SOFI options"
                                       - Look for the symbol format: `SOFI251126C00025000`
                                    
                                    3. **Common SOFI Strike Prices** (as of recent):
                                       - $20, $22.50, $25, $27.50, $30, $32.50, $35
                                    
                                    4. **Common Expiration Dates**:
                                       - Weekly: Every Friday
                                       - Monthly: Third Friday of each month
                                    
                                    **Symbol Format**: `SOFI + YYMMDD + C/P + 8-digit strike`
                                    - Example: `SOFI251126C00025000` = SOFI $25 Call expiring 11/26/25
                                    """)
                                
                                # Generate options contract symbol automatically
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Auto-generate options symbol if we have the required data
                                    auto_generated_symbol = ""
                                    if selected_rec and selected_rec.get('strike_suggestion') and selected_rec.get('dte_suggestion'):
                                        try:
                                            # Parse strike price and round to nearest $0.50 or $1.00
                                            strike = float(selected_rec['strike_suggestion'].replace('$', '').split()[0])
                                            
                                            # Round to nearest $0.50 for better strike price availability
                                            if strike < 10:
                                                strike = round(strike * 2) / 2  # Round to nearest $0.50
                                            else:
                                                strike = round(strike)  # Round to nearest $1.00
                                            
                                            # Calculate expiration date (DTE from today)
                                            dte_text = selected_rec['dte_suggestion'].split()[0]
                                            # Handle range format like "30-45" by taking the first number
                                            if '-' in dte_text:
                                                dte = int(dte_text.split('-')[0])
                                            else:
                                                dte = int(dte_text)
                                            
                                            # Round to common options expiration dates (Fridays)
                                            exp_date = datetime.now() + timedelta(days=dte)
                                            # Find the next Friday (options typically expire on Fridays)
                                            days_until_friday = (4 - exp_date.weekday()) % 7
                                            if days_until_friday == 0 and exp_date.weekday() != 4:  # If today is not Friday
                                                days_until_friday = 7
                                            exp_date = exp_date + timedelta(days=days_until_friday)
                                            
                                            # Determine option type (P for PUT, C for CALL)
                                            option_type = "P" if "PUT" in selected_rec.get('strategy', '') else "C"
                                            
                                            # Format: SYMBOL + YYMMDD + P/C + 8-digit strike (padded with zeros)
                                            auto_generated_symbol = f"{trade_symbol.upper()}{exp_date.strftime('%y%m%d')}{option_type}{int(strike * 1000):08d}"
                                            
                                            # Set the session state before creating the widget
                                            if 'modal_option_symbol' not in st.session_state or not st.session_state['modal_option_symbol']:
                                                st.session_state['modal_option_symbol'] = auto_generated_symbol
                                        except:
                                            pass
                                    
                                    # Use temp generated symbol if available, otherwise use existing value
                                    default_value = st.session_state.get('temp_generated_symbol', st.session_state.get('modal_option_symbol', ''))
                                    if st.session_state.get('temp_generated_symbol'):
                                        # Clear the temp value after using it
                                        st.session_state['modal_option_symbol'] = st.session_state['temp_generated_symbol']
                                        del st.session_state['temp_generated_symbol']
                                    
                                    option_symbol = st.text_input(
                                        "Options Contract Symbol", 
                                        value=st.session_state.get('modal_option_symbol', ''), 
                                        placeholder="e.g., SOFI250117P00029000",
                                        help="Enter the full options contract symbol (e.g., SOFI250117P00029000 for SOFI $29 Put expiring 01/17/25)",
                                        key="modal_option_symbol"
                                    )
                                
                                with col2:
                                    if st.button("🔧 Auto-Generate", help="Generate options symbol from strike and DTE"):
                                        if selected_rec and selected_rec.get('strike_suggestion') and selected_rec.get('dte_suggestion'):
                                            try:
                                                # Parse strike price and round to nearest $0.50 or $1.00
                                                strike = float(selected_rec['strike_suggestion'].replace('$', '').split()[0])
                                                
                                                # Round to nearest $0.50 for better strike price availability
                                                if strike < 10:
                                                    strike = round(strike * 2) / 2  # Round to nearest $0.50
                                                else:
                                                    strike = round(strike)  # Round to nearest $1.00
                                                
                                                # Calculate expiration date (DTE from today)
                                                dte_text = selected_rec['dte_suggestion'].split()[0]
                                                # Handle range format like "30-45" by taking the first number
                                                if '-' in dte_text:
                                                    dte = int(dte_text.split('-')[0])
                                                else:
                                                    dte = int(dte_text)
                                                
                                                # Round to common options expiration dates (Fridays)
                                                exp_date = datetime.now() + timedelta(days=dte)
                                                # Find the next Friday (options typically expire on Fridays)
                                                days_until_friday = (4 - exp_date.weekday()) % 7
                                                if days_until_friday == 0 and exp_date.weekday() != 4:  # If today is not Friday
                                                    days_until_friday = 7
                                                exp_date = exp_date + timedelta(days=days_until_friday)
                                                
                                                # Determine option type (P for PUT, C for CALL)
                                                option_type = "P" if "PUT" in selected_rec.get('strategy', '') else "C"
                                                
                                                # Format: SYMBOL + YYMMDD + P/C + 8-digit strike (padded with zeros)
                                                generated_symbol = f"{trade_symbol.upper()}{exp_date.strftime('%y%m%d')}{option_type}{int(strike * 1000):08d}"
                                                # Store the generated symbol in a temporary session state key
                                                st.session_state['temp_generated_symbol'] = generated_symbol
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error generating symbol: {e}")
                                        else:
                                            st.error("Need strike and DTE suggestions to auto-generate")
                                    
                                    # Add validation button
                                    if st.button("✅ Validate Symbol", help="Check if the options symbol exists"):
                                        if st.session_state.get('modal_option_symbol'):
                                            symbol = st.session_state['modal_option_symbol']
                                            
                                            # Basic format validation first
                                            if len(symbol) < 15:
                                                st.error("❌ Options symbol too short. Expected format: SYMBOL + YYMMDD + C/P + 8-digit strike")
                                            elif not any(c.isdigit() for c in symbol):
                                                st.error("❌ Options symbol must contain numbers for date and strike")
                                            elif not any(c in ['C', 'P'] for c in symbol):
                                                st.error("❌ Options symbol must contain 'C' for Call or 'P' for Put")
                                            else:
                                                with st.spinner("Validating options symbol..."):
                                                    success, message = st.session_state.tradier_client.validate_options_symbol(symbol)
                                                    if success:
                                                        st.success(f"✅ {message}")
                                                    else:
                                                        # Check if it's an API limitation
                                                        if "API limitation" in message:
                                                            st.warning(f"⚠️ {message}")
                                                            st.info("💡 The symbol format looks correct. You can proceed with the trade, but verify the symbol exists on your broker's platform.")
                                                        else:
                                                            st.error(f"❌ {message}")
                                        else:
                                            st.error("Please enter an options symbol first")
                                
                                trade_side = st.selectbox("Action", 
                                                        ["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"],
                                                        index=0 if 'buy' in default_action else 1,
                                                        key="modal_trade_side")
                                trade_quantity = st.number_input("Contracts", min_value=1, value=default_qty, step=1, key="modal_trade_qty")
                            else:
                                trade_side = st.selectbox("Action", ["buy", "sell", "sell_short", "buy_to_cover"], key="modal_trade_side2")
                                trade_quantity = st.number_input("Quantity (shares)", min_value=1, value=default_qty, step=1, key="modal_trade_qty2")
                                option_symbol = None
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
                        if st.button("✅ Place Order", type="primary", width="stretch", key="modal_place_order"):
                            with st.spinner("Placing order..."):
                                try:
                                    # Validate required fields
                                    if not trade_symbol:
                                        st.error("❌ Please enter a symbol")
                                        st.stop()
                                    elif trade_quantity <= 0:
                                        st.error("❌ Quantity must be greater than 0")
                                        st.stop()
                                    elif trade_type == "limit" and (not trade_price or trade_price <= 0):
                                        st.error("❌ Please enter a valid limit price")
                                        st.stop()
                                    elif trade_class == "option" and (not st.session_state.get('modal_option_symbol', '')):
                                        st.error("❌ Please enter the options contract symbol (e.g., SOFI250117P00029000)")
                                        st.stop()
                                    
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
                                        
                                        # Prepare bracket order parameters
                                        bracket_params = {
                                            "symbol": trade_symbol.upper(),
                                            "side": trade_side,
                                            "quantity": trade_quantity,
                                            "entry_price": trade_price,
                                            "take_profit_price": target,
                                            "stop_loss_price": stop_loss,
                                            "duration": 'gtc',  # Use GTC for bracket orders
                                            "tag": f"AIREC{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                        }
                                        
                                        # Add option_symbol if this is an options trade
                                        if trade_class == "option" and st.session_state.get('modal_option_symbol'):
                                            bracket_params["option_symbol"] = st.session_state['modal_option_symbol'].upper()
                                        
                                        success, result = st.session_state.tradier_client.place_bracket_order(**bracket_params)
                                    else:
                                        # Fallback to regular order for market orders or options
                                        order_data = {
                                            "class": trade_class,
                                            "side": trade_side,
                                            "quantity": str(trade_quantity),
                                            "type": trade_type,
                                            "duration": "day",
                                            "tag": f"AIREC{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                        }
                                        
                                        # Use appropriate symbol field based on trade class
                                        if trade_class == "option" and st.session_state.get('modal_option_symbol'):
                                            order_data["option_symbol"] = st.session_state['modal_option_symbol'].upper()
                                            trade_symbol_display = st.session_state['modal_option_symbol']
                                        else:
                                            order_data["symbol"] = trade_symbol.upper()
                                            trade_symbol_display = trade_symbol
                                        
                                        if trade_type == "limit" and trade_price:
                                            order_data["price"] = str(trade_price)
                                        
                                        # Explain why bracket wasn't used
                                        reason = "market order" if trade_type == "market" else "options trade" if trade_class != "equity" else "non-standard side"
                                        logger.info(f"🚀 Placing REGULAR order ({reason}): {trade_symbol_display} {trade_side} {trade_quantity} @ {trade_type}")
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
                        if st.button("❌ Cancel", width="stretch", key="modal_cancel"):
                            st.session_state.show_quick_trade = False
                            st.session_state.selected_recommendation = None
                            st.rerun()
            st.divider()
        
        if analyze_btn and search_ticker:
            # Clear previous analysis from session state
            if 'analysis' in st.session_state:
                del st.session_state['analysis']

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
                
                st.write("📄 Fetching SEC filings (8-K, 10-Q, 10-K)...")
                time.sleep(0.3)
                
                st.write(f"🤖 Generating {trading_style_display} recommendations...")
                analysis = ComprehensiveAnalyzer.analyze_stock(search_ticker, trading_style)
                
                # Fetch SEC filings and enhanced catalyst data
                sec_filings = []
                enhanced_catalysts = []
                if analysis:
                    try:
                        logger.info(f"📄 Fetching SEC filings for {search_ticker}...")
                        # Create a temporary SEC detector instance (we don't need alert system for just fetching)
                        from services.event_detectors.base_detector import BaseEventDetector
                        
                        # Get company CIK for SEC filings
                        try:
                            stock = yf.Ticker(search_ticker)
                            info = stock.info
                            # Try to get CIK from info or look it up
                            cik = None
                            if 'cik' in info:
                                cik = str(info['cik']).zfill(10)  # CIK should be 10 digits
                            
                            # If no CIK in info, try to look it up from SEC
                            if not cik:
                                logger.info(f"🔍 CIK not found in yfinance info, looking up from SEC...")
                                try:
                                    import requests
                                    url = "https://www.sec.gov/files/company_tickers.json"
                                    headers = {'User-Agent': "Sentient Trader/1.0 (trading@example.com)"}
                                    response = requests.get(url, headers=headers, timeout=10)
                                    response.raise_for_status()
                                    companies = response.json()
                                    for company in companies.values():
                                        if company.get('ticker', '').upper() == search_ticker.upper():
                                            cik = str(company.get('cik_str', '')).zfill(10)
                                            logger.info(f"✅ Found CIK for {search_ticker}: {cik}")
                                            break
                                except Exception as lookup_error:
                                    logger.warning(f"Could not lookup CIK for {search_ticker}: {lookup_error}")
                            
                            # Log CIK status
                            if cik:
                                logger.info(f"✅ CIK found for {search_ticker}: {cik}")
                            else:
                                logger.warning(f"⚠️ No CIK available for {search_ticker}, skipping SEC filings")
                            
                            if cik:
                                # Create SEC detector instance (using None for alert_system since we just want data)
                                class TempSECDetector:
                                    def __init__(self):
                                        self.user_agent = "Sentient Trader/1.0 (trading@example.com)"
                                    
                                    def get_company_cik(self, ticker: str):
                                        # Already have it
                                        return cik
                                    
                                    def get_recent_filings(self, ticker: str, cik: str, hours_back: int = 168):
                                        """Get recent SEC filings (last 7 days)"""
                                        try:
                                            import requests
                                            from datetime import datetime, timedelta
                                            
                                            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                                            headers = {'User-Agent': self.user_agent}
                                            
                                            response = requests.get(url, headers=headers, timeout=10)
                                            response.raise_for_status()
                                            
                                            data = response.json()
                                            recent_filings = data.get('filings', {}).get('recent', {})
                                            
                                            if not recent_filings:
                                                return []
                                            
                                            filings = []
                                            cutoff_time = datetime.now() - timedelta(hours=hours_back)
                                            
                                            filing_dates = recent_filings.get('filingDate', [])
                                            form_types = recent_filings.get('form', [])
                                            accession_numbers = recent_filings.get('accessionNumber', [])
                                            primary_documents = recent_filings.get('primaryDocument', [])
                                            
                                            # Check last 20 filings
                                            for i in range(min(len(filing_dates), 20)):
                                                try:
                                                    filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d')
                                                    
                                                    if filing_date >= cutoff_time:
                                                        form_type = form_types[i]
                                                        accession = accession_numbers[i]
                                                        primary_doc = primary_documents[i] if i < len(primary_documents) else ''
                                                        
                                                        # Build filing URL
                                                        accession_clean = accession.replace('-', '')
                                                        cik_clean = cik.lstrip('0')  # Remove leading zeros
                                                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{primary_doc}"
                                                        
                                                        # Get filing description
                                                        filing_descriptions = {
                                                            '8-K': 'Material Event Report',
                                                            '8-K/A': 'Amended Material Event',
                                                            '4': 'Insider Trading Statement',
                                                            '10-Q': 'Quarterly Report',
                                                            '10-K': 'Annual Report',
                                                            '10-Q/A': 'Amended Quarterly Report',
                                                            '10-K/A': 'Amended Annual Report',
                                                            'S-1': 'IPO Registration',
                                                            'S-3': 'Securities Registration',
                                                            'DEF 14A': 'Proxy Statement'
                                                        }
                                                        
                                                        filings.append({
                                                            'ticker': ticker,
                                                            'form_type': form_type,
                                                            'filing_date': filing_date.strftime('%Y-%m-%d'),
                                                            'description': filing_descriptions.get(form_type, form_type),
                                                            'url': filing_url,
                                                            'days_ago': (datetime.now() - filing_date).days,
                                                            'is_critical': form_type in ['8-K', '8-K/A', '4', 'S-1']
                                                        })
                                                except Exception as e:
                                                    logger.debug(f"Error parsing filing {i}: {e}")
                                                    continue
                                            
                                            return sorted(filings, key=lambda x: x['filing_date'], reverse=True)[:10]  # Last 10
                                            
                                        except Exception as e:
                                            logger.error(f"Error fetching SEC filings: {e}")
                                            return []
                                
                                sec_detector = TempSECDetector()
                                sec_filings = sec_detector.get_recent_filings(search_ticker, cik, hours_back=168)  # Last 7 days
                                logger.info(f"✅ Retrieved {len(sec_filings)} recent SEC filings for {search_ticker}")
                                
                                # Analyze filings for catalysts
                                if sec_filings:
                                    for filing in sec_filings:
                                        if filing['form_type'] == '8-K':
                                            # Parse 8-K for material events
                                            try:
                                                # Note: Full parsing would require fetching filing content
                                                # For now, we'll just flag 8-Ks as material events
                                                enhanced_catalysts.append({
                                                    'type': 'SEC Filing - 8-K',
                                                    'date': filing['filing_date'],
                                                    'days_away': -filing['days_ago'],  # Negative means in the past
                                                    'impact': 'HIGH',
                                                    'description': f"Material event filing: {filing['description']}",
                                                    'filing_url': filing['url'],
                                                    'is_critical': filing['is_critical']
                                                })
                                            except Exception as e:
                                                logger.debug(f"Error parsing 8-K: {e}")
                        except Exception as e:
                            logger.warning(f"Could not fetch SEC filings for {search_ticker}: {e}")
                            sec_filings = []
                    except Exception as e:
                        logger.error(f"Error fetching SEC filings data for {search_ticker}: {e}", exc_info=True)
                        sec_filings = []
                
                # Store filings in session state
                st.session_state.sec_filings = sec_filings
                st.session_state.enhanced_catalysts = enhanced_catalysts
                
                # Run unified penny stock analysis if applicable
                penny_stock_analysis = None
                if analysis and is_penny_stock(analysis.price):
                    logger.info(f"💰 PENNY STOCK DETECTED: {search_ticker} @ ${analysis.price:.2f} (< ${PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE})")
                    st.write("💰 Running enhanced penny stock analysis...")
                    try:
                        unified_analyzer = UnifiedPennyStockAnalysis()
                        logger.info(f"✅ UnifiedPennyStockAnalysis initialized for {search_ticker}")
                        
                        # Map trading style for penny stock analysis
                        penny_style_map = {
                            "DAY_TRADE": "SCALP",
                            "SWING_TRADE": "SWING",
                            "SCALP": "SCALP",
                            "BUY_HOLD": "POSITION",
                            "OPTIONS": "SWING"
                        }
                        penny_trading_style = penny_style_map.get(trading_style, "SWING")
                        logger.info(f"📊 Trading style mapped: {trading_style} -> {penny_trading_style} for penny stock analysis")
                        
                        logger.info(f"🔍 Starting comprehensive penny stock analysis for {search_ticker}...")
                        penny_stock_analysis = unified_analyzer.analyze_comprehensive(
                            ticker=search_ticker,
                            trading_style=penny_trading_style,
                            include_backtest=False,  # Skip backtest for speed
                            check_options=(trading_style == "OPTIONS")
                        )
                        
                        if penny_stock_analysis:
                            if 'error' in penny_stock_analysis:
                                logger.error(f"❌ Penny stock analysis error for {search_ticker}: {penny_stock_analysis['error']}")
                                st.error(f"⚠️ Penny stock analysis encountered an error: {penny_stock_analysis['error']}")
                            else:
                                logger.info(f"✅ Penny stock analysis completed for {search_ticker}")
                                logger.info(f"   Classification: {penny_stock_analysis.get('classification', 'N/A')}")
                                logger.info(f"   ATR Stop: ${penny_stock_analysis.get('atr_stop_loss', 'N/A')} ({penny_stock_analysis.get('atr_stop_pct', 0):.1f}%)")
                                logger.info(f"   Risk Level: {penny_stock_analysis.get('risk_level', 'N/A')}")
                                
                                if 'final_recommendation' in penny_stock_analysis:
                                    final_rec = penny_stock_analysis['final_recommendation']
                                    logger.info(f"   Final Decision: {final_rec.get('decision', 'N/A')}")
                        else:
                            logger.warning(f"⚠️ Penny stock analysis returned None for {search_ticker}")
                        
                        st.session_state.penny_stock_analysis = penny_stock_analysis
                        logger.info(f"💾 Penny stock analysis stored in session state for {search_ticker}")
                    except Exception as e:
                        logger.error(f"❌ ERROR running unified penny stock analysis for {search_ticker}: {e}", exc_info=True)
                        st.error(f"⚠️ Error running enhanced penny stock analysis: {str(e)}")
                        penny_stock_analysis = None
                else:
                    if analysis:
                        logger.info(f"ℹ️ {search_ticker} @ ${analysis.price:.2f} is NOT a penny stock (>= $5.0)")
                    else:
                        logger.warning(f"⚠️ No analysis available for {search_ticker} to check penny stock status")

                # --- Generate Premium AI Trading Signal ---
                st.session_state.ai_trading_signal = None
                if analysis:
                    st.write("🤖 Generating Premium AI Trading Signal with Gemini...")
                    signal_generator = AITradingSignalGenerator()

                    # Prepare data for the signal generator
                    technical_data = {
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
                    news_data = analysis.recent_news
                    sentiment_data = {
                        'score': analysis.sentiment_score,
                        'signals': analysis.sentiment_signals
                    }
                    social_data = None  # Social sentiment not available in StockAnalysis

                    # Generate the signal using the configured premium model
                    ai_signal = signal_generator.generate_signal(
                        symbol=analysis.ticker,
                        technical_data=technical_data,
                        news_data=news_data,
                        sentiment_data=sentiment_data,
                        social_data=social_data
                    )
                    st.session_state.ai_trading_signal = ai_signal
                # ----------------------------------------

                if analysis:
                    status.update(label=f"✅ Analysis complete for {search_ticker}", state="complete")
                else:
                    status.update(label=f"❌ Analysis failed for {search_ticker}", state="error")
                
                if analysis:
                    logger.info(f"💾 Storing analysis in session state: {analysis.ticker} @ ${analysis.price:.2f}")
                    st.session_state.current_analysis = analysis
                    logger.info(f"✅ Analysis stored. Quick trade flag status: {st.session_state.get('show_quick_trade', False)}")
                    
                    # Detect penny stock and runner characteristics
                    is_penny_stock_flag = is_penny_stock(analysis.price)
                    is_otc = analysis.ticker.endswith(('.OTC', '.PK', '.QB'))
                    volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
                    is_runner = volume_vs_avg > 200 and analysis.change_pct > 10  # 200%+ volume spike and 10%+ gain
                    
                    # Get unified penny stock analysis if available
                    penny_stock_analysis = st.session_state.get('penny_stock_analysis')
                    if is_penny_stock_flag:
                        if penny_stock_analysis:
                            logger.info(f"✅ Found penny stock analysis in session state for {analysis.ticker}")
                        else:
                            logger.warning(f"⚠️ Penny stock detected but no enhanced analysis found for {analysis.ticker}")
                    
                    # Header metrics
                    st.success(f"✅ Analysis complete for {analysis.ticker}")

                    # --- Display Premium AI Trading Signal ---
                    if 'ai_trading_signal' in st.session_state and st.session_state.ai_trading_signal:
                        signal = st.session_state.ai_trading_signal
                        st.subheader("🤖 Premium AI Trading Signal (Gemini)")
                        
                        signal_color = "green" if signal.signal == "BUY" else "red" if signal.signal == "SELL" else "orange"
                        st.markdown(f"## <span style='color:{signal_color};'>{signal.signal}</span>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{signal.confidence:.1f}%")
                        with col2:
                            st.metric("Risk Level", signal.risk_level)
                        with col3:
                            st.metric("Time Horizon", signal.time_horizon.replace('_', ' ').title())

                        with st.expander("View AI Reasoning and Price Targets"):
                            st.write(f"**Reasoning:** {signal.reasoning}")
                            st.write("-")
                            st.metric("Entry Price", f"${signal.entry_price:.2f}" if signal.entry_price else "N/A")
                            st.metric("Target Price", f"${signal.target_price:.2f}" if signal.target_price else "N/A")
                            st.metric("Stop Loss", f"${signal.stop_loss:.2f}" if signal.stop_loss else "N/A")
                        st.divider()
                    # ----------------------------------------
                    
                    # Special alerts for penny stocks and runners
                    if is_runner:
                        st.warning(f"🚀 **RUNNER DETECTED!** {volume_vs_avg:+.0f}% volume spike with {analysis.change_pct:+.1f}% price move!")
                    
                    if is_penny_stock_flag:
                        if penny_stock_analysis and 'classification' in penny_stock_analysis:
                            classification = penny_stock_analysis.get('classification', 'PENNY_STOCK')
                            if classification == 'LOW_PRICED':
                                st.info(f"💰 **LOW-PRICED STOCK** (${analysis.price:.2f}) - Price < $5 but market cap suggests established company. Moderate risk.")
                            else:
                                st.warning(f"💰 **{classification}** (${analysis.price:.4f}) - High risk/high reward. Use enhanced risk management.")
                        else:
                            st.info(f"💰 **PENNY STOCK** (${analysis.price:.4f}) - High risk/high reward. Use caution and proper position sizing.")
                    
                    if is_otc:
                        st.warning("⚠️ **OTC STOCK** - Lower liquidity, wider spreads, higher risk. Limited data may be available.")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                    
                    with metric_col1:
                        price_display = f"${analysis.price:.4f}" if is_penny_stock_flag else f"${analysis.price:.2f}"
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
                            runner_risk = "EXTREME" if is_penny_stock_flag and volume_vs_avg > 300 else "VERY HIGH" if volume_vs_avg > 200 else "HIGH"
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
                    
                    # Entropy Analysis (Market Noise Detection)
                    st.subheader("🔬 Entropy Analysis (Market Noise Detection)")
                    
                    entropy_col1, entropy_col2, entropy_col3 = st.columns(3)
                    
                    with entropy_col1:
                        entropy_value = analysis.entropy if analysis.entropy is not None else 50.0
                        st.metric("Entropy Score", f"{entropy_value:.1f}/100")
                        if entropy_value < 30:
                            st.caption("✅ Highly Structured - Ideal for trading")
                        elif entropy_value < 50:
                            st.caption("✅ Structured - Good patterns")
                        elif entropy_value < 70:
                            st.caption("⚠️ Mixed - Trade with caution")
                        else:
                            st.caption("❌ Noisy - High risk/choppy")
                    
                    with entropy_col2:
                        entropy_state = analysis.entropy_state if analysis.entropy_state else "UNKNOWN"
                        st.metric("Market State", entropy_state)
                        st.caption("Pattern predictability")
                    
                    with entropy_col3:
                        entropy_signal = analysis.entropy_signal if analysis.entropy_signal else "CAUTION"
                        signal_emoji = {"FAVORABLE": "✅", "CAUTION": "⚠️", "AVOID": "❌"}
                        st.metric("Trade Signal", f"{signal_emoji.get(entropy_signal, '⚠️')} {entropy_signal}")
                        if entropy_signal == "FAVORABLE":
                            st.caption("🟢 Low entropy - Trade normally")
                        elif entropy_signal == "CAUTION":
                            st.caption("🟡 Moderate entropy - Reduce size")
                        else:
                            st.caption("🔴 High entropy - Avoid or skip")
                    
                    # Entropy explanation
                    with st.expander("ℹ️ What is Entropy?"):
                        st.write("""
                        **Entropy measures market unpredictability and noise:**
                        
                        - **Low Entropy (< 30)**: Clear patterns, predictable moves → Trade with confidence
                        - **Medium Entropy (30-70)**: Some noise, mixed signals → Trade with caution
                        - **High Entropy (> 70)**: Random/choppy price action → Avoid or reduce size significantly
                        
                        Entropy helps filter out false signals and whipsaws by identifying when the market is 
                        too noisy for reliable pattern recognition.
                        """)
                    
                    # Catalysts
                    st.subheader("📅 Upcoming Catalysts")
                    
                    # Combine regular catalysts with enhanced catalysts from SEC filings
                    all_catalysts = list(analysis.catalysts) if analysis.catalysts else []
                    enhanced_catalysts = st.session_state.get('enhanced_catalysts', [])
                    if enhanced_catalysts:
                        all_catalysts.extend(enhanced_catalysts)
                        logger.info(f"✅ Added {len(enhanced_catalysts)} enhanced catalysts from SEC filings")
                    
                    if all_catalysts:
                        for catalyst in all_catalysts:
                            impact_color = {
                                'HIGH': '🔴',
                                'MEDIUM': '🟡',
                                'LOW': '🟢'
                            }.get(catalyst['impact'], '⚪')
                            
                            # Format days away display
                            days_away = catalyst.get('days_away', 'N/A')
                            if isinstance(days_away, int):
                                if days_away < 0:
                                    days_text = f"{abs(days_away)} days ago"
                                elif days_away == 0:
                                    days_text = "Today"
                                else:
                                    days_text = f"{days_away} days away"
                            else:
                                days_text = str(days_away)
                            
                            expander_title = f"{impact_color} {catalyst['type']} - {catalyst['date']} ({days_text})"
                            
                            with st.expander(expander_title):
                                st.write(f"**Impact Level:** {catalyst['impact']}")
                                st.write(f"**Details:** {catalyst['description']}")
                                
                                # Add filing URL if available
                                if 'filing_url' in catalyst:
                                    st.write(f"[📄 View SEC Filing]({catalyst['filing_url']})")
                                
                                if catalyst['type'] == 'Earnings Report' and isinstance(days_away, int) and days_away >= 0 and days_away <= 7:
                                    st.warning("⚠️ Earnings within 7 days - expect high volatility!")
                                
                                if catalyst.get('is_critical'):
                                    st.error("🔴 **CRITICAL FILING** - Review immediately for material events")
                    else:
                        st.info("No major catalysts identified in the next 60 days")
                    
                    # SEC Filings Section
                    sec_filings = st.session_state.get('sec_filings', [])
                    if sec_filings:
                        st.subheader("📄 Recent SEC Filings (Last 7 Days)")
                        logger.info(f"📄 Displaying {len(sec_filings)} SEC filings for {analysis.ticker}")
                        
                        filings_col1, filings_col2 = st.columns([3, 1])
                        
                        with filings_col1:
                            for filing in sec_filings[:10]:  # Show last 10
                                filing_icon = "🔴" if filing['is_critical'] else "🟡"
                                filing_desc = f"{filing_icon} **{filing['form_type']}** - {filing['description']}"
                                
                                with st.expander(f"{filing_desc} ({filing['filing_date']}, {filing['days_ago']} days ago)"):
                                    st.write(f"**Filing Type:** {filing['form_type']}")
                                    st.write(f"**Description:** {filing['description']}")
                                    st.write(f"**Filing Date:** {filing['filing_date']}")
                                    
                                    if filing['is_critical']:
                                        st.error("⚠️ **CRITICAL FILING** - Material event (8-K) or significant filing")
                                    
                                    if filing.get('url'):
                                        st.write(f"[📄 View Filing on SEC.gov]({filing['url']})")
                        
                        with filings_col2:
                            critical_count = sum(1 for f in sec_filings if f['is_critical'])
                            total_count = len(sec_filings)
                            
                            st.metric("Total Filings", total_count)
                            if critical_count > 0:
                                st.metric("Critical Filings", critical_count, delta="Review Required")
                            
                            # Filing type breakdown
                            filing_types = {}
                            for f in sec_filings:
                                form_type = f['form_type']
                                filing_types[form_type] = filing_types.get(form_type, 0) + 1
                            
                            if filing_types:
                                st.write("**Filing Types:**")
                                for form_type, count in sorted(filing_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                                    st.caption(f"{form_type}: {count}")
                    else:
                        logger.info(f"ℹ️ No recent SEC filings found for {analysis.ticker} in the last 7 days")
                    
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
                    
                    # Enhanced Penny Stock Analysis (if applicable)
                    logger.info(f"🔍 Checking enhanced penny stock display: is_penny_stock={is_penny_stock_flag}, has_penny_analysis={penny_stock_analysis is not None}")
                    if is_penny_stock_flag and penny_stock_analysis:
                        logger.info(f"✅ DISPLAYING Enhanced Penny Stock Analysis for {analysis.ticker}")
                        st.subheader("💰 Enhanced Penny Stock Analysis")
                        st.success(f"✅ Enhanced analysis available for {analysis.ticker} - Showing detailed results below")
                        
                        # Classification
                        if 'classification' in penny_stock_analysis:
                            classification = penny_stock_analysis.get('classification', 'UNKNOWN')
                            risk_level = penny_stock_analysis.get('risk_level', 'UNKNOWN')
                            
                            class_col1, class_col2 = st.columns(2)
                            with class_col1:
                                st.metric("Stock Classification", classification)
                            with class_col2:
                                st.metric("Risk Level", risk_level)
                        
                        # ATR-Based Stop Loss & Targets
                        if 'atr_stop_loss' in penny_stock_analysis and penny_stock_analysis['atr_stop_loss']:
                            st.subheader("🎯 ATR-Based Risk Management")
                            
                            stop_loss = penny_stock_analysis.get('atr_stop_loss')
                            target = penny_stock_analysis.get('atr_target')
                            stop_pct = penny_stock_analysis.get('atr_stop_pct', 0)
                            target_pct = penny_stock_analysis.get('atr_target_pct', 0)
                            rr_ratio = penny_stock_analysis.get('atr_risk_reward', 0)
                            
                            stop_col1, stop_col2, stop_col3 = st.columns(3)
                            with stop_col1:
                                st.metric("Stop Loss", f"${stop_loss:.4f}", f"{stop_pct:.1f}%")
                                if stop_pct > 12:
                                    st.error("⚠️ STOP EXCEEDS 12% MAX - Consider skipping or reducing position")
                                elif stop_pct > 8:
                                    st.warning("⚠️ Wide stop - Use smaller position size")
                                else:
                                    st.success("✅ Acceptable stop width")
                            with stop_col2:
                                st.metric("Target", f"${target:.4f}", f"{target_pct:.1f}%")
                            with stop_col3:
                                st.metric("Risk/Reward", f"{rr_ratio:.1f}:1")
                                if rr_ratio >= 2.0:
                                    st.success("✅ Good R/R ratio")
                                else:
                                    st.warning("⚠️ R/R below 2:1")
                            
                            # Stop recommendation
                            if 'stop_recommendation' in penny_stock_analysis:
                                st.info(penny_stock_analysis['stop_recommendation'])
                        
                        # Stock Liquidity Check
                        if 'liquidity_check' in penny_stock_analysis:
                            liquidity = penny_stock_analysis['liquidity_check']
                            st.subheader("💧 Stock Liquidity Analysis")
                            
                            liq_col1, liq_col2, liq_col3 = st.columns(3)
                            with liq_col1:
                                overall_risk = liquidity.get('overall_risk', 'UNKNOWN')
                                risk_color = {
                                    'CRITICAL': '🔴',
                                    'HIGH': '🟠',
                                    'MEDIUM': '🟡',
                                    'LOW': '🟢'
                                }.get(overall_risk, '⚪')
                                st.metric("Overall Risk", f"{risk_color} {overall_risk}")
                            with liq_col2:
                                max_pos_pct = liquidity.get('max_position_pct_of_volume', 0)
                                st.metric("Max Position", f"{max_pos_pct:.1f}% of daily volume")
                            with liq_col3:
                                avg_vol = liquidity.get('avg_daily_volume', 0)
                                st.metric("Avg Volume", f"{avg_vol:,}")
                            
                            if overall_risk == "CRITICAL":
                                st.error("❌ **CRITICAL LIQUIDITY RISK** - Cannot execute safely. AVOID or use extreme caution.")
                            elif overall_risk == "HIGH":
                                st.warning("⚠️ **HIGH RISK** - Use limit orders only, small position size")
                            
                            if liquidity.get('warnings'):
                                for warning in liquidity['warnings']:
                                    st.warning(warning)
                        
                        # Final Recommendation
                        if 'final_recommendation' in penny_stock_analysis:
                            final_rec = penny_stock_analysis['final_recommendation']
                            st.subheader("📊 Final Recommendation")
                            
                            decision = final_rec.get('decision', 'UNKNOWN')
                            emoji = final_rec.get('emoji', '⚠️')
                            reason = final_rec.get('reason', 'N/A')
                            
                            st.markdown(f"## {emoji} **{decision}**")
                            st.write(f"**Reason:** {reason}")
                            
                            if final_rec.get('blockers'):
                                st.error("**Blockers:**")
                                for blocker in final_rec['blockers']:
                                    st.write(f"  {blocker}")
                            
                            if final_rec.get('warnings'):
                                st.warning("**Warnings:**")
                                for warning in final_rec['warnings']:
                                    st.write(f"  {warning}")
                            
                            if final_rec.get('signals'):
                                st.success("**Positive Signals:**")
                                for signal in final_rec['signals']:
                                    st.write(f"  {signal}")
                        
                        logger.info(f"✅ Enhanced penny stock analysis display completed for {analysis.ticker}")
                    elif is_penny_stock_flag:
                        logger.warning(f"⚠️ Penny stock detected but enhanced analysis not available - using fallback display")
                        # Fallback to basic penny stock assessment if enhanced analysis not available
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
                    
                    # Timeframe-Specific Analysis
                    st.subheader(f"⏰ {trading_style_display} Analysis")
                    
                    # Calculate timeframe-specific metrics
                    if trading_style == "DAY_TRADE":
                        # Day trading focus: quick moves, tight stops
                        timeframe_score = 0
                        reasons = []
                        
                        # ENTROPY CHECK (CRITICAL FOR DAY TRADING)
                        entropy_value = analysis.entropy if analysis.entropy is not None else 50.0
                        if entropy_value < 30:
                            timeframe_score += 30
                            reasons.append(f"✅ LOW ENTROPY ({entropy_value:.0f}) - Clean price action, ideal for day trading")
                        elif entropy_value < 50:
                            timeframe_score += 15
                            reasons.append(f"✅ Moderate entropy ({entropy_value:.0f}) - Structured patterns present")
                        elif entropy_value < 70:
                            timeframe_score -= 10
                            reasons.append(f"⚠️ Moderate-high entropy ({entropy_value:.0f}) - Use wider stops and reduce size 30%")
                        else:
                            timeframe_score -= 25
                            reasons.append(f"❌ HIGH ENTROPY ({entropy_value:.0f}) - CHOPPY MARKET - Avoid day trading or reduce size 50%+")
                        
                        if volume_vs_avg > 100:
                            timeframe_score += 20
                            reasons.append(f"✅ High volume (+{volume_vs_avg:.0f}%) - good for day trading")
                        else:
                            reasons.append(f"⚠️ Volume only +{volume_vs_avg:.0f}% - may lack intraday momentum")
                        
                        if abs(analysis.change_pct) > 2:
                            timeframe_score += 15
                            reasons.append(f"✅ Strong intraday move ({analysis.change_pct:+.1f}%)")
                        else:
                            reasons.append("⚠️ Low intraday volatility - limited profit potential")
                        
                        if 30 < analysis.rsi < 70:
                            timeframe_score += 15
                            reasons.append("✅ RSI in tradeable range (not overbought/oversold)")
                        
                        if not is_penny_stock_flag:
                            timeframe_score += 10
                            reasons.append("✅ Not a penny stock - better liquidity for day trading")
                        else:
                            reasons.append("⚠️ Penny stock - higher risk, use smaller size")
                        
                        if analysis.trend != "NEUTRAL":
                            timeframe_score += 10
                            reasons.append(f"✅ Clear trend ({analysis.trend}) - easier to trade")
                        
                        st.metric("Day Trading Suitability", f"{timeframe_score}/100")
                        
                        for reason in reasons:
                            st.write(reason)
                        
                        # Overall verdict based on entropy-adjusted score
                        if entropy_value >= 70:
                            st.error("🔴 **NOT RECOMMENDED** for day trading - High entropy (choppy market) will cause whipsaws")
                        elif timeframe_score > 70:
                            st.success("🟢 **EXCELLENT** for day trading - strong setup!")
                        elif timeframe_score > 50:
                            st.info("🟡 **GOOD** for day trading - proceed with caution")
                        elif timeframe_score > 30:
                            st.warning("🟡 **MARGINAL** for day trading - not ideal; multiple divergent signals")
                        else:
                            st.error("🔴 **POOR** for day trading - consider swing/position trading instead")
                        
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
                        
                        if not is_penny_stock_flag or (is_penny_stock_flag and volume_vs_avg > 200):
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
                        
                        if not is_penny_stock_flag:
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
                        
                        if is_penny_stock_flag:
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
                        # ENTROPY OVERRIDE: Block day trading/scalping recommendations if entropy is too high
                        entropy_value = analysis.entropy if analysis.entropy is not None else 50.0
                        
                        if trading_style in ["DAY_TRADE", "SCALP"] and entropy_value >= 70:
                            st.error("❌ **DAY TRADING/SCALPING NOT RECOMMENDED**")
                            st.warning(f"""
                            **High Entropy Alert ({entropy_value:.0f}/100):**
                            
                            The market is currently too choppy and unpredictable for day trading or scalping. 
                            High entropy means random price movements that will cause whipsaws and false signals.
                            
                            **Recommendation:** 
                            - ⏸️ Skip this trade for day trading
                            - 🔄 Consider swing trading or options strategies instead
                            - ⏰ Wait for entropy to drop below 50 before day trading
                            - 📊 Current state: {analysis.entropy_state}
                            """)
                        elif trading_style == "DAY_TRADE" and entropy_value >= 50:
                            st.warning(f"""
                            **⚠️ Moderate Entropy Warning ({entropy_value:.0f}/100):**
                            
                            Market noise is elevated. Day trading is risky in these conditions.
                            
                            **If you still trade:**
                            - Reduce position size by 50%
                            - Use wider stops (1.5-2x normal)
                            - Take profits quickly
                            - Expect lower win rate
                            """)
                            st.markdown("---")
                            st.markdown(f"**{trading_style_display} Strategy (Use Caution):**")
                            st.markdown(analysis.recommendation)
                        else:
                            # Add penny stock context to recommendation
                            if is_penny_stock_flag and trading_style in ["BUY_HOLD", "SWING_TRADE"]:
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
                    
                    if is_penny_stock_flag:
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
                    # Ensure alpha_factors is defined
                    if 'alpha_factors' not in locals():
                        alpha_factors = None
                    
                    # Get current timestamp safely
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    st.caption(f"""**Analysis completed at:** {current_time} | 
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
                                        width="stretch", 
                                        type="primary",
                                        on_click=execute_trade_callback
                                    )
                    else:
                        st.warning("⚠️ No trade recommendations - Verdict score too low. Consider waiting for a better setup.")
                    
                    # Other quick actions
                    st.divider()
                    action_col1, action_col2 = st.columns(2)
                    
                    with action_col1:
                        if st.button("🎯 Get More Strategy Ideas", width="stretch"):
                            st.session_state.goto_strategy_advisor = True
                            st.rerun()
                    
                    with action_col2:
                        if st.button("📊 View in Strategy Analyzer", width="stretch"):
                            st.session_state.analyzer_ticker = analysis.ticker
                            st.rerun()
                    
                else:
                    st.error(f"❌ Could not analyze {search_ticker}. Please check the ticker symbol.")
        
        elif st.session_state.current_analysis:
            st.info("💡 Previous analysis is displayed. Enter a new ticker and click Analyze to update.")
    
    with tab2:
        st.header("🚀 Advanced Opportunity Scanner")
        st.write("**All-in-one scanner** with AI/ML analysis, powerful filters, reverse split detection, and merger candidate identification!")
        
        # Get cached scanners (only initialized once, reused on reruns)
        scanner = get_advanced_scanner()
        ai_scanner = get_ai_scanner()
        ml_scanner = get_ml_scanner()
        
        # Analysis mode selector
        analysis_mode = st.radio(
            "🔬 Analysis Mode:",
            options=["⚡ Quick Scan (Fast)", "🧠 AI+ML Enhanced (Comprehensive)"],
            horizontal=True,
            help="Quick Scan uses technical analysis only. AI+ML Enhanced adds AI confidence ratings and ML predictions for maximum accuracy."
        )
        
        use_ai_ml = analysis_mode == "🧠 AI+ML Enhanced (Comprehensive)"
        
        if use_ai_ml:
            with st.expander("ℹ️ What does AI+ML Enhanced include?", expanded=False):
                st.markdown("""
                **AI+ML Enhanced Mode** combines three powerful analysis systems:
                - **🤖 AI Confidence Analysis**: LLM-powered reasoning and risk assessment
                - **🧠 ML Predictions**: 158 alpha factors from Qlib (if installed)
                - **📊 Technical Analysis**: All standard indicators plus reverse splits and merger detection
                
                This provides the **highest confidence** trading signals by requiring agreement across multiple systems.
                """)
        
        st.divider()
        
        # Scan configuration
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Scan Type")
            # Use session state to prevent unnecessary reruns on dropdown change
            if 'scan_type_display' not in st.session_state:
                st.session_state.scan_type_display = "🎯 All Opportunities"
            
            scan_type_display = st.selectbox(
                "What to scan for:",
                options=[
                    "🎯 All Opportunities",
                    "📈 Options Plays", 
                    "💰 Penny Stocks (<$5)",
                    "💥 Breakouts",
                    "🚀 Momentum Plays",
                    "🔥 Buzzing Stocks",
                    "🌶️ Hottest Stocks"
                ],
                key="scan_type_display",
                help="Select the type of opportunities to find"
            )
            
            scan_type_map = {
                "🎯 All Opportunities": ScanType.ALL,
                "📈 Options Plays": ScanType.OPTIONS,
                "💰 Penny Stocks (<$5)": ScanType.PENNY_STOCKS,
                "💥 Breakouts": ScanType.BREAKOUTS,
                "🚀 Momentum Plays": ScanType.MOMENTUM,
                "🔥 Buzzing Stocks": ScanType.BUZZING,
                "🌶️ Hottest Stocks": ScanType.HOTTEST_STOCKS
            }
            scan_type = scan_type_map[scan_type_display]
            
            # Trading style selector
            st.subheader("📈 Trading Style")
            # Use session state to prevent unnecessary reruns on dropdown change
            if 'trading_style_display' not in st.session_state:
                st.session_state.trading_style_display = "📈 Swing Trading (days-weeks)"
            
            trading_style_display = st.selectbox(
                "Strategy recommendations for:",
                options=[
                    "📊 Options Trading",
                    "⚡ Scalping (seconds-minutes)",
                    "🎯 Day Trading (intraday)",
                    "📈 Swing Trading (days-weeks)",
                    "💎 Buy & Hold (long-term)"
                ],
                key="trading_style_display",
                help="Choose your preferred trading style for strategy recommendations"
            )
            
            trading_style_map = {
                "📊 Options Trading": "OPTIONS",
                "⚡ Scalping (seconds-minutes)": "SCALP",
                "🎯 Day Trading (intraday)": "DAY_TRADE",
                "📈 Swing Trading (days-weeks)": "SWING_TRADE",
                "💎 Buy & Hold (long-term)": "BUY_HOLD"
            }
            trading_style = trading_style_map[trading_style_display]
            
            num_results = st.slider("Number of results", 5, 50, 20, 5, key="num_results_slider")
            
            # Performance control for buzzing and hottest stocks scans
            max_tickers_to_scan = None
            if scan_type in [ScanType.BUZZING, ScanType.HOTTEST_STOCKS]:
                max_tickers_to_scan = st.slider(
                    "Max tickers to scan (performance)", 
                    min_value=50, 
                    max_value=300, 
                    value=150, 
                    step=25,
                    key="max_tickers_slider",
                    help="Limit the number of tickers to scan for faster results. More tickers = wider net but slower scan."
                )
        
        with col2:
            st.subheader("🎚️ Quick Filters")
            use_extended_universe = st.checkbox("Use Extended Universe (200+ tickers)", value=True, 
                                               key="use_extended_universe_cb",
                                               help="Includes obscure plays and emerging stocks")
            
            # Initialize quick_filter variable
            quick_filter = "None - Show All"  # Default value
            
            # Strategy-Based Hybrid Approach
            use_hybrid_approach = st.checkbox("🧬 Use Strategy-Based Hybrid Approach", value=False,
                                            key="use_hybrid_approach_cb",
                                            help="Use proven strategy combinations for balanced risk and opportunity")
            
            if use_hybrid_approach:
                st.info("💡 **Strategy-Based Mode**: Uses proven filter combinations with AI analysis and personalized strategy recommendations")
                
                # Strategy selection
                st.markdown("### 🎯 Choose Your Strategy")
                strategy_choice = st.radio(
                    "Select a proven strategy combination:",
                    options=[
                        "🎯 Quality Momentum (Recommended)",
                        "📈 Aggressive Growth", 
                        "⚡ Conservative Income",
                        "🔥 High-Volatility Plays",
                        "🔧 Custom Combination"
                    ],
                    key="strategy_choice"
                )
                
                # Strategy-specific configurations
                if strategy_choice == "🎯 Quality Momentum (Recommended)":
                    st.success("**Best for balanced risk and opportunity**")
                    st.markdown("""
                    **Combines:**
                    - High Confidence Only (≥70) - Foundation filter
                    - Volume Surge (>2x avg) - Confirmation of interest  
                    - Power Zone Stocks OR EMA Reclaim Setups - Technical validation
                    
                    **Why it works:** Risk management through confidence scores + growth potential through momentum + clear entry/exit points
                    """)
                    
                    primary_approach = "High Confidence Only (Score ≥70)"
                    secondary_approach = ["Volume Surge (>2x avg)", "Power Zone Stocks Only"]
                    technical_choice = st.radio("Technical Confirmation:", ["Power Zone Stocks Only", "EMA Reclaim Setups"], key="quality_tech")
                    if technical_choice == "EMA Reclaim Setups":
                        secondary_approach = ["Volume Surge (>2x avg)", "EMA Reclaim Setups"]
                
                elif strategy_choice == "📈 Aggressive Growth":
                    st.warning("**Higher risk, higher reward - for experienced traders**")
                    st.markdown("""
                    **Combines:**
                    - Penny Stocks ($1-$5) - Price range for upside potential
                    - Volume Surge (>2x avg) - Liquidity filter
                    - High Confidence (≥70) - Quality control
                    
                    **Why it works:** Penny stock upside + volume ensures you can enter/exit + confidence filter reduces risk
                    """)
                    
                    primary_approach = "Penny Stocks ($1-$5)"
                    secondary_approach = ["Volume Surge (>2x avg)", "High Confidence Only (Score ≥70)"]
                
                elif strategy_choice == "⚡ Conservative Income":
                    st.info("**Lower risk, steady returns**")
                    st.markdown("""
                    **Combines:**
                    - High Confidence Only (≥70) - Quality foundation
                    - EMA Reclaim Setups - High-probability entries
                    - RSI Oversold (<30) - Mean reversion opportunities
                    
                    **Why it works:** Conservative screening + technical confirmation + oversold bounce potential
                    """)
                    
                    primary_approach = "High Confidence Only (Score ≥70)"
                    secondary_approach = ["EMA Reclaim Setups", "RSI Oversold (<30)"]
                
                elif strategy_choice == "🔥 High-Volatility Plays":
                    st.error("**Highest risk, highest reward - for aggressive traders only**")
                    st.markdown("""
                    **Combines:**
                    - Ultra-Low Price (<$1) - Maximum upside potential
                    - Volume Surge (>2x avg) - Confirmation of interest
                    - Strong Momentum (>5% change) - Already moving stocks
                    
                    **Why it works:** Maximum upside + volume confirmation + momentum continuation
                    """)
                    
                    primary_approach = "Ultra-Low Price (<$1)"
                    secondary_approach = ["Volume Surge (>2x avg)", "Strong Momentum (>5% change)"]
                
                else:  # Custom Combination
                    st.markdown("**Build your own strategy combination**")
                    
                    col_custom1, col_custom2 = st.columns(2)
                    
                    with col_custom1:
                        st.markdown("**Primary Filter:**")
                        primary_approach = st.selectbox(
                            "Main Filter:",
                            options=[
                                "High Confidence Only (Score ≥70)",
                                "Ultra-Low Price (<$1)",
                                "Penny Stocks ($1-$5)",
                                "Volume Surge (>2x avg)",
                                "Strong Momentum (>5% change)",
                                "Power Zone Stocks Only",
                                "EMA Reclaim Setups"
                            ],
                            key="custom_primary"
                        )
                    
                    with col_custom2:
                        st.markdown("**Additional Filters:**")
                        secondary_approach = st.multiselect(
                            "Secondary Filters:",
                            options=[
                                "High Confidence Only (Score ≥70)",
                                "Volume Surge (>2x avg)",
                                "Strong Momentum (>5% change)",
                                "Power Zone Stocks Only",
                                "EMA Reclaim Setups",
                                "RSI Oversold (<30)",
                                "RSI Overbought (>70)",
                                "High IV Rank (>60)",
                                "Low IV Rank (<40)"
                            ],
                            default=["Volume Surge (>2x avg)"],
                            key="custom_secondary"
                        )
                
                # Strategy recommendation integration
                st.divider()
                include_strategy_recs = st.checkbox("🎯 Include Strategy Recommendations", value=True,
                                                  help="Get specific trading strategy recommendations for each found opportunity")
                
                if include_strategy_recs:
                    st.markdown("**Your Trading Profile:**")
                    strategy_col1, strategy_col2 = st.columns(2)
                    
                    with strategy_col1:
                        user_experience_hybrid = st.selectbox(
                            "Experience Level",
                            options=["Beginner", "Intermediate", "Advanced"],
                            key='hybrid_experience'
                        )
                        
                        risk_tolerance_hybrid = st.selectbox(
                            "Risk Tolerance",
                            options=["Conservative", "Moderate", "Aggressive"],
                            key='hybrid_risk'
                        )
                    
                    with strategy_col2:
                        capital_available_hybrid = st.number_input(
                            "Available Capital ($)",
                            min_value=500,
                            max_value=1000000,
                            value=5000,
                            step=500,
                            key='hybrid_capital'
                        )
                        
                        outlook_hybrid = st.selectbox(
                            "Market Outlook",
                            options=["Bullish", "Bearish", "Neutral"],
                            key='hybrid_outlook'
                        )
            else:
                # Original single filter approach
                quick_filter = st.selectbox(
                    "Filter Preset:",
                    options=[
                        "None - Show All",
                        "High Confidence Only (Score ≥70)",
                        "Ultra-Low Price (<$1)",
                        "Penny Stocks ($1-$5)",
                        "Volume Surge (>2x avg)",
                        "Strong Momentum (>5% change)",
                        "Power Zone Stocks Only",
                        "EMA Reclaim Setups"
                    ]
                )
        
        # Advanced Filters (Expandable)
        with st.expander("🔧 Advanced Filters", expanded=False):
            fcol1, fcol2, fcol3 = st.columns(3)
            
            with fcol1:
                st.markdown("**Price Filters**")
                min_price = st.number_input("Min Price ($)", min_value=0.0, value=None, step=0.1, key="adv_min_price")
                max_price = st.number_input("Max Price ($)", min_value=0.0, value=None, step=1.0, key="adv_max_price")
                
                st.markdown("**Volume Filters**")
                min_volume = st.number_input("Min Volume", min_value=0, value=None, step=100000, key="adv_min_vol")
                min_volume_ratio = st.number_input("Min Volume Ratio (x avg)", min_value=0.0, value=None, step=0.5, key="adv_vol_ratio")
            
            with fcol2:
                st.markdown("**Momentum Filters**")
                min_change = st.number_input("Min Change %", value=None, step=1.0, key="adv_min_change")
                max_change = st.number_input("Max Change %", value=None, step=1.0, key="adv_max_change")
                
                st.markdown("**Score Filters**")
                min_score = st.slider("Min Score", 0, 100, 50, 5, key="adv_min_score")
                min_confidence = st.number_input("Min Confidence Score", min_value=0, max_value=100, value=None, step=5, key="adv_min_conf")
            
            with fcol3:
                st.markdown("**Technical Filters**")
                require_power_zone = st.checkbox("Require Power Zone (8>21 EMA)", key="adv_power")
                require_reclaim = st.checkbox("Require EMA Reclaim", key="adv_reclaim")
                require_alignment = st.checkbox("Require Timeframe Alignment", key="adv_align")
                
                st.markdown("**Entropy Filters** 🔬")
                require_low_entropy = st.checkbox("Require Low Entropy (< 50)", key="adv_low_entropy", 
                                                 help="Only show structured markets, ideal for day trading")
                max_entropy = st.number_input("Max Entropy", min_value=0, max_value=100, value=None, step=5, key="adv_max_entropy",
                                             help="Filter out high-noise markets above this threshold")
                
                st.markdown("**RSI Filters**")
                rsi_range = st.slider("RSI Range", 0, 100, (0, 100), key="adv_rsi")
        
        # Build filters object
        filters = ScanFilters(
            min_price=min_price,
            max_price=max_price,
            min_volume=min_volume,
            min_volume_ratio=min_volume_ratio,
            min_change_pct=min_change,
            max_change_pct=max_change,
            min_score=min_score,
            min_confidence_score=min_confidence,
            require_power_zone=require_power_zone,
            require_ema_reclaim=require_reclaim,
            require_timeframe_alignment=require_alignment,
            min_rsi=rsi_range[0] if rsi_range[0] > 0 else None,
            max_rsi=rsi_range[1] if rsi_range[1] < 100 else None,
            require_low_entropy=require_low_entropy,
            max_entropy=max_entropy
        )
        
        # Apply filter presets (hybrid or single)
        min_buzz_score = 30.0  # Default
        
        if use_hybrid_approach:
            # Apply hybrid approach filters
            st.session_state.hybrid_approach_active = True
            # Store values in different session state keys to avoid widget conflicts
            st.session_state.hybrid_primary_value = primary_approach
            st.session_state.hybrid_secondary_value = secondary_approach
            st.session_state.include_strategy_recs_value = include_strategy_recs
            st.session_state.strategy_choice_value = strategy_choice
            
            # Apply primary approach
            _apply_filter_preset(filters, primary_approach)
            
            # Apply secondary approaches
            for secondary in secondary_approach:
                _apply_secondary_filter(filters, secondary)
            
            # Set buzz score for hybrid
            min_buzz_score = 50.0  # Higher threshold for hybrid approach
            
        else:
            # Apply single filter preset (original logic)
            st.session_state.hybrid_approach_active = False
            
            if quick_filter == "None - Show All":
                filters.min_score = 0.0  # Show everything
                min_buzz_score = 10.0  # Very low threshold for buzzing scan
            elif quick_filter == "High Confidence Only (Score ≥70)":
                filters.min_score = 70.0
                min_buzz_score = 60.0
            elif quick_filter == "Ultra-Low Price (<$1)":
                filters.max_price = PENNY_THRESHOLDS.ULTRA_LOW_PRICE
            elif quick_filter == "Penny Stocks ($1-$5)":
                filters.min_price = PENNY_THRESHOLDS.ULTRA_LOW_PRICE
                filters.max_price = PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE
            elif quick_filter == "Volume Surge (>2x avg)":
                filters.min_volume_ratio = 2.0
                min_buzz_score = 40.0  # Higher threshold for volume surge
            elif quick_filter == "Strong Momentum (>5% change)":
                filters.min_change_pct = 5.0
                min_buzz_score = 40.0
            elif quick_filter == "Power Zone Stocks Only":
                filters.require_power_zone = True
            elif quick_filter == "EMA Reclaim Setups":
                filters.require_ema_reclaim = True
        
        # Scan button
        st.divider()
        scan_col1, scan_col2 = st.columns([1, 3])
        with scan_col1:
            scan_button = st.button("🔍 Scan Markets", type="primary", width="stretch", key="advanced_scan_button")
        with scan_col2:
            if scan_type == ScanType.BUZZING:
                st.info(f"💡 **Buzzing scan** detects unusual volume, volatility, price action + **Reddit/news sentiment** (min score: {min_buzz_score:.0f})")
            else:
                if use_hybrid_approach:
                    strategy_name = st.session_state.get('strategy_choice_value', 'Custom Strategy')
                    st.info(f"💡 Scanning for **{scan_type_display}** using **{strategy_name}** strategy")
                else:
                    st.info(f"💡 Scanning for **{scan_type_display}** with {quick_filter}")
        
        # Execute scan
        if scan_button:
            with st.status("🔍 Scanning markets...", expanded=True) as status:
                try:
                    st.write(f"Analyzing {scan_type_display}...")
                    
                    if use_ai_ml:
                        # AI+ML Enhanced Mode
                        st.write("🧠 Running ML analysis with 158 alpha factors...")
                        st.write("🤖 Generating AI confidence ratings...")
                        st.write("⚡ Calculating technical indicators, reverse splits, and merger candidates...")
                        
                        # Use ML scanner for Options or Penny Stocks scans
                        if scan_type in [ScanType.OPTIONS, ScanType.PENNY_STOCKS]:
                            if scan_type == ScanType.OPTIONS:
                                opportunities = st.session_state.ml_scanner.scan_top_options_with_ml(
                                    top_n=num_results,
                                    min_ensemble_score=filters.min_score if filters.min_score else 60.0
                                )
                            else:  # Penny stocks
                                opportunities = st.session_state.ml_scanner.scan_top_penny_stocks_with_ml(
                                    top_n=num_results,
                                    min_ensemble_score=filters.min_score if filters.min_score else 50.0
                                )
                            
                            # Convert to OpportunityResult format if needed
                            # ML scanner returns different format, wrap in simple display
                            st.session_state.adv_scan_ai_results = opportunities
                            st.session_state.adv_scan_mode = "AI+ML"
                        else:
                            # For other scan types, use standard scanner with AI enabled
                            if scan_type == ScanType.BUZZING:
                                opportunities = scanner.scan_buzzing_stocks(
                                    top_n=num_results,
                                    trading_style=trading_style,
                                    min_buzz_score=min_buzz_score,
                                    max_tickers_to_scan=max_tickers_to_scan
                                )
                            else:
                                opportunities = scanner.scan_opportunities(
                                    scan_type=scan_type,
                                    top_n=num_results,
                                    trading_style=trading_style,
                                    filters=filters,
                                    use_extended_universe=use_extended_universe
                                )
                            st.session_state.adv_scan_results = opportunities
                            st.session_state.adv_scan_mode = "Standard"
                    else:
                        # Quick Scan Mode
                        if scan_type == ScanType.BUZZING:
                            opportunities = scanner.scan_buzzing_stocks(
                                top_n=num_results,
                                trading_style=trading_style,
                                min_buzz_score=min_buzz_score,
                                max_tickers_to_scan=max_tickers_to_scan
                            )
                        else:
                            opportunities = scanner.scan_opportunities(
                                scan_type=scan_type,
                                top_n=num_results,
                                trading_style=trading_style,
                                filters=filters,
                                use_extended_universe=use_extended_universe
                            )
                        st.session_state.adv_scan_results = opportunities
                        st.session_state.adv_scan_mode = "Standard"
                    
                    # Store scan type
                    st.session_state.adv_scan_type = scan_type_display
                    
                    # Get count
                    result_count = len(opportunities) if hasattr(opportunities, '__len__') else len(st.session_state.get('adv_scan_results', []))
                    
                    status.update(label=f"✅ Found {result_count} opportunities!", state="complete")
                    if use_ai_ml:
                        st.success(f"✅ AI+ML Scan complete! Found {result_count} quality {scan_type_display}")
                    else:
                        st.success(f"✅ Scan complete! Found {result_count} {scan_type_display}")
                    
                except Exception as e:
                    status.update(label="❌ Scan failed", state="error")
                    st.error(f"Error during scan: {str(e)}")
                    logger.error(f"Advanced scan error: {e}", exc_info=True)
        
        # Display AI+ML results (if available)
        if 'adv_scan_ai_results' in st.session_state and st.session_state.adv_scan_ai_results:
            ai_results = st.session_state.adv_scan_ai_results
            
            st.divider()
            st.subheader(f"🧠 AI+ML Enhanced Results: {st.session_state.adv_scan_type}")
            
            st.info(f"📡 **Real-time AI+ML analysis** - Found {len(ai_results)} quality opportunities with high confidence")
            
            for i, trade in enumerate(ai_results, 1):
                # Determine emoji based on rating
                if trade.ai_rating >= 8.0:
                    emoji = "🟢"
                elif trade.ai_rating >= 6.5:
                    emoji = "🟡"
                else:
                    emoji = "🟠"
                
                # Build header
                if hasattr(trade, 'combined_score'):
                    header = f"{emoji} #{i} **{trade.ticker}** - Ensemble: {trade.combined_score:.1f}/100 | ML: {trade.ml_prediction_score:.1f} | AI: {trade.ai_rating:.1f}/10"
                else:
                    header = f"{emoji} #{i} **{trade.ticker}** - AI: {trade.ai_rating:.1f}/10 | Score: {trade.score:.1f}/100"
                
                with st.expander(header, expanded=(i==1)):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("💵 Price", f"${trade.price:.2f}", f"{trade.change_pct:+.1f}%")
                        st.metric("📊 Volume", f"{trade.volume_ratio:.1f}x", "Above Avg" if trade.volume_ratio > 1.5 else "Normal")
                    
                    with col2:
                        st.metric("🎯 AI Rating", f"{trade.ai_rating:.1f}/10", trade.ai_confidence)
                        st.metric("⚠️ Risk", trade.risk_level)
                    
                    with col3:
                        if hasattr(trade, 'ml_prediction_score'):
                            st.metric("🧠 ML Score", f"{trade.ml_prediction_score:.1f}/100")
                        # Add to My Tickers button
                        if st.button(f"⭐ Add to My Tickers", key=f"add_ai_{trade.ticker}_{i}"):
                            if 'ticker_manager' not in st.session_state:
                                st.session_state.ticker_manager = TickerManager()
                            st.session_state.ticker_manager.add_ticker(trade.ticker, "AI+ML Scanner")
                            st.success(f"✅ Added {trade.ticker} to My Tickers!")
                    
                    st.divider()
                    
                    if trade.ai_reasoning:
                        st.markdown("**🤖 AI Analysis**")
                        st.info(trade.ai_reasoning)
                    
                    if trade.ai_risks:
                        st.markdown("**⚠️ Risk Assessment**")
                        st.warning(trade.ai_risks)
        
        # Display standard results
        elif 'adv_scan_results' in st.session_state and st.session_state.adv_scan_results:
            opportunities = st.session_state.adv_scan_results
            scan_summary = scanner.get_scan_summary(opportunities)
            
            st.divider()
            
            # Hybrid approach summary
            if st.session_state.get('hybrid_approach_active', False):
                strategy_name = st.session_state.get('strategy_choice_value', 'Custom Strategy')
                st.subheader(f"🧬 {strategy_name} Results: {st.session_state.adv_scan_type}")
                st.info(f"**Strategy:** {strategy_name} | **Primary:** {st.session_state.get('hybrid_primary_value', 'N/A')} | **Secondary:** {', '.join(st.session_state.get('hybrid_secondary_value', []))}")
            else:
                st.subheader(f"📊 Results: {st.session_state.adv_scan_type}")
            
            # Summary metrics
            mcol1, mcol2, mcol3, mcol4, mcol5, mcol6, mcol7 = st.columns(7)
            with mcol1:
                st.metric("Total", scan_summary['total'])
            with mcol2:
                st.metric("Avg Score", f"{scan_summary['avg_score']:.1f}")
            with mcol3:
                st.metric("High Confidence", scan_summary['high_confidence'])
            with mcol4:
                st.metric("Breakouts", scan_summary['breakouts'])
            with mcol5:
                st.metric("Buzzing", scan_summary['buzzing'])
            with mcol6:
                reverse_split_count = scan_summary.get('reverse_split_stocks', 0)
                st.metric("⚠️ Rev Splits", reverse_split_count)
            with mcol7:
                merger_count = scan_summary.get('merger_candidates', 0)
                st.metric("🔄 Mergers", merger_count)
            
            # Results table
            st.markdown("### 📋 Top Opportunities")
            
            for i, opp in enumerate(opportunities, 1):
                with st.expander(f"#{i} {opp.ticker} - Score: {opp.score:.1f} | ${opp.price:.2f} ({opp.change_pct:+.1f}%)", expanded=(i <= 3)):
                    rcol1, rcol2, rcol3 = st.columns([2, 2, 1])
                    
                    with rcol1:
                        st.markdown(f"**{opp.ticker}**")
                        st.write(f"💰 **Price:** ${opp.price:.2f} ({opp.change_pct:+.1f}%)")
                        st.write(f"📊 **Volume:** {opp.volume:,} ({opp.volume_ratio:.1f}x avg)")
                        if opp.market_cap:
                            st.write(f"💼 **Market Cap:** ${opp.market_cap:.1f}M")
                        if opp.sector:
                            st.write(f"🏢 **Sector:** {opp.sector}")
                    
                    with rcol2:
                        st.write(f"🎯 **Score:** {opp.score:.1f}/100")
                        st.write(f"✅ **Confidence:** {opp.confidence}")
                        st.write(f"⚠️ **Risk:** {opp.risk_level}")
                        if opp.entropy is not None:
                            entropy_emoji = "✅" if opp.entropy < 50 else "⚠️" if opp.entropy < 70 else "❌"
                            st.write(f"🔬 **Entropy:** {entropy_emoji} {opp.entropy:.0f}/100")
                            if opp.entropy_state:
                                st.caption(f"State: {opp.entropy_state}")
                        if opp.trend:
                            st.write(f"📈 **Trend:** {opp.trend}")
                        if opp.rsi:
                            st.write(f"📉 **RSI:** {opp.rsi:.1f}")
                    
                    with rcol3:
                        if opp.is_breakout:
                            st.success("💥 BREAKOUT")
                        if opp.is_buzzing:
                            st.warning(f"🔥 BUZZING\n{opp.buzz_score:.0f}")
                        if opp.is_merger_candidate:
                            st.info(f"🔄 MERGER\n{opp.merger_score:.0f}")
                    
                    st.markdown(f"**Reason:** {opp.reason}")
                    
                    # Reverse split warning (prominent for penny stocks)
                    if opp.reverse_split_warning:
                        st.error(f"⚠️ **{opp.reverse_split_warning}**")
                        if opp.reverse_splits:
                            split_history = ", ".join([f"{s['ratio_str']} on {s['date']}" for s in opp.reverse_splits[:3]])
                            st.caption(f"Split History: {split_history}")
                    
                    if opp.breakout_signals:
                        st.info(f"🎯 **Breakout Signals:** {', '.join(opp.breakout_signals)}")
                    
                    if opp.buzz_reasons:
                        st.warning(f"🔥 **Buzz Reasons:** {', '.join(opp.buzz_reasons)}")
                    
                    if opp.is_merger_candidate and opp.merger_signals:
                        st.info(f"🔄 **Merger Signals:** {', '.join(opp.merger_signals)}")
                    
                    # Hybrid approach: Add strategy recommendations
                    if st.session_state.get('hybrid_approach_active', False) and st.session_state.get('include_strategy_recs_value', False):
                        st.markdown("---")
                        st.markdown("**🎯 Strategy Recommendations**")
                        
                        # Get analysis for strategy recommendations
                        try:
                            analysis = ComprehensiveAnalyzer.analyze_stock(opp.ticker, "SWING_TRADE")
                            if analysis:
                                recommendations = StrategyAdvisor.get_recommendations(
                                    analysis=analysis,
                                    user_experience=st.session_state.get('hybrid_experience', 'Intermediate'),
                                    risk_tolerance=st.session_state.get('hybrid_risk', 'Moderate'),
                                    capital_available=st.session_state.get('hybrid_capital', 5000),
                                    outlook=st.session_state.get('hybrid_outlook', 'Neutral')
                                )
                                
                                if recommendations:
                                    # Show top 2 recommendations
                                    for j, rec in enumerate(recommendations[:2], 1):
                                        confidence_pct = int(rec.confidence * 100)
                                        st.markdown(f"**{j}. {rec.strategy_name}** ({confidence_pct}% match)")
                                        st.caption(f"Risk: {rec.risk_level} | Best for: {', '.join(rec.best_conditions[:2])}")
                                        
                                        if st.button(f"Use Strategy", key=f"use_strategy_{opp.ticker}_{j}"):
                                            st.session_state.selected_strategy = rec.action
                                            st.session_state.selected_ticker = opp.ticker
                                            st.success(f"✅ Selected {rec.strategy_name} for {opp.ticker}")
                                else:
                                    st.info("No specific strategies recommended for this stock")
                        except Exception as e:
                            st.warning(f"Could not generate strategy recommendations: {str(e)}")
                    
                    # Action buttons
                    acol1, acol2 = st.columns(2)
                    with acol1:
                        if st.button(f"📊 Full Analysis", key=f"analyze_{opp.ticker}_{i}"):
                            st.info(f"Switch to '🔍 Stock Intelligence' tab and analyze {opp.ticker}")
                    with acol2:
                        if st.button(f"⭐ Add to My Tickers", key=f"add_{opp.ticker}_{i}"):
                            if 'ticker_manager' not in st.session_state:
                                st.session_state.ticker_manager = TickerManager()
                            st.session_state.ticker_manager.add_ticker(opp.ticker, "Advanced Scanner")
                            st.success(f"✅ Added {opp.ticker} to My Tickers!")
            
            # Export option
            st.divider()
            export_col1, export_col2 = st.columns([1, 3])
            with export_col1:
                if st.button("📥 Export to CSV"):
                    df = pd.DataFrame([{
                        'Ticker': o.ticker,
                        'Score': o.score,
                        'Price': o.price,
                        'Change %': o.change_pct,
                        'Volume': o.volume,
                        'Volume Ratio': o.volume_ratio,
                        'Confidence': o.confidence,
                        'Risk': o.risk_level,
                        'Entropy': o.entropy if o.entropy is not None else '',
                        'Entropy State': o.entropy_state if o.entropy_state else '',
                        'Trend': o.trend,
                        'RSI': o.rsi,
                        'Breakout': o.is_breakout,
                        'Buzzing': o.is_buzzing,
                        'Reverse Split Warning': o.reverse_split_warning if o.reverse_split_warning else '',
                        'Reverse Splits Count': len(o.reverse_splits),
                        'Merger Candidate': o.is_merger_candidate,
                        'Merger Score': o.merger_score,
                        'Reason': o.reason
                    } for o in opportunities])
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download CSV",
                        csv,
                        f"advanced_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
            with export_col2:
                st.caption(f"💡 Export {len(opportunities)} opportunities to CSV for further analysis")
        
        else:
            st.info("👆 Configure your scan settings and click 'Scan Markets' to find opportunities")
    
    with tab3:
        st.header("⭐ My Tickers")
        st.write("Manage your saved tickers and watchlists.")
        
        # Use cached ticker manager instance
        tm = get_ticker_manager()
        
        # Debug: Show connection status
        if tm.supabase:
            if tm.test_connection():
                st.success("✅ Supabase connected and table accessible")
            else:
                st.error("❌ Supabase connected but table not accessible - check table exists")
                st.stop()
        else:
            st.error("❌ Supabase not connected - check your secrets")
            
            # Show debug info about secrets
            try:
                if 'supabase' in st.secrets:
                    st.info(f"✅ Found supabase section in secrets")
                    st.info(f"URL: {st.secrets['supabase'].get('url', 'Not set')}")
                    st.info(f"Service key: {'Set' if st.secrets['supabase'].get('service_key') else 'Not set'}")
                else:
                    st.error("❌ No [supabase] section found in Streamlit secrets")
            except Exception as e:
                st.error(f"❌ Error reading secrets: {e}")
            st.stop()
        
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
                    st.error("❌ Failed to add ticker. Check the logs for details.")
                    # Show debug info
                    if tm.supabase:
                        st.info("✅ Supabase client is connected")
                    else:
                        st.error("❌ Supabase client is not connected - check your secrets")
            else:
                st.warning("⚠️ Ticker symbol is required.")
        
        st.divider()
        
        # Trading Style Selector
        st.subheader("⚙️ Analysis Settings")
        col_style, col_refresh_all = st.columns([2, 1])
        
        with col_style:
            analysis_style = st.selectbox(
                "Trading Style for Analysis",
                ["AI", "OPTIONS", "DAY_TRADE", "SWING_TRADE", "SCALP", "WARRIOR_SCALPING", "BUY_AND_HOLD"],
                index=0,
                help="Select the trading style to analyze your tickers with"
            )
            st.session_state.analysis_timeframe = analysis_style
        
        with col_refresh_all:
            if st.button("🔄 Refresh All", help="Refresh analysis for all tickers"):
                st.session_state.refresh_all_tickers = True
                st.rerun()
        
        # View saved tickers with pagination
        st.subheader("📋 Your Saved Tickers")
        
        # Pagination settings
        import math
        items_per_page = 10
        
        # Initialize pagination state
        if 'ticker_page' not in st.session_state:
            st.session_state.ticker_page = 1
        
        # Get total count for pagination (get all to count, but only display page)
        all_tickers_full = tm.get_all_tickers(limit=100)
        total_tickers = len(all_tickers_full)
        total_pages = max(1, math.ceil(total_tickers / items_per_page))
        
        # Ensure current page is within bounds
        if st.session_state.ticker_page > total_pages:
            st.session_state.ticker_page = total_pages
        
        # Display pagination controls at top
        if total_pages > 1:
            col_p1, col_p2, col_p3, col_p4 = st.columns([1, 2, 2, 1])
            with col_p1:
                if st.button("◀ Previous", disabled=st.session_state.ticker_page == 1, key="prev_top"):
                    st.session_state.ticker_page -= 1
                    st.rerun()
            with col_p2:
                st.write(f"**Page {st.session_state.ticker_page} of {total_pages}**")
            with col_p3:
                st.write(f"**Showing {min(items_per_page, total_tickers)} of {total_tickers} tickers**")
            with col_p4:
                if st.button("Next ▶", disabled=st.session_state.ticker_page == total_pages, key="next_top"):
                    st.session_state.ticker_page += 1
                    st.rerun()
        
        # Get only the tickers for current page
        start_idx = (st.session_state.ticker_page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        all_tickers = all_tickers_full[start_idx:end_idx]

        if all_tickers:
            for ticker in all_tickers:
                # Create enhanced ticker card with better visual design
                ticker_symbol = ticker['ticker']
                ticker_name = ticker.get('name', 'Unknown Company')
                ticker_type = ticker.get('type', 'stock')
                ml_score = ticker.get('ml_score')
                notes = ticker.get('notes', '')
                
                # Determine card header styling based on ML score
                if ml_score is not None:
                    if ml_score >= 70:
                        score_emoji = "🟢"
                        score_color = "green"
                        confidence_label = "HIGH"
                    elif ml_score >= 50:
                        score_emoji = "🟡"
                        score_color = "orange"
                        confidence_label = "MEDIUM"
                    else:
                        score_emoji = "🔴"
                        score_color = "red"
                        confidence_label = "LOW"
                    expander_title = f"{score_emoji} **{ticker_symbol}** · {ticker_name[:30]}{'...' if len(ticker_name) > 30 else ''} · **{confidence_label}** {ml_score:.0f}/100"
                else:
                    expander_title = f"📊 **{ticker_symbol}** · {ticker_name[:30]}{'...' if len(ticker_name) > 30 else ''}"
                
                # Main card container
                with st.container():
                    # Card header with action buttons
                    col_header, col_actions = st.columns([5, 2])
                    
                    with col_header:
                        with st.expander(expander_title, expanded=False):
                            # Company info section
                            info_col1, info_col2 = st.columns(2)
                            
                            with info_col1:
                                st.markdown("**📊 Company Details**")
                                st.write(f"**Symbol:** `{ticker_symbol}`")
                                st.write(f"**Name:** {ticker_name}")
                                st.write(f"**Type:** {ticker_type.replace('_', ' ').title()}")
                                
                                # Sector and tags if available
                                sector = ticker.get('sector')
                                if sector:
                                    st.write(f"**Sector:** {sector}")
                                
                                tags = ticker.get('tags')
                                if tags and isinstance(tags, list):
                                    st.write(f"**Tags:** {', '.join(tags)}")
                            
                            with info_col2:
                                st.markdown("**📈 Activity & Stats**")
                                
                                # Access count with emoji
                                access_count = ticker.get('access_count', 0)
                                activity_emoji = "🔥" if access_count > 10 else "📊" if access_count > 5 else "📋"
                                st.write(f"**Views:** {activity_emoji} {access_count}")
                                
                                # Date added
                                date_added_str = ticker.get('date_added', 'Unknown')
                                if date_added_str != 'Unknown':
                                    try:
                                        dt_utc = datetime.fromisoformat(date_added_str).replace(tzinfo=timezone.utc)
                                        dt_local = dt_utc.astimezone()
                                        friendly_date = dt_local.strftime('%b %d, %Y')
                                        days_ago = (datetime.now(timezone.utc) - dt_utc).days
                                        if days_ago == 0:
                                            time_label = "Today"
                                        elif days_ago == 1:
                                            time_label = "Yesterday"
                                        elif days_ago < 7:
                                            time_label = f"{days_ago} days ago"
                                        else:
                                            time_label = friendly_date
                                        st.write(f"**Added:** {time_label}")
                                    except (ValueError, TypeError):
                                        st.write(f"**Added:** {date_added_str}")
                                else:
                                    st.write(f"**Added:** Unknown")
                                
                                # Last accessed if available
                                last_accessed = ticker.get('last_accessed')
                                if last_accessed:
                                    try:
                                        dt_accessed = datetime.fromisoformat(last_accessed).replace(tzinfo=timezone.utc)
                                        dt_local_accessed = dt_accessed.astimezone()
                                        accessed_ago = (datetime.now(timezone.utc) - dt_accessed).days
                                        if accessed_ago == 0:
                                            access_label = "Today"
                                        elif accessed_ago == 1:
                                            access_label = "Yesterday"
                                        elif accessed_ago < 7:
                                            access_label = f"{accessed_ago} days ago"
                                        else:
                                            access_label = dt_local_accessed.strftime('%b %d, %Y')
                                        st.write(f"**Last View:** {access_label}")
                                    except:
                                        pass
                            
                            # Notes section if available
                            if notes and notes.strip():
                                st.markdown("**📝 Notes**")
                                with st.container():
                                    st.info(notes)
                            
                            # Comprehensive Analysis section - Enhanced with Dashboard-level data
                            st.divider()
                            st.markdown("**📊 Comprehensive Analysis**")
                            
                            # Check if analysis is stale (older than 1 hour)
                            # Ensure tm is properly initialized and has the method
                            try:
                                if tm and hasattr(tm, 'should_update_analysis'):
                                    needs_update = tm.should_update_analysis(ticker_symbol, max_age_hours=1.0)
                                else:
                                    # Fallback: check last_analyzed directly from ticker data
                                    needs_update = True
                            except (AttributeError, TypeError) as e:
                                logger.error(f"Error checking analysis staleness for {ticker_symbol}: {e}")
                                needs_update = True  # Default to needing update if check fails
                            
                            last_analyzed_str = ticker.get('last_analyzed')
                            
                            # Display last analysis timestamp
                            if last_analyzed_str:
                                try:
                                    last_analyzed_dt = datetime.fromisoformat(last_analyzed_str).replace(tzinfo=timezone.utc)
                                    age_hours = (datetime.now(timezone.utc) - last_analyzed_dt).total_seconds() / 3600
                                    if age_hours < 1:
                                        st.info(f"📊 Analysis cached ({age_hours*60:.0f} minutes ago) - Click 'Refresh' for latest data")
                                    else:
                                        st.warning(f"⚠️ Analysis is {age_hours:.1f} hours old - Click 'Refresh' for latest data")
                                except:
                                    pass
                            else:
                                st.info("📊 No recent analysis - Click 'Analyze' to generate")
                            
                            # Show analysis button instead of auto-analyzing
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                analyze_btn = st.button(
                                    f"🔍 {'Refresh' if not needs_update else 'Analyze'} {ticker_symbol}",
                                    key=f"analyze_btn_{ticker_symbol}",
                                    help="Run comprehensive analysis with latest data",
                                    type="primary" if needs_update else "secondary"
                                )
                            
                            with col_btn2:
                                show_cached = st.button(
                                    "👁️ View Cached Data",
                                    key=f"view_cached_{ticker_symbol}",
                                    help="View last saved analysis data",
                                    disabled=not last_analyzed_str
                                )
                            
                            # Get fresh analysis data only if requested
                            analysis = None
                            try:
                                # Check if user requested fresh analysis
                                should_refresh = (
                                    analyze_btn or
                                    st.session_state.get(f"refresh_{ticker_symbol}", False) or 
                                    st.session_state.get('refresh_all_tickers', False) or
                                    st.session_state.get('ml_ticker_to_analyze') == ticker_symbol
                                )
                                
                                if should_refresh:
                                    with st.spinner(f"🔄 Analyzing {ticker_symbol} with latest data..."):
                                        analysis = ComprehensiveAnalyzer.analyze_stock(ticker_symbol, st.session_state.get('analysis_timeframe', 'OPTIONS'))
                                        # Fetch historical data for trading style analysis
                                        hist, _ = get_cached_stock_data(ticker_symbol)
                                    
                                    # Clear refresh flags
                                    if f"refresh_{ticker_symbol}" in st.session_state:
                                        del st.session_state[f"refresh_{ticker_symbol}"]
                                    
                                    st.success(f"✅ Fresh analysis completed for {ticker_symbol}!")
                                elif show_cached or not needs_update:
                                    # Show cached data from database without re-analyzing
                                    # Display metrics from database directly
                                    st.info(f"📊 Displaying cached analysis data for {ticker_symbol}")
                                    
                                    # Show basic metrics from database
                                    db_col1, db_col2, db_col3, db_col4 = st.columns(4)
                                    with db_col1:
                                        st.metric("ML Score", f"{ticker.get('ml_score', 'N/A')}/100" if ticker.get('ml_score') else "N/A")
                                    with db_col2:
                                        momentum = ticker.get('momentum', 0)
                                        st.metric("Momentum", f"{momentum:+.2f}%" if momentum else "N/A")
                                    with db_col3:
                                        rsi = ticker.get('rsi')
                                        st.metric("RSI", f"{rsi:.1f}" if rsi else "N/A")
                                    with db_col4:
                                        sentiment = ticker.get('sentiment_score')
                                        st.metric("Sentiment", f"{sentiment:.2f}" if sentiment is not None else "N/A")
                                    
                                    st.info("💡 Click 'Analyze' above to run fresh comprehensive analysis with latest market data")
                                    
                                    # Don't show full analysis - just the cached summary
                                    analysis = None
                                    
                                if analysis:
                                    # Update ticker with fresh analysis data
                                    tm.update_analysis(ticker_symbol, analysis.__dict__)
                                    
                                    # Detect characteristics
                                    is_penny_stock_check = is_penny_stock(analysis.price)
                                    is_otc = analysis.ticker.endswith(('.OTC', '.PK', '.QB'))
                                    volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
                                    is_runner = volume_vs_avg > 200 and analysis.change_pct > 10
                                    
                                    # Header metrics (same as dashboard)
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
                                    
                                    # Special alerts for penny stocks and runners
                                    if is_runner:
                                        st.warning(f"🚀 **RUNNER DETECTED!** {volume_vs_avg:+.0f}% volume spike with {analysis.change_pct:+.1f}% price move!")
                                    
                                    if is_penny_stock:
                                        st.info(f"💰 **PENNY STOCK** (${analysis.price:.4f}) - High risk/high reward. Use caution and proper position sizing.")
                                    
                                    if is_otc:
                                        st.warning("⚠️ **OTC STOCK** - Lower liquidity, wider spreads, higher risk. Limited data may be available.")
                                    
                                    # Technical Indicators (same as dashboard)
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
                                    
                                    # IV Analysis (same as dashboard)
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
                                    
                                    # Catalysts (same as dashboard)
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
                                    
                                    # News & Sentiment (same as dashboard)
                                    st.subheader("📰 Recent News & Sentiment")
                                    
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
                                            for idx, article in enumerate(analysis.recent_news[:3]):  # Show top 3
                                                # Create a more informative expander
                                                expander_title = f"📰 {article['title'][:50]}..." if len(article['title']) > 50 else f"📰 {article['title']}"
                                                
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
                                        else:
                                            st.info("📭 No recent news found for this ticker.")
                                    
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
                                    
                                    # Timeframe-Specific Analysis
                                    st.subheader("⏰ Trading Style Analysis")
                                    
                                    # Get trading style from session state or default to OPTIONS
                                    trading_style = st.session_state.get('analysis_timeframe', 'OPTIONS')
                                    
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
                                            st.success("🟢 **EXCELLENT** for day trading!")
                                        elif timeframe_score > 50:
                                            st.info("🟡 **GOOD** for day trading with caution")
                                        else:
                                            st.warning("🔴 **POOR** for day trading - consider other timeframes")
                                    
                                    elif trading_style == "AI":
                                        # AI Analysis
                                        st.write("🤖 **AI-Powered Analysis**")
                                        ai_results = TradingStyleAnalyzer.analyze_ai_style(analysis, hist)
                                        
                                        # Display AI score and prediction
                                        ai_col1, ai_col2, ai_col3 = st.columns(3)
                                        with ai_col1:
                                            st.metric("AI Score", f"{ai_results['score']}/100")
                                        with ai_col2:
                                            st.metric("ML Prediction", ai_results.get('ml_prediction', 'N/A'))
                                        with ai_col3:
                                            st.metric("Risk Level", ai_results.get('risk_level', 'UNKNOWN'))
                                        
                                        # Display signals
                                        if ai_results.get('signals'):
                                            st.write("**📊 AI Signals:**")
                                            for signal in ai_results['signals']:
                                                st.write(signal)
                                        
                                        # Display recommendations
                                        if ai_results.get('recommendations'):
                                            st.write("**💡 AI Recommendations:**")
                                            for rec in ai_results['recommendations']:
                                                st.write(rec)

                                    elif trading_style == "SWING_TRADE":
                                        # Swing trading focus: multi-day moves, wider stops
                                        timeframe_score = 0
                                        reasons = []
                                        
                                        if abs(analysis.change_pct) > 5:
                                            timeframe_score += 25
                                            reasons.append(f"✅ Strong momentum ({analysis.change_pct:+.1f}%) - good for swing trades")
                                        
                                        if analysis.trend != "NEUTRAL":
                                            timeframe_score += 30
                                            reasons.append(f"✅ Clear trend ({analysis.trend}) - ideal for swing trading")
                                        
                                        if 40 < analysis.rsi < 60:
                                            timeframe_score += 20
                                            reasons.append("✅ RSI in good swing range")
                                        
                                        if analysis.iv_rank > 30:
                                            timeframe_score += 15
                                            reasons.append("✅ Sufficient volatility for swing moves")
                                        
                                        if len(analysis.recent_news) > 0:
                                            timeframe_score += 10
                                            reasons.append("✅ News catalyst available")
                                        
                                        st.metric("Swing Trading Suitability", f"{timeframe_score}/100")
                                        
                                        for reason in reasons:
                                            st.write(reason)
                                        
                                        if timeframe_score > 70:
                                            st.success("🟢 **EXCELLENT** for swing trading!")
                                        elif timeframe_score > 50:
                                            st.info("🟡 **GOOD** for swing trading")
                                        else:
                                            st.warning("🔴 **POOR** for swing trading - wait for better setup")
                                    
                                    elif trading_style == "SCALP":
                                        # Scalp Analysis
                                        st.write("⚡ **Scalping Analysis**")
                                        scalp_results = TradingStyleAnalyzer.analyze_scalp_style(analysis, hist)
                                        
                                        # Display scalp score and risk
                                        scalp_col1, scalp_col2 = st.columns(2)
                                        with scalp_col1:
                                            st.metric("Scalping Score", f"{scalp_results['score']}/100")
                                        with scalp_col2:
                                            st.metric("Risk Level", scalp_results.get('risk_level', 'UNKNOWN'))
                                        
                                        # Display signals
                                        if scalp_results.get('signals'):
                                            st.write("**📊 Scalping Signals:**")
                                            for signal in scalp_results['signals']:
                                                st.write(signal)
                                        
                                        # Display targets
                                        if scalp_results.get('targets'):
                                            st.write("**🎯 Scalping Targets:**")
                                            for target in scalp_results['targets']:
                                                st.write(target)
                                        
                                        # Display recommendations
                                        if scalp_results.get('recommendations'):
                                            st.write("**💡 Scalping Strategy:**")
                                            for rec in scalp_results['recommendations']:
                                                st.write(rec)

                                    elif trading_style == "WARRIOR_SCALPING":
                                        # Warrior Scalping Analysis
                                        st.write("⚔️ **Warrior Scalping Analysis**")
                                        warrior_results = TradingStyleAnalyzer.analyze_warrior_scalping_style(analysis, hist)
                                        
                                        # Display warrior score and setup type
                                        warrior_col1, warrior_col2, warrior_col3 = st.columns(3)
                                        with warrior_col1:
                                            st.metric("Warrior Score", f"{warrior_results['score']}/100")
                                        with warrior_col2:
                                            st.metric("Setup Type", warrior_results.get('setup_type', 'N/A'))
                                        with warrior_col3:
                                            st.metric("Risk Level", warrior_results.get('risk_level', 'UNKNOWN'))
                                        
                                        # Display signals
                                        if warrior_results.get('signals'):
                                            st.write("**📊 Warrior Signals:**")
                                            for signal in warrior_results['signals']:
                                                st.write(signal)
                                        
                                        # Display targets
                                        if warrior_results.get('targets'):
                                            st.write("**🎯 Warrior Targets:**")
                                            for target in warrior_results['targets']:
                                                st.write(target)
                                        
                                        # Display recommendations
                                        if warrior_results.get('recommendations'):
                                            st.write("**💡 Warrior Strategy:**")
                                            for rec in warrior_results['recommendations']:
                                                st.write(rec)

                                    elif trading_style == "BUY_AND_HOLD":
                                        # Buy & Hold Analysis
                                        st.write("💎 **Buy & Hold Analysis**")
                                        hold_results = TradingStyleAnalyzer.analyze_buy_and_hold_style(analysis, hist)
                                        
                                        # Display hold score and risk
                                        hold_col1, hold_col2 = st.columns(2)
                                        with hold_col1:
                                            st.metric("Investment Score", f"{hold_results['score']}/100")
                                        with hold_col2:
                                            st.metric("Risk Level", hold_results.get('risk_level', 'UNKNOWN'))
                                        
                                        # Display valuation metrics
                                        if hold_results.get('valuation'):
                                            st.write("**📊 Valuation Metrics:**")
                                            val_col1, val_col2, val_col3 = st.columns(3)
                                            valuation = hold_results['valuation']
                                            with val_col1:
                                                if '200_day_ma' in valuation:
                                                    st.metric("200-Day MA", valuation['200_day_ma'])
                                            with val_col2:
                                                if 'pe_ratio' in valuation:
                                                    st.metric("P/E Ratio", valuation['pe_ratio'])
                                            with val_col3:
                                                if 'dividend_yield' in valuation:
                                                    st.metric("Dividend Yield", valuation['dividend_yield'])
                                        
                                        # Display signals
                                        if hold_results.get('signals'):
                                            st.write("**📊 Investment Signals:**")
                                            for signal in hold_results['signals']:
                                                st.write(signal)
                                        
                                        # Display long-term targets
                                        if hold_results.get('targets'):
                                            st.write("**🎯 Long-Term Targets:**")
                                            for target in hold_results['targets']:
                                                st.write(target)
                                        
                                        # Display recommendations
                                        if hold_results.get('recommendations'):
                                            st.write("**💡 Investment Strategy:**")
                                            for rec in hold_results['recommendations']:
                                                st.write(rec)

                                    else:  # OPTIONS trading
                                        # Options trading focus: IV, time decay, volatility
                                        timeframe_score = 0
                                        reasons = []
                                        
                                        if analysis.iv_rank > 60:
                                            timeframe_score += 30
                                            reasons.append(f"✅ High IV Rank ({analysis.iv_rank}%) - great for selling premium")
                                        elif analysis.iv_rank < 40:
                                            timeframe_score += 20
                                            reasons.append(f"✅ Low IV Rank ({analysis.iv_rank}%) - good for buying options")
                                        else:
                                            reasons.append(f"⚠️ Moderate IV Rank ({analysis.iv_rank}%) - mixed signals")
                                        
                                        if analysis.trend != "NEUTRAL":
                                            timeframe_score += 25
                                            reasons.append(f"✅ Clear trend ({analysis.trend}) - easier to pick direction")
                                        
                                        if 30 < analysis.rsi < 70:
                                            timeframe_score += 20
                                            reasons.append("✅ RSI in tradeable range")
                                        
                                        if volume_vs_avg > 50:
                                            timeframe_score += 15
                                            reasons.append(f"✅ Good volume activity (+{volume_vs_avg:.0f}%)")
                                        
                                        if len(analysis.catalysts) > 0:
                                            timeframe_score += 10
                                            reasons.append("✅ Upcoming catalysts for volatility")
                                        
                                        st.metric("Options Trading Suitability", f"{timeframe_score}/100")
                                        
                                        for reason in reasons:
                                            st.write(reason)
                                        
                                        if timeframe_score > 70:
                                            st.success("🟢 **EXCELLENT** for options trading!")
                                        elif timeframe_score > 50:
                                            st.info("🟡 **GOOD** for options trading")
                                        else:
                                            st.warning("🔴 **POOR** for options trading - wait for better setup")
                                        
                                        # Options-specific recommendations
                                        if analysis.iv_rank > 60:
                                            st.info("💡 **High IV Strategy:** Consider selling puts, covered calls, or iron condors")
                                        elif analysis.iv_rank < 40:
                                            st.info("💡 **Low IV Strategy:** Consider buying calls/puts or debit spreads")
                                    
                                    # Last analysis timestamp
                                    st.caption(f"🕒 Analysis updated: Just now")
                                    
                                else:
                                    st.error(f"❌ Could not analyze {ticker_symbol}. Check ticker symbol or try again.")
                                    
                            except Exception as e:
                                st.error(f"❌ Error analyzing {ticker_symbol}: {str(e)}")
                                # Fallback to stored analysis if available
                                if ml_score is not None:
                                    st.info("📊 Showing cached analysis data:")
                                    
                                    # Confidence score with color coding
                                    if ml_score >= 70:
                                        st.success(f"✅ **HIGH CONFIDENCE** - Score: {ml_score:.0f}/100")
                                    elif ml_score >= 50:
                                        st.info(f"📊 **MEDIUM CONFIDENCE** - Score: {ml_score:.0f}/100")
                                    else:
                                        st.warning(f"⚠️ **LOW CONFIDENCE** - Score: {ml_score:.0f}/100")
                                    
                                    # Analysis metrics in grid
                                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                                    
                                    with metric_col1:
                                        trend = ticker.get('trend', 'N/A')
                                        trend_emoji = "📈" if trend == "BULLISH" else "📉" if trend == "BEARISH" else "➡️"
                                        st.metric("Trend", f"{trend_emoji} {trend}")
                                    
                                    with metric_col2:
                                        sentiment = ticker.get('sentiment_score')
                                        if sentiment is not None:
                                            sentiment_emoji = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😟"
                                            st.metric("Sentiment", f"{sentiment_emoji} {sentiment:.2f}")
                                        else:
                                            st.metric("Sentiment", "N/A")
                                    
                                    with metric_col3:
                                        rsi = ticker.get('rsi')
                                        if rsi is not None:
                                            rsi_emoji = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                                            st.metric("RSI", f"{rsi_emoji} {rsi:.1f}")
                                        else:
                                            st.metric("RSI", "N/A")
                                    
                                    with metric_col4:
                                        momentum = ticker.get('momentum')
                                        if momentum is not None:
                                            momentum_emoji = "🚀" if momentum > 5 else "📈" if momentum > 0 else "📉"
                                            st.metric("Momentum", f"{momentum_emoji} {momentum:.1f}%")
                                        else:
                                            st.metric("Momentum", "N/A")
                                    
                                    # Recommendation if available
                                    recommendation = ticker.get('recommendation')
                                    if recommendation and recommendation != 'N/A':
                                        rec_emoji = "💰" if "BUY" in recommendation.upper() else "⏱️" if "HOLD" in recommendation.upper() else "🚨"
                                        st.markdown(f"**💡 Recommendation:** {rec_emoji} {recommendation}")
                                    
                                    # Last analysis timestamp
                                    last_analyzed_str = ticker.get('last_analyzed')
                                    if last_analyzed_str:
                                        try:
                                            dt_analyzed = datetime.fromisoformat(last_analyzed_str).replace(tzinfo=timezone.utc)
                                            analyzed_ago = (datetime.now(timezone.utc) - dt_analyzed).total_seconds() / 3600
                                            if analyzed_ago < 1:
                                                time_str = "Just now"
                                            elif analyzed_ago < 24:
                                                time_str = f"{analyzed_ago:.0f} hours ago"
                                            else:
                                                time_str = f"{analyzed_ago/24:.0f} days ago"
                                            st.caption(f"🕒 Analysis updated: {time_str}")
                                        except:
                                            pass
                    
                    with col_actions:
                        st.write("")  # Add some spacing
                        
                        # Quick analyze button with timeframe selection
                        if st.button("🔍 Analyze", key=f"analyze_{ticker_symbol}", help="Run fresh comprehensive analysis", width="stretch"):
                            st.session_state.ml_ticker_to_analyze = ticker_symbol
                            st.session_state.analysis_timeframe = "OPTIONS"  # Default to options
                            st.rerun()
                        
                        # Refresh analysis button
                        if st.button("🔄 Refresh", key=f"refresh_{ticker_symbol}", help="Refresh analysis data", width="stretch"):
                            st.session_state[f"refresh_{ticker_symbol}"] = True
                            st.rerun()
                        
                        # Quick trade button
                        if st.button("⚡ Trade", key=f"trade_{ticker_symbol}", help="Open quick trade interface", width="stretch"):
                            st.session_state.selected_ticker = ticker_symbol
                            st.session_state.show_quick_trade = True
                            st.info(f"💡 Switch to '🚀 Quick Trade' tab to trade {ticker_symbol}")
                        
                        # Edit notes button
                        if st.button("✏️ Edit", key=f"edit_{ticker_symbol}", help="Edit ticker details", width="stretch"):
                            st.session_state[f"editing_{ticker_symbol}"] = True
                            st.rerun()
                        
                        # Remove button
                        if st.button("🗑️ Remove", key=f"remove_{ticker_symbol}", help="Remove from saved tickers", width="stretch"):
                            if tm.remove_ticker(ticker_symbol):
                                st.success(f"🗑️ Removed {ticker_symbol}!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to remove {ticker_symbol}.")
                    
                    # Edit mode popup
                    if st.session_state.get(f"editing_{ticker_symbol}", False):
                        with st.expander(f"✏️ Edit {ticker_symbol}", expanded=True):
                            edit_col1, edit_col2 = st.columns(2)
                            with edit_col1:
                                new_name = st.text_input("Company Name", value=ticker_name, key=f"edit_name_{ticker_symbol}")
                                new_notes = st.text_area("Notes", value=notes, key=f"edit_notes_{ticker_symbol}")
                            with edit_col2:
                                new_sector = st.text_input("Sector", value=ticker.get('sector', ''), key=f"edit_sector_{ticker_symbol}")
                                new_type = st.selectbox("Type", ["stock", "option", "penny_stock", "crypto"], 
                                                       index=["stock", "option", "penny_stock", "crypto"].index(ticker_type) if ticker_type in ["stock", "option", "penny_stock", "crypto"] else 0,
                                                       key=f"edit_type_{ticker_symbol}")
                            
                            button_col1, button_col2 = st.columns(2)
                            with button_col1:
                                if st.button("� Save Changes", key=f"save_{ticker_symbol}"):
                                    if tm.add_ticker(ticker_symbol, name=new_name, sector=new_sector, ticker_type=new_type, notes=new_notes):
                                        st.success(f"✅ Updated {ticker_symbol}!")
                                        st.session_state[f"editing_{ticker_symbol}"] = False
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to update ticker.")
                            with button_col2:
                                if st.button("❌ Cancel", key=f"cancel_{ticker_symbol}"):
                                    st.session_state[f"editing_{ticker_symbol}"] = False
                                    st.rerun()
                    
                    st.divider()  # Separator between cards
            
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
            
            # Pagination controls at bottom
            if total_pages > 1:
                st.divider()
                col_p1_btm, col_p2_btm, col_p3_btm, col_p4_btm = st.columns([1, 2, 2, 1])
                with col_p1_btm:
                    if st.button("◀ Previous", disabled=st.session_state.ticker_page == 1, key="prev_bottom"):
                        st.session_state.ticker_page -= 1
                        st.rerun()
                with col_p2_btm:
                    st.write(f"**Page {st.session_state.ticker_page} of {total_pages}**")
                with col_p3_btm:
                    st.write(f"**Showing {min(items_per_page, total_tickers)} of {total_tickers} tickers**")
                with col_p4_btm:
                    if st.button("Next ▶", disabled=st.session_state.ticker_page == total_pages, key="next_bottom"):
                        st.session_state.ticker_page += 1
                        st.rerun()
        else:
            st.info("No saved tickers yet. Add some above!")
        
        # Clear refresh_all_tickers flag if it was set
        if st.session_state.get('refresh_all_tickers', False):
            st.session_state.refresh_all_tickers = False
        
        # Bulk Analysis Section with Trading Style Support
        if all_tickers:
            st.divider()
            st.subheader("🧠 Bulk Analysis")
            
            # Get the selected trading style from above
            selected_style = st.session_state.get('analysis_timeframe', 'OPTIONS')
            
            bulk_col1, bulk_col2 = st.columns([3, 1])
            with bulk_col1:
                st.write(f"Run comprehensive analysis on all your saved tickers using **{selected_style}** trading style.")
            with bulk_col2:
                max_tickers = st.number_input("Max tickers", min_value=5, max_value=50, value=10, step=5, key="bulk_max")
            
            if st.button("🚀 Analyze All My Tickers", type="primary", width="stretch"):
                # Logging is now handled by Loguru - no need for manual handlers
                results = []
                style_results = []  # Store trading style-specific results
                
                with st.expander("📊 Live Analysis Logs", expanded=True):
                    log_container = st.empty()
                    with st.status(f"Analyzing your tickers with {selected_style} style...", expanded=True) as status:
                        ticker_list = [t['ticker'] for t in all_tickers[:max_tickers]]
                        
                        for i, ticker_symbol in enumerate(ticker_list):
                            status.update(label=f"Analyzing {ticker_symbol} ({i+1}/{len(ticker_list)})...", state="running")
                            try:
                                # Run comprehensive analysis with selected trading style
                                analysis = ComprehensiveAnalyzer.analyze_stock(ticker_symbol, selected_style)
                                
                                if analysis:
                                    results.append(analysis.__dict__)
                                    tm.update_analysis(ticker_symbol, analysis.__dict__)
                                    
                                    # Run trading style-specific analysis
                                    hist, _ = get_cached_stock_data(ticker_symbol)
                                    if not hist.empty:
                                        if selected_style == "AI":
                                            style_result = TradingStyleAnalyzer.analyze_ai_style(analysis, hist)
                                        elif selected_style == "SCALP":
                                            style_result = TradingStyleAnalyzer.analyze_scalp_style(analysis, hist)
                                        elif selected_style == "WARRIOR_SCALPING":
                                            style_result = TradingStyleAnalyzer.analyze_warrior_scalping_style(analysis, hist)
                                        elif selected_style == "BUY_AND_HOLD":
                                            style_result = TradingStyleAnalyzer.analyze_buy_and_hold_style(analysis, hist)
                                        else:
                                            style_result = None
                                        
                                        if style_result:
                                            style_result['ticker'] = ticker_symbol
                                            style_result['analysis'] = analysis
                                            style_results.append(style_result)
                                            
                            except Exception as e:
                                logger.error(f"⚠️ Error analyzing {ticker_symbol}: {e}")
                            log_container.code(log_stream.getvalue())
                        
                        status.update(label="✅ Analysis complete!", state="complete")
                
                alpha_factors_logger.removeHandler(st_handler)

                # Display results based on trading style
                if style_results:
                    # Sort by style-specific score
                    style_results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    st.success(f"✅ Analyzed {len(style_results)} tickers with {selected_style} style")
                    
                    st.subheader(f"🏆 Top {selected_style} Opportunities")
                    
                    for i, result in enumerate(style_results[:5], 1):
                        ticker_sym = result['ticker']
                        score = result.get('score', 0)
                        analysis = result['analysis']
                        
                        # Score-based emoji
                        emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                        
                        with st.expander(f"{emoji} #{i} - **{ticker_sym}** - Score: {score}/100", expanded=(i <= 3)):
                            # Basic metrics
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            with metric_col1:
                                st.metric("Price", f"${analysis.price:.2f}", f"{analysis.change_pct:+.2f}%")
                            with metric_col2:
                                st.metric(f"{selected_style} Score", f"{score}/100")
                            with metric_col3:
                                st.metric("Risk Level", result.get('risk_level', 'N/A'))
                            with metric_col4:
                                st.metric("Trend", analysis.trend)
                            
                            # Style-specific signals
                            if result.get('signals'):
                                st.write("**📊 Key Signals:**")
                                for signal in result['signals'][:5]:  # Top 5 signals
                                    st.write(signal)
                            
                            # Style-specific recommendations
                            if result.get('recommendations'):
                                st.write("**💡 Recommendations:**")
                                for rec in result['recommendations'][:3]:  # Top 3 recommendations
                                    st.write(rec)
                            
                            # Targets if available
                            if result.get('targets'):
                                st.write("**🎯 Targets:**")
                                for target in result['targets']:
                                    st.write(target)
                            
                            # Special fields for specific styles
                            if selected_style == "AI" and result.get('ml_prediction'):
                                st.info(f"🤖 **ML Prediction:** {result['ml_prediction']}")
                            elif selected_style == "WARRIOR_SCALPING" and result.get('setup_type'):
                                st.info(f"⚔️ **Setup Type:** {result['setup_type']}")
                            elif selected_style == "BUY_AND_HOLD" and result.get('valuation'):
                                st.info(f"📊 **Valuation:** {result['valuation']}")
                
                elif results:
                    # Fallback to standard results if no style-specific analysis
                    results.sort(key=lambda x: x['confidence_score'], reverse=True)
                    st.success(f"✅ Analyzed {len(results)} tickers")
                    st.subheader("🏆 Top Opportunities from Your Tickers")
                    for i, result in enumerate(results[:5], 1):
                        score = result['confidence_score']
                        emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns([1, 1, 1, 1, 1])
                        with col_r1:
                            st.write(f"{emoji} **{result['ticker']}**")
                        with col_r2:
                            st.write(f"Score: **{score:.0f}**/100")
                        with col_r3:
                            st.write(f"Trend: {result['trend']}")
                        with col_r4:
                            st.write(f"Sentiment: {result['sentiment_score']:.2f}")
                        with col_r5:
                            st.write(f"RSI: {result['rsi']:.0f}")
                else:
                    st.warning("No results to display after analysis.")
    
    with tab4:
        st.header("🔍 Stock Intelligence")
        st.write("Analyze stocks in-depth with AI-powered insights. Use Dashboard tab for quick analysis.")
        st.info("💡 Tip: Use the Dashboard tab for comprehensive stock intelligence and analysis.")
    
    with tab5:
        st.header("🎯 Intelligent Strategy Advisor")
        st.write("Get personalized strategy recommendations based on comprehensive analysis.")
        
        # Check if we have a current analysis to work with
        current_analysis = st.session_state.get('current_analysis', None)
        
        if current_analysis:
            st.success(f"📊 **Current Analysis Available:** {current_analysis.ticker} @ ${current_analysis.price:.2f}")
            
            # Quick analysis summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price Change", f"{current_analysis.change_pct:+.2f}%")
            with col2:
                st.metric("RSI", f"{current_analysis.rsi:.1f}")
            with col3:
                st.metric("IV Rank", f"{current_analysis.iv_rank:.1f}%")
            with col4:
                st.metric("Trend", current_analysis.trend)
            
            # Generate strategy recommendations based on current analysis
            st.subheader("🎯 AI-Powered Strategy Recommendations")
            st.write("Based on your current analysis and market conditions:")
            
            # Get user preferences
            col1, col2 = st.columns(2)
            with col1:
                user_experience = st.selectbox(
                    "Your Experience Level",
                    ["Beginner", "Intermediate", "Advanced"],
                    index=1,
                    key="advisor_exp"
                )
                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    ["Low", "Moderate", "High"],
                    index=1,
                    key="advisor_risk"
                )
            with col2:
                capital_available = st.number_input(
                    "Available Capital ($)",
                    min_value=100,
                    max_value=1000000,
                    value=5000,
                    step=100,
                    key="advisor_capital"
                )
                market_outlook = st.selectbox(
                    "Market Outlook",
                    ["Bullish", "Bearish", "Neutral"],
                    index=2,
                    key="advisor_outlook"
                )
            
            # Generate recommendations
            if st.button("🔍 Generate Strategy Recommendations", type="primary"):
                with st.spinner("Analyzing market conditions and generating recommendations..."):
                    try:
                        recommendations = StrategyAdvisor.get_recommendations(
                            analysis=current_analysis,
                            user_experience=user_experience,
                            risk_tolerance=risk_tolerance,
                            capital_available=capital_available,
                            outlook=market_outlook
                        )
                        
                        if recommendations:
                            st.success(f"✅ Generated {len(recommendations)} strategy recommendations!")
                            
                            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                                with st.expander(f"#{i} {rec.name} (Score: {int(rec.score * 100)}/100)", expanded=i==1):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Risk Level", rec.risk_level)
                                        st.metric("Max Loss", rec.max_loss)
                                    with col2:
                                        st.metric("Max Gain", rec.max_gain)
                                        st.metric("Win Rate", rec.win_rate)
                                    with col3:
                                        st.metric("Capital Req", rec.capital_req)
                                        st.metric("Experience", rec.experience)
                                    
                                    st.write(f"**Description:** {rec.description}")
                                    st.write(f"**Best For:** {', '.join(rec.best_for)}")
                                    
                                    if rec.reasoning:
                                        st.write("**Why This Strategy:**")
                                        for reason in rec.reasoning_list:
                                            st.write(f"• {reason}")
                                    
                                    if rec.setup_steps:
                                        st.write("**Setup Steps:**")
                                        for j, step in enumerate(rec.setup_steps, 1):
                                            st.write(f"{j}. {step}")
                                    
                                    if rec.warnings:
                                        st.write("**⚠️ Warnings:**")
                                        for warning in rec.warnings:
                                            st.warning(warning)
                                    
                                    # Action buttons
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button(f"Use This Strategy", key=f"use_rec_{i}"):
                                            st.session_state.selected_strategy = rec.name
                                            st.session_state.selected_ticker = current_analysis.ticker
                                            st.success(f"✅ Strategy '{rec.name}' selected for {current_analysis.ticker}")
                                    with col2:
                                        if st.button(f"View Details", key=f"details_rec_{i}"):
                                            st.info("Navigate to 'Generate Signal' tab to configure this strategy")
                        else:
                            st.warning("No suitable strategies found for current market conditions. Try adjusting your preferences.")
                            
                    except Exception as e:
                        st.error(f"Error generating recommendations: {e}")
        else:
            st.info("💡 **No stock analysis available.** Go to the Dashboard tab to analyze a stock first, then return here for strategy recommendations.")
            if st.button("Go to Dashboard"):
                st.info("Navigate to the 'Dashboard' tab above to analyze a stock")
        
        st.divider()
        
        # Add educational section about filtered investment approaches
        with st.expander("📚 Understanding Filtered Investment Approaches", expanded=False):
            st.markdown("""
            ### What are Filtered Investment Approaches?
            
            Filtered investment approaches are pre-configured scanning strategies that help you find specific types of trading opportunities based on your risk tolerance, market conditions, and investment goals. Each approach applies different criteria to filter stocks and identify the most relevant opportunities for your trading style.
            
            ### Available Investment Approaches:
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 High Confidence Only (Score ≥70)**
                - **What it does**: Shows only stocks with high AI confidence scores
                - **Best for**: Conservative traders, beginners, reliable setups
                - **Risk level**: Low to Medium
                - **Why use**: Reduces false signals, focuses on quality setups
                
                **💰 Ultra-Low Price (<$1)**
                - **What it does**: Finds stocks trading under $1 per share
                - **Best for**: High-risk, high-reward traders, penny stock enthusiasts
                - **Risk level**: Very High
                - **Why use**: Maximum upside potential, but requires careful risk management
                
                **💵 Penny Stocks ($1-$5)**
                - **What it does**: Targets stocks between $1-$5 per share
                - **Best for**: Growth-focused traders, small-cap investors
                - **Risk level**: High
                - **Why use**: Classic penny stock range with moderate risk/reward
                
                **📈 Volume Surge (>2x avg)**
                - **What it does**: Identifies stocks with unusually high trading volume
                - **Best for**: Momentum traders, breakout specialists
                - **Risk level**: Medium to High
                - **Why use**: High volume often precedes significant price movements
                """)
            
            with col2:
                st.markdown("""
                **🚀 Strong Momentum (>5% change)**
                - **What it does**: Finds stocks with significant price movements
                - **Best for**: Trend followers, momentum traders
                - **Risk level**: Medium to High
                - **Why use**: Captures stocks already in motion with strong directional bias
                
                **⚡ Power Zone Stocks Only**
                - **What it does**: Filters for stocks in EMA 8>21 power zone
                - **Best for**: Technical traders, trend followers
                - **Risk level**: Medium
                - **Why use**: EMA power zones indicate strong uptrend momentum
                
                **🔄 EMA Reclaim Setups**
                - **What it does**: Finds stocks that have reclaimed key EMA levels
                - **Best for**: Mean reversion traders, technical analysts
                - **Risk level**: Low to Medium
                - **Why use**: High-probability entry points with defined risk levels
                """)
            
            st.markdown("""
            ### ⚠️ Important Risk Considerations:
            
            - **Penny Stocks & Ultra-Low Price**: These stocks are highly volatile and can experience rapid price swings. Many penny stocks have low liquidity and may be difficult to exit quickly.
            
            - **Volume Surge & Momentum**: While high volume and momentum can indicate strong moves, they can also signal the end of a trend. Always use proper risk management.
            
            - **Technical Setups**: Power zones and EMA reclaims are based on historical patterns and may not always predict future performance.
            
            - **Diversification**: Don't put all your capital into one approach. Consider spreading risk across different strategies and timeframes.
            
            ### 💡 Pro Tips:
            
            1. **Start Conservative**: Begin with "High Confidence Only" to understand the platform
            2. **Combine Approaches**: Use multiple filters together for more targeted results
            3. **Risk Management**: Never risk more than you can afford to lose
            4. **Research First**: Always do your own due diligence before trading
            5. **Paper Trade**: Test strategies with paper trading before using real money
            """)
        
        # Check if we have analysis (optional for traditional strategies)
        if st.session_state.current_analysis:
            analysis = st.session_state.current_analysis
            st.success(f"Using analysis for: **{analysis.ticker}** (${analysis.price}, {analysis.change_pct:+.2f}%)")
            
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Your Trading Profile")
                
                # Create tabs for different input modes
                profile_tab, comparison_tab = st.tabs(["Single Profile", "Compare Scenarios"])

                with profile_tab:
                    user_experience = st.selectbox(
                        "Experience Level",
                        options=["Beginner", "Intermediate", "Advanced"],
                        key='user_experience_select',
                        help="Affects which strategies are recommended"
                    )

                    risk_tolerance = st.selectbox(
                        "Risk Tolerance",
                        options=["Conservative", "Moderate", "Aggressive"],
                        key='risk_tolerance_select',
                        help="Conservative = Lower risk strategies, Aggressive = Higher risk/reward"
                    )

                    capital_available = st.number_input(
                        "Available Capital ($)",
                        min_value=100,
                        max_value=1000000,
                        value=500,
                        step=100,
                        help="Total capital you're willing to risk on this trade"
                    )
                    
                    # Add position sizing controls
                    st.subheader("Position Sizing")
                    max_position_pct = st.slider(
                        "Max % of Capital per Trade",
                        min_value=1,
                        max_value=50,
                        value=10,
                        help="Maximum percentage of capital to risk on a single trade"
                    )
                    
                    max_position_amount = capital_available * (max_position_pct / 100)
                    st.info(f"💰 Max position size: ${max_position_amount:,.0f}")
                    
                    # Risk calculator
                    st.subheader("Risk Calculator")
                    risk_per_trade = st.number_input(
                        "Risk per Trade ($)",
                        min_value=10.0,
                        max_value=float(max_position_amount),
                        value=float(min(100, max_position_amount)),
                        step=10.0,
                        help="Maximum amount you're willing to lose on this single trade"
                    )
                    
                    risk_percentage = (risk_per_trade / capital_available) * 100
                    st.metric("Risk as % of Capital", f"{risk_percentage:.1f}%")
                    
                    if risk_percentage > 5:
                        st.warning("⚠️ Risk is high (>5% of capital). Consider reducing position size.")
                    elif risk_percentage > 2:
                        st.info("ℹ️ Moderate risk level (2-5% of capital).")
                    else:
                        st.success("✅ Conservative risk level (<2% of capital).")

                with comparison_tab:
                    st.subheader("Compare Different Scenarios")
                    
                    # Scenario 1
                    st.write("**Scenario 1 (Conservative)**")
                    col1a, col1b, col1c = st.columns(3)
                    with col1a:
                        exp1 = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], key="exp1")
                    with col1b:
                        risk1 = st.selectbox("Risk", ["Conservative", "Moderate", "Aggressive"], key="risk1")
                    with col1c:
                        cap1 = st.number_input("Capital ($)", 100, 1000000, 500, 100, key="cap1")
                    
                    # Scenario 2
                    st.write("**Scenario 2 (Moderate)**")
                    col2a, col2b, col2c = st.columns(3)
                    with col2a:
                        exp2 = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], key="exp2")
                    with col2b:
                        risk2 = st.selectbox("Risk", ["Conservative", "Moderate", "Aggressive"], key="risk2")
                    with col2c:
                        cap2 = st.number_input("Capital ($)", 100, 1000000, 2000, 100, key="cap2")
                    
                    # Scenario 3
                    st.write("**Scenario 3 (Aggressive)**")
                    col3a, col3b, col3c = st.columns(3)
                    with col3a:
                        exp3 = st.selectbox("Experience", ["Beginner", "Intermediate", "Advanced"], key="exp3")
                    with col3b:
                        risk3 = st.selectbox("Risk", ["Conservative", "Moderate", "Aggressive"], key="risk3")
                    with col3c:
                        cap3 = st.number_input("Capital ($)", 100, 1000000, 10000, 100, key="cap3")
                    
                    # Store scenarios for comparison
                    scenarios = [
                        {"name": "Conservative", "exp": exp1, "risk": risk1, "cap": cap1},
                        {"name": "Moderate", "exp": exp2, "risk": risk2, "cap": cap2},
                        {"name": "Aggressive", "exp": exp3, "risk": risk3, "cap": cap3}
                    ]

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

            # Generate recommendations based on selected tab
            if st.button("🚀 Generate Strategy Recommendations", type="primary", width="stretch"):
                with st.spinner("Analyzing optimal strategies..."):
                    # Check which tab is active by looking at the current tab selection
                    # For now, we'll generate both single and comparison views
                    
                    # Single profile recommendations
                    single_recommendations = StrategyAdvisor.get_recommendations(
                        analysis=analysis,
                        user_experience=user_experience,
                        risk_tolerance=risk_tolerance,
                        capital_available=capital_available,
                        outlook=outlook
                    )
                    
                    # Comparison recommendations
                    comparison_results = []
                    for scenario in scenarios:
                        scenario_recs = StrategyAdvisor.get_recommendations(
                            analysis=analysis,
                            user_experience=scenario["exp"],
                            risk_tolerance=scenario["risk"],
                            capital_available=scenario["cap"],
                            outlook=outlook
                        )
                        comparison_results.append({
                            "scenario": scenario,
                            "recommendations": scenario_recs
                        })
                    
                    # Display results
                    if single_recommendations:
                        st.subheader(f"📋 Recommended Strategies for {analysis.ticker}")
                        
                        # Show single profile results
                        st.write("**Your Profile Results:**")
                        for idx, rec in enumerate(single_recommendations, 1):
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
                                        if st.button(f"Load Example Trade", key=f"strategy_load_example_{idx}"):
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
                    
                    # Display comparison results
                    st.divider()
                    st.subheader("📊 Strategy Comparison Across Different Scenarios")
                    
                    if comparison_results:
                        # Create a comparison table
                        comparison_data = []
                        for result in comparison_results:
                            scenario = result["scenario"]
                            recommendations = result["recommendations"]
                            
                            if recommendations:
                                top_rec = recommendations[0]  # Get the top recommendation
                                comparison_data.append({
                                    "Scenario": scenario["name"],
                                    "Experience": scenario["exp"],
                                    "Risk": scenario["risk"],
                                    "Capital": f"${scenario['cap']:,}",
                                    "Top Strategy": top_rec.strategy_name,
                                    "Confidence": f"{int(top_rec.confidence * 100)}%",
                                    "Risk Level": top_rec.risk_level,
                                    "Max Loss": top_rec.max_loss,
                                    "Max Gain": top_rec.max_gain
                                })
                        
                        if comparison_data:
                            # Display as a table
                            df_comparison = pd.DataFrame(comparison_data)
                            st.dataframe(df_comparison, width="stretch")
                            
                            # Show detailed comparison for each scenario
                            st.subheader("🔍 Detailed Strategy Analysis by Scenario")
                            
                            for i, result in enumerate(comparison_results):
                                scenario = result["scenario"]
                                recommendations = result["recommendations"]
                                
                                with st.expander(f"Scenario {i+1}: {scenario['name']} ({scenario['exp']} + {scenario['risk']} + ${scenario['cap']:,})", expanded=False):
                                    if recommendations:
                                        for idx, rec in enumerate(recommendations[:3], 1):  # Show top 3
                                            confidence_pct = int(rec.confidence * 100)
                                            badge = "🟢 High" if confidence_pct >= 70 else "🟡 Moderate" if confidence_pct >= 50 else "🟠 Low"
                                            
                                            st.write(f"**#{idx} {rec.strategy_name}** - {badge} ({confidence_pct}%)")
                                            st.write(f"Risk: {rec.risk_level} | Max Loss: {rec.max_loss} | Max Gain: {rec.max_gain}")
                                            st.write(f"*{rec.reasoning}*")
                                            st.write("---")
                                    else:
                                        st.write("No suitable strategies found for this scenario.")
                        else:
                            st.warning("No strategies found for any scenario. Try adjusting parameters.")
        else:
            # No analysis available - show message
            st.info("💡 **Traditional Strategy Recommendations** require stock analysis. Analyze a stock in the Dashboard tab first, or use Advanced Strategies below.")
        
        # Custom Template Strategies Section
        st.divider()
        # Enhanced Custom Strategy Templates with Analysis Integration
        st.subheader("📚 Your Custom Strategy Templates")
        st.caption("Strategies you've saved in the Strategy Templates tab")
        
        try:
            from models.option_strategy_templates import template_manager
            
            custom_templates = template_manager.get_all_templates()
            
            if custom_templates:
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    template_exp_filter = st.selectbox(
                        "Experience Level",
                        ["All", "Beginner", "Intermediate", "Advanced", "Professional"],
                        key="template_exp_filter"
                    )
                with col2:
                    template_direction_filter = st.selectbox(
                        "Direction",
                        ["All", "Bullish", "Bearish", "Neutral", "Volatility"],
                        key="template_direction_filter"
                    )
                with col3:
                    template_oa_filter = st.checkbox(
                        "Option Alpha Compatible Only",
                        value=True,
                        key="template_oa_filter"
                    )
                
                # Apply filters
                filtered_templates = custom_templates
                if template_exp_filter != "All":
                    filtered_templates = [t for t in filtered_templates if t.experience_level == template_exp_filter]
                if template_direction_filter != "All":
                    filtered_templates = [t for t in filtered_templates if template_direction_filter.upper() in t.direction.upper()]
                if template_oa_filter:
                    filtered_templates = [t for t in filtered_templates if t.option_alpha_compatible]
                
                if filtered_templates:
                    st.write(f"**{len(filtered_templates)} template(s) available**")
                    
                    # If we have current analysis, show compatibility scores
                    if current_analysis:
                        st.info("🎯 **Analysis-Based Recommendations:** Templates are scored based on current market conditions")
                        
                        # Score templates based on current analysis
                        scored_templates = []
                        for template in filtered_templates:
                            score = 0
                            reasoning = []
                            
                            # IV Rank compatibility
                            if template.ideal_iv_rank == "High (>60)" and current_analysis.iv_rank > 60:
                                score += 30
                                reasoning.append(f"✅ High IV Rank ({current_analysis.iv_rank}%) - perfect for premium selling")
                            elif template.ideal_iv_rank == "Low (<30)" and current_analysis.iv_rank < 30:
                                score += 30
                                reasoning.append(f"✅ Low IV Rank ({current_analysis.iv_rank}%) - good for option buying")
                            elif template.ideal_iv_rank == "Medium (30-60)" and 30 <= current_analysis.iv_rank <= 60:
                                score += 25
                                reasoning.append(f"✅ Medium IV Rank ({current_analysis.iv_rank}%) - balanced conditions")
                            
                            # RSI compatibility
                            if template.direction == "BULLISH" and current_analysis.rsi < 30:
                                score += 20
                                reasoning.append(f"✅ Oversold RSI ({current_analysis.rsi:.1f}) - bullish opportunity")
                            elif template.direction == "BEARISH" and current_analysis.rsi > 70:
                                score += 20
                                reasoning.append(f"✅ Overbought RSI ({current_analysis.rsi:.1f}) - bearish opportunity")
                            elif template.direction == "NEUTRAL" and 30 <= current_analysis.rsi <= 70:
                                score += 15
                                reasoning.append(f"✅ Neutral RSI ({current_analysis.rsi:.1f}) - good for neutral strategies")
                            
                            # Trend compatibility
                            if template.direction == "BULLISH" and current_analysis.trend == "Uptrend":
                                score += 25
                                reasoning.append("✅ Uptrending stock - bullish strategies favorable")
                            elif template.direction == "BEARISH" and current_analysis.trend == "Downtrend":
                                score += 25
                                reasoning.append("✅ Downtrending stock - bearish strategies favorable")
                            elif template.direction == "NEUTRAL" and current_analysis.trend == "Sideways":
                                score += 20
                                reasoning.append("✅ Sideways movement - neutral strategies ideal")
                            
                            # Price movement compatibility
                            if template.direction == "BULLISH" and current_analysis.change_pct > 0:
                                score += 10
                                reasoning.append(f"✅ Positive price movement ({current_analysis.change_pct:+.1f}%)")
                            elif template.direction == "BEARISH" and current_analysis.change_pct < 0:
                                score += 10
                                reasoning.append(f"✅ Negative price movement ({current_analysis.change_pct:+.1f}%)")
                            
                            scored_templates.append((template, score, reasoning))
                        
                        # Sort by score
                        scored_templates.sort(key=lambda x: x[1], reverse=True)
                        
                        for template, score, reasoning in scored_templates:
                            with st.expander(f"📋 {template.name} (Compatibility: {score}/100) ({template.experience_level} | {template.risk_level} Risk)", expanded=score>70):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Direction", template.direction)
                                    st.metric("Risk Level", template.risk_level)
                                with col2:
                                    st.metric("Capital Required", template.capital_requirement)
                                    if template.typical_win_rate:
                                        st.metric("Win Rate", template.typical_win_rate)
                                with col3:
                                    st.metric("IV Rank", template.ideal_iv_rank)
                                    st.metric("Type", template.strategy_type)
                                
                                st.markdown(f"**Description:** {template.description}")
                                st.markdown(f"**Max Loss:** {template.max_loss}")
                                st.markdown(f"**Max Gain:** {template.max_gain}")
                                
                                if reasoning:
                                    st.write("**Why This Strategy Works Now:**")
                                    for reason in reasoning:
                                        st.write(f"• {reason}")
                                
                                if template.setup_steps:
                                    st.write("**Setup Steps:**")
                                    for i, step in enumerate(template.setup_steps, 1):
                                        st.write(f"{i}. {step}")
                                
                                if template.warnings:
                                    st.write("**⚠️ Warnings:**")
                                    for warning in template.warnings:
                                        st.warning(warning)
                                
                                if template.option_alpha_compatible:
                                    st.success(f"✅ Option Alpha Compatible - Action: `{template.option_alpha_action}`")
                                
                                # Action buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"Use This Template", key=f"use_template_{template.strategy_id}"):
                                        st.session_state.selected_template = template.strategy_id
                                        st.session_state.selected_strategy = template.name
                                        st.session_state.selected_ticker = current_analysis.ticker
                                        st.success(f"✅ Template '{template.name}' selected for {current_analysis.ticker}")
                                with col2:
                                    if st.button(f"View Full Details", key=f"details_template_{template.strategy_id}"):
                                        st.info("Navigate to 'Generate Signal' tab to configure this strategy")
                    else:
                        # No current analysis - show basic template list
                        for template in filtered_templates:
                            with st.expander(f"📋 {template.name} ({template.experience_level} | {template.risk_level} Risk)"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Direction", template.direction)
                                    st.metric("Risk Level", template.risk_level)
                                with col2:
                                    st.metric("Capital Required", template.capital_requirement)
                                    if template.typical_win_rate:
                                        st.metric("Win Rate", template.typical_win_rate)
                                with col3:
                                    st.metric("IV Rank", template.ideal_iv_rank)
                                    st.metric("Type", template.strategy_type)
                                
                                st.markdown(f"**Description:** {template.description}")
                                st.markdown(f"**Max Loss:** {template.max_loss}")
                                st.markdown(f"**Max Gain:** {template.max_gain}")
                                
                                if template.option_alpha_compatible:
                                    st.success(f"✅ Option Alpha Compatible - Action: `{template.option_alpha_action}`")
                                
                                if st.button(f"Use This Template", key=f"use_template_{template.strategy_id}"):
                                    st.session_state.selected_template = template.strategy_id
                                    st.success(f"✅ Template selected! Configure in Generate Signal tab.")
                else:
                    st.info("No templates match your filters. Adjust filters or add templates in the Strategy Templates tab.")
            else:
                st.info("💡 No custom templates yet. Create your first template in the **Strategy Templates** tab!")
                if st.button("Go to Strategy Templates"):
                    st.info("Navigate to the 'Strategy Templates' tab above to create templates")
        
        except Exception as e:
            st.error(f"Error loading custom templates: {e}")
        
        # Strategy Testing and Comparison Section
        if current_analysis:
            st.divider()
            st.subheader("🧪 Strategy Testing & Comparison")
            st.caption("Test and compare different strategies for the current analysis")
            
            # Quick strategy comparison
            if st.button("🔍 Compare All Strategies", type="secondary"):
                with st.spinner("Analyzing all strategies for current market conditions..."):
                    try:
                        from models.option_strategy_templates import template_manager
                        
                        all_templates = template_manager.get_all_templates()
                        comparison_results = []
                        
                        for template in all_templates:
                            score = 0
                            reasoning = []
                            
                            # IV Rank compatibility
                            if template.ideal_iv_rank == "High (>60)" and current_analysis.iv_rank > 60:
                                score += 30
                                reasoning.append(f"High IV Rank ({current_analysis.iv_rank}%) - perfect for premium selling")
                            elif template.ideal_iv_rank == "Low (<30)" and current_analysis.iv_rank < 30:
                                score += 30
                                reasoning.append(f"Low IV Rank ({current_analysis.iv_rank}%) - good for option buying")
                            elif template.ideal_iv_rank == "Medium (30-60)" and 30 <= current_analysis.iv_rank <= 60:
                                score += 25
                                reasoning.append(f"Medium IV Rank ({current_analysis.iv_rank}%) - balanced conditions")
                            
                            # RSI compatibility
                            if template.direction == "BULLISH" and current_analysis.rsi < 30:
                                score += 20
                                reasoning.append(f"Oversold RSI ({current_analysis.rsi:.1f}) - bullish opportunity")
                            elif template.direction == "BEARISH" and current_analysis.rsi > 70:
                                score += 20
                                reasoning.append(f"Overbought RSI ({current_analysis.rsi:.1f}) - bearish opportunity")
                            elif template.direction == "NEUTRAL" and 30 <= current_analysis.rsi <= 70:
                                score += 15
                                reasoning.append(f"Neutral RSI ({current_analysis.rsi:.1f}) - good for neutral strategies")
                            
                            # Trend compatibility
                            if template.direction == "BULLISH" and current_analysis.trend == "Uptrend":
                                score += 25
                                reasoning.append("Uptrending stock - bullish strategies favorable")
                            elif template.direction == "BEARISH" and current_analysis.trend == "Downtrend":
                                score += 25
                                reasoning.append("Downtrending stock - bearish strategies favorable")
                            elif template.direction == "NEUTRAL" and current_analysis.trend == "Sideways":
                                score += 20
                                reasoning.append("Sideways movement - neutral strategies ideal")
                            
                            comparison_results.append({
                                'template': template,
                                'score': score,
                                'reasoning': reasoning
                            })
                        
                        # Sort by score
                        comparison_results.sort(key=lambda x: x['score'], reverse=True)
                        
                        # Display results
                        st.success(f"✅ Analyzed {len(comparison_results)} strategies")
                        
                        # Create a comparison table
                        # pandas already imported at module level
                        comparison_data = []
                        for result in comparison_results[:10]:  # Top 10
                            template = result['template']
                            comparison_data.append({
                                'Strategy': template.name,
                                'Direction': template.direction,
                                'Risk': template.risk_level,
                                'Score': result['score'],
                                'IV Match': template.ideal_iv_rank,
                                'Experience': template.experience_level,
                                'Capital Req': template.capital_requirement
                            })
                        
                        df = pd.DataFrame(comparison_data)
                        st.dataframe(df, width="stretch")
                        
                        # Show top 3 strategies in detail
                        st.subheader("🏆 Top 3 Recommended Strategies")
                        for i, result in enumerate(comparison_results[:3], 1):
                            template = result['template']
                            with st.expander(f"#{i} {template.name} (Score: {result['score']}/100)", expanded=i==1):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Direction", template.direction)
                                    st.metric("Risk Level", template.risk_level)
                                with col2:
                                    st.metric("Capital Required", template.capital_requirement)
                                    if template.typical_win_rate:
                                        st.metric("Win Rate", template.typical_win_rate)
                                with col3:
                                    st.metric("IV Rank", template.ideal_iv_rank)
                                    st.metric("Type", template.strategy_type)
                                
                                st.markdown(f"**Description:** {template.description}")
                                
                                if result['reasoning']:
                                    st.write("**Why This Strategy Works Now:**")
                                    for reason in result['reasoning']:
                                        st.write(f"• {reason}")
                                
                                if template.setup_steps:
                                    st.write("**Setup Steps:**")
                                    for j, step in enumerate(template.setup_steps, 1):
                                        st.write(f"{j}. {step}")
                                
                                # Action buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"Use This Strategy", key=f"use_compare_{template.strategy_id}"):
                                        st.session_state.selected_template = template.strategy_id
                                        st.session_state.selected_strategy = template.name
                                        st.session_state.selected_ticker = current_analysis.ticker
                                        st.success(f"✅ Strategy '{template.name}' selected for {current_analysis.ticker}")
                                with col2:
                                    if st.button(f"Test Strategy", key=f"test_compare_{template.strategy_id}"):
                                        st.info("Navigate to 'Generate Signal' tab to test this strategy")
                        
                    except Exception as e:
                        st.error(f"Error comparing strategies: {e}")
        
        # Advanced Strategies Section (works independently)
        st.divider()
        st.subheader("🚀 Advanced Professional Strategies")
        st.caption("Professional-grade strategies with AI validation")
        
        # Import advanced strategy modules
        try:
            from models.reddit_strategies import get_all_custom_strategies, get_custom_strategy
            from services.reddit_strategy_validator import StrategyValidator
            from analyzers.strategy import StrategyAdvisor as AdvancedAdvisor
            
            # Get available strategies
            user_exp_advanced = st.selectbox(
                "Your Experience Level for Advanced Strategies",
                ["Beginner", "Intermediate", "Advanced", "Professional"],
                index=1,
                key="advanced_exp_level"
            )
            
            advanced_strategies = AdvancedAdvisor.get_custom_strategies(user_exp_advanced)
            
            if not advanced_strategies:
                st.info(f"ℹ️ No advanced strategies available for {user_exp_advanced} level. Try selecting a higher experience level.")
            else:
                # Strategy selection
                strategy_names = [s.name for s in advanced_strategies]
                selected_name = st.selectbox(
                    "Select Advanced Strategy",
                    strategy_names,
                    key="advanced_strategy_select"
                )
                
                # Get selected strategy
                selected_strategy = next(s for s in advanced_strategies if s.name == selected_name)
                
                # Display strategy overview
                with st.expander("📋 Strategy Overview", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Source", selected_strategy.source)
                        st.metric("Experience", selected_strategy.experience_level)
                    with col2:
                        st.metric("Risk Level", selected_strategy.risk_level)
                        st.metric("Capital Required", selected_strategy.capital_requirement)
                    with col3:
                        if selected_strategy.typical_win_rate:
                            st.metric("Win Rate", selected_strategy.typical_win_rate)
                        st.metric("Products", len(selected_strategy.suitable_products))
                    
                    st.markdown(f"**Description:** {selected_strategy.description}")
                    st.markdown(f"**Philosophy:** {selected_strategy.philosophy}")
                
                # Key metrics
                with st.expander("📊 Performance Metrics"):
                    metric_cols = st.columns(len(selected_strategy.key_metrics))
                    for i, (key, value) in enumerate(selected_strategy.key_metrics.items()):
                        with metric_cols[i]:
                            st.metric(key.replace("_", " ").title(), value)
                
                # Setup rules
                with st.expander("📖 Step-by-Step Playbook"):
                    for rule in sorted(selected_strategy.setup_rules, key=lambda r: r.priority):
                        st.markdown(f"**{rule.priority}. {rule.condition}**")
                        st.info(rule.action)
                        if rule.notes:
                            st.caption(f"📝 {rule.notes}")
                        st.markdown("---")
                
                # Risk management
                with st.expander("🛡️ Risk Management"):
                    for rule in selected_strategy.risk_management:
                        mandatory = "🔴 MANDATORY" if rule.mandatory else "🟡 Optional"
                        st.markdown(f"**{rule.rule_type.upper()}** {mandatory}")
                        st.write(f"Value: `{rule.value}`")
                        st.caption(rule.description)
                        st.markdown("---")
                
                # Warnings
                if selected_strategy.warnings:
                    with st.expander("⚠️ Important Warnings", expanded=True):
                        for warning in selected_strategy.warnings:
                            st.warning(warning)
                
                # AI Validation Section
                st.markdown("---")
                st.markdown("### 🤖 AI Strategy Validation")
                st.caption("Validate this strategy for a specific ticker with AI analysis")
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    ticker_input = st.text_input(
                        "Ticker to Validate",
                        value=st.session_state.current_analysis.ticker if st.session_state.current_analysis else "",
                        placeholder="SPY",
                        key="advanced_strat_ticker"
                    )
                with col2:
                    include_context = st.checkbox("Include Market Context", value=True, key="include_market_ctx")
                
                market_context = None
                if include_context:
                    with st.expander("📊 Market Context (Optional)"):
                        ctx_col1, ctx_col2 = st.columns(2)
                        with ctx_col1:
                            vix = st.number_input("VIX Level", 0.0, 100.0, 20.0, key="vix_input")
                            sentiment = st.selectbox(
                                "Market Sentiment",
                                ["Bullish", "Neutral", "Bearish", "Panic", "Euphoric"],
                                key="sentiment_input"
                            )
                        with ctx_col2:
                            events = st.text_area(
                                "Upcoming Events",
                                placeholder="Fed meeting, earnings season, etc.",
                                key="events_input"
                            )
                        market_context = {
                            "vix": vix,
                            "sentiment": sentiment,
                            "upcoming_events": events
                        }
                
                if st.button("🚀 Validate Strategy with AI", type="primary", key="validate_advanced_btn"):
                    if not ticker_input:
                        st.error("Please enter a ticker symbol")
                    else:
                        with st.spinner(f"AI analyzing {selected_strategy.name} for {ticker_input}..."):
                            try:
                                # Get stock analysis if available
                                analysis_for_validation = None
                                if st.session_state.current_analysis and st.session_state.current_analysis.ticker == ticker_input:
                                    analysis_for_validation = st.session_state.current_analysis
                                else:
                                    # Try to get fresh analysis
                                    try:
                                        analysis_for_validation = ComprehensiveAnalyzer.analyze_stock(ticker_input, "SWING_TRADE")
                                    except Exception as e:
                                        st.warning(f"Could not get fresh analysis: {e}")
                                
                                # Run validation
                                validator = StrategyValidator()
                                validation = validator.validate_strategy(
                                    strategy=selected_strategy,
                                    ticker=ticker_input,
                                    analysis=analysis_for_validation,
                                    market_context=market_context
                                )
                                
                                # Display results
                                st.markdown("---")
                                st.markdown("### 📊 Validation Results")
                                
                                # Overall verdict
                                if validation.is_viable:
                                    st.success(f"✅ **VIABLE** - {validation.market_alignment} Market Alignment")
                                else:
                                    st.error(f"❌ **NOT VIABLE** - {validation.market_alignment} Market Alignment")
                                
                                # Metrics
                                metric_col1, metric_col2, metric_col3 = st.columns(3)
                                with metric_col1:
                                    st.metric("Viability Score", f"{validation.viability_score:.1%}")
                                with metric_col2:
                                    st.metric("Confidence", f"{validation.confidence:.1%}")
                                with metric_col3:
                                    st.metric("Alignment", validation.market_alignment)
                                
                                # Reasoning
                                with st.expander("🧠 AI Reasoning", expanded=True):
                                    st.markdown(validation.reasoning)
                                
                                # Strengths
                                if validation.strengths:
                                    with st.expander("✅ Strengths", expanded=True):
                                        for strength in validation.strengths:
                                            st.success(f"✅ {strength}")
                                
                                # Concerns
                                if validation.concerns:
                                    with st.expander("⚠️ Concerns", expanded=True):
                                        for concern in validation.concerns:
                                            st.warning(f"⚠️ {concern}")
                                
                                # Missing conditions
                                if validation.missing_conditions:
                                    with st.expander("❌ Missing Conditions", expanded=True):
                                        st.markdown("**The following required conditions are NOT currently met:**")
                                        for condition in validation.missing_conditions:
                                            st.error(f"❌ {condition}")
                                
                                # Red flags
                                if validation.red_flags_detected:
                                    with st.expander("🚩 Red Flags Detected", expanded=True):
                                        st.markdown("**⚠️ WARNING: The following red flags were detected:**")
                                        for flag in validation.red_flags_detected:
                                            st.error(f"🚩 {flag}")
                                
                                # Recommendations
                                if validation.recommendations:
                                    with st.expander("💡 Recommendations", expanded=True):
                                        for rec in validation.recommendations:
                                            st.info(f"💡 {rec}")
                            
                            except Exception as e:
                                st.error(f"Validation failed: {str(e)}")
                                with st.expander("Error Details"):
                                    import traceback
                                    st.code(traceback.format_exc())
        
        except ImportError as e:
            st.error(f"Advanced strategies module not available: {e}")
            st.info("Make sure the following files exist:\n- models/reddit_strategies.py\n- services/reddit_strategy_validator.py")
        
        # Strategy Explanation Section
        st.divider()
        st.subheader("📚 Understanding Option Strategies")
        
        with st.expander("🔍 Learn About Different Option Strategies", expanded=False):
            st.markdown("""
            ### For Beginners (Low Risk):
            - **Long Call/Put**: Buy options with limited risk (premium paid)
            - **Covered Call**: Sell calls against stock you own
            - **Cash-Secured Put**: Sell puts with cash backing (like your NOK example)
            
            ### For Intermediate (Medium Risk):
            - **Credit Spreads**: Sell one option, buy another to limit risk
            - **Debit Spreads**: Buy one option, sell another to reduce cost
            - **Iron Condors**: Range-bound strategies for sideways markets
            
            ### For Advanced (Higher Risk):
            - **Straddles/Strangles**: Profit from big moves in either direction
            - **Calendar Spreads**: Time-based strategies
            - **Wheel Strategy**: Systematic put selling and call writing
            """)
            
            st.markdown("""
            ### Risk Management Tips:
            1. **Start Small**: Use only 1-2% of capital per trade initially
            2. **Define Risk**: Always know your maximum loss before entering
            3. **Diversify**: Don't put all capital in one strategy or stock
            4. **Learn Gradually**: Master one strategy before trying others
            5. **Use Stops**: Set mental or actual stop losses
            """)
        
        # Quick Reference Section
        st.divider()
        st.subheader("📋 Quick Reference: Investment Approaches")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Conservative Approaches**
            - High Confidence Only
            - EMA Reclaim Setups
            - Power Zone Stocks
            """)
        
        with col2:
            st.markdown("""
            **⚡ Active Trading**
            - Volume Surge
            - Strong Momentum
            - Power Zone Stocks
            """)
        
        with col3:
            st.markdown("""
            **💰 High Risk/Reward**
            - Ultra-Low Price
            - Penny Stocks
            - Volume Surge + Momentum
            """)
        
        st.info("💡 **Tip**: Use the Advanced Scanner tab to apply these approaches to find specific opportunities in the market!")
        
        # Hybrid Approach Explanation
        st.divider()
        st.subheader("🧬 Hybrid Approach: Holistic Stock Assessment")
        
        st.markdown("""
        ### What is the Hybrid Approach?
        
        The **Hybrid Approach** in the Advanced Scanner combines multiple investment approaches with AI analysis and strategy recommendations to provide a **comprehensive, holistic assessment** of the best stocks to invest in and the optimal strategies to use.
        
        ### Key Features:
        
        **🔗 Multi-Filter Combination:**
        - **Primary Approach**: Choose your main investment focus (e.g., "High Confidence Only")
        - **Secondary Filters**: Add additional criteria (e.g., "Volume Surge" + "Power Zone")
        - **Smart Filtering**: Combines all criteria to find stocks that meet multiple conditions
        
        **🎯 Strategy Integration:**
        - **Personalized Recommendations**: Get specific trading strategies for each found opportunity
        - **Risk-Adjusted**: Strategies are tailored to your experience level and risk tolerance
        - **Capital-Aware**: Recommendations consider your available capital
        - **Market Outlook**: Strategies align with your market expectations
        
        **🤖 AI-Enhanced Analysis:**
        - **Comprehensive Scoring**: Combines technical, fundamental, and sentiment analysis
        - **Confidence Ratings**: Each opportunity gets an AI confidence score
        - **Risk Assessment**: Detailed risk analysis for each recommendation
        
        ### Example Hybrid Scenarios:
        
        **Conservative Hybrid:**
        - Primary: "High Confidence Only (Score ≥70)"
        - Secondary: "EMA Reclaim Setups" + "RSI Oversold (<30)"
        - Result: High-quality, low-risk setups with strong technical confirmation
        
        **Momentum Hybrid:**
        - Primary: "Strong Momentum (>5% change)"
        - Secondary: "Volume Surge (>2x avg)" + "Power Zone Stocks Only"
        - Result: High-momentum stocks with strong volume and technical confirmation
        
        **Penny Stock Hybrid:**
        - Primary: "Penny Stocks ($1-$5)"
        - Secondary: "Volume Surge (>2x avg)" + "High Confidence Only"
        - Result: Quality penny stocks with strong volume and AI confidence
        
        ### Benefits of Hybrid Approach:
        
        1. **Higher Quality Results**: Multiple filters reduce false signals
        2. **Personalized Strategies**: Get specific trading recommendations for each stock
        3. **Risk Management**: Built-in risk assessment and strategy matching
        4. **Comprehensive Analysis**: Combines technical, fundamental, and AI analysis
        5. **Actionable Insights**: Not just what to buy, but how to trade it
        
        ### How to Use:
        
        1. **Enable Hybrid Mode**: Check "🧬 Use Hybrid Approach" in the Advanced Scanner
        2. **Set Primary Filter**: Choose your main investment approach
        3. **Add Secondary Filters**: Select additional criteria to combine
        4. **Configure Strategy Preferences**: Set your experience, risk tolerance, and capital
        5. **Run Scan**: Get comprehensive results with strategy recommendations
        6. **Review Results**: Each opportunity shows both analysis and recommended strategies
        7. **Take Action**: Use the recommended strategies or get full analysis
        
        This hybrid approach gives you the **most comprehensive and actionable** stock analysis available, combining the best of technical analysis, AI insights, and personalized strategy recommendations.
        """)
    
    with tab6:
        st.header("📊 Generate Trading Signal")
        
        # Get current trading mode
        mode_manager = get_trading_mode_manager()
        paper_mode = mode_manager.is_paper_mode()
        
        # Show current trading mode with switch option
        col1, col2 = st.columns([3, 1])
        with col1:
            if paper_mode:
                st.info("🔒 **Paper Trading Mode** - Signals will be logged only")
            else:
                st.warning("⚠️ **LIVE TRADING MODE** - Real trades will be executed!")
        with col2:
            if paper_mode:
                if st.button("Switch to Live Trading", type="primary"):
                    if switch_to_production_mode():
                        st.success("Switched to Live Trading Mode!")
                        st.rerun()
                    else:
                        st.error("Failed to switch to Live Trading Mode")
            else:
                if st.button("Switch to Paper Trading"):
                    if switch_to_paper_mode():
                        st.success("Switched to Paper Trading Mode!")
                        st.rerun()
                    else:
                        st.error("Failed to switch to Paper Trading Mode")
        
        # Check if we have a selected strategy template
        selected_template_id = st.session_state.get('selected_template')
        selected_strategy = st.session_state.get('selected_strategy')
        selected_ticker = st.session_state.get('selected_ticker', 'N/A')
        
        if selected_template_id:
            try:
                from models.option_strategy_templates import template_manager
                template = template_manager.get_template(selected_template_id)
                
                if template:
                    st.success(f"🎯 **Using Strategy Template:** {template.name}")
                    
                    # Show template details in an expandable section
                    with st.expander("📋 Template Details", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Direction", template.direction)
                            st.metric("Risk Level", template.risk_level)
                        with col2:
                            st.metric("Capital Required", template.capital_requirement)
                            if template.typical_win_rate:
                                st.metric("Win Rate", template.typical_win_rate)
                        with col3:
                            st.metric("IV Rank", template.ideal_iv_rank)
                            st.metric("Type", template.strategy_type)
                        
                        st.markdown(f"**Description:** {template.description}")
                        st.markdown(f"**Max Loss:** {template.max_loss}")
                        st.markdown(f"**Max Gain:** {template.max_gain}")
                        
                        if template.setup_steps:
                            st.write("**Setup Steps:**")
                            for i, step in enumerate(template.setup_steps, 1):
                                st.write(f"{i}. {step}")
                        
                        if template.warnings:
                            st.write("**⚠️ Warnings:**")
                            for warning in template.warnings:
                                st.warning(warning)
                        
                        if template.option_alpha_compatible:
                            st.success(f"✅ Option Alpha Compatible - Action: `{template.option_alpha_action}`")
                    
                    # Pre-fill strategy selection
                    if template.option_alpha_action:
                        st.session_state.selected_strategy = template.option_alpha_action
                        selected_strategy = template.option_alpha_action
                    
                    st.divider()
                else:
                    st.warning("Selected template not found. Please select a valid template.")
            except Exception as e:
                st.error(f"Error loading template: {e}")
        
        if selected_strategy:
            st.info(f"💡 Using recommended strategy: **{selected_strategy}** for **{selected_ticker}**")
        
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
    
    with tab7:
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
    
    with tab8:
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

                    if st.button("Load from selected strategy/example", key=f"sensitivity_load_example_{i}"):
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
    
    
    with tab9:
        # Strategy Templates Manager
        from ui.strategy_template_manager import render_template_manager
        render_template_manager()
    
    
    with tab10:
        # Initialize Tradier client
        from src.integrations.tradier_client import create_tradier_client_from_env
        mode_manager = get_trading_mode_manager()
        current_mode = mode_manager.get_mode()
        
        # Check if we need to initialize or refresh the client
        should_refresh_client = (
            'tradier_client' not in st.session_state or
            st.session_state.tradier_client is None or
            st.session_state.tradier_client.trading_mode != current_mode
        )
        
        if should_refresh_client:
            logger.info("Initializing/refreshing Tradier client from environment")
            logger.info("Current trading mode: %s", current_mode.value)
            try:
                # Use trading mode manager to get client for current mode
                st.session_state.tradier_client = create_tradier_client_from_env(trading_mode=current_mode)
                logger.info("Tradier client initialized successfully: %s", bool(st.session_state.tradier_client))
                logger.info("Client trading mode: %s", st.session_state.tradier_client.trading_mode.value if st.session_state.tradier_client else "None")
                # Clear cached account data when switching modes
                if 'account_summary' in st.session_state:
                    del st.session_state.account_summary
                    logger.info("Cleared cached account summary due to mode change")
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
                        # Group orders by class (show bracket orders specially)
                        for order in orders:
                            order_class = order.get('class', 'equity')
                            order_id = order.get('id', 'N/A')
                            symbol = order.get('symbol', 'N/A')
                            status = order.get('status', 'N/A')
                            
                            if order_class in ['otoco', 'oco']:
                                # Bracket order - show all legs
                                with st.expander(f"🎯 Bracket Order: {symbol} (ID: {order_id}) - {status}"):
                                    st.write(f"**Order Class:** {order_class.upper()}")
                                    st.write(f"**Status:** {status}")
                                    
                                    # Get legs if available
                                    legs = order.get('leg', [])
                                    if not isinstance(legs, list):
                                        legs = [legs] if legs else []
                                    
                                    if legs:
                                        st.write("**Order Legs:**")
                                        for i, leg in enumerate(legs, 1):
                                            leg_type = leg.get('type', 'N/A')
                                            leg_side = leg.get('side', 'N/A')
                                            leg_qty = leg.get('quantity', 'N/A')
                                            leg_price = leg.get('price', leg.get('avg_fill_price', ''))
                                            leg_stop = leg.get('stop', '')
                                            leg_status = leg.get('status', 'N/A')
                                            
                                            # Determine leg purpose based on type and position
                                            if leg_type == 'limit' and i == 1:
                                                price_str = f"${leg_price}" if leg_price else "N/A"
                                                st.info(f"**Leg {i} - Entry:** {leg_side.upper()} {leg_qty} @ {price_str} ({leg_status})")
                                            elif leg_type == 'limit' and i == 2:
                                                price_str = f"${leg_price}" if leg_price else "N/A"
                                                st.success(f"**Leg {i} - Take Profit:** {leg_side.upper()} {leg_qty} @ {price_str} ({leg_status})")
                                            elif leg_type in ['stop', 'stop_limit'] or i == 3:
                                                # For stop orders, show stop price
                                                if leg_stop:
                                                    price_display = f"${leg_stop}"
                                                elif leg_price:
                                                    price_display = f"${leg_price}"
                                                else:
                                                    price_display = "N/A"
                                                st.error(f"**Leg {i} - Stop Loss:** {leg_side.upper()} {leg_qty} @ {price_display} ({leg_status})")
                                            else:
                                                # Fallback display
                                                price_info = f"Price: ${leg_price}" if leg_price else ""
                                                stop_info = f", Stop: ${leg_stop}" if leg_stop else ""
                                                st.write(f"**Leg {i}:** {leg_type.upper()} {leg_side.upper()} {leg_qty} - {price_info}{stop_info} ({leg_status})")
                                    
                                    # Show full order details
                                    with st.expander("View Full Order JSON"):
                                        st.json(order)
                            else:
                                # Simple order
                                with st.expander(f"📝 {order_class.upper()}: {symbol} (ID: {order_id}) - {status}"):
                                    st.write(f"**Side:** {order.get('side', 'N/A')}")
                                    st.write(f"**Quantity:** {order.get('quantity', 'N/A')}")
                                    st.write(f"**Type:** {order.get('type', 'N/A')}")
                                    st.write(f"**Price:** ${order.get('price', 'N/A')}")
                                    st.write(f"**Status:** {status}")
                                    
                                    with st.expander("View Full Order JSON"):
                                        st.json(order)
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
                if st.button("🔗 Connect to IBKR", type="primary", width="stretch"):
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
                if st.button("🔌 Disconnect", width="stretch"):
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
                
                if st.button("🔄 Refresh Positions", width="stretch"):
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
                        st.dataframe(positions_df, width="stretch")
                        
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
                        st.dataframe(orders_df, width="stretch")
                        
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
                if st.button("🚀 Place Order", type="primary", width="stretch"):
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
                    if st.button("📈 Get Quote", width="stretch"):
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
                scalp_analyze_btn = st.button("⚡ Get Scalp Signal", type="primary", width="stretch")
            
            if scalp_analyze_btn and scalp_ticker:
                with st.status(f"⚡ Analyzing {scalp_ticker} for scalping...", expanded=True) as scalp_status:
                    st.write("📊 Fetching real-time data...")
                    
                    try:
                        # Get analysis with scalp trading style
                        analysis = ComprehensiveAnalyzer.analyze_stock(scalp_ticker, "SCALP")
                        
                        if analysis:
                            scalp_status.update(label=f"✅ Scalp analysis complete for {scalp_ticker}", state="complete")
                            
                            # Detect characteristics
                            is_penny = is_penny_stock(analysis.price)
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
                                if st.button(f"📋 Copy {signal} Order to Form", width="stretch"):
                                    st.session_state['scalp_prefill_symbol'] = scalp_ticker
                                    st.session_state['scalp_prefill_side'] = signal
                                    st.session_state['scalp_prefill_entry'] = entry_price
                                    st.session_state['scalp_prefill_target'] = target_price
                                    st.session_state['scalp_prefill_stop'] = stop_loss
                                    st.success("✅ Copied to order form below!")
                            
                            with action_col2:
                                if st.button("🔄 Refresh Signal", width="stretch"):
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
            
            if st.button("🧠 Generate AI Signals", type="primary", width="stretch"):
                symbols_list = [s.strip().upper() for s in ai_symbols.split(',') if s.strip()]
                
                if not symbols_list:
                    st.error("Please enter at least one symbol")
                else:
                    with st.status("🤖 AI analyzing market data...", expanded=True) as status:
                        try:
                            # Import and initialize AI signal generator
                            from services.ai_trading_signals import create_ai_signal_generator
                            
                            st.write("Initializing AI engine...")
                            ai_generator = create_ai_signal_generator(provider=ai_provider)  # noqa: F841
                            
                            # Verify AI generator is ready and immediately test functionality
                            if not ai_generator or not hasattr(ai_generator, 'batch_analyze'):
                                raise Exception("Failed to initialize AI signal generator or missing batch_analyze method")
                            
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
                                    analysis = ComprehensiveAnalyzer.analyze_stock(symbol)
                                    if not analysis:
                                        st.write(f"⚠️ Error analyzing {symbol}: Could not retrieve analysis.")
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
                            
                            # Generate signals using the AI generator
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
                                            if st.button(f"✅ Execute {signal.signal} Order", key=f"exec_{signal.symbol}_{idx}", type="primary", width="stretch"):
                                                st.session_state[f'execute_signal_{signal.symbol}'] = signal
                                                st.success(f"Ready to execute! Go to order entry below to place {signal.signal} order for {signal.symbol}")
                                        
                                        with col_exec2:
                                            if st.button(f"📋 Copy to Order Form", key=f"copy_{signal.symbol}_{idx}", width="stretch"):
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
                    if st.button("🔗 Connect to Tradier", type="primary", width="stretch"):
                        try:
                            from src.integrations.tradier_client import create_tradier_client_from_env
                            # Get current trading mode and create client
                            mode_manager = get_trading_mode_manager()
                            client = create_tradier_client_from_env()
                            if client:
                                st.session_state.tradier_client = client
                                logger.info("Tradier client connected successfully")
                                logger.info("Trading mode: %s", mode_manager.get_mode().value)
                                st.success(f"✅ Connected to Tradier ({mode_manager.get_mode().value.title()} Mode)!")
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
                if st.button("🚀 Market Order", type="primary", width="stretch", key="market_tradier"):
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
                if st.button("📊 Limit Order", width="stretch", key="limit_tradier"):
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
                if st.button("🛑 Stop Order", width="stretch", key="stop_tradier"):
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
                if st.button("🔄 Refresh", width="stretch", key="refresh_pos_tradier"):
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
                    # pandas already imported at module level
                    df_positions = pd.DataFrame(positions_data)
                    
                    # Style the dataframe
                    def color_pnl(val):
                        if 'P&L' in val.name or 'P&L %' in val.name:
                            return ['color: green' if '+' in str(x) or (isinstance(x, (int, float)) and x > 0) else 'color: red' if '-' in str(x) or (isinstance(x, (int, float)) and x < 0) else '' for x in val]
                        return ['' for _ in val]
                    
                    # Remove hidden column before display
                    display_df = df_positions.drop(columns=['_pnl_raw'])
                    st.dataframe(display_df, width="stretch", height=300)
                    
                    # Quick close buttons
                    st.write("**Quick Actions:**")
                    cols = st.columns(min(len(positions), 4))
                    
                    for idx, pos in enumerate(positions[:4]):
                        with cols[idx]:
                            symbol = pos['symbol']
                            qty = int(float(pos.get('quantity', 0)))
                            
                            if st.button(f"❌ Close {symbol}", key=f"close_{symbol}_tradier", width="stretch"):
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
                        st.dataframe(df_orders, width="stretch")
                        
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
                    if st.button("🚀 Market Order", type="primary", width="stretch", key="market_ibkr"):
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
                    if st.button("📊 Limit Order", width="stretch", key="limit_ibkr"):
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
                    if st.button("🛑 Stop Order", width="stretch", key="stop_ibkr"):
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
                    if st.button("🔄 Refresh", width="stretch", key="refresh_pos_ibkr"):
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
                        
                        # pandas already imported at module level
                        df_positions = pd.DataFrame(positions_data)
                        st.dataframe(df_positions, width="stretch", height=300)
                        
                        # Quick close
                        st.write("**Quick Actions:**")
                        cols = st.columns(min(len(positions), 4))
                        for idx, pos in enumerate(positions[:4]):
                            with cols[idx]:
                                if st.button(f"❌ Close {pos.symbol}", key=f"close_{pos.symbol}_ibkr", width="stretch"):
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
            provider = st.selectbox("LLM Provider", options=["openai", "anthropic", "google", "openrouter"], index=3, key='tab12_llm_provider_select')
            model = st.text_input("Model (leave blank for default)", value=os.getenv("AI_ANALYZER_MODEL", ""))
            api_key_input = st.text_input("API Key (optional, will override env var)", value="", type="password")
            run_btn = st.button("🔎 Run Analysis", type="primary")

        with col2:
            st.subheader("Sample Bot Configuration")
            sample_config = extract_bot_config_from_screenshot()
            st.json(sample_config)

        if run_btn:
            logger.info("--- 'Run Analysis' button clicked ---")
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
                        
                        if st.button("📊 Load into Signal Generator", type="primary", width="stretch"):
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
    
    with tab14:
        st.header("🤖 Automated Trading Bot")
        st.write("Set up automated trading that monitors your watchlist and executes high-confidence signals.")
        
        st.warning("⚠️ **IMPORTANT**: Start with Paper Trading mode to test before using real money!")
        
        # Initialize auto-trader in session state
        if 'auto_trader' not in st.session_state:
            st.session_state.auto_trader = None
        
        # ========================================================================
        # BACKGROUND TRADER CONFIGURATION MANAGER - DYNAMIC STRATEGY SYSTEM
        # ========================================================================
        
        st.divider()
        st.subheader("⚙️ Dynamic Strategy Configuration")
        st.write("Select a strategy, modify its settings, and save to its specific config file. The background trader will automatically use your selection.")
        
        # Helper functions for .env file management
        def update_env_file_for_paper_trading():
            """Update .env file to set paper trading mode"""
            try:
                env_file = '.env'
                if not os.path.exists(env_file):
                    logger.warning(f".env file not found at {env_file}")
                    return False
                
                with open(env_file, 'r') as f:
                    content = f.read()
                
                # Replace paper trading settings
                content = content.replace('IS_PAPER_TRADING=False', 'IS_PAPER_TRADING=True')
                content = content.replace('PAPER_TRADING_MODE=False', 'PAPER_TRADING_MODE=True')
                
                # Ensure settings exist if they don't
                if 'IS_PAPER_TRADING=' not in content:
                    content += '\nIS_PAPER_TRADING=True\n'
                if 'PAPER_TRADING_MODE=' not in content:
                    content += '\nPAPER_TRADING_MODE=True\n'
                
                with open(env_file, 'w') as f:
                    f.write(content)
                
                logger.info("✅ Updated .env file for paper trading")
                return True
            except Exception as e:
                logger.error(f"Error updating .env file for paper trading: {e}")
                return False
        
        def update_env_file_for_live_trading():
            """Update .env file to set live trading mode"""
            try:
                env_file = '.env'
                if not os.path.exists(env_file):
                    logger.warning(f".env file not found at {env_file}")
                    return False
                
                with open(env_file, 'r') as f:
                    content = f.read()
                
                # Replace live trading settings
                content = content.replace('IS_PAPER_TRADING=True', 'IS_PAPER_TRADING=False')
                content = content.replace('PAPER_TRADING_MODE=True', 'PAPER_TRADING_MODE=False')
                
                # Ensure settings exist if they don't
                if 'IS_PAPER_TRADING=' not in content:
                    content += '\nIS_PAPER_TRADING=False\n'
                if 'PAPER_TRADING_MODE=' not in content:
                    content += '\nPAPER_TRADING_MODE=False\n'
                
                with open(env_file, 'w') as f:
                    f.write(content)
                
                logger.info("✅ Updated .env file for live trading")
                return True
            except Exception as e:
                logger.error(f"Error updating .env file for live trading: {e}")
                return False
        
        # Load active strategy selector
        def load_active_strategy():
            """Load the currently active strategy from active_strategy.json"""
            import json  # Local import to avoid closure issues
            try:
                with open('active_strategy.json', 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                # Create default if not exists
                default_strategy = {
                    "active_strategy": "GENERAL_TRADING",
                    "config_file": "config_background_trader.py",
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "available_strategies": {
                        "WARRIOR_SCALPING": {
                            "name": "Warrior Scalping",
                            "config_file": "config_warrior_scalping.py",
                            "description": "Gap & Go strategy (9:30-10:00 AM, $2-$20 stocks, 2-20% gaps)",
                            "trading_mode": "WARRIOR_SCALPING"
                        },
                        "GENERAL_TRADING": {
                            "name": "General Trading",
                            "config_file": "config_background_trader.py",
                            "description": "Standard scalping, stocks, or options trading",
                            "trading_mode": "SCALPING"
                        },
                        "OPTIONS_PREMIUM": {
                            "name": "Options Premium Selling",
                            "config_file": "config_options_premium.py",
                            "description": "Wheel strategy, credit spreads, iron condors",
                            "trading_mode": "OPTIONS"
                        },
                        "SWING_TRADING": {
                            "name": "Swing Trading",
                            "config_file": "config_swing_trader.py",
                            "description": "Medium-term positions (1-5 days)",
                            "trading_mode": "SWING_TRADE"
                        }
                    }
                }
                with open('active_strategy.json', 'w') as f:
                    json.dump(default_strategy, f, indent=2)
                return default_strategy
        
        def save_active_strategy(strategy_key):
            """Save the active strategy selection"""
            import json  # Local import to avoid closure issues
            try:
                logger.info(f"💾 Attempting to save active strategy: {strategy_key}")
                strategy_config = load_active_strategy()
                
                if strategy_key not in strategy_config['available_strategies']:
                    logger.error(f"❌ Strategy key '{strategy_key}' not found in available strategies")
                    st.error(f"Strategy '{strategy_key}' not found!")
                    return False
                
                # Update strategy config
                strategy_config['active_strategy'] = strategy_key
                strategy_config['config_file'] = strategy_config['available_strategies'][strategy_key]['config_file']
                strategy_config['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                logger.info(f"📝 Updating active_strategy.json: strategy={strategy_key}, config_file={strategy_config['config_file']}")
                
                # Write to file with explicit flush and error handling
                file_path = 'active_strategy.json'
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(strategy_config, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    
                    # Verify the write worked
                    import time
                    time.sleep(0.1)  # Brief pause to ensure file system sync
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        verification = json.load(f)
                    
                    if verification.get('active_strategy') == strategy_key:
                        logger.info(f"✅ Successfully saved and verified active strategy: {strategy_key}")
                        logger.info(f"   Config file: {verification.get('config_file')}")
                        return True
                    else:
                        logger.error(f"❌ Verification failed! Saved '{strategy_key}' but file shows '{verification.get('active_strategy')}'")
                        st.error(f"⚠️ Saved {strategy_key} but verification failed! File may be locked.")
                        return False
                        
                except PermissionError as pe:
                    logger.error(f"❌ Permission denied writing to {file_path}: {pe}")
                    st.error(f"⚠️ Permission denied! Make sure {file_path} is not open in another program.")
                    return False
                except Exception as write_error:
                    logger.error(f"❌ Error writing to file: {write_error}", exc_info=True)
                    st.error(f"Error writing to file: {write_error}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Error saving active strategy: {e}", exc_info=True)
                st.error(f"Error saving active strategy: {e}")
            return False
        
        # Helper functions for config file management
        def load_config_file(config_filename):
            """Load settings from any config file dynamically"""
            try:
                # Remove .py extension and import dynamically
                module_name = config_filename.replace('.py', '')
                import importlib
                import sys
                
                # Reload module if already imported to get fresh data
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    cfg = sys.modules[module_name]
                else:
                    cfg = importlib.import_module(module_name)
                
                return {
                    'trading_mode': getattr(cfg, 'TRADING_MODE', 'SCALPING'),
                    'scan_interval': getattr(cfg, 'SCAN_INTERVAL_MINUTES', 15),
                    'min_confidence': getattr(cfg, 'MIN_CONFIDENCE', 70),
                    'max_daily_orders': getattr(cfg, 'MAX_DAILY_ORDERS', 10),
                    'max_position_size_pct': getattr(cfg, 'MAX_POSITION_SIZE_PCT', 15.0),
                    'use_bracket_orders': getattr(cfg, 'USE_BRACKET_ORDERS', True),
                    'scalping_take_profit_pct': getattr(cfg, 'SCALPING_TAKE_PROFIT_PCT', 2.0),
                    'scalping_stop_loss_pct': getattr(cfg, 'SCALPING_STOP_LOSS_PCT', 1.0),
                    'risk_per_trade_pct': getattr(cfg, 'RISK_PER_TRADE_PCT', 0.02),
                    'max_daily_loss_pct': getattr(cfg, 'MAX_DAILY_LOSS_PCT', 0.04),
                    'use_smart_scanner': getattr(cfg, 'USE_SMART_SCANNER', False),
                    'watchlist': getattr(cfg, 'WATCHLIST', ['SPY', 'QQQ', 'AAPL']),
                    'allow_short_selling': getattr(cfg, 'ALLOW_SHORT_SELLING', False),
                    'use_settled_funds_only': getattr(cfg, 'USE_SETTLED_FUNDS_ONLY', True),
                    # Capital Management
                    'total_capital': getattr(cfg, 'TOTAL_CAPITAL', 10000.0),
                    'reserve_cash_pct': getattr(cfg, 'RESERVE_CASH_PCT', 10.0),
                    'max_capital_utilization_pct': getattr(cfg, 'MAX_CAPITAL_UTILIZATION_PCT', 80.0),
                    # AI-Powered Hybrid Mode (NEW)
                    'use_ml_enhanced_scanner': getattr(cfg, 'USE_ML_ENHANCED_SCANNER', True),
                    'use_ai_validation': getattr(cfg, 'USE_AI_VALIDATION', True),
                    'min_ensemble_score': getattr(cfg, 'MIN_ENSEMBLE_SCORE', 70.0),
                    'min_ai_validation_confidence': getattr(cfg, 'MIN_AI_VALIDATION_CONFIDENCE', 0.7),
                    # Extra fields for special strategies
                    'config_filename': config_filename,
                }
            except Exception as e:
                st.error(f"Error loading {config_filename}: {e}")
                return None
        
        def save_config_to_file(config_dict, config_filename):
            """Save settings to any config file dynamically"""
            try:
                # Determine which template to use based on filename
                module_name = config_filename.replace('.py', '')
                # Read the template
                config_content = f'''"""
Configuration for Background Auto-Trader
Customize your trading bot settings here
Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

# ==============================================================================
# TRADING CONFIGURATION
# ==============================================================================

# Trading Mode: "SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"
TRADING_MODE = "{config_dict['trading_mode']}"

# Scan Interval (minutes)
SCAN_INTERVAL_MINUTES = {config_dict['scan_interval']}

# Minimum Confidence % (only execute signals above this)
MIN_CONFIDENCE = {config_dict['min_confidence']}

# ==============================================================================
# CAPITAL MANAGEMENT
# ==============================================================================

# Total capital allocated to auto-trading
TOTAL_CAPITAL = {config_dict['total_capital']}  # ${config_dict['total_capital']:,.0f}

# Reserve cash percentage (kept aside, not used for trading)
RESERVE_CASH_PCT = {config_dict['reserve_cash_pct']}  # {config_dict['reserve_cash_pct']}% = ${config_dict['total_capital'] * config_dict['reserve_cash_pct'] / 100:,.0f} reserved

# Maximum capital utilization (% of usable capital that can be deployed)
MAX_CAPITAL_UTILIZATION_PCT = {config_dict['max_capital_utilization_pct']}  # Max {config_dict['max_capital_utilization_pct']}% of usable capital in positions

# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================

MAX_DAILY_ORDERS = {config_dict['max_daily_orders']}
MAX_POSITION_SIZE_PCT = {config_dict['max_position_size_pct']}  # Max % per position
RISK_PER_TRADE_PCT = {config_dict['risk_per_trade_pct']}  # {config_dict['risk_per_trade_pct'] * 100:.1f}% risk per trade
MAX_DAILY_LOSS_PCT = {config_dict['max_daily_loss_pct']}  # {config_dict['max_daily_loss_pct'] * 100:.1f}% max daily loss

# Bracket Orders (Stop-Loss & Take-Profit)
USE_BRACKET_ORDERS = {config_dict['use_bracket_orders']}
SCALPING_TAKE_PROFIT_PCT = {config_dict['scalping_take_profit_pct']}
SCALPING_STOP_LOSS_PCT = {config_dict['scalping_stop_loss_pct']}

# PDT-Safe Cash Management
USE_SETTLED_FUNDS_ONLY = {config_dict['use_settled_funds_only']}
CASH_BUCKETS = 3
T_PLUS_SETTLEMENT_DAYS = 2

# ==============================================================================
# TICKER SELECTION
# ==============================================================================

# Use Smart Scanner (finds best tickers automatically)
USE_SMART_SCANNER = {config_dict['use_smart_scanner']}

# Your Custom Watchlist (used only if USE_SMART_SCANNER = False)
WATCHLIST = {config_dict['watchlist']}

# ==============================================================================
# AI-POWERED HYBRID MODE (1-2 KNOCKOUT COMBO) 🥊
# ==============================================================================

# Enable ML-Enhanced Scanner for triple validation (40% ML + 35% LLM + 25% Quant)
USE_ML_ENHANCED_SCANNER = {config_dict.get('use_ml_enhanced_scanner', True)}  # RECOMMENDED: Superior trade quality

# Enable AI Pre-Trade Validation (final risk check before execution)
USE_AI_VALIDATION = {config_dict.get('use_ai_validation', True)}  # RECOMMENDED: Blocks high-risk trades

# Minimum ensemble score for ML-Enhanced Scanner (0-100)
MIN_ENSEMBLE_SCORE = {config_dict.get('min_ensemble_score', 70.0)}  # Only trades passing all 3 systems with 70%+ score

# Minimum AI validation confidence (0-1.0)
MIN_AI_VALIDATION_CONFIDENCE = {config_dict.get('min_ai_validation_confidence', 0.7)}  # AI must be 70%+ confident to approve

# NOTE: When both are enabled, you get the 1-2 KNOCKOUT COMBO:
#   PUNCH 1: ML-Enhanced Scanner filters trades (triple validation)
#   PUNCH 2: AI Validator performs final risk check
#   Result: Only the highest quality, lowest risk trades execute!

# ==============================================================================
# ADVANCED OPTIONS
# ==============================================================================

# Short Selling (ONLY works in paper trading)
ALLOW_SHORT_SELLING = {config_dict['allow_short_selling']}

# Multi-Agent System
USE_AGENT_SYSTEM = False
'''
                
                with open(config_filename, 'w', encoding='utf-8') as f:
                    f.write(config_content)
                return True
            except Exception as e:
                st.error(f"Error saving config: {e}")
                return False
        
        # ========================================================================
        # STRATEGY SELECTOR
        # ========================================================================
        
        st.markdown("""
        ### 📋 Configuration Workflow
        
        <div style="background: linear-gradient(90deg, #1f77b4, #ff7f0e, #2ca02c); padding: 2px; border-radius: 10px; margin: 10px 0;">
            <div style="background: white; padding: 20px; border-radius: 8px;">
                <h4 style="margin: 0 0 10px 0;">Simple 3-Step Process:</h4>
                <p style="margin: 5px 0;"><strong>1️⃣ SELECT</strong> → Choose which strategy to configure (dropdown loads its settings)</p>
                <p style="margin: 5px 0;"><strong>2️⃣ EDIT</strong> → Modify settings in tabs below (changes are temporary)</p>
                <p style="margin: 5px 0;"><strong>3️⃣ SAVE</strong> → Write changes to config file (permanent)</p>
                <hr style="margin: 10px 0;">
                <p style="margin: 5px 0; color: #ff6b6b;"><strong>⚠️ To Use Saved Settings:</strong> Click "Activate" (if not active), then restart background trader</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Load strategy config
        active_strategy_data = load_active_strategy()
        available_strategies = active_strategy_data['available_strategies']
        current_active_strategy = active_strategy_data['active_strategy']
        
        # Create strategy selector with better visual hierarchy
        st.subheader("🎯 Step 1: Select Strategy to Configure")
        
        col_strategy, col_status, col_action = st.columns([3, 1, 1])
        
        with col_strategy:
            strategy_options = {k: v['name'] for k, v in available_strategies.items()}
            strategy_descriptions = {k: v['description'] for k, v in available_strategies.items()}
            
            selected_strategy = st.selectbox(
                "Choose which strategy's settings to edit:",
                options=list(strategy_options.keys()),
                index=list(strategy_options.keys()).index(current_active_strategy) if current_active_strategy in strategy_options else 0,
                format_func=lambda x: f"{strategy_options[x]}",
                help="Selecting a strategy will LOAD its current settings below for editing",
                key="strategy_selector",
                label_visibility="collapsed"
            )
        
        with col_status:
            st.write("")  # Spacing
            if selected_strategy == current_active_strategy:
                st.success("✅ **ACTIVE**")
            else:
                st.warning("📝 **Editing**")
        
        with col_action:
            st.write("")  # Spacing
            if selected_strategy != current_active_strategy:
                if st.button("🎯 Activate", help="Make this the active strategy for background trader"):
                    if save_active_strategy(selected_strategy):
                        st.success("✅ Activated!")
                        # Show what will be loaded
                        config_file = available_strategies[selected_strategy]['config_file']
                        st.info(f"📁 Background trader will load: `{config_file}`")
                        st.info(f"🎯 Trading Mode: {available_strategies[selected_strategy].get('trading_mode', 'UNKNOWN')}")
                        st.success("🔄 **Config change detected!** Background trader will auto-restart within 60 seconds.")
                        st.info("💡 If using `start_autotrader_auto_restart.bat`, it will restart automatically. Otherwise, manually restart.")
                        
                        # Verify file was actually updated
                        try:
                            import json
                            with open('active_strategy.json', 'r') as f:
                                verify = json.load(f)
                            if verify.get('active_strategy') == selected_strategy:
                                st.success(f"✅ Verified: File updated correctly to `{verify.get('active_strategy')}`")
                            else:
                                st.error(f"❌ Mismatch! File shows `{verify.get('active_strategy')}` but expected `{selected_strategy}`")
                        except Exception as e:
                            st.warning(f"⚠️ Could not verify file: {e}")
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Failed to save! Check logs for details.")
        
        # Show active strategy info with better formatting
        st.markdown("---")
        
        # CRITICAL: Show what background trader will ACTUALLY load
        try:
            import json
            with open('active_strategy.json', 'r') as f:
                file_content = json.load(f)
            file_strategy = file_content.get('active_strategy', 'UNKNOWN')
            file_config = file_content.get('config_file', 'UNKNOWN')
            
            st.markdown("### 🔍 Background Trader Configuration")
            if file_strategy == current_active_strategy:
                st.success(f"✅ **Active Strategy:** `{file_strategy}` → Config: `{file_config}`")
            else:
                st.error(f"⚠️ **MISMATCH DETECTED!**")
                st.error(f"- Streamlit shows: `{current_active_strategy}`")
                st.error(f"- File contains: `{file_strategy}` → Config: `{file_config}`")
                st.warning("🚨 Background trader will use what's in the FILE, not what Streamlit shows!")
                st.info("💡 Click '🎯 Activate' to sync Streamlit with the file")
        except Exception as e:
            st.warning(f"⚠️ Could not read active_strategy.json: {e}")
        selected_config_file = available_strategies[selected_strategy]['config_file']
        
        st.info(f"""
        **📝 Now Editing:** `{strategy_options[selected_strategy]}`  
        **📁 Config File:** `{selected_config_file}`  
        **📖 Description:** {strategy_descriptions[selected_strategy]}
        
        ℹ️ **Settings below are loaded from this config file. Changes are NOT saved until you click "💾 Save Configuration" at the bottom.**
        """)
        
        st.divider()
        
        st.subheader("🎯 Step 2: Edit Settings Below")
        st.caption("Make your changes in the tabs below. Settings are loaded from the selected strategy's config file.")
        
        # Load config for selected strategy
        current_config = load_config_file(selected_config_file)
        
        if current_config:
            st.success(f"✅ Loaded configuration from `{selected_config_file}`")
            
            # Create tabs for organization
            cfg_tab1, cfg_tab2, cfg_tab3 = st.tabs([
                "📊 Strategy & Tickers", 
                "⚖️ Risk & AI Settings", 
                "💾 Step 3: Save"
            ])
            
            with cfg_tab1:
                st.subheader("Strategy Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    trading_mode = st.selectbox(
                        "Trading Mode",
                        options=["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"],
                        index=["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"].index(current_config.get('trading_mode', 'SCALPING')) if current_config.get('trading_mode', 'SCALPING') in ["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"] else 0,
                        help="SCALPING: Fast intraday | WARRIOR_SCALPING: Gap & Go (9:30-10:00 AM) | STOCKS: Swing trades | OPTIONS: Options trading"
                    )
                    
                    scan_interval = st.slider(
                        "Scan Interval (minutes)",
                        min_value=5,
                        max_value=60,
                        value=int(current_config['scan_interval']),
                        step=5,
                        help="How often to scan for new opportunities"
                    )
                
                with col2:
                    min_confidence = st.slider(
                        "Minimum Confidence %",
                        min_value=60,
                        max_value=95,
                        value=int(current_config['min_confidence']),
                        step=5,
                        help="Only execute signals above this confidence level"
                    )
                    
                    use_bracket_orders = st.checkbox(
                        "Use Bracket Orders (Stop-Loss + Take-Profit)",
                        value=current_config['use_bracket_orders'],
                        help="Automatically set protective orders"
                    )
                
                st.divider()
                st.subheader("Ticker Selection")
                
                use_smart_scanner = st.checkbox(
                    "🧠 Use Smart Scanner (Auto-discover best tickers)",
                    value=current_config.get('use_smart_scanner', True),  # Default to True if config missing
                    help="When enabled, ignores watchlist and automatically finds opportunities"
                )
                
                # Get checked tickers from the watchlist section
                def get_selected_tickers_from_ui():
                    """Get tickers that are checked in the main watchlist"""
                    selected = []
                    try:
                        ticker_mgr = TickerManager()
                        all_tickers = ticker_mgr.get_all_tickers()
                        if all_tickers:
                            for t in all_tickers:
                                ticker = t['ticker']
                                checkbox_key = f"auto_trade_{ticker}"
                                if checkbox_key in st.session_state and st.session_state[checkbox_key]:
                                    selected.append(ticker)
                    except Exception:
                        pass
                    return selected
                
                # Add sync button
                col_sync1, col_sync2 = st.columns([3, 1])
                with col_sync1:
                    st.write("**Quick Actions:**")
                with col_sync2:
                    if st.button("📋 Copy Checked Tickers", help="Copy tickers you checked in the Watchlist section below"):
                        checked_tickers = get_selected_tickers_from_ui()
                        if checked_tickers:
                            # Update both session state keys to ensure text area updates
                            st.session_state['synced_watchlist'] = ", ".join(checked_tickers)
                            st.session_state['watchlist_text_area'] = ", ".join(checked_tickers)
                            st.success(f"✅ Copied {len(checked_tickers)} tickers!")
                            st.info(f"📋 **Tickers ready to save:** {', '.join(checked_tickers[:10])}{'...' if len(checked_tickers) > 10 else ''}")
                            st.rerun()
                        else:
                            st.warning("⚠️ No tickers checked in Watchlist section below. Scroll down and check some first!")
                
                # Use synced watchlist if available
                if 'synced_watchlist' in st.session_state:
                    default_watchlist = st.session_state['synced_watchlist']
                    # Clear the synced state so it doesn't persist forever
                    if st.session_state.get('clear_sync', False):
                        del st.session_state['synced_watchlist']
                        st.session_state['clear_sync'] = False
                else:
                    default_watchlist = ", ".join(current_config['watchlist'])
                
                if not use_smart_scanner:
                    st.info("💡 Smart Scanner disabled - will use your custom watchlist below")
                    watchlist_str = st.text_area(
                        "Custom Watchlist (comma-separated)",
                        value=default_watchlist,
                        help="Enter tickers separated by commas. Example: TSLA, NVDA, AMD, AAPL\nTip: Use '📋 Copy Checked Tickers' to auto-fill from your checked tickers below!",
                        height=100,
                        key="watchlist_text_area"
                    )
                else:
                    st.warning("⚠️ Smart Scanner enabled - watchlist below will be IGNORED")
                    watchlist_str = st.text_area(
                        "Custom Watchlist (not used when Smart Scanner enabled)",
                        value=default_watchlist,
                        help="These tickers are ignored while Smart Scanner is enabled",
                        height=100,
                        disabled=True,
                        key="watchlist_text_area_disabled"
                    )
            
            with cfg_tab2:
                st.subheader("💰 Capital Management")
                st.markdown("_Control how much capital the bot can use for trading_")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    total_capital = st.number_input(
                        "Total Capital Allocated to Bot",
                        min_value=100.0,
                        max_value=1000000.0,
                        value=float(current_config.get('total_capital', 10000.0)),
                        step=100.0,
                        help="💵 Total account balance or capital allocated for auto-trading"
                    )
                    
                    reserve_cash_pct = st.slider(
                        "Reserve Cash %",
                        min_value=0.0,
                        max_value=50.0,
                        value=float(current_config.get('reserve_cash_pct', 10.0)),
                        step=5.0,
                        help="💰 Percentage kept aside, not used for trading (emergency cash)"
                    )
                    
                    st.info(f"**Usable Capital:** ${total_capital * (1 - reserve_cash_pct/100):,.2f}")
                
                with col2:
                    max_capital_utilization_pct = st.slider(
                        "Max Capital Utilization %",
                        min_value=20.0,
                        max_value=100.0,
                        value=float(current_config.get('max_capital_utilization_pct', 80.0)),
                        step=5.0,
                        help="📊 Maximum % of usable capital that can be deployed in positions"
                    )
                    
                    usable_capital = total_capital * (1 - reserve_cash_pct/100)
                    max_deployed = usable_capital * (max_capital_utilization_pct/100)
                    st.info(f"**Max Deployed:** ${max_deployed:,.2f}")
                    st.info(f"**Always Available:** ${usable_capital - max_deployed:,.2f}")
                
                st.divider()
                st.subheader("⚖️ Risk Management")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    max_daily_orders = st.number_input(
                        "Max Daily Orders",
                        min_value=1,
                        max_value=50,
                        value=int(current_config['max_daily_orders']),
                        help="Maximum number of trades per day"
                    )
                    
                    max_position_size_pct = st.slider(
                        "Max Position Size %",
                        min_value=1.0,
                        max_value=50.0,
                        value=float(current_config['max_position_size_pct']),
                        step=1.0,
                        help="Maximum % of total capital per single trade"
                    )
                    
                    risk_per_trade_pct = st.slider(
                        "Risk Per Trade %",
                        min_value=0.5,
                        max_value=5.0,
                        value=float(current_config['risk_per_trade_pct'] * 100),
                        step=0.5,
                        help="Risk % of account per trade"
                    ) / 100.0
                
                with col4:
                    max_daily_loss_pct = st.slider(
                        "Max Daily Loss %",
                        min_value=1.0,
                        max_value=10.0,
                        value=float(current_config['max_daily_loss_pct'] * 100),
                        step=0.5,
                        help="Stop trading if down this % in a day"
                    ) / 100.0
                    
                    scalping_take_profit_pct = st.slider(
                        "Take-Profit % (Scalping)",
                        min_value=0.5,
                        max_value=10.0,
                        value=float(current_config['scalping_take_profit_pct']),
                        step=0.5,
                        help="Target profit % for scalping mode"
                    )
                    
                    scalping_stop_loss_pct = st.slider(
                        "Stop-Loss % (Scalping)",
                        min_value=0.25,
                        max_value=5.0,
                        value=float(current_config['scalping_stop_loss_pct']),
                        step=0.25,
                        help="Stop loss % for scalping mode"
                    )
                
                st.divider()
                st.subheader("Advanced Options")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    use_settled_funds_only = st.checkbox(
                        "PDT-Safe: Use Settled Funds Only",
                        value=current_config['use_settled_funds_only'],
                        help="Avoids Pattern Day Trader restrictions"
                    )
                
                with col4:
                    allow_short_selling = st.checkbox(
                        "Allow Short Selling (Paper Only)",
                        value=current_config['allow_short_selling'],
                        help="⚠️ Advanced: Enable short selling in paper trading"
                    )
                
                st.divider()
                st.subheader("🥊 AI-Powered Hybrid Mode (1-2 KNOCKOUT COMBO)")
                st.markdown("""
                **The ultimate trade quality system** - Only the best trades survive double validation!
                
                - **PUNCH 1**: ML-Enhanced Scanner (40% ML + 35% LLM + 25% Quant)
                - **PUNCH 2**: AI Pre-Trade Validation (final risk check)
                - **Result**: Maximum trade quality + risk control
                """)
                
                col_ai1, col_ai2 = st.columns(2)
                
                with col_ai1:
                    use_ml_enhanced_scanner = st.checkbox(
                        "🧠 Enable ML-Enhanced Scanner (PUNCH 1)",
                        value=current_config.get('use_ml_enhanced_scanner', True),
                        help="Triple validation: 40% ML + 35% LLM + 25% Quantitative analysis"
                    )
                    
                    if use_ml_enhanced_scanner:
                        min_ensemble_score = st.slider(
                            "Min Ensemble Score %",
                            min_value=50,
                            max_value=95,
                            value=int(current_config.get('min_ensemble_score', 70)),
                            step=5,
                            help="Minimum combined score from ML+LLM+Quant (70%+ recommended)"
                        )
                    else:
                        min_ensemble_score = 70.0
                
                with col_ai2:
                    use_ai_validation = st.checkbox(
                        "🛡️ Enable AI Pre-Trade Validation (PUNCH 2)",
                        value=current_config.get('use_ai_validation', True),
                        help="LLM validates risk/reward, portfolio fit, and red flags before execution"
                    )
                    
                    if use_ai_validation:
                        min_ai_validation_confidence = st.slider(
                            "Min AI Validation Confidence",
                            min_value=0.5,
                            max_value=0.95,
                            value=float(current_config.get('min_ai_validation_confidence', 0.7)),
                            step=0.05,
                            format="%.2f",
                            help="Minimum confidence for AI to approve trade (0.7+ recommended)"
                        )
                    else:
                        min_ai_validation_confidence = 0.7
                
                if use_ml_enhanced_scanner and use_ai_validation:
                    st.success("🥊 **KNOCKOUT COMBO ACTIVE!** Maximum trade quality & risk control enabled.")
                elif use_ml_enhanced_scanner:
                    st.info("🧠 ML-Enhanced Scanner active. Enable AI Validation for full knockout combo!")
                elif use_ai_validation:
                    st.info("🛡️ AI Validation active. Enable ML-Enhanced Scanner for full knockout combo!")
                else:
                    st.warning("⚠️ AI features disabled. Enable for superior trade quality!")
            
            with cfg_tab3:
                st.subheader("🎯 Step 3: Save Configuration")
                
                st.markdown(f"""
                ### 💾 Ready to Save?
                
                You are editing: **`{strategy_options[selected_strategy]}`**  
                Config file: **`{selected_config_file}`**
                
                **What happens when you click "Save":**
                1. ✅ Your changes are written to **`{selected_config_file}`**
                2. ✅ The config file is permanently updated
                3. ⚠️ Changes take effect only after you **restart the background trader**
                
                **To apply saved changes:**
                ```powershell
                # Stop trader
                .\\stop_autotrader.bat
                
                # Start trader (loads the updated config)
                .\\start_autotrader_background.bat
                ```
                
                💡 **Tip:** You can edit multiple strategies, save them all, then activate the one you want to use.
                """)
                
                st.divider()
                
                # Show what will be saved
                with st.expander("👁️ Preview Configuration"):
                    preview_config = {
                        'Trading Mode': trading_mode,
                        'Scan Interval': f"{scan_interval} minutes",
                        'Min Confidence': f"{min_confidence}%",
                        'Smart Scanner': "Enabled" if use_smart_scanner else "Disabled",
                        'Watchlist': watchlist_str if not use_smart_scanner else "(Using Smart Scanner)",
                        '--- Capital Management ---': '---',
                        'Total Capital': f"${total_capital:,.2f}",
                        'Reserve Cash': f"{reserve_cash_pct}% (${total_capital * reserve_cash_pct / 100:,.2f})",
                        'Usable Capital': f"${total_capital * (1 - reserve_cash_pct/100):,.2f}",
                        'Max Capital Utilization': f"{max_capital_utilization_pct}%",
                        'Max Deployed Capital': f"${total_capital * (1 - reserve_cash_pct/100) * max_capital_utilization_pct / 100:,.2f}",
                        '--- Risk Management ---': '---',
                        'Max Daily Orders': max_daily_orders,
                        'Max Position Size': f"{max_position_size_pct}% (${total_capital * max_position_size_pct / 100:,.2f})",
                        'Risk Per Trade': f"{risk_per_trade_pct * 100:.1f}%",
                        'Max Daily Loss': f"{max_daily_loss_pct * 100:.1f}%",
                        'Use Bracket Orders': "Yes" if use_bracket_orders else "No",
                        'Take-Profit': f"{scalping_take_profit_pct}%",
                        'Stop-Loss': f"{scalping_stop_loss_pct}%",
                        '--- AI-Powered Hybrid Mode (1-2 KNOCKOUT COMBO) ---': '🥊',
                        'ML-Enhanced Scanner (PUNCH 1)': "✅ Enabled" if use_ml_enhanced_scanner else "❌ Disabled",
                        'Min Ensemble Score': f"{min_ensemble_score}%" if use_ml_enhanced_scanner else "N/A",
                        'AI Pre-Trade Validation (PUNCH 2)': "✅ Enabled" if use_ai_validation else "❌ Disabled",
                        'Min AI Validation Confidence': f"{min_ai_validation_confidence:.2f}" if use_ai_validation else "N/A",
                        'Knockout Combo Status': "🥊 ACTIVE - Maximum Quality!" if (use_ml_enhanced_scanner and use_ai_validation) else "⚠️ Partial" if (use_ml_enhanced_scanner or use_ai_validation) else "❌ Disabled",
                    }
                    st.json(preview_config)
                
                st.divider()
                
                # Save button
                if st.button("💾 Save Configuration to File", type="primary", width="stretch"):
                    # Parse watchlist
                    if not use_smart_scanner:
                        watchlist_tickers = [t.strip().upper() for t in watchlist_str.split(',') if t.strip()]
                        if not watchlist_tickers:
                            st.error("❌ Watchlist cannot be empty when Smart Scanner is disabled!")
                            st.stop()
                    else:
                        watchlist_tickers = [t.strip().upper() for t in watchlist_str.split(',') if t.strip()]
                        if not watchlist_tickers:
                            watchlist_tickers = ['SPY', 'QQQ', 'AAPL']  # Fallback
                    
                    # Prepare config dict
                    new_config = {
                        'trading_mode': trading_mode,
                        'scan_interval': scan_interval,
                        'min_confidence': min_confidence,
                        'max_daily_orders': max_daily_orders,
                        'max_position_size_pct': max_position_size_pct,
                        'use_bracket_orders': use_bracket_orders,
                        'scalping_take_profit_pct': scalping_take_profit_pct,
                        'scalping_stop_loss_pct': scalping_stop_loss_pct,
                        'risk_per_trade_pct': risk_per_trade_pct,
                        'max_daily_loss_pct': max_daily_loss_pct,
                        'use_smart_scanner': use_smart_scanner,
                        'watchlist': watchlist_tickers,
                        'allow_short_selling': allow_short_selling,
                        'use_settled_funds_only': use_settled_funds_only,
                        # Capital Management (NEW)
                        'total_capital': total_capital,
                        'reserve_cash_pct': reserve_cash_pct,
                        'max_capital_utilization_pct': max_capital_utilization_pct,
                        # AI-Powered Hybrid Mode (NEW)
                        'use_ml_enhanced_scanner': use_ml_enhanced_scanner,
                        'use_ai_validation': use_ai_validation,
                        'min_ensemble_score': min_ensemble_score,
                        'min_ai_validation_confidence': min_ai_validation_confidence,
                    }
                    
                    # Save to file
                    if save_config_to_file(new_config, selected_config_file):
                        st.success(f"✅ Configuration saved successfully to `{selected_config_file}`!")
                        
                        st.warning("""
                        **⚠️ RESTART REQUIRED**
                        
                        To apply these changes, restart the background trader:
                        
                        **Windows PowerShell:**
                        ```powershell
                        .\\stop_autotrader.bat
                        .\\start_autotrader_background.bat
                        ```
                        
                        **Or manually:**
                        ```powershell
                        Stop-Process -Name pythonw -Force
                        Start-Process pythonw -ArgumentList "run_autotrader_background.py" -WorkingDirectory "C:\\Users\\seaso\\Sentient Trader"
                        ```
                        
                        **Verify new settings in logs:**
                        ```powershell
                        Get-Content logs\\autotrader_background.log -Tail 20
                        ```
                        """)
                        
                        # Clear synced watchlist after saving
                        if 'synced_watchlist' in st.session_state:
                            del st.session_state['synced_watchlist']
                        
                        # Force page reload to show updated config
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Failed to save configuration. Check file permissions.")
        else:
            st.warning(f"""
            ⚠️ Configuration file not found!
            
            The file `{selected_config_file}` doesn't exist yet.
            
            **To create it:**
            1. Use the configuration below to set your preferences
            2. Click "Save Configuration" 
            3. File will be created automatically
            
            **Alternatively:**
            - Select a different strategy that has an existing config file
            - Or copy an existing config file and rename it to `{selected_config_file}`
            """)
            
            # Show default form for creating new config
            st.info("📝 Using default settings. Customize below and save to create the config file.")
            
            trading_mode = st.selectbox("Trading Mode", ["SCALPING", "WARRIOR_SCALPING", "STOCKS", "OPTIONS", "ALL"], index=0, help="SCALPING: Fast intraday | WARRIOR_SCALPING: Gap & Go (9:30-10:00 AM)")
            scan_interval = st.slider("Scan Interval (minutes)", 5, 60, 15, 5)
            min_confidence = st.slider("Min Confidence %", 60, 95, 75, 5)
            use_smart_scanner = st.checkbox("Use Smart Scanner", value=True)
            watchlist_str = st.text_area("Watchlist", value="SPY, QQQ, AAPL, TSLA, NVDA")
            
            if st.button("💾 Create Configuration File"):
                watchlist_tickers = [t.strip().upper() for t in watchlist_str.split(',') if t.strip()]
                new_config = {
                    'trading_mode': trading_mode,
                    'scan_interval': scan_interval,
                    'min_confidence': min_confidence,
                    'max_daily_orders': 10,
                    'max_position_size_pct': 20.0,
                    'use_bracket_orders': True,
                    'scalping_take_profit_pct': 2.0,
                    'scalping_stop_loss_pct': 1.0,
                    'risk_per_trade_pct': 0.02,
                    'max_daily_loss_pct': 0.04,
                    'use_smart_scanner': use_smart_scanner,
                    'watchlist': watchlist_tickers,
                    'allow_short_selling': False,
                    'use_settled_funds_only': True,
                }
                if save_background_config(new_config):
                    st.success("✅ Configuration file created!")
                    st.rerun()
        
        st.divider()
        
        # ========================================================================
        # END BACKGROUND TRADER CONFIGURATION MANAGER
        # ========================================================================
        
        # Configuration section
        st.subheader("⚙️ Configuration")
        
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
        
        with col_cfg1:
            scan_interval = st.number_input(
                "Scan Interval (minutes)",
                min_value=5,
                max_value=60,
                value=15,
                help="How often to scan for new signals"
            )
            min_confidence = st.slider(
                "Min Confidence %",
                min_value=60,
                max_value=95,
                value=75,
                help="Only execute signals above this confidence"
            )
        
        with col_cfg2:
            max_daily_orders = st.number_input(
                "Max Daily Orders",
                min_value=1,
                max_value=50,
                value=10,
                help="Maximum orders per day"
            )
            use_bracket_orders = st.checkbox(
                "Use Bracket Orders",
                value=True,
                help="Automatically set stop-loss and take-profit"
            )
        
        with col_cfg3:
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                options=["LOW", "MEDIUM", "HIGH"],
                index=1
            )
            paper_trading = st.checkbox(
                "Paper Trading Mode",
                value=True,
                help="HIGHLY RECOMMENDED: Test with paper trading first"
            )
            allow_short_selling = st.checkbox(
                "Allow Short Selling (Advanced)",
                value=False,
                help="⚠️ Enable short selling for SELL signals. NOT recommended for scalping or cash accounts. Only for margin accounts and advanced strategies.",
                disabled=not paper_trading
            )
            test_mode = st.checkbox(
                "🧪 Test Mode (Bypass Market Hours)",
                value=False,
                help="⚠️ TESTING ONLY: Allows trading when market is closed. Use this to test your scalping setup while the market is closed. Make sure you're in Paper Trading mode!"
            )
        
        if test_mode:
            st.warning("""
            🧪 **Test Mode Enabled**
            
            - ✅ Market hours check is **DISABLED** - you can test even when the market is closed
            - ⚠️ **FOR TESTING ONLY** - Only use this when testing your setup
            - ✅ Make sure **Paper Trading** is enabled to avoid real trades
            - 📝 The scalper will run and scan for signals even outside market hours
            """)
        
        # Smart Scanner option
        st.divider()
        use_smart_scanner = st.checkbox(
            "🧠 Use Smart Scanner (Advanced)",
            value=True,  # Default to True - automatically finds opportunities
            help="IGNORES your ticker selections and automatically finds the best tickers using the Advanced Scanner. Leave unchecked to only scan YOUR selected tickers."
        )
        
        if use_smart_scanner:
            st.warning("""
            ⚠️ **Smart Scanner Mode:**
            - **IGNORES** your ticker checkboxes below
            - Automatically scans 24-33 curated tickers based on strategy
            - Uses Advanced Scanner to find top opportunities
            - Updates dynamically each scan cycle
            
            **Strategy Mapping:**
            - SCALPING → Scans 24 high-volume tickers, returns top 10
            - STOCKS → Scans 33 swing trade candidates, returns top 15
            - OPTIONS → Scans 24 high IV tickers, returns top 15
            - ALL → Scans 24 mixed tickers, returns top 20
            
            💡 **Recommended if:** You want automated ticker discovery
            ❌ **Not recommended if:** You want to control which tickers to trade
            """)
        else:
            st.success("✅ Using YOUR selected tickers below (manual control)")
        
        # Trading mode selection
        st.subheader("📈 Trading Mode")
        col_mode1, col_mode2 = st.columns(2)
        
        with col_mode1:
            trading_mode = st.selectbox(
                "Strategy Type",
                options=["STOCKS", "OPTIONS", "SCALPING", "WARRIOR_SCALPING", "ALL"],
                index=2,  # Default to SCALPING
                help="SCALPING: Fast intraday trades | WARRIOR_SCALPING: Gap & Go strategy (9:30-10:00 AM)"
            )
        
        with col_mode2:
            if trading_mode == "SCALPING":
                scalp_take_profit = st.number_input(
                    "Scalp Take Profit %",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                    help="Target profit percentage for scalping"
                )
                scalp_stop_loss = st.number_input(
                    "Scalp Stop Loss %",
                    min_value=0.5,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    help="Stop loss percentage for scalping"
                )
            else:
                scalp_take_profit = 2.0
                scalp_stop_loss = 1.0
        
        st.divider()
        
        # Watchlist selection
        st.subheader("📋 Watchlist")
        st.write("Select tickers to monitor for automated trading:")
        
        # Get tickers from database
        try:
            ticker_mgr = TickerManager()
            all_tickers = ticker_mgr.get_all_tickers()
            ticker_symbols = [t['ticker'] for t in all_tickers] if all_tickers else []
        except Exception:
            ticker_symbols = []
        
        if ticker_symbols:
            st.write("**Enable/Disable Auto-Trading Per Ticker:**")
            
            # Show checkboxes for each ticker
            selected_tickers = []
            cols_per_row = 4
            ticker_rows = [ticker_symbols[i:i+cols_per_row] for i in range(0, len(ticker_symbols), cols_per_row)]
            
            # Track if we need to show migration warning
            needs_migration = False
            
            for row in ticker_rows:
                cols = st.columns(cols_per_row)
                for idx, ticker in enumerate(row):
                    with cols[idx]:
                        # Get current auto-trade status
                        ticker_data = ticker_mgr.get_ticker(ticker)
                        current_enabled = ticker_data.get('auto_trade_enabled', False) if ticker_data else False
                        
                        # Checkbox for enabling auto-trade
                        enabled = st.checkbox(
                            f"✅ {ticker}",
                            value=current_enabled,
                            key=f"auto_trade_{ticker}",
                            help=f"Enable auto-trading for {ticker}"
                        )
                        
                        if enabled:
                            selected_tickers.append(ticker)
                            # Update database if changed
                            if enabled != current_enabled:
                                success = ticker_mgr.set_auto_trade(ticker, enabled, trading_mode)
                                if not success:
                                    needs_migration = True
            
            if needs_migration:
                st.error("⚠️ **Database Migration Required**")
                st.warning("The auto-trade columns are missing from your database. Please run this SQL in your Supabase SQL Editor:")
                st.code("""
ALTER TABLE saved_tickers 
ADD COLUMN IF NOT EXISTS auto_trade_enabled BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS auto_trade_strategy TEXT;
                """, language="sql")
                st.info("📁 Full migration script available at: `migrations/add_auto_trade_columns.sql`")
            
            if not selected_tickers:
                st.info("👆 Check the boxes above to enable auto-trading for specific tickers")
        else:
            st.warning("No tickers in your watchlist. Add some in the '⭐ My Tickers' tab first!")
            selected_tickers = []
        
        st.divider()
        
        # Control buttons
        st.subheader("🎮 Controls")
        
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🚀 Start Auto-Trader", type="primary", disabled=len(selected_tickers) == 0):
                if not st.session_state.tradier_client:
                    st.error("❌ Tradier not connected! Go to 🏦 Tradier Account tab to connect.")
                else:
                    try:
                        from services.auto_trader import create_auto_trader, AutoTraderConfig
                        from services.ai_trading_signals import create_ai_signal_generator
                        
                        # Create config
                        config = AutoTraderConfig(
                            enabled=True,
                            scan_interval_minutes=scan_interval,
                            min_confidence=min_confidence,
                            max_daily_orders=max_daily_orders,
                            use_bracket_orders=use_bracket_orders,
                            risk_tolerance=risk_tolerance,
                            paper_trading=paper_trading,
                            trading_mode=trading_mode,
                            scalping_take_profit_pct=scalp_take_profit,
                            scalping_stop_loss_pct=scalp_stop_loss,
                            allow_short_selling=allow_short_selling if paper_trading else False,
                            test_mode=test_mode
                        )
                        
                        # Create signal generator
                        signal_gen = create_ai_signal_generator()
                        
                        # Create and start auto-trader
                        auto_trader = create_auto_trader(
                            tradier_client=st.session_state.tradier_client,
                            signal_generator=signal_gen,
                            watchlist=selected_tickers,
                            config=config,
                            use_smart_scanner=use_smart_scanner
                        )
                        
                        auto_trader.start()
                        st.session_state.auto_trader = auto_trader
                        
                        st.success("✅ Auto-Trader started successfully!")
                        if test_mode:
                            st.warning("🧪 Test Mode enabled: Market hours check is bypassed. You can test while the market is closed.")
                        if use_smart_scanner:
                            st.info(f"🧠 Smart Scanner enabled: Will dynamically find top tickers for {trading_mode} strategy each scan")
                        else:
                            st.info(f"Monitoring {len(selected_tickers)} tickers: {', '.join(selected_tickers)}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to start Auto-Trader: {e}")
                        logger.error(f"Auto-trader start error: {e}", exc_info=True)
        
        with col_btn2:
            if st.button("🛑 Stop Auto-Trader", disabled=st.session_state.auto_trader is None):
                if st.session_state.auto_trader:
                    st.session_state.auto_trader.stop()
                    st.session_state.auto_trader = None
                    st.success("Auto-Trader stopped")
                    st.rerun()
        
        with col_btn3:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        st.divider()
        
        # Check for background auto-trader
        def check_background_trader():
            """Check if background auto-trader is running"""
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if proc.info['name'] in ['pythonw.exe', 'python.exe']:
                            cmdline = proc.info.get('cmdline', [])
                            if cmdline and any('run_autotrader_background' in str(cmd) for cmd in cmdline):
                                return True, proc.info['pid']
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except ImportError:
                pass  # psutil not installed
            return False, None
        
        bg_running, bg_pid = check_background_trader()
        
        if bg_running:
            st.info(f"""
            🟢 **Background Auto-Trader Detected**
            
            A background auto-trader is currently running (PID: {bg_pid})
            
            ⚠️ **IMPORTANT**: Don't start another auto-trader here to avoid duplicate trades!
            
            📊 **Monitor it via:**
            - Logs: `logs/autotrader_background.log`
            - State: `data/trade_state.json`
            - Command: `Get-Content logs\\autotrader_background.log -Tail 50 -Wait`
            
            🛑 **To stop it:** Run `stop_autotrader.bat` or kill process {bg_pid}
            """)
        
        # Status display
        st.subheader("📊 Status")
        
        if st.session_state.auto_trader:
            status = st.session_state.auto_trader.get_status()
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                status_icon = "🟢" if status['is_running'] else "🔴"
                st.metric("Status", f"{status_icon} {'Running' if status['is_running'] else 'Stopped'}")
            
            with col_stat2:
                st.metric("Daily Orders", f"{status['daily_orders']}/{status['max_daily_orders']}")
            
            with col_stat3:
                st.metric("Watchlist Size", status['watchlist_size'])
            
            with col_stat4:
                if status['config'].get('test_mode', False):
                    hours_status = "🧪 Test Mode"
                else:
                    hours_status = "✅ Yes" if status['in_trading_hours'] else "❌ No"
                st.metric("Trading Hours", hours_status)
            
            # Short positions display (if enabled)
            if status.get('short_positions', 0) > 0:
                st.divider()
                st.subheader("📉 Active Short Positions")
                short_details = status.get('short_positions_details', [])
                for short_pos in short_details:
                    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                    with col_s1:
                        st.write(f"**{short_pos['symbol']}**")
                    with col_s2:
                        st.write(f"Qty: {short_pos['quantity']}")
                    with col_s3:
                        st.write(f"Entry: ${short_pos['entry_price']:.2f}")
                    with col_s4:
                        st.write(f"Time: {short_pos['entry_time'][:16]}")
            
            # Configuration display
            with st.expander("⚙️ Current Configuration"):
                st.json(status['config'])
            
            # Execution history
            st.subheader("📜 Execution History")
            history = st.session_state.auto_trader.get_execution_history()
            
            if history:
                st.write(f"**Total Executions:** {len(history)}")
                
                for idx, execution in enumerate(reversed(history[-10:]), 1):  # Show last 10
                    with st.expander(f"{idx}. {execution['symbol']} - {execution['signal']} ({execution['timestamp']})"):
                        col_ex1, col_ex2, col_ex3 = st.columns(3)
                        
                        with col_ex1:
                            st.write(f"**Confidence:** {execution['confidence']:.1f}%")
                            st.write(f"**Quantity:** {execution['quantity']}")
                        
                        with col_ex2:
                            st.write(f"**Entry:** ${execution['entry_price']:.2f}")
                            st.write(f"**Target:** ${execution['target_price']:.2f}")
                        
                        with col_ex3:
                            st.write(f"**Stop Loss:** ${execution['stop_loss']:.2f}")
                            
                            profit_pct = ((execution['target_price'] - execution['entry_price']) / execution['entry_price'] * 100) if execution['entry_price'] else 0
                            st.write(f"**Potential:** {profit_pct:+.1f}%")
            else:
                st.info("No executions yet. The bot will execute when it finds high-confidence signals.")
        else:
            st.info("Auto-Trader is not running. Configure settings above and click 'Start Auto-Trader'.")
        
        # Help section
        with st.expander("❓ How It Works"):
            st.markdown("""
### Automated Trading Process

1. **Monitoring**: The bot scans your watchlist every X minutes
2. **Analysis**: Generates AI signals using comprehensive analysis
3. **Filtering**: Only executes signals above your confidence threshold
4. **Execution**: Places bracket orders with stop-loss and take-profit
5. **Safety**: Respects daily limits and trading hours

### Safety Features

- ✅ **Trading Hours**: Only trades during market hours (9:30 AM - 3:30 PM ET)
- ✅ **Daily Limits**: Stops after max daily orders reached
- ✅ **Confidence Filter**: Only executes high-confidence signals
- ✅ **Bracket Orders**: Automatic stop-loss protection
- ✅ **Paper Trading**: Test mode before using real money
- ✅ **Position Checks**: Won't add to existing positions
- ✅ **Short Selling**: Supports shorting in paper trading mode

### Short Selling (Advanced - Disabled by Default)

⚠️ **NOT recommended for scalping or cash account strategies!**

When enabled (paper trading only), SELL signals can open short positions:
- **Requires**: Margin account with sufficient equity
- **Best for**: Advanced swing trading or hedge strategies
- **Not for**: Scalping, day trading with cash accounts
- **BUY signals**: Opens long positions or covers shorts
- **SELL signals**: Closes long positions or opens shorts

**For scalping**: Keep this DISABLED. Only sell stocks you own!

### Trading Modes

**STOCKS**: Standard stock trading with AI signals
**OPTIONS**: Options strategies (coming soon)
**SCALPING**: Fast intraday trades with tight stops
- Default: 2% profit target, 1% stop loss
- Orders close same day
- Scan interval: 5-15 minutes recommended
- Best for: High-volume, liquid stocks

**WARRIOR_SCALPING**: Gap & Go strategy (Ross Cameron's approach)
- Focus: 9:30-10:00 AM momentum window
- Filters: $2-$20 price, 4-10% gap, 2-3x volume
- Setups: Gap & Go, Micro Pullback, Red-to-Green, Bull Flag
- Targets: 2% profit, 1% stop loss
- Best for: Premarket gappers with morning momentum

**ALL**: Combines all strategies

### Best Practices

1. **Start with Paper Trading** - Test for at least a week
2. **Monitor Daily** - Check execution history regularly
3. **Start Small** - Use low max daily orders (5-10)
4. **High Confidence** - Keep min confidence at 75%+
5. **Diversify** - Monitor 5-10 different tickers
6. **Review Results** - Analyze what works and adjust
7. **Scalping Tips**: Use 5-10 min intervals, liquid stocks only

### Risk Warning

⚠️ Automated trading carries significant risk. Past performance doesn't guarantee future results. 
Always start with paper trading and only risk capital you can afford to lose.
            """)
    
    with tab15:
        st.header("₿ Cryptocurrency Trading (Kraken Integration)")
        st.write("Trade cryptocurrencies 24/7 with AI-powered signals and automated strategies.")
        
        # Check if Kraken is configured
        kraken_key = os.getenv('KRAKEN_API_KEY')
        kraken_secret = os.getenv('KRAKEN_API_SECRET')
        
        if not kraken_key or not kraken_secret:
            st.error("🔑 **Kraken API credentials not found!**")
            st.info("Please set up your Kraken API keys in the `.env` file.")
            st.markdown("""
            ### 🚀 Quick Setup Guide:
            
            1. **Create Kraken Account**: Visit [kraken.com](https://www.kraken.com/)
            2. **Generate API Keys**: Go to Settings > API
            3. **Set Permissions**: 
               - ✅ Query Funds
               - ✅ Query Orders
               - ✅ Create/Modify Orders (for live trading)
               - ❌ Withdraw Funds (keep disabled!)
            4. **Add to `.env` file**:
               ```
               KRAKEN_API_KEY=your_api_key_here
               KRAKEN_API_SECRET=your_private_key_here
               ```
            5. **Read Full Guide**: `documentation/KRAKEN_SETUP_GUIDE.md`
            
            ⚠️ **IMPORTANT**: Kraken has NO paper trading mode. Start with $100-200 "learning capital" 
            and use $20-30 position sizes for testing. See `documentation/CRYPTO_QUICK_START.md`
            
            🔒 **Security**: Never share your API keys! Store them only in `.env` file.
            """)
            return
        
        # Import crypto modules
        try:
            from clients.kraken_client import KrakenClient
            from services.crypto_scanner import CryptoOpportunityScanner
            from services.crypto_trading_signals import CryptoTradingSignalGenerator
            import config_crypto_trading as crypto_config
        except ImportError as e:
            st.error(f"Error importing crypto modules: {e}")
            st.info("Please ensure all crypto trading files are in place.")
            return
        
        # Initialize Kraken client (cached to avoid 4-5 second delay on reruns)
        try:
            kraken_client = get_kraken_client(api_key=kraken_key, api_secret=kraken_secret)
            st.success("✅ Connected to Kraken (cached)")
        except Exception as e:
            st.error(f"❌ **Failed to initialize Kraken client**: {e}")
            return
        
        # Initialize crypto watchlist manager
        if 'crypto_watchlist_manager' not in st.session_state:
            from services.crypto_watchlist_manager import CryptoWatchlistManager
            st.session_state.crypto_watchlist_manager = CryptoWatchlistManager()
        
        crypto_wl_manager = st.session_state.crypto_watchlist_manager
        
        # Create subtabs for crypto features
        crypto_tab1, crypto_tab2, crypto_tab3, crypto_tab4, crypto_tab5, crypto_tab6, crypto_tab7 = st.tabs([
            "📊 Market Overview",
            "🔍 Crypto Scanner",
            "💰 Penny Cryptos (<$1)",
            "⭐ My Watchlist",
            "🎯 Signal Generator",
            "⚡ Quick Trade",
            "📈 Portfolio & Settings"
        ])
        
        with crypto_tab1:
            st.subheader("📊 Crypto Market Overview")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("💡 **24/7 Trading**: Crypto markets never close! Trade anytime, any day.")
            
            with col2:
                if st.button("🔄 Refresh Data", key="crypto_refresh"):
                    st.rerun()
            
            # Get account balance
            try:
                balances = kraken_client.get_account_balance()
                total_usd = kraken_client.get_total_balance_usd()
                
                st.markdown("### 💰 Account Balance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Value (USD)", f"${total_usd:,.2f}")
                
                # Find USD and crypto balances
                usd_balance = next((b for b in balances if b.currency in ['USD', 'ZUSD']), None)
                crypto_holdings = [b for b in balances if b.balance > 0 and b.currency not in ['USD', 'ZUSD']]
                
                with col2:
                    if usd_balance:
                        st.metric("Available USD", f"${usd_balance.available:,.2f}")
                    else:
                        st.metric("Available USD", "$0.00")
                
                with col3:
                    st.metric("Crypto Assets", len(crypto_holdings))
                
                # Show crypto holdings
                if crypto_holdings:
                    st.markdown("### 📊 Your Crypto Holdings")
                    
                    holdings_data = []
                    for balance in crypto_holdings:
                        try:
                            pair = f"{balance.currency}/USD"
                            ticker = kraken_client.get_ticker_data(pair)
                            
                            if ticker:
                                value_usd = balance.balance * ticker['last_price']
                                holdings_data.append({
                                    'Asset': balance.currency,
                                    'Balance': f"{balance.balance:.4f}",
                                    'Price': f"${ticker['last_price']:,.2f}",
                                    'Value (USD)': f"${value_usd:,.2f}",
                                    '24h Change': f"{((ticker['high_24h'] - ticker['low_24h']) / ticker['low_24h'] * 100):.2f}%"
                                })
                        except:
                            holdings_data.append({
                                'Asset': balance.currency,
                                'Balance': f"{balance.balance:.4f}",
                                'Price': 'N/A',
                                'Value (USD)': 'N/A',
                                '24h Change': 'N/A'
                            })
                    
                    if holdings_data:
                        st.dataframe(holdings_data, width="stretch")
                
            except Exception as e:
                st.error(f"Error fetching account data: {e}")
            
            # Show top crypto prices
            st.markdown("### 📈 Top Cryptocurrencies")
            
            major_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD']
            
            price_data = []
            for pair in major_pairs:
                try:
                    ticker = kraken_client.get_ticker_data(pair)
                    if ticker:
                        change_pct = ((ticker['last_price'] - ticker['low_24h']) / ticker['low_24h']) * 100
                        price_data.append({
                            'Pair': pair,
                            'Price': f"${ticker['last_price']:,.2f}",
                            '24h High': f"${ticker['high_24h']:,.2f}",
                            '24h Low': f"${ticker['low_24h']:,.2f}",
                            '24h Change': f"{change_pct:.2f}%",
                            'Volume': f"{ticker['volume_24h']:,.2f}"
                        })
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error fetching {pair}: {e}")
            
            if price_data:
                st.dataframe(price_data, width="stretch")
        
        with crypto_tab2:
            logger.info("🏁 CRYPTO_TAB2 (Scanner) RENDERING")
            st.subheader("🔍 Advanced Crypto Opportunity Scanner")
            st.write("**AI-powered scanner** to find the best crypto trading opportunities with multiple analysis modes.")
            
            # Initialize scanners (cached to avoid repeated initialization)
            crypto_scanner = get_crypto_scanner(kraken_client, crypto_config)
            ai_crypto_scanner = get_ai_crypto_scanner(kraken_client, crypto_config)
            
            # Analysis mode selector
            analysis_mode = st.radio(
                "🔬 Analysis Mode:",
                options=["⚡ Quick Scan (Technical Only)", "🧠 AI-Enhanced (LLM Analysis)", "🔥 Buzzing Cryptos", "🌶️ Hottest Cryptos", "💥 Breakout Detection"],
                horizontal=False,
                help="Choose analysis mode:\n- Quick: Fast technical analysis\n- AI-Enhanced: Adds LLM reasoning\n- Buzzing: High volume surges\n- Hottest: Strong momentum\n- Breakout: Technical breakouts"
            )
            
            use_ai = "AI-Enhanced" in analysis_mode
            
            if use_ai:
                with st.expander("ℹ️ What does AI-Enhanced include?", expanded=False):
                    st.markdown("""
                    **AI-Enhanced Mode** adds intelligent analysis:
                    - **🤖 LLM Reasoning**: Natural language explanations
                    - **🎯 Risk Assessment**: Crypto-specific risk analysis
                    - **📊 Market Cycle**: Where we are in the cycle (accumulation, markup, etc.)
                    - **💬 Social Narrative**: Current market sentiment
                    - **⭐ AI Rating**: 0-10 confidence score from AI
                    
                    This provides **deeper insights** beyond pure technical analysis.
                    """)
            
            st.divider()
            
            # Scan configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "Buzzing" in analysis_mode or "Hottest" in analysis_mode or "Breakout" in analysis_mode:
                    top_n = st.number_input("Top N Results", min_value=5, max_value=20, value=10)
                else:
                    scan_strategy = st.selectbox(
                        "Strategy",
                        ['ALL', 'SCALP', 'MOMENTUM', 'SWING'],
                        help="Select trading strategy to scan for"
                    )
            
            with col2:
                if "Buzzing" not in analysis_mode and "Hottest" not in analysis_mode and "Breakout" not in analysis_mode:
                    top_n = st.number_input("Top N Results", min_value=3, max_value=20, value=10)
            
            with col3:
                if "Buzzing" not in analysis_mode and "Hottest" not in analysis_mode and "Breakout" not in analysis_mode:
                    min_score = st.slider("Min Score", min_value=50, max_value=90, value=60)
            
            # Filters
            with st.expander("🎚️ Advanced Filters", expanded=False):
                fcol1, fcol2 = st.columns(2)
                
                with fcol1:
                    st.markdown("**Volume Filters**")
                    min_volume_ratio = st.slider("Min Volume Ratio (x avg)", 1.0, 5.0, 1.0, 0.5, key="crypto_vol_ratio")
                    
                    st.markdown("**Momentum Filters**")
                    min_momentum = st.slider("Min 24h Change %", 0.0, 10.0, 0.0, 1.0, key="crypto_momentum")
                
                with fcol2:
                    st.markdown("**Volatility Filters**")
                    max_volatility = st.slider("Max Volatility %", 5.0, 20.0, 20.0, 1.0, key="crypto_vol")
                    
                    if use_ai:
                        st.markdown("**AI Confidence Filter**")
                        ai_confidence_filter = st.selectbox(
                            "Min AI Confidence",
                            ['ALL', 'MEDIUM', 'HIGH'],
                            help="Filter by AI confidence level"
                        )
            
            # Scan button
            button_label = {
                "⚡ Quick Scan (Technical Only)": "🚀 Quick Scan",
                "🧠 AI-Enhanced (LLM Analysis)": "🧠 AI-Enhanced Scan",
                "🔥 Buzzing Cryptos": "🔥 Find Buzzing Cryptos",
                "🌶️ Hottest Cryptos": "🌶️ Find Hottest Cryptos",
                "💥 Breakout Detection": "💥 Find Breakouts"
            }
            
            # Initialize session state for scan results
            if 'crypto_scan_results' not in st.session_state:
                st.session_state.crypto_scan_results = None
            
            if st.button(button_label[analysis_mode], key="crypto_scan", type="primary"):
                logger.info(f"🔍 CRYPTO SCAN BUTTON CLICKED - Mode: {analysis_mode}")
                print(f"\n{'='*80}\n🔍 CRYPTO SCAN BUTTON CLICKED - Mode: {analysis_mode}\n{'='*80}\n", flush=True)
                with st.spinner(f"Scanning crypto markets with {analysis_mode}..."):
                    try:
                        opportunities = []
                        
                        # Execute appropriate scan based on mode
                        if analysis_mode == "⚡ Quick Scan (Technical Only)":
                            opportunities = crypto_scanner.scan_opportunities(
                                strategy=scan_strategy,
                                top_n=top_n,
                                min_score=min_score
                            )
                        
                        elif analysis_mode == "🧠 AI-Enhanced (LLM Analysis)":
                            ai_conf_filter = None if ai_confidence_filter == 'ALL' else ai_confidence_filter
                            opportunities = ai_crypto_scanner.scan_with_ai_confidence(
                                strategy=scan_strategy,
                                top_n=top_n,
                                min_score=min_score,
                                min_ai_confidence=ai_conf_filter
                            )
                        
                        elif analysis_mode == "🔥 Buzzing Cryptos":
                            if use_ai:
                                opportunities = ai_crypto_scanner.get_buzzing_cryptos(top_n=top_n)
                            else:
                                opportunities = crypto_scanner.scan_buzzing_cryptos(
                                    top_n=top_n,
                                    min_volume_ratio=min_volume_ratio
                                )
                        
                        elif analysis_mode == "🌶️ Hottest Cryptos":
                            if use_ai:
                                opportunities = ai_crypto_scanner.get_hottest_cryptos(top_n=top_n)
                            else:
                                opportunities = crypto_scanner.scan_hottest_cryptos(
                                    top_n=top_n,
                                    min_momentum=min_momentum
                                )
                        
                        elif analysis_mode == "💥 Breakout Detection":
                            opportunities = crypto_scanner.scan_breakout_cryptos(top_n=top_n)
                        
                        # Apply filters
                        if opportunities:
                            # Volume ratio filter
                            if min_volume_ratio > 1.0:
                                opportunities = [opp for opp in opportunities if opp.volume_ratio >= min_volume_ratio]
                            
                            # Momentum filter
                            if min_momentum > 0:
                                opportunities = [opp for opp in opportunities if abs(opp.change_pct_24h) >= min_momentum]
                            
                            # Volatility filter
                            if max_volatility < 20.0:
                                opportunities = [opp for opp in opportunities if opp.volatility_24h <= max_volatility]
                        
                        # Store results in session state
                        st.session_state.crypto_scan_results = opportunities
                        logger.info(f"📊 Scan complete - Found {len(opportunities)} opportunities")
                        
                    except Exception as e:
                        st.error(f"Scanner error: {e}")
                        logger.error(f"Crypto scanner error: {e}", exc_info=True)
            
            # Display results from session state (persists across button clicks)
            if st.session_state.crypto_scan_results is not None:
                opportunities = st.session_state.crypto_scan_results
                if opportunities:
                    logger.info(f"🎯 Rendering {len(opportunities)} crypto cards...")
                    st.success(f"✅ Found {len(opportunities)} crypto opportunities!")
                    
                    # Summary metrics
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    
                    with scol1:
                        avg_score = sum(opp.score for opp in opportunities) / len(opportunities)
                        st.metric("Avg Score", f"{avg_score:.1f}/100")
                    
                    with scol2:
                        avg_vol_ratio = sum(opp.volume_ratio for opp in opportunities) / len(opportunities)
                        st.metric("Avg Volume Ratio", f"{avg_vol_ratio:.2f}x")
                    
                    with scol3:
                        high_conf = sum(1 for opp in opportunities if opp.confidence == 'HIGH')
                        st.metric("High Confidence", f"{high_conf}/{len(opportunities)}")
                    
                    with scol4:
                        if use_ai and hasattr(opportunities[0], 'ai_rating'):
                            avg_ai_rating = sum(opp.ai_rating for opp in opportunities) / len(opportunities)
                            st.metric("Avg AI Rating", f"{avg_ai_rating:.1f}/10")
                        else:
                            st.metric("Analysis Mode", "Technical" if not use_ai else "AI")
                    
                    st.divider()
                    
                    # Display each opportunity
                    logger.info(f"🔁 Starting loop to render {len(opportunities)} expanders")
                    for i, opp in enumerate(opportunities, 1):
                        logger.info(f"📋 Rendering card {i}/{len(opportunities)}: {opp.symbol}")
                        # Build expander title
                        title_parts = [
                            f"#{i}",
                            f"{opp.symbol}",
                            f"Score: {opp.score:.1f}",
                            f"{opp.confidence} Conf",
                            f"{opp.risk_level} Risk"
                        ]
                        
                        if use_ai and hasattr(opp, 'ai_rating'):
                            title_parts.append(f"AI: {opp.ai_rating:.1f}/10")
                        
                        expander_title = " | ".join(title_parts)
                        logger.info(f"🔽 Creating expander for {opp.symbol}: {expander_title}")
                        
                        with st.expander(expander_title, expanded=(i <= 3)):
                            logger.info(f"📂 Inside expander for {opp.symbol}, rendering button...")
                            # Metrics row
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Current Price", f"${opp.current_price:,.2f}")
                                st.metric("Strategy", opp.strategy.upper())
                            
                            with col2:
                                direction = "🟢" if opp.change_pct_24h > 0 else "🔴"
                                st.metric("24h Change", f"{direction} {opp.change_pct_24h:.2f}%")
                                st.metric("Volatility", f"{opp.volatility_24h:.2f}%")
                            
                            with col3:
                                st.metric("Volume 24h", f"${opp.volume_24h:,.0f}")
                                st.metric("Vol Ratio", f"{opp.volume_ratio:.2f}x")
                            
                            with col4:
                                st.metric("Confidence", opp.confidence)
                                st.metric("Risk Level", opp.risk_level)
                            
                            st.divider()
                            
                            # Analysis
                            st.markdown("**📊 Technical Analysis:**")
                            st.info(opp.reason)
                            
                            # AI Analysis (if available)
                            if use_ai and hasattr(opp, 'ai_reasoning'):
                                st.divider()
                                
                                acol1, acol2 = st.columns(2)
                                
                                with acol1:
                                    st.markdown("**🤖 AI Analysis:**")
                                    st.markdown(f"**AI Confidence:** {opp.ai_confidence}")
                                    st.markdown(f"**AI Rating:** {opp.ai_rating:.1f}/10")
                                    st.markdown(f"**Reasoning:** {opp.ai_reasoning}")
                                
                                with acol2:
                                    st.markdown("**⚠️ AI Risk Assessment:**")
                                    st.warning(opp.ai_risks)
                                    
                                    if hasattr(opp, 'market_cycle_phase'):
                                        st.markdown(f"**Market Cycle:** {opp.market_cycle_phase}")
                                    
                                    if hasattr(opp, 'social_narrative') and opp.social_narrative:
                                        st.markdown(f"**Social Narrative:** {opp.social_narrative}")
                            
                            st.divider()
                            
                            # Action buttons
                            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                            
                            with bcol1:
                                button_key = f"save_wl_{i}"
                                logger.info(f"🔘 Creating button widget for {opp.symbol} with key={button_key}")
                                print(f"🔘 Creating button widget for {opp.symbol} with key={button_key}", flush=True)
                                button_clicked = st.button(f"⭐ Save to Watchlist", key=button_key)
                                logger.info(f"🎯 Button state for {opp.symbol}: clicked={button_clicked}")
                                print(f"🎯 Button state for {opp.symbol}: clicked={button_clicked}", flush=True)
                                if button_clicked:
                                    logger.info(f"🔵 WATCHLIST BUTTON CLICKED for {opp.symbol}")
                                    print(f"\n{'='*80}\n🔵🔵🔵 WATCHLIST BUTTON CLICKED for {opp.symbol} 🔵🔵🔵\n{'='*80}\n", flush=True)
                                    try:
                                        with st.spinner(f"Saving {opp.symbol}..."):
                                            # Create opportunity data dict for watchlist
                                            opp_data = {
                                                'symbol': opp.symbol,
                                                'current_price': opp.current_price,
                                                'change_pct_24h': opp.change_pct_24h,
                                                'volume_24h': opp.volume_24h,
                                                'volume_ratio': opp.volume_ratio,
                                                'volatility_24h': opp.volatility_24h,
                                                'rsi': opp.rsi if hasattr(opp, 'rsi') else None,
                                                'momentum_score': opp.momentum_score if hasattr(opp, 'momentum_score') else None,
                                                'technical_score': opp.technical_score if hasattr(opp, 'technical_score') else None,
                                                'score': opp.score,
                                                'confidence': opp.confidence,
                                                'risk_level': opp.risk_level,
                                                'strategy': opp.strategy,
                                                'reason': opp.reason
                                            }
                                            logger.info(f"📦 Prepared data dict for {opp.symbol}: keys={list(opp_data.keys())}")
                                            logger.info(f"📊 Data values: price=${opp_data['current_price']}, confidence={opp_data['confidence']}, strategy={opp_data['strategy']}")
                                            
                                            logger.info(f"🔄 Calling crypto_wl_manager.add_crypto({opp.symbol}, opp_data)")
                                            success = crypto_wl_manager.add_crypto(opp.symbol, opp_data)
                                            logger.info(f"✨ add_crypto returned: {success}")
                                            
                                            if success:
                                                st.success(f"✅ Added {opp.symbol} to watchlist!")
                                                logger.info(f"✅ SUCCESS: {opp.symbol} added to watchlist")
                                            else:
                                                st.warning(f"⚠️ {opp.symbol} might already be in watchlist")
                                                logger.warning(f"⚠️ FAILED: {opp.symbol} not added (returned False)")
                                    except Exception as e:
                                        error_msg = f"Error saving {opp.symbol} to watchlist: {e}"
                                        st.error(f"❌ {error_msg}")
                                        logger.error(f"❌ EXCEPTION in watchlist save: {error_msg}", exc_info=True)
                            
                            with bcol2:
                                if st.button(f"📊 Generate Signal", key=f"gen_signal_{i}"):
                                    st.session_state.crypto_signal_symbol = opp.symbol
                                    st.info(f"Navigate to Signal Generator tab to see {opp.symbol} signals!")
                            
                            with bcol3:
                                if st.button(f"💹 View Chart", key=f"view_chart_{i}"):
                                    st.info(f"Chart viewing for {opp.symbol} - Coming soon!")
                            
                            with bcol4:
                                if st.button(f"⚡ Quick Trade", key=f"quick_trade_{i}"):
                                    st.session_state.crypto_quick_trade_pair = opp.symbol
                                    st.info(f"Quick trade setup for {opp.symbol}")

                else:
                    st.warning("No opportunities found matching your criteria. Try adjusting filters.")
            
            # Show scanner help
            with st.expander("❓ Scanner Help", expanded=False):
                st.markdown("""
                ### 🔍 Scanner Modes Explained
                
                **⚡ Quick Scan (Technical Only)**
                - Fast technical analysis based on price, volume, momentum
                - No AI overhead, great for quick checks
                - Use when you want fast results
                
                **🧠 AI-Enhanced (LLM Analysis)**
                - Adds LLM reasoning and risk assessment
                - Crypto-specific insights (24/7 market, social sentiment)
                - Best for high-confidence trades
                - Requires OpenRouter API key
                
                **🔥 Buzzing Cryptos**
                - Focus on volume surges (2x+ average)
                - Indicates strong interest and liquidity
                - Good for momentum plays
                
                **🌶️ Hottest Cryptos**
                - Focus on strong price momentum (3%+ moves)
                - Identifies trending assets
                - Good for breakout continuation
                
                **💥 Breakout Detection**
                - Technical breakouts (price > EMAs with volume)
                - High probability setups
                - Good entry points for new trends
                
                ### 🎯 Strategy Types
                
                - **SCALP**: Quick 1-3% moves, high frequency
                - **MOMENTUM**: Ride 5-10% trends with volume
                - **SWING**: Hold 1-7 days for larger moves
                - **ALL**: Mixed approach based on conditions
                
                ### ⚙️ Filter Tips
                
                - **Volume Ratio**: Higher = more interest, better liquidity
                - **Momentum**: Higher = stronger trend, more volatile
                - **Volatility**: Lower = more predictable, safer
                - **AI Confidence**: HIGH = AI agrees with technical analysis
                """)
        
        with crypto_tab3:
            logger.info("🏁 CRYPTO_TAB3 (Penny Cryptos) RENDERING")
            st.subheader("💰 Penny Crypto Scanner - Monster Runners Under $1")
            st.write("**Find sub-$1 cryptocurrencies with extreme runner potential** - including sub-penny coins (0.0000000+)")
            
            # Initialize penny crypto scanner (cached to avoid re-initialization)
            penny_crypto_scanner = get_penny_crypto_scanner(kraken_client, crypto_config)
            
            # Show scanner coverage
            scanner_stats = penny_crypto_scanner.get_scanner_stats()
            with st.expander("📊 Scanner Coverage & Method", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Coins Scanned", scanner_stats['total_coins_scanned'])
                    st.markdown("**Categories Covered:**")
                    for cat, count in scanner_stats['categories'].items():
                        st.caption(f"• {cat.replace('_', ' ').title()}: {count}")
                with col2:
                    st.markdown("**Scan Method:**")
                    st.caption(scanner_stats['scan_method'])
                    st.markdown("**Update Frequency:**")
                    st.caption(scanner_stats['update_frequency'])
                    st.warning(f"**Note:** {scanner_stats['note']}")
            
            # Scan mode selector
            scan_mode = st.radio(
                "🎯 Scan Mode:",
                options=["💰 All Penny Cryptos (<$1)", "🔥 Sub-Penny Cryptos (<$0.01)"],
                horizontal=False,
                help="Choose scan mode:\n- All Penny: Cryptos under $1\n- Sub-Penny: Extreme runners under $0.01"
            )
            
            st.divider()
            
            # Scan configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if "Sub-Penny" in scan_mode:
                    max_price = st.slider("Max Price", 0.001, 0.01, 0.01, 0.001, key="sub_penny_price")
                else:
                    max_price = st.slider("Max Price", 0.01, 1.0, 1.0, 0.1, key="penny_price")
            
            with col2:
                top_n = st.number_input("Top N Results", min_value=5, max_value=30, value=15)
            
            with col3:
                min_runner_score = st.slider("Min Runner Score", 40, 100, 60, 5, key="runner_score")
            
            # Advanced filters
            with st.expander("🎚️ Advanced Filters", expanded=False):
                fcol1, fcol2 = st.columns(2)
                
                with fcol1:
                    st.markdown("**Momentum Filters**")
                    min_volume_ratio = st.slider("Min Volume Ratio (x avg)", 1.0, 5.0, 1.5, 0.5, key="penny_vol_ratio")
                    min_volatility = st.slider("Min Volatility %", 1.0, 30.0, 5.0, 1.0, key="penny_vol_min")
                
                with fcol2:
                    st.markdown("**Price Action**")
                    min_momentum = st.slider("Min 24h Change %", 0.0, 50.0, 5.0, 5.0, key="penny_momentum")
                    max_volatility = st.slider("Max Volatility %", 10.0, 50.0, 50.0, 5.0, key="penny_vol_max")
            
            # Initialize session state for penny scan results
            if 'penny_crypto_scan_results' not in st.session_state:
                st.session_state.penny_crypto_scan_results = None
            if 'trending_runners_results' not in st.session_state:
                st.session_state.trending_runners_results = None
            if 'sub_penny_discovery_results' not in st.session_state:
                st.session_state.sub_penny_discovery_results = None
            
            # Scan buttons
            col_scan1, col_scan2, col_scan3 = st.columns(3)
            
            with col_scan1:
                scan_penny = st.button("🚀 Scan Watchlist for Runners", key="penny_scan", type="primary")
            
            with col_scan2:
                scan_trending = st.button("🔥 Scan CoinGecko Trending", key="trending_scan", type="secondary")
            
            with col_scan3:
                scan_sub_penny = st.button("🔬 Discover Sub-Penny (<$0.01)", key="sub_penny_scan", type="secondary")
            
            # Handle sub-penny discovery
            if scan_sub_penny:
                logger.info("🔬 SUB-PENNY DISCOVERY BUTTON CLICKED")
                with st.spinner("🔬 Discovering ultra-low coins from CoinGecko (this may take 30-60s)..."):
                    try:
                        import asyncio
                        # Get cached discovery engine
                        discovery = get_sub_penny_discovery()
                        
                        # Discover sub-penny runners
                        sub_penny_coins = asyncio.run(discovery.discover_sub_penny_runners(
                            max_price=0.01,
                            min_market_cap=0,
                            max_market_cap=10_000_000,
                            top_n=20,
                            sort_by="runner_potential"
                        ))
                        
                        st.session_state.sub_penny_discovery_results = sub_penny_coins
                        logger.info(f"📊 Sub-penny discovery complete - Found {len(sub_penny_coins)} opportunities")
                        
                    except Exception as e:
                        st.error(f"Sub-penny discovery error: {e}")
                        logger.error(f"Sub-penny discovery error: {e}", exc_info=True)
            
            # Handle trending runners scan
            if scan_trending:
                logger.info("🔍 TRENDING RUNNERS SCAN BUTTON CLICKED")
                with st.spinner("🔍 Fetching CoinGecko trending + sentiment analysis..."):
                    try:
                        import asyncio
                        
                        # Get trending runners
                        trending_results = asyncio.run(
                            penny_crypto_scanner.scan_trending_runners(top_n=10)
                        )
                        
                        st.session_state.trending_runners_results = trending_results
                        logger.info(f"📊 Trending scan complete - Found {len(trending_results)} opportunities")
                        
                    except Exception as e:
                        st.error(f"Trending scanner error: {e}")
                        logger.error(f"Trending crypto scanner error: {e}", exc_info=True)
            
            # Scan button
            if scan_penny:
                logger.info(f"🔍 PENNY CRYPTO SCAN BUTTON CLICKED - Mode: {scan_mode}")
                with st.spinner(f"Scanning for penny cryptos with {scan_mode}..."):
                    try:
                        if "Sub-Penny" in scan_mode:
                            opportunities = penny_crypto_scanner.scan_sub_penny_cryptos(
                                max_price=max_price,
                                top_n=top_n
                            )
                        else:
                            opportunities = penny_crypto_scanner.scan_penny_cryptos(
                                max_price=max_price,
                                top_n=top_n,
                                min_runner_score=min_runner_score
                            )
                        
                        # Apply advanced filters
                        if opportunities:
                            # Volume ratio filter
                            opportunities = [opp for opp in opportunities if opp.volume_ratio >= min_volume_ratio]
                            
                            # Momentum filter
                            opportunities = [opp for opp in opportunities if abs(opp.change_pct_24h) >= min_momentum]
                            
                            # Volatility filters
                            opportunities = [opp for opp in opportunities if opp.volatility_24h >= min_volatility and opp.volatility_24h <= max_volatility]
                        
                        # Store results
                        st.session_state.penny_crypto_scan_results = opportunities
                        logger.info(f"📊 Penny scan complete - Found {len(opportunities)} opportunities")
                        
                    except Exception as e:
                        st.error(f"Scanner error: {e}")
                        logger.error(f"Penny crypto scanner error: {e}", exc_info=True)
            
            # Display trending runners results
            if st.session_state.trending_runners_results is not None:
                trending_results = st.session_state.trending_runners_results
                if trending_results:
                    logger.info(f"🎯 Rendering {len(trending_results)} trending runner cards...")
                    st.success(f"✅ Found {len(trending_results)} trending monster runners from CoinGecko!")
                    
                    # Summary metrics
                    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                    
                    with tcol1:
                        avg_runner_score = sum(r['overall_runner_score'] for r in trending_results) / len(trending_results)
                        st.metric("Avg Runner Score", f"{avg_runner_score:.1f}/100")
                    
                    with tcol2:
                        bullish_count = sum(1 for r in trending_results if r['social_sentiment'].get('overall_sentiment') == 'BULLISH')
                        st.metric("Bullish Sentiment", f"{bullish_count}/{len(trending_results)}")
                    
                    with tcol3:
                        avg_trending_rank = sum(r['trending_score'] for r in trending_results) / len(trending_results)
                        st.metric("Avg Trending Rank", f"#{avg_trending_rank:.1f}")
                    
                    with tcol4:
                        st.metric("Data Source", "CoinGecko + Reddit + Twitter")
                    
                    st.divider()
                    
                    # Display each trending runner
                    for i, result in enumerate(trending_results, 1):
                        symbol = result['symbol']
                        name = result['name']
                        trending_rank = result['trending_score']
                        runner_score = result['overall_runner_score']
                        social = result['social_sentiment']
                        runner_potential = result['runner_potential']
                        
                        # Build expander title
                        title_parts = [
                            f"#{i}",
                            f"🔥 {symbol}",
                            f"({name})",
                            f"Trending: #{trending_rank}",
                            f"Runner: {runner_score:.1f}",
                            f"{runner_potential['confidence']} Conf"
                        ]
                        
                        expander_title = " | ".join(title_parts)
                        
                        with st.expander(expander_title, expanded=(i <= 3)):
                            # Metrics row
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Price (USD)", f"${result['price_usd']:.8f}")
                                st.metric("Market Cap Rank", result['market_cap_rank'] or "N/A")
                            
                            with col2:
                                st.metric("Trending Rank", f"#{trending_rank}")
                                st.metric("Trending Sentiment", result['trending_sentiment'])
                            
                            with col3:
                                st.metric("Overall Sentiment", social['overall_sentiment'])
                                st.metric("Sentiment Score", f"{social['overall_sentiment_score']:.2f}")
                            
                            with col4:
                                st.metric("Reddit Mentions", social['reddit_mentions'])
                                st.metric("Twitter Mentions", social['twitter_mentions'])
                            
                            st.divider()
                            
                            # Runner Potential Analysis
                            st.markdown("### 🚀 Monster Runner Potential")
                            st.metric("Runner Score", f"{runner_score:.1f}/100")
                            st.markdown(f"**Confidence:** {runner_potential['confidence']}")
                            st.markdown(f"**Recommendation:** {runner_potential['recommendation']}")
                            
                            # Signals
                            st.markdown("**Key Signals:**")
                            for signal in runner_potential['signals']:
                                st.caption(f"• {signal}")
                            
                            st.divider()
                            
                            # Technical data (if available)
                            if result['technical']:
                                tech = result['technical']
                                st.markdown("### 📊 Technical Analysis")
                                
                                tcol1, tcol2, tcol3 = st.columns(3)
                                
                                with tcol1:
                                    st.metric("24h Change", f"{tech['change_24h']:+.2f}%")
                                    st.metric("7d Change", f"{tech['change_7d']:+.2f}%")
                                
                                with tcol2:
                                    st.metric("Volume 24h", f"${tech['volume_24h']:,.0f}")
                                    st.metric("Vol Ratio", f"{tech['volume_ratio']:.2f}x")
                                
                                with tcol3:
                                    st.metric("Volatility", f"{tech['volatility']:.2f}%")
                                    st.metric("RSI", f"{tech['rsi']:.0f}")
                                
                                st.divider()
                            
                            # Action buttons
                            bcol1, bcol2, bcol3 = st.columns(3)
                            
                            with bcol1:
                                if st.button(f"⭐ Save to Watchlist", key=f"save_trending_{i}"):
                                    try:
                                        with st.spinner(f"Saving {symbol}..."):
                                            opp_data = {
                                                'symbol': f"{symbol}/USD",
                                                'current_price': result['price_usd'],
                                                'change_pct_24h': result['technical']['change_24h'] if result['technical'] else 0,
                                                'volume_24h': result['technical']['volume_24h'] if result['technical'] else 0,
                                                'volume_ratio': result['technical']['volume_ratio'] if result['technical'] else 0,
                                                'volatility_24h': result['technical']['volatility'] if result['technical'] else 0,
                                                'rsi': result['technical']['rsi'] if result['technical'] else 0,
                                                'momentum_score': result['technical']['momentum_score'] if result['technical'] else 0,
                                                'score': runner_score,
                                                'confidence': runner_potential['confidence'],
                                                'risk_level': result['technical']['risk_level'] if result['technical'] else 'MEDIUM',
                                                'strategy': 'trending_runner',
                                                'reason': f"CoinGecko Trending #{trending_rank} | {' | '.join(runner_potential['signals'][:2])}"
                                            }
                                            
                                            success = crypto_wl_manager.add_crypto(f"{symbol}/USD", opp_data)
                                            
                                            if success:
                                                st.success(f"✅ Added {symbol} to watchlist!")
                                            else:
                                                st.warning(f"⚠️ {symbol} might already be in watchlist")
                                    except Exception as e:
                                        st.error(f"❌ Error saving {symbol}: {e}")
                                        logger.error(f"Error saving trending runner to watchlist: {e}", exc_info=True)
                            
                            with bcol2:
                                if st.button(f"📊 Generate Signal", key=f"gen_trending_signal_{i}"):
                                    st.session_state.crypto_signal_symbol = f"{symbol}/USD"
                                    st.info(f"Navigate to Signal Generator tab to see {symbol} signals!")
                            
                            with bcol3:
                                if st.button(f"⚡ Quick Trade", key=f"quick_trending_trade_{i}"):
                                    st.session_state.crypto_quick_trade_pair = f"{symbol}/USD"
                                    st.info(f"Quick trade setup for {symbol}")
                
                else:
                    st.warning("No trending runners found. Try again later or check your CoinGecko API key.")
            
            # Display sub-penny discovery results
            if st.session_state.sub_penny_discovery_results is not None:
                sub_penny_coins = st.session_state.sub_penny_discovery_results
                if sub_penny_coins:
                    logger.info(f"🎯 Rendering {len(sub_penny_coins)} sub-penny coins...")
                    st.success(f"✅ Discovered {len(sub_penny_coins)} ultra-low coins under $0.01!")
                    
                    # Summary metrics
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    
                    with scol1:
                        avg_runner_score = sum(c.runner_potential_score for c in sub_penny_coins) / len(sub_penny_coins)
                        st.metric("Avg Runner Score", f"{avg_runner_score:.1f}/100")
                    
                    with scol2:
                        avg_price = sum(c.price_usd for c in sub_penny_coins) / len(sub_penny_coins)
                        st.metric("Avg Price", f"${avg_price:.8f}")
                    
                    with scol3:
                        avg_decimals = sum(c.price_decimals for c in sub_penny_coins) / len(sub_penny_coins)
                        st.metric("Avg Decimals", f"{avg_decimals:.1f}")
                    
                    with scol4:
                        total_market_cap = sum(c.market_cap for c in sub_penny_coins)
                        st.metric("Total Market Cap", f"${total_market_cap:,.0f}")
                    
                    st.divider()
                    
                    # Display each sub-penny coin
                    for i, coin in enumerate(sub_penny_coins, 1):
                        # Build expander title
                        title_parts = [
                            f"#{i}",
                            f"🔬 {coin.symbol}",
                            f"${coin.price_usd:.{min(coin.price_decimals, 8)}f}",
                            f"Runner: {coin.runner_potential_score:.1f}",
                            f"MC: ${coin.market_cap:,.0f}"
                        ]
                        
                        expander_title = " | ".join(title_parts)
                        
                        with st.expander(expander_title, expanded=(i <= 3)):
                            # Metrics row
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Price (USD)", f"${coin.price_usd:.{min(coin.price_decimals, 8)}f}")
                                st.metric("Decimals", f"{coin.price_decimals}")
                            
                            with col2:
                                st.metric("24h Change", f"{coin.change_24h:+.2f}%")
                                st.metric("7d Change", f"{coin.change_7d:+.2f}%")
                            
                            with col3:
                                st.metric("Market Cap", f"${coin.market_cap:,.0f}")
                                st.metric("Market Cap Rank", coin.market_cap_rank or "N/A")
                            
                            with col4:
                                st.metric("Volume 24h", f"${coin.volume_24h:,.0f}")
                                st.metric("Market Cap Change", f"{coin.market_cap_change_24h:+.2f}%")
                            
                            st.divider()
                            
                            # Recovery Potential
                            st.markdown("### 🚀 Recovery Potential")
                            if coin.ath > 0:
                                recovery = (coin.ath - coin.price_usd) / coin.price_usd * 100
                                st.metric("ATH Recovery Potential", f"+{recovery:.0f}%")
                                st.caption(f"ATH: ${coin.ath:.8f} | ATL: ${coin.atl:.8f}")
                            
                            st.divider()
                            
                            # Supply Analysis
                            st.markdown("### 📊 Supply Analysis")
                            scol1, scol2, scol3 = st.columns(3)
                            
                            with scol1:
                                st.metric("Circulating Supply", f"{coin.circulating_supply:,.0f}")
                            
                            with scol2:
                                st.metric("Total Supply", f"{coin.total_supply:,.0f}")
                            
                            with scol3:
                                if coin.total_supply > 0:
                                    circ_pct = (coin.circulating_supply / coin.total_supply) * 100
                                    st.metric("Circulating %", f"{circ_pct:.1f}%")
                            
                            st.divider()
                            
                            # Discovery Reason
                            st.markdown("### 💡 Why This Coin?")
                            st.info(coin.discovery_reason)
                            
                            st.divider()
                            
                            # Action buttons
                            bcol1, bcol2 = st.columns(2)
                            
                            with bcol1:
                                if st.button(f"⭐ Save to Watchlist", key=f"save_sub_penny_{i}"):
                                    try:
                                        with st.spinner(f"Saving {coin.symbol}..."):
                                            opp_data = {
                                                'symbol': f"{coin.symbol}/USD",
                                                'current_price': coin.price_usd,
                                                'change_pct_24h': coin.change_24h,
                                                'volume_24h': coin.volume_24h,
                                                'volatility_24h': 0,  # Not available from CoinGecko
                                                'score': coin.runner_potential_score,
                                                'confidence': 'MEDIUM',
                                                'risk_level': 'HIGH',
                                                'strategy': 'sub_penny_runner',
                                                'reason': coin.discovery_reason
                                            }
                                            
                                            success = crypto_wl_manager.add_crypto(f"{coin.symbol}/USD", opp_data)
                                            
                                            if success:
                                                st.success(f"✅ Added {coin.symbol} to watchlist!")
                                            else:
                                                st.warning(f"⚠️ {coin.symbol} might already be in watchlist")
                                    except Exception as e:
                                        st.error(f"❌ Error saving {coin.symbol}: {e}")
                                        logger.error(f"Error saving sub-penny to watchlist: {e}", exc_info=True)
                            
                            with bcol2:
                                st.info(f"💡 Tip: Research {coin.symbol} on CoinGecko before trading")
                
                else:
                    st.warning("No sub-penny coins found. Try adjusting filters or check back later.")
            
            # Display penny scan results
            if st.session_state.penny_crypto_scan_results is not None:
                opportunities = st.session_state.penny_crypto_scan_results
                if opportunities:
                    logger.info(f"🎯 Rendering {len(opportunities)} penny crypto cards...")
                    st.success(f"✅ Found {len(opportunities)} monster runner opportunities!")
                    
                    # Summary metrics
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    
                    with scol1:
                        avg_runner_score = sum(opp.runner_potential_score for opp in opportunities) / len(opportunities)
                        st.metric("Avg Runner Score", f"{avg_runner_score:.1f}/100")
                    
                    with scol2:
                        avg_vol_ratio = sum(opp.volume_ratio for opp in opportunities) / len(opportunities)
                        st.metric("Avg Volume Ratio", f"{avg_vol_ratio:.2f}x")
                    
                    with scol3:
                        high_conf = sum(1 for opp in opportunities if opp.confidence == 'HIGH')
                        st.metric("High Confidence", f"{high_conf}/{len(opportunities)}")
                    
                    with scol4:
                        avg_volatility = sum(opp.volatility_24h for opp in opportunities) / len(opportunities)
                        st.metric("Avg Volatility", f"{avg_volatility:.1f}%")
                    
                    st.divider()
                    
                    # Display each opportunity
                    for i, opp in enumerate(opportunities, 1):
                        # Build expander title with price precision indicator
                        price_indicator = "🔬" if opp.price_decimals > 6 else "💰"
                        title_parts = [
                            f"#{i}",
                            f"{price_indicator} {opp.symbol}",
                            f"${opp.current_price:.{min(opp.price_decimals, 8)}f}",
                            f"Runner: {opp.runner_potential_score:.1f}",
                            f"{opp.confidence} Conf",
                            f"{opp.risk_level} Risk"
                        ]
                        
                        expander_title = " | ".join(title_parts)
                        
                        with st.expander(expander_title, expanded=(i <= 3)):
                            # Metrics row
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Entry Price", f"${opp.current_price:.{min(opp.price_decimals, 8)}f}")
                                st.metric("Decimals", f"{opp.price_decimals}")
                            
                            with col2:
                                direction = "🟢" if opp.change_pct_24h > 0 else "🔴"
                                st.metric("24h Change", f"{direction} {opp.change_pct_24h:+.2f}%")
                                st.metric("7d Change", f"{opp.change_pct_7d:+.2f}%")
                            
                            with col3:
                                st.metric("Volume 24h", f"${opp.volume_24h:,.0f}")
                                st.metric("Vol Ratio", f"{opp.volume_ratio:.2f}x")
                            
                            with col4:
                                st.metric("Volatility", f"{opp.volatility_24h:.2f}%")
                                st.metric("RSI", f"{opp.rsi:.0f}")
                            
                            st.divider()
                            
                            # Monster Runner Targets
                            st.markdown("### 🎯 Monster Runner Targets")
                            tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                            
                            with tcol1:
                                gain_1 = ((opp.target_1 - opp.entry_price) / opp.entry_price) * 100
                                st.metric("Target 1 (50%)", f"${opp.target_1:.{min(opp.price_decimals, 8)}f}", f"+{gain_1:.1f}%")
                            
                            with tcol2:
                                gain_2 = ((opp.target_2 - opp.entry_price) / opp.entry_price) * 100
                                st.metric("Target 2 (100%)", f"${opp.target_2:.{min(opp.price_decimals, 8)}f}", f"+{gain_2:.1f}%")
                            
                            with tcol3:
                                gain_3 = ((opp.target_3 - opp.entry_price) / opp.entry_price) * 100
                                st.metric("Target 3 (200%+)", f"${opp.target_3:.{min(opp.price_decimals, 8)}f}", f"+{gain_3:.1f}%")
                            
                            with tcol4:
                                st.metric("Momentum", f"{opp.momentum_score:.0f}/100")
                            
                            st.divider()
                            
                            # Analysis
                            st.markdown("**📊 Runner Potential Analysis:**")
                            st.info(opp.reason)
                            
                            st.divider()
                            
                            # Action buttons
                            bcol1, bcol2, bcol3 = st.columns(3)
                            
                            with bcol1:
                                if st.button(f"⭐ Save to Watchlist", key=f"save_penny_{i}"):
                                    try:
                                        with st.spinner(f"Saving {opp.symbol}..."):
                                            opp_data = {
                                                'symbol': opp.symbol,
                                                'current_price': opp.current_price,
                                                'change_pct_24h': opp.change_pct_24h,
                                                'volume_24h': opp.volume_24h,
                                                'volume_ratio': opp.volume_ratio,
                                                'volatility_24h': opp.volatility_24h,
                                                'rsi': opp.rsi,
                                                'momentum_score': opp.momentum_score,
                                                'score': opp.runner_potential_score,
                                                'confidence': opp.confidence,
                                                'risk_level': opp.risk_level,
                                                'strategy': 'penny_runner',
                                                'reason': opp.reason
                                            }
                                            
                                            success = crypto_wl_manager.add_crypto(opp.symbol, opp_data)
                                            
                                            if success:
                                                st.success(f"✅ Added {opp.symbol} to watchlist!")
                                            else:
                                                st.warning(f"⚠️ {opp.symbol} might already be in watchlist")
                                    except Exception as e:
                                        st.error(f"❌ Error saving {opp.symbol}: {e}")
                                        logger.error(f"Error saving penny crypto to watchlist: {e}", exc_info=True)
                            
                            with bcol2:
                                if st.button(f"📊 Generate Signal", key=f"gen_penny_signal_{i}"):
                                    st.session_state.crypto_signal_symbol = opp.symbol
                                    st.info(f"Navigate to Signal Generator tab to see {opp.symbol} signals!")
                            
                            with bcol3:
                                if st.button(f"⚡ Quick Trade", key=f"quick_penny_trade_{i}"):
                                    st.session_state.crypto_quick_trade_pair = opp.symbol
                                    st.info(f"Quick trade setup for {opp.symbol}")
                
                else:
                    st.warning("No penny cryptos found matching your criteria. Try adjusting filters.")
            
            # Show scanner help
            with st.expander("❓ Penny Crypto Scanner Help", expanded=False):
                st.markdown("""
                ### 💰 Penny Crypto Scanner Guide
                
                **What are Penny Cryptos?**
                - Cryptocurrencies trading under $1.00
                - Often have extreme volatility and runner potential
                - Can move 100%+ in hours or days
                
                **Sub-Penny Cryptos (🔬)**
                - Cryptos under $0.01 with highest runner potential
                - Display with extreme precision (0.0000000+)
                - Highest risk/reward ratio
                
                **Monster Runner Potential Score**
                - Combines: momentum, volume, volatility, RSI, price action
                - 60-75: MEDIUM potential
                - 75-85: HIGH potential
                - 85+: EXTREME potential
                
                **Key Signals for Runners:**
                - 🚀 EXTREME 24h moves (>15%)
                - 💥 Volume surges (3x+ average)
                - ⚡ High volatility (>15%)
                - 🎯 Oversold RSI (<30)
                - 📊 Strong 7-day trends
                
                **Risk Management:**
                - ⚠️ Penny cryptos are HIGHLY VOLATILE
                - Use EXTREME CAUTION with leverage
                - Set tight stop losses
                - Position size accordingly
                - Monitor 24/7 (crypto never sleeps)
                
                **Target Strategy:**
                - **Target 1 (50%)**: Quick scalp profit
                - **Target 2 (100%)**: Swing trade target
                - **Target 3 (200%+)**: Monster runner target
                
                **Best Practices:**
                - Start with small positions
                - Take profits at targets
                - Don't hold through resistance
                - Watch volume for confirmation
                - Monitor social sentiment
                """)
        
        with crypto_tab4:
            st.subheader("⭐ My Crypto Watchlist")
            
            # Import and render watchlist UI
            try:
                from ui.crypto_watchlist_ui import render_crypto_watchlist_tab
                render_crypto_watchlist_tab(crypto_wl_manager)
            except Exception as e:
                st.error(f"Error loading watchlist: {e}")
                logger.error(f"Crypto watchlist UI error: {e}", exc_info=True)
        
        with crypto_tab5:
            st.subheader("🎯 Crypto Signal Generator")
            
            # Import and render signal generation UI
            try:
                from ui.crypto_signal_ui import render_signal_generation_tab
                render_signal_generation_tab(crypto_wl_manager, kraken_client)
            except Exception as e:
                st.error(f"Error loading signal generator: {e}")
                logger.error(f"Crypto signal UI error: {e}", exc_info=True)
        
        with crypto_tab6:
            # Import and render quick trade UI
            try:
                from ui.crypto_quick_trade_ui import render_quick_trade_tab
                render_quick_trade_tab(kraken_client, crypto_config)
            except Exception as e:
                st.error(f"Error loading quick trade: {e}")
                logger.error(f"Crypto quick trade UI error: {e}", exc_info=True)
        
        with crypto_tab7:
            st.subheader("📈 Crypto Portfolio & Settings")
            
            # Configuration display
            st.markdown("### ⚙️ Current Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Capital Management**")
                st.text(f"Total Capital: ${crypto_config.TOTAL_CAPITAL:,.2f}")
                st.text(f"Reserve Cash: {crypto_config.RESERVE_CASH_PCT}%")
                st.text(f"Max Position Size: {crypto_config.MAX_POSITION_SIZE_PCT}%")
            
            with col2:
                st.markdown("**Risk Management**")
                st.text(f"Risk per Trade: {crypto_config.RISK_PER_TRADE_PCT * 100}%")
                st.text(f"Max Daily Loss: {crypto_config.MAX_DAILY_LOSS_PCT * 100}%")
                st.text(f"Max Daily Orders: {crypto_config.MAX_DAILY_ORDERS}")
            
            st.markdown("### 📚 Resources")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **📖 Documentation**
                - [Kraken Setup Guide](documentation/KRAKEN_SETUP_GUIDE.md)
                - [Crypto Trading Config](config_crypto_trading.py)
                - [Trading Strategies](#)
                """)
            
            with col2:
                st.markdown("""
                **🛠️ Tools**
                - Test Kraken Connection
                - View Trading History
                - Export Trade Data
                """)
            
            with col3:
                st.markdown("""
                **⚠️ Safety**
                - Always use stop losses
                - Start with paper trading
                - Never risk more than 2%
                - Only invest what you can lose
                """)
            
            st.markdown("### 🎯 Strategy Recommendations")
            
            st.info("""
            **For Beginners** (Low Risk):
            - Strategy: CRYPTO_SWING
            - Pairs: BTC/USD, ETH/USD only
            - Position Size: 8-10%
            - Hold Time: 1-3 days
            - Take Profit: 6-8%
            - Stop Loss: 3%
            
            **For Active Traders** (Medium Risk):
            - Strategy: CRYPTO_SCALPING
            - Pairs: Top 5-10 by volume
            - Position Size: 10-12%
            - Hold Time: 15-30 minutes
            - Take Profit: 2-3%
            - Stop Loss: 1-1.5%
            
            **For Experienced** (High Risk):
            - Strategy: CRYPTO_MOMENTUM
            - Pairs: Volatile altcoins
            - Position Size: 8-12%
            - Hold Time: Few hours
            - Take Profit: 8-12%
            - Stop Loss: 4-5%
            """)
            
            st.warning("""
            ⚠️ **CRYPTO RISK WARNING**:
            
            Cryptocurrency trading is HIGHLY VOLATILE and RISKY:
            - Prices can swing 10%+ in hours
            - 24/7 market means no "safe" closing bell
            - Flash crashes and pumps are common
            - Regulation is still evolving
            - Exchanges can have downtime
            - **NO PAPER TRADING MODE on Kraken**
            
            **Start with "learning capital" ($100-200) you can afford to lose completely!**
            
            **Recommended for testing:**
            - Allocate only $100-200 initially
            - Use $20-30 position sizes
            - Make 10-20 small test trades
            - Expect to lose some money while learning
            - Scale up ONLY after consistent profitability
            
            This is REAL money from day 1. Trade conservatively!
            """)


if __name__ == "__main__":
    main()

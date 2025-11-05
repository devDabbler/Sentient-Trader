"""
Enhanced Ticker Discovery System
Multi-source intelligent ticker discovery using:
- Technical momentum (Smart Scanner)
- News sentiment analysis
- Social media trends (if available)
- Market screeners (volume, volatility, gaps)
- Sector rotation analysis
"""

from loguru import logger
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
from collections import Counter



@dataclass
class TickerSignal:
    """A signal from a discovery source"""
    ticker: str
    source: str  # 'scanner', 'sentiment', 'screener', 'social', 'sector'
    confidence: float  # 0-100
    reason: str
    timestamp: datetime
    metadata: Dict = None


class EnhancedTickerDiscovery:
    """
    Multi-source ticker discovery that combines:
    - Technical analysis (momentum, volume, patterns)
    - Sentiment analysis (news, social media)
    - Market screeners (unusual activity)
    - Sector rotation
    """
    
    def __init__(self, tradier_client=None, min_confidence: float = 60.0):
        """
        Initialize enhanced ticker discovery
        
        Args:
            tradier_client: Optional TradierClient for market data
            min_confidence: Minimum confidence score to include ticker
        """
        self.tradier_client = tradier_client
        self.min_confidence = min_confidence
        
    def discover_tickers(
        self,
        strategy: str = "WARRIOR_SCALPING",
        use_sentiment: bool = True,
        use_screeners: bool = True,
        use_social: bool = False,
        max_tickers: int = 20,
        fallback_universe: List[str] = None
    ) -> List[str]:
        """
        Discover optimal tickers using multiple sources
        
        Args:
            strategy: Trading strategy to optimize for
            use_sentiment: Enable sentiment-based discovery
            use_screeners: Enable market screener discovery
            use_social: Enable social media trend discovery
            max_tickers: Maximum number of tickers to return
            fallback_universe: Fallback list if discovery fails
            
        Returns:
            List of ticker symbols ranked by confidence
        """
        logger.info(f"🔍 Enhanced Ticker Discovery: Strategy={strategy}, Sources=Smart+{'Sentiment+' if use_sentiment else ''}{'Screeners+' if use_screeners else ''}{'Social' if use_social else ''}")
        
        all_signals: List[TickerSignal] = []
        
        # Source 1: Smart Scanner (Technical)
        try:
            scanner_tickers = self._discover_via_scanner(strategy)
            all_signals.extend(scanner_tickers)
            logger.info(f"✅ Scanner found {len(scanner_tickers)} signals")
        except Exception as e:
            logger.error(f"Scanner discovery failed: {e}")
        
        # Source 2: Sentiment Analysis
        if use_sentiment:
            try:
                sentiment_tickers = self._discover_via_sentiment(strategy)
                all_signals.extend(sentiment_tickers)
                logger.info(f"✅ Sentiment found {len(sentiment_tickers)} signals")
            except Exception as e:
                logger.error(f"Sentiment discovery failed: {e}")
        
        # Source 3: Market Screeners
        if use_screeners:
            try:
                screener_tickers = self._discover_via_screeners(strategy)
                all_signals.extend(screener_tickers)
                logger.info(f"✅ Screeners found {len(screener_tickers)} signals")
            except Exception as e:
                logger.error(f"Screener discovery failed: {e}")
        
        # Source 4: Social Media Trends (optional)
        if use_social:
            try:
                social_tickers = self._discover_via_social(strategy)
                all_signals.extend(social_tickers)
                logger.info(f"✅ Social found {len(social_tickers)} signals")
            except Exception as e:
                logger.error(f"Social discovery failed: {e}")
        
        # Aggregate and rank signals
        if not all_signals:
            logger.warning("⚠️ No signals from any source, using fallback universe")
            return fallback_universe[:max_tickers] if fallback_universe else []
        
        ranked_tickers = self._aggregate_and_rank(all_signals, max_tickers)
        
        logger.info(f"🎯 Enhanced Discovery Result: {len(ranked_tickers)} tickers from {len(all_signals)} signals")
        for i, (ticker, score, sources) in enumerate(ranked_tickers[:10], 1):
            logger.info(f"  #{i}. {ticker}: {score:.1f} confidence ({', '.join(sources)})")
        
        return [t[0] for t in ranked_tickers]
    
    def _discover_via_scanner(self, strategy: str) -> List[TickerSignal]:
        """Use existing Smart Scanner for technical discovery"""
        from services.advanced_opportunity_scanner import AdvancedOpportunityScanner, ScanType
        
        # Strategy-specific universes
        strategy_universes = {
            "SCALPING": [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                'PLTR', 'SOFI', 'RIVN', 'PLUG', 'NOK', 'AMC', 'GME', 'MARA',
                'RIOT', 'COIN', 'HOOD', 'SNAP', 'UBER', 'LYFT', 'NIO', 'LCID',
                'SPCE', 'CLOV', 'WISH', 'BB', 'TLRY', 'SNDL', 'FCEL', 'WKHS'
            ],
            "WARRIOR_SCALPING": [
                'AAPL', 'AMD', 'TSLA', 'NVDA', 'PLTR', 'SOFI', 'RIVN',
                'MARA', 'RIOT', 'NOK', 'AMC', 'GME', 'SNAP', 'HOOD',
                'NIO', 'LCID', 'PLUG', 'FCEL', 'TLRY', 'SNDL', 'AFRM',
                'PINS', 'RBLX', 'DASH', 'UBER', 'LYFT', 'SPCE', 'CLOV',
                'WISH', 'BB', 'WKHS', 'RIDE', 'GOEV', 'FSR', 'QS'
            ],
            "STOCKS": [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
                'NFLX', 'DIS', 'PLTR', 'SOFI', 'COIN', 'RBLX', 'ABNB', 'DASH',
                'SHOP', 'SNOW', 'CRWD', 'ZS', 'DDOG', 'NIO', 'RIVN', 'PLUG',
                'MRNA', 'BNTX', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'XOM', 'CVX',
                'COST', 'WMT', 'HD', 'LOW', 'TGT', 'PG', 'KO', 'PEP'
            ],
        }
        
        universe = strategy_universes.get(strategy, strategy_universes["WARRIOR_SCALPING"])
        
        scan_config = {
            "SCALPING": {"scan_type": ScanType.MOMENTUM, "trading_style": "SCALP", "top_n": 15},
            "WARRIOR_SCALPING": {"scan_type": ScanType.MOMENTUM, "trading_style": "SCALP", "top_n": 15},
            "STOCKS": {"scan_type": ScanType.ALL, "trading_style": "SWING_TRADE", "top_n": 20},
        }
        
        config = scan_config.get(strategy, scan_config["WARRIOR_SCALPING"])
        
        scanner = AdvancedOpportunityScanner(use_ai=False)
        opportunities = scanner.scan_opportunities(
            scan_type=config["scan_type"],
            trading_style=config["trading_style"],
            top_n=config["top_n"],
            custom_tickers=universe,
            use_extended_universe=False
        )
        
        signals = []
        for opp in opportunities:
            # OpportunityResult is an object, not a dict - use attributes
            signals.append(TickerSignal(
                ticker=opp.ticker,
                source='scanner',
                confidence=opp.score if hasattr(opp, 'score') else 75.0,
                reason=opp.rationale if hasattr(opp, 'rationale') else 'Technical momentum',
                timestamp=datetime.now(),
                metadata={'ticker': opp.ticker, 'score': opp.score if hasattr(opp, 'score') else 75.0}
            ))
        
        return signals
    
    def _discover_via_sentiment(self, strategy: str) -> List[TickerSignal]:
        """Discover tickers based on news sentiment and buzz"""
        from analyzers.news import NewsAnalyzer
        
        # Expanded universe for sentiment scanning
        sentiment_universe = [
            # Tech mega caps
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
            # High buzz stocks
            'PLTR', 'SOFI', 'COIN', 'HOOD', 'RIVN', 'LCID', 'NIO', 'PLUG',
            # Meme/Social
            'GME', 'AMC', 'BB', 'NOK', 'WISH', 'CLOV', 'SPCE',
            # Crypto proxies
            'MARA', 'RIOT', 'COIN', 'HOOD', 'SQ', 'PYPL',
            # Cannabis
            'TLRY', 'SNDL', 'APHA', 'CGC', 'ACB', 'HEXO',
            # EV
            'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'FSR', 'GOEV', 'QS',
            # Fintech
            'SQ', 'PYPL', 'AFRM', 'UPST', 'SOFI', 'HOOD',
            # High growth
            'RBLX', 'DASH', 'UBER', 'LYFT', 'ABNB', 'SNAP', 'PINS', 'SPOT'
        ]
        
        signals = []
        news_analyzer = NewsAnalyzer()
        
        # Sample subset for performance (you can increase this)
        sample_size = min(30, len(sentiment_universe))
        logger.info(f"📰 Analyzing sentiment for {sample_size} tickers...")
        
        for ticker in sentiment_universe[:sample_size]:
            try:
                # Get recent news
                articles = news_analyzer.get_stock_news(ticker, max_articles=5)
                if not articles:
                    continue
                
                # Analyze sentiment
                sentiment_score, sentiment_signals = news_analyzer.analyze_sentiment(articles)
                article_count = len(articles)
                
                # Strong positive sentiment = potential opportunity
                # Lowered from 0.6 to 0.5 to capture more signals (0.8 sentiment seen in logs)
                if sentiment_score >= 0.5 and article_count >= 3:
                    confidence = min(90, 60 + (sentiment_score * 30) + (article_count * 2))
                    
                    signals.append(TickerSignal(
                        ticker=ticker,
                        source='sentiment',
                        confidence=confidence,
                        reason=f"Strong positive sentiment ({sentiment_score:.2f}) with {article_count} recent articles",
                        timestamp=datetime.now(),
                        metadata={
                            'sentiment_score': sentiment_score,
                            'article_count': article_count,
                            'articles': articles[:3]  # Top 3 headlines
                        }
                    ))
                # Strong negative sentiment = potential short or avoid
                elif sentiment_score <= 0.3 and article_count >= 3:
                    # For now, we'll skip bearish signals unless short selling is enabled
                    pass
                    
            except Exception as e:
                logger.debug(f"Error analyzing sentiment for {ticker}: {e}")
                continue
        
        return signals
    
    def _discover_via_screeners(self, strategy: str) -> List[TickerSignal]:
        """
        Discover tickers using market screener criteria:
        - Unusual volume
        - Price gaps
        - Volatility expansion
        - Breakouts
        """
        signals = []
        
        # For WARRIOR_SCALPING, focus on gap detection
        if strategy == "WARRIOR_SCALPING":
            signals.extend(self._scan_for_gappers())
        
        # For all strategies, scan for unusual volume
        signals.extend(self._scan_for_unusual_volume(strategy))
        
        return signals
    
    def _scan_for_gappers(self) -> List[TickerSignal]:
        """Scan for stocks with significant premarket/overnight gaps"""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            # Curated list of gap-prone stocks
            gap_universe = [
                'TSLA', 'AMD', 'NVDA', 'PLTR', 'SOFI', 'RIVN', 'LCID',
                'MARA', 'RIOT', 'COIN', 'GME', 'AMC', 'SNAP', 'HOOD',
                'NIO', 'XPEV', 'LI', 'PLUG', 'FCEL', 'TLRY', 'SNDL',
                'RBLX', 'DASH', 'UBER', 'LYFT', 'AFRM', 'UPST', 'PINS',
                'SPCE', 'CLOV', 'WISH', 'BB', 'WKHS', 'RIDE', 'GOEV'
            ]
            
            signals = []
            
            for ticker in gap_universe[:25]:  # Sample 25 for performance
                try:
                    stock = yf.Ticker(ticker)
                    
                    # Get last 5 days of data
                    hist = stock.history(period='5d', interval='1d')
                    if len(hist) < 2:
                        continue
                    
                    # Calculate gap from previous close to current open/price
                    prev_close = hist['Close'].iloc[-2]
                    current_price = hist['Close'].iloc[-1]
                    
                    # Check if there's a gap
                    gap_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    # For gaps >= 2%, this is interesting
                    if abs(gap_pct) >= 2.0:
                        # Calculate volume ratio
                        avg_volume = hist['Volume'].iloc[:-1].mean()
                        current_volume = hist['Volume'].iloc[-1]
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                        
                        # Higher confidence for larger gaps with volume
                        confidence = min(95, 60 + abs(gap_pct) * 2 + (volume_ratio - 1) * 10)
                        
                        signals.append(TickerSignal(
                            ticker=ticker,
                            source='screener_gap',
                            confidence=confidence,
                            reason=f"Gap {gap_pct:+.1f}% with {volume_ratio:.1f}x volume",
                            timestamp=datetime.now(),
                            metadata={
                                'gap_pct': gap_pct,
                                'volume_ratio': volume_ratio,
                                'prev_close': prev_close,
                                'current_price': current_price
                            }
                        ))
                
                except Exception as e:
                    logger.debug(f"Error checking gap for {ticker}: {e}")
                    continue
            
            return signals
            
        except ImportError:
            logger.warning("yfinance not available for gap scanning")
            return []
        except Exception as e:
            logger.error(f"Error in gap scanner: {e}")
            return []
    
    def _scan_for_unusual_volume(self, strategy: str) -> List[TickerSignal]:
        """Scan for stocks with unusual volume (potential catalysts)"""
        try:
            import yfinance as yf
            
            # Strategy-specific universes
            volume_universe = {
                "SCALPING": [
                    'AAPL', 'MSFT', 'TSLA', 'AMD', 'NVDA', 'META',
                    'PLTR', 'SOFI', 'COIN', 'GME', 'AMC', 'HOOD'
                ],
                "WARRIOR_SCALPING": [
                    'PLTR', 'SOFI', 'RIVN', 'LCID', 'NIO', 'PLUG',
                    'GME', 'AMC', 'MARA', 'RIOT', 'SNAP', 'HOOD'
                ]
            }
            
            universe = volume_universe.get(strategy, volume_universe["WARRIOR_SCALPING"])
            signals = []
            
            for ticker in universe[:15]:  # Sample 15 for performance
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period='10d', interval='1d')
                    
                    if len(hist) < 5:
                        continue
                    
                    # Calculate volume ratio
                    avg_volume = hist['Volume'].iloc[:-1].mean()
                    current_volume = hist['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    # Unusual volume = 2x+ average
                    if volume_ratio >= 2.0:
                        confidence = min(90, 50 + (volume_ratio - 1) * 15)
                        
                        signals.append(TickerSignal(
                            ticker=ticker,
                            source='screener_volume',
                            confidence=confidence,
                            reason=f"Unusual volume: {volume_ratio:.1f}x average",
                            timestamp=datetime.now(),
                            metadata={
                                'volume_ratio': volume_ratio,
                                'avg_volume': avg_volume,
                                'current_volume': current_volume
                            }
                        ))
                
                except Exception as e:
                    logger.debug(f"Error checking volume for {ticker}: {e}")
                    continue
            
            return signals
            
        except ImportError:
            logger.warning("yfinance not available for volume scanning")
            return []
        except Exception as e:
            logger.error(f"Error in volume scanner: {e}")
            return []
    
    def _discover_via_social(self, strategy: str) -> List[TickerSignal]:
        """
        Discover trending tickers from social media
        Currently a placeholder - can integrate with:
        - Reddit WallStreetBets API
        - Twitter/X trends
        - StockTwits
        """
        # TODO: Integrate social media APIs
        logger.debug("Social media discovery not yet implemented")
        return []
    
    def _aggregate_and_rank(
        self, 
        signals: List[TickerSignal], 
        max_tickers: int
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Aggregate signals from multiple sources and rank by consensus
        
        Returns:
            List of (ticker, aggregate_confidence, sources)
        """
        # Group signals by ticker
        ticker_signals: Dict[str, List[TickerSignal]] = {}
        for signal in signals:
            if signal.ticker not in ticker_signals:
                ticker_signals[signal.ticker] = []
            ticker_signals[signal.ticker].append(signal)
        
        # Calculate aggregate score for each ticker
        ticker_scores = []
        for ticker, sigs in ticker_signals.items():
            # Multiple sources = higher confidence
            source_bonus = min(20, len(sigs) * 10)
            
            # Average confidence from all sources
            avg_confidence = sum(s.confidence for s in sigs) / len(sigs)
            
            # Aggregate score = average + bonus for multiple sources
            aggregate_score = min(100, avg_confidence + source_bonus)
            
            # Get list of sources
            sources = [s.source for s in sigs]
            
            # Only include if meets minimum confidence
            if aggregate_score >= self.min_confidence:
                ticker_scores.append((ticker, aggregate_score, sources))
        
        # Sort by score descending
        ticker_scores.sort(key=lambda x: x[1], reverse=True)
        
        return ticker_scores[:max_tickers]


def get_enhanced_discovery(tradier_client=None, min_confidence: float = 60.0) -> EnhancedTickerDiscovery:
    """Get singleton instance of enhanced ticker discovery"""
    return EnhancedTickerDiscovery(tradier_client=tradier_client, min_confidence=min_confidence)


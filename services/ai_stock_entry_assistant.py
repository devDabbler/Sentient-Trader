"""
AI Stock Entry Assistant - Intelligent Trade Entry Timing for Stocks
Analyzes stock market conditions BEFORE entry to optimize timing and reduce bad trades
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List, Tuple
from loguru import logger
import json
import time
import pandas as pd
import threading


class EntryAction(Enum):
    """Recommended entry actions"""
    ENTER_NOW = "ENTER_NOW"
    WAIT_FOR_PULLBACK = "WAIT_FOR_PULLBACK"
    WAIT_FOR_BREAKOUT = "WAIT_FOR_BREAKOUT"
    DO_NOT_ENTER = "DO_NOT_ENTER"
    PLACE_LIMIT_ORDER = "PLACE_LIMIT_ORDER"


@dataclass
class EntryAnalysis:
    """AI entry recommendation for stocks"""
    symbol: str
    action: str
    confidence: float
    reasoning: str
    urgency: str
    current_price: float
    suggested_entry: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    risk_reward_ratio: float = 0.0
    technical_score: float = 0.0
    trend_score: float = 0.0
    timing_score: float = 0.0
    risk_score: float = 0.0
    wait_for_price: Optional[float] = None
    wait_for_rsi: Optional[float] = None
    analysis_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.analysis_time is None:
            self.analysis_time = datetime.now()


@dataclass
class MonitoredEntryOpportunity:
    """Stock entry opportunity being monitored"""
    symbol: str
    side: str
    target_price: Optional[float]
    target_rsi: Optional[float]
    target_conditions: Dict
    position_size: float
    risk_pct: float
    take_profit_pct: float
    original_analysis: EntryAnalysis
    created_time: datetime
    last_check_time: datetime
    current_price: float = 0.0
    notification_sent: bool = False
    auto_execute: bool = False


class AIStockEntryAssistant:
    """
    AI-powered stock entry assistant that analyzes market conditions
    before entering trades to optimize timing
    """
    
    def __init__(
        self,
        broker_client,  # TradierClient or IBKRClient or BrokerAdapter
        llm_analyzer=None,
        check_interval_seconds: int = 60,
        enable_auto_entry: bool = False,
        min_confidence_for_entry: float = 85.0,
        min_confidence_for_auto: float = 90.0
    ):
        """
        Initialize AI Stock Entry Assistant
        
        Args:
            broker_client: Broker API client (Tradier, IBKR, or BrokerAdapter)
            llm_analyzer: LLM strategy analyzer for AI decisions
            check_interval_seconds: How often to check monitored opportunities
            enable_auto_entry: Allow auto-execution on high confidence
            min_confidence_for_entry: Minimum confidence to recommend entry
            min_confidence_for_auto: Minimum confidence for auto-execution
        """
        self.broker_client = broker_client
        self.llm_analyzer = llm_analyzer
        self.check_interval = check_interval_seconds
        self.enable_auto_entry = enable_auto_entry
        self.min_confidence_entry = min_confidence_for_entry
        self.min_confidence_auto = min_confidence_for_auto
        
        # Monitored opportunities
        self.opportunities: Dict[str, MonitoredEntryOpportunity] = {}
        
        # State persistence (different file from crypto)
        self.state_file = "ai_stock_entry_monitors.json"
        
        # Monitoring thread
        self.monitoring_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        logger.info("🎯 AI Stock Entry Assistant initialized")
        logger.info(f"   Check interval: {check_interval_seconds}s")
        logger.info(f"   Auto-entry: {'Enabled' if enable_auto_entry else 'Disabled'}")
        logger.info(f"   Min confidence (entry): {min_confidence_for_entry}%")
        logger.info(f"   Min confidence (auto): {min_confidence_for_auto}%")
        
        # Load saved monitors from previous session
        self._load_state()
    
    def analyze_entry(
        self,
        symbol: str,
        side: str,
        position_size: float,
        risk_pct: float,
        take_profit_pct: float
    ) -> EntryAnalysis:
        """
        Analyze if NOW is a good time to enter this stock trade
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            side: "BUY" or "SELL"
            position_size: Position size in USD
            risk_pct: Risk percentage for stop loss
            take_profit_pct: Take profit percentage
        
        Returns:
            EntryAnalysis with recommendation
        """
        try:
            logger.info(f"🎯 Analyzing entry for {symbol} ({side})")
            
            # Get current market price
            current_price = self._get_current_price(symbol)
            if not current_price or current_price <= 0:
                return self._create_error_analysis(symbol, "Invalid price data")
            
            # Get technical indicators
            technical_data = self._get_technical_indicators(symbol)
            if not technical_data:
                logger.warning(f"⚠️ Limited technical data for {symbol}, using price action only")
            
            # ENHANCED: Fetch real-time news with FinBERT sentiment
            recent_news, sentiment_score = self._get_stock_news_sentiment(symbol)
            
            # Build AI prompt for entry analysis (with news context)
            prompt = self._build_entry_analysis_prompt(
                symbol=symbol,
                side=side,
                current_price=current_price,
                position_size=position_size,
                risk_pct=risk_pct,
                take_profit_pct=take_profit_pct,
                technical_data=technical_data,
                recent_news=recent_news,  # NEW
                sentiment_score=sentiment_score  # NEW
            )
            
            # Get AI recommendation
            if not self.llm_analyzer:
                logger.error("LLM analyzer not configured")
                return self._create_error_analysis(symbol, "LLM analyzer not initialized")
            
            # Use appropriate method based on provider
            if getattr(self.llm_analyzer, 'provider', '') == 'ollama':
                response = self.llm_analyzer.analyze_with_llm(prompt)
            else:
                response = self.llm_analyzer._call_openrouter(prompt, max_retries=2, try_fallbacks=True)
            
            if not response:
                logger.error("Failed to get AI response")
                return self._create_error_analysis(symbol, "LLM API call failed")
            
            decision_data = self._parse_ai_response(response)
            
            if decision_data:
                analysis = EntryAnalysis(
                    symbol=symbol,
                    action=decision_data.get('action', 'DO_NOT_ENTER'),
                    confidence=float(decision_data.get('confidence', 0)),
                    reasoning=decision_data.get('reasoning', ''),
                    urgency=decision_data.get('urgency', 'LOW'),
                    current_price=current_price,
                    suggested_entry=decision_data.get('suggested_entry'),
                    suggested_stop=decision_data.get('suggested_stop'),
                    suggested_target=decision_data.get('suggested_target'),
                    risk_reward_ratio=float(decision_data.get('risk_reward_ratio', 0)),
                    technical_score=float(decision_data.get('technical_score', 0)),
                    trend_score=float(decision_data.get('trend_score', 0)),
                    timing_score=float(decision_data.get('timing_score', 0)),
                    risk_score=float(decision_data.get('risk_score', 0)),
                    wait_for_price=decision_data.get('wait_for_price'),
                    wait_for_rsi=decision_data.get('wait_for_rsi')
                )
                
                logger.info(f"✅ Entry analysis complete: {analysis.action} (Confidence: {analysis.confidence:.1f}%)")
                logger.info(f"   Reasoning: {analysis.reasoning}")
                
                return analysis
            else:
                return self._create_error_analysis(symbol, "Failed to parse AI response")
                
        except Exception as e:
            logger.error(f"Error analyzing entry for {symbol}: {e}", exc_info=True)
            return self._create_error_analysis(symbol, str(e))
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current stock price from broker"""
        try:
            # Try to get quote from broker
            # BrokerAdapter/Tradier method
            if hasattr(self.broker_client, 'get_quote'):
                quote = self.broker_client.get_quote(symbol)
                if quote:
                    # Tradier format
                    if isinstance(quote, dict):
                        return float(quote.get('last', 0))
                    # Other formats
                    return float(quote)
            
            # IBKR method
            if hasattr(self.broker_client, 'get_market_data'):
                data = self.broker_client.get_market_data(symbol)
                if data and 'last' in data:
                    return float(data['last'])
            
            logger.warning(f"Unable to fetch price for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    def _get_technical_indicators(self, symbol: str) -> Dict:
        """Get technical indicators for stock entry analysis"""
        try:
            # Try to get historical data from broker
            bars = self._get_historical_bars(symbol)
            
            if not bars or len(bars) < 20:
                logger.warning(f"Insufficient historical data for {symbol}")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            
            # Ensure we have required columns
            if 'close' not in df.columns:
                logger.warning(f"Missing 'close' column for {symbol}")
                return {}
            
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            indicators['ema_20'] = df['close'].ewm(span=20).mean().iloc[-1]
            indicators['ema_50'] = df['close'].ewm(span=50).mean().iloc[-1] if len(df) >= 50 else indicators['ema_20']
            indicators['price_vs_ema20'] = ((df['close'].iloc[-1] - indicators['ema_20']) / indicators['ema_20']) * 100
            
            # RSI
            delta = df['close'].diff().astype(float)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_histogram'] = (macd - signal).iloc[-1]
            
            # Volume
            indicators['volume_avg'] = df['volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_current'] = df['volume'].iloc[-1]
            indicators['volume_change_pct'] = ((indicators['volume_current'] / indicators['volume_avg']) - 1) * 100 if indicators['volume_avg'] > 0 else 0
            
            # Support/Resistance (recent highs/lows)
            indicators['support'] = df['low'].tail(20).min()
            indicators['resistance'] = df['high'].tail(20).max()
            indicators['distance_to_support'] = ((df['close'].iloc[-1] - indicators['support']) / df['close'].iloc[-1]) * 100
            indicators['distance_to_resistance'] = ((indicators['resistance'] - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
            
            # Recent momentum (last 5 bars)
            indicators['recent_move_pct'] = ((df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100 if len(df) >= 5 else 0
            
            # Trend
            if indicators['ema_20'] > indicators['ema_50']:
                indicators['trend'] = 'BULLISH'
            elif indicators['ema_20'] < indicators['ema_50']:
                indicators['trend'] = 'BEARISH'
            else:
                indicators['trend'] = 'NEUTRAL'
            
            # Volatility (recent price swings)
            indicators['volatility_5d'] = (df['high'].tail(5).max() - df['low'].tail(5).min()) / df['close'].iloc[-1] * 100
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}", exc_info=True)
            return {}
    
    def _get_stock_news_sentiment(self, symbol: str) -> Tuple[List[Dict], Optional[float]]:
        """
        Fetch recent stock news and analyze sentiment with FinBERT (ENHANCED)
        
        Returns:
            (recent_news, sentiment_score) where news is list of dicts and score is 0-100
        """
        try:
            import yfinance as yf
            from services.finbert_sentiment import get_finbert_analyzer
            
            # Get stock info and news
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                logger.debug(f"No news available for {symbol}")
                return [], None
            
            # Initialize FinBERT analyzer
            finbert = get_finbert_analyzer()
            
            recent_news = []
            sentiment_scores = []
            
            # Process top 5 most recent news articles
            for article in news[:5]:
                try:
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    
                    if not title:
                        continue
                    
                    # Analyze sentiment with FinBERT
                    text_to_analyze = f"{title}. {summary}" if summary else title
                    sentiment_result = finbert.analyze_sentiment(text_to_analyze)
                    
                    # Map FinBERT sentiment to trading sentiment
                    trading_sentiment = {
                        'positive': 'BULLISH',
                        'negative': 'BEARISH',
                        'neutral': 'NEUTRAL'
                    }.get(sentiment_result.sentiment, 'NEUTRAL')
                    
                    recent_news.append({
                        'title': title,
                        'sentiment': trading_sentiment,
                        'confidence': sentiment_result.confidence
                    })
                    
                    # Calculate score for aggregation (-1 to 1)
                    if sentiment_result.sentiment == 'positive':
                        sentiment_scores.append(sentiment_result.confidence)
                    elif sentiment_result.sentiment == 'negative':
                        sentiment_scores.append(-sentiment_result.confidence)
                    else:
                        sentiment_scores.append(0.0)
                        
                except Exception as e:
                    logger.debug(f"Error processing news article: {e}")
                    continue
            
            # Calculate aggregate sentiment score (0-100 scale)
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                sentiment_score = (avg_sentiment + 1.0) * 50  # -1 to 1 -> 0 to 100
                logger.info(f"📰 Fetched {len(recent_news)} news articles for {symbol} (sentiment: {sentiment_score:.1f}/100)")
            else:
                sentiment_score = 50.0  # Neutral
            
            return recent_news, sentiment_score
            
        except ImportError:
            logger.debug("yfinance not installed, skipping news analysis")
            return [], None
        except Exception as e:
            logger.debug(f"Error fetching news sentiment for {symbol}: {e}")
            return [], None
    
    def _get_historical_bars(self, symbol: str) -> List[Dict]:
        """Get historical bars from broker or fallback to yfinance"""
        try:
            # Try IBKR method first (if available)
            if hasattr(self.broker_client, 'get_historical_data'):
                bars = self.broker_client.get_historical_data(
                    symbol,
                    duration='2 M',
                    bar_size='1 day'
                )
                
                if bars:
                    # Convert IBKR format to standard format
                    return [{
                        'date': bar.get('date'),
                        'open': bar.get('open'),
                        'high': bar.get('high'),
                        'low': bar.get('low'),
                        'close': bar.get('close'),
                        'volume': bar.get('volume')
                    } for bar in bars]
            
            # Fallback to yfinance for Tradier and other brokers
            logger.debug(f"Using yfinance fallback for historical data: {symbol}")
            try:
                import yfinance as yf
                from datetime import datetime, timedelta
                
                # Get 60 days of data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if hist.empty:
                    logger.warning(f"No historical data from yfinance for {symbol}")
                    return []
                
                # Convert to list of dicts
                bars = []
                for date, row in hist.iterrows():
                    bars.append({
                        'date': date,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume'])
                    })
                
                logger.debug(f"Retrieved {len(bars)} bars from yfinance for {symbol}")
                return bars
                
            except ImportError:
                logger.warning("yfinance not installed. Install with: pip install yfinance")
                return []
            except Exception as e:
                logger.error(f"Error fetching from yfinance for {symbol}: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Error fetching historical bars for {symbol}: {e}")
            return []
    
    def _build_entry_analysis_prompt(
        self,
        symbol: str,
        side: str,
        current_price: float,
        position_size: float,
        risk_pct: float,
        take_profit_pct: float,
        technical_data: Dict,
        recent_news: Optional[List[Dict]] = None,
        sentiment_score: Optional[float] = None
    ) -> str:
        """Build AI prompt for stock entry analysis (ENHANCED with real-time news)"""
        
        prompt = f"""
Analyze this STOCK trade with REAL-TIME MARKET CONTEXT and determine if NOW is optimal entry timing or WAIT for better conditions.

**Stock Trade Request:**
- Symbol: {symbol}
- Side: {side} (BUY or SELL/SHORT)
- Current Price: ${current_price:,.2f}
- Position Size: ${position_size:,.2f}
- Risk: {risk_pct}% | Target: {take_profit_pct}%
"""
        
        if technical_data:
            prompt += f"""
**Technical Indicators:**
- RSI: {technical_data.get('rsi', 0):.2f} {'(OVERSOLD ✓)' if technical_data.get('rsi', 50) < 30 else '(OVERBOUGHT ⚠️)' if technical_data.get('rsi', 50) > 70 else '(Neutral)'}
- MACD: {technical_data.get('macd', 0):.4f} (Signal: {technical_data.get('macd_signal', 0):.4f})
- MACD Histogram: {technical_data.get('macd_histogram', 0):.4f} {'(Bullish ✓)' if technical_data.get('macd_histogram', 0) > 0 else '(Bearish ⚠️)'}
- EMA 20: ${technical_data.get('ema_20', 0):,.2f} | EMA 50: ${technical_data.get('ema_50', 0):,.2f}
- Price vs EMA20: {technical_data.get('price_vs_ema20', 0):+.2f}%
- Trend: {technical_data.get('trend', 'NEUTRAL')}
- Volume Change: {technical_data.get('volume_change_pct', 0):+.1f}% {'(High volume ✓)' if technical_data.get('volume_change_pct', 0) > 50 else '(Low volume ⚠️)' if technical_data.get('volume_change_pct', 0) < -30 else ''}
- Recent Move (5d): {technical_data.get('recent_move_pct', 0):+.2f}%
- Support: ${technical_data.get('support', 0):,.2f} (Distance: {technical_data.get('distance_to_support', 0):.2f}%)
- Resistance: ${technical_data.get('resistance', 0):,.2f} (Distance: {technical_data.get('distance_to_resistance', 0):.2f}%)
- Volatility (5d): {technical_data.get('volatility_5d', 0):.2f}%
"""
        else:
            prompt += "\n**Technical Indicators:** Unavailable (analyze price action)\n"
        
        # ENHANCED: Add real-time news and sentiment context
        if recent_news and len(recent_news) > 0:
            prompt += f"""
**📰 BREAKING NEWS & MARKET SENTIMENT (Recent):**
"""
            for news in recent_news[:5]:  # Top 5 most recent
                title = news.get('title', 'No title')
                sentiment = news.get('sentiment', 'NEUTRAL')
                confidence = news.get('confidence', 0.0)
                
                # Add emoji indicators
                sentiment_emoji = '🟢' if sentiment == 'BULLISH' else '🔴' if sentiment == 'BEARISH' else '⚪'
                
                prompt += f"""
- {sentiment_emoji} {title[:90]}
  Sentiment: {sentiment} (Confidence: {confidence:.0%})
"""
            
            if sentiment_score is not None:
                sentiment_emoji = '🟢 BULLISH' if sentiment_score > 65 else '🔴 BEARISH' if sentiment_score < 35 else '⚪ NEUTRAL'
                prompt += f"""
**Aggregate News Sentiment: {sentiment_score:.1f}/100** ({sentiment_emoji})
"""
        else:
            prompt += """
**News Context:** No significant recent news
"""
        
        prompt += f"""
**Stock-Specific Considerations:**
- Market hours (optimal 9:30am-4pm ET)
- Liquidity (volume > 1M shares/day preferred)
- Spread (tight spreads < 0.5%)
- News/earnings events
- Sector sentiment

**Entry Timing Framework:**
Evaluate these critical factors:

1. **Trend Quality** (0-100):
   - Is the trend strong and confirmed?
   - Are we entering WITH the trend or against it?
   - Is momentum building or weakening?

2. **Entry Price** (0-100):
   - Are we buying at support or resistance?
   - Is this a good price or chasing a rally?
   - Is there room to the upside?

3. **Technical Setup** (0-100):
   - Are indicators aligned for entry?
   - Is RSI in a good zone (not overbought)?
   - Is MACD confirming?

4. **Risk/Reward** (0-100):
   - Is the R:R favorable at this price?
   - Would waiting for pullback improve R:R?
   - Is the stop loss appropriate?

5. **Volume** (0-100):
   - Is volume confirming the move?
   - Is there sufficient liquidity?
   - Or is volume dying (low conviction)?

**Available Actions:**
1. **ENTER_NOW** - Excellent setup, execute immediately (85%+ confidence)
2. **WAIT_FOR_PULLBACK** - Good stock but overbought, wait for better entry (provide wait_for_price and wait_for_rsi)
3. **WAIT_FOR_BREAKOUT** - Consolidating, wait for confirmed move (provide wait_for_price)
4. **PLACE_LIMIT_ORDER** - Set limit at better price (provide suggested_entry)
5. **DO_NOT_ENTER** - Poor setup, avoid this trade (<50% confidence)

**Critical Rules for Stock Entry Timing:**
- Only ENTER_NOW if 85%+ confidence AND good technical setup
- WAIT_FOR_PULLBACK if RSI > 70 (overbought) even if trend is good
- WAIT_FOR_PULLBACK if price just moved >5% in last few days (let it reset)
- WAIT_FOR_BREAKOUT if price is consolidating near resistance
- DO_NOT_ENTER if trend is against you or setup is poor
- Consider market hours - best entries during 9:30am-3:30pm ET

**Respond ONLY with valid JSON (no other text):**
{{
    "action": "ENTER_NOW|WAIT_FOR_PULLBACK|WAIT_FOR_BREAKOUT|PLACE_LIMIT_ORDER|DO_NOT_ENTER",
    "confidence": 0-100,
    "reasoning": "2-3 sentence explanation of timing analysis",
    "urgency": "LOW|MEDIUM|HIGH",
    "suggested_entry": current_or_better_price,
    "suggested_stop": price_for_stop_loss,
    "suggested_target": price_for_take_profit,
    "risk_reward_ratio": calculated_rr_ratio,
    "wait_for_price": price_to_wait_for_or_null,
    "wait_for_rsi": rsi_level_to_wait_for_or_null,
    "technical_score": 0-100,
    "trend_score": 0-100,
    "timing_score": 0-100,
    "risk_score": 0-100
}}
"""
        
        return prompt
    
    def _parse_ai_response(self, response: str) -> Optional[Dict]:
        """Parse AI JSON response"""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in AI response")
                return None
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response JSON: {e}")
            logger.debug(f"Response: {response}")
            return None
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return None
    
    def _create_error_analysis(self, symbol: str, error_msg: str) -> EntryAnalysis:
        """Create error analysis response"""
        return EntryAnalysis(
            symbol=symbol,
            action=EntryAction.DO_NOT_ENTER.value,
            confidence=0.0,
            reasoning=f"Analysis error: {error_msg}",
            urgency="LOW",
            current_price=0.0,
            technical_score=0.0,
            trend_score=0.0,
            timing_score=0.0,
            risk_score=100.0
        )
    
    def monitor_entry_opportunity(
        self,
        symbol: str,
        side: str,
        position_size: float,
        risk_pct: float,
        take_profit_pct: float,
        analysis: EntryAnalysis,
        auto_execute: bool = False
    ) -> str:
        """
        Start monitoring for stock entry opportunity
        
        Args:
            symbol: Stock symbol
            side: BUY or SELL
            position_size: Position size in USD
            risk_pct: Risk percentage
            take_profit_pct: Take profit percentage
            analysis: Original entry analysis
            auto_execute: Auto-execute when conditions met
        
        Returns:
            Opportunity ID
        """
        opportunity_id = f"{symbol}_{int(time.time())}"
        
        # Extract target conditions from analysis
        target_conditions = {
            'wait_for_price': analysis.wait_for_price,
            'wait_for_rsi': analysis.wait_for_rsi,
            'action': analysis.action
        }
        
        opportunity = MonitoredEntryOpportunity(
            symbol=symbol,
            side=side,
            target_price=analysis.wait_for_price,
            target_rsi=analysis.wait_for_rsi,
            target_conditions=target_conditions,
            position_size=position_size,
            risk_pct=risk_pct,
            take_profit_pct=take_profit_pct,
            original_analysis=analysis,
            created_time=datetime.now(),
            last_check_time=datetime.now(),
            auto_execute=auto_execute
        )
        
        self.opportunities[opportunity_id] = opportunity
        
        logger.info(f"📊 Monitoring stock entry opportunity: {symbol} (ID: {opportunity_id})")
        logger.info(f"   Target price: {analysis.wait_for_price if analysis.wait_for_price else 'Any'}")
        logger.info(f"   Target RSI: {analysis.wait_for_rsi if analysis.wait_for_rsi else 'Any'}")
        logger.info(f"   Auto-execute: {auto_execute}")
        
        # Save state
        self._save_state()
        
        # Start monitoring thread if not running
        if not self.is_running:
            self.start_monitoring()
        
        return opportunity_id
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.is_running:
            logger.warning("Stock monitoring already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🎯 Stock entry opportunity monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running and not self.stop_event.is_set():
            try:
                if self.opportunities:
                    self._check_opportunities()
                
                # Wait for next check
                self.stop_event.wait(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in stock entry monitoring loop: {e}", exc_info=True)
                time.sleep(self.check_interval)
    
    def _check_opportunities(self):
        """Check all monitored stock opportunities"""
        for opp_id, opportunity in list(self.opportunities.items()):
            try:
                # Get current market price
                current_price = self._get_current_price(opportunity.symbol)
                if not current_price:
                    continue
                
                opportunity.current_price = current_price
                opportunity.last_check_time = datetime.now()
                
                # Get technical indicators
                technical_data = self._get_technical_indicators(opportunity.symbol)
                
                # Check if conditions are met
                conditions_met = self._check_entry_conditions(opportunity, current_price, technical_data)
                
                if conditions_met:
                    logger.info(f"✅ Entry conditions met for {opportunity.symbol}!")
                    logger.info(f"   Price: ${current_price:,.2f}")
                    if technical_data:
                        logger.info(f"   RSI: {technical_data.get('rsi', 0):.2f}")
                    
                    # Send notification
                    if not opportunity.notification_sent:
                        self._send_entry_notification(opportunity, current_price, technical_data)
                        opportunity.notification_sent = True
                    
                    # Auto-execute if enabled
                    if opportunity.auto_execute and self.enable_auto_entry:
                        logger.info(f"🤖 Auto-executing entry for {opportunity.symbol}")
                        # TODO: Integrate with trade execution via callback
                        # This would be handled by auto-trader
                        self.remove_opportunity(opp_id)
                
            except Exception as e:
                logger.error(f"Error checking opportunity {opp_id}: {e}", exc_info=True)
    
    def _check_entry_conditions(
        self,
        opportunity: MonitoredEntryOpportunity,
        current_price: float,
        technical_data: Dict
    ) -> bool:
        """Check if entry conditions are met"""
        conditions_met = True
        
        # Check price condition
        if opportunity.target_price:
            if opportunity.side == "BUY":
                # For BUY, we want price to come down to target
                conditions_met = conditions_met and (current_price <= opportunity.target_price)
            else:
                # For SELL, we want price to go up to target
                conditions_met = conditions_met and (current_price >= opportunity.target_price)
        
        # Check RSI condition
        if opportunity.target_rsi and technical_data:
            current_rsi = technical_data.get('rsi', 0)
            if opportunity.side == "BUY":
                # For BUY, we want RSI to reset (come down)
                conditions_met = conditions_met and (current_rsi <= opportunity.target_rsi)
            else:
                # For SELL, we want RSI to increase
                conditions_met = conditions_met and (current_rsi >= opportunity.target_rsi)
        
        return conditions_met
    
    def _send_entry_notification(
        self,
        opportunity: MonitoredEntryOpportunity,
        current_price: float,
        technical_data: Dict
    ):
        """Send notification that entry conditions are met"""
        try:
            # Try to import Discord webhook (optional dependency)
            try:
                from src.integrations.discord_webhook import send_discord_alert
                from models.alerts import TradingAlert, AlertType, AlertPriority
            except (ImportError, ModuleNotFoundError):
                logger.debug("Discord webhook not available, logging notification instead")
                rsi_text = f"RSI: {technical_data.get('rsi', 0):.2f}" if technical_data else "N/A"
                trend_text = f"Trend: {technical_data.get('trend', 'N/A')}" if technical_data else "N/A"
                logger.info(f"📢 ENTRY NOTIFICATION: {opportunity.symbol} {opportunity.side} @ ${current_price:,.2f} | {rsi_text} | {trend_text}")
                return
            
            rsi_text = f"RSI: {technical_data.get('rsi', 0):.2f}" if technical_data else ""
            trend_text = f"Trend: {technical_data.get('trend', 'N/A')}" if technical_data else ""
            
            alert = TradingAlert(
                ticker=opportunity.symbol,
                alert_type=AlertType.AI_SIGNAL,
                message=f"🎯 Stock Entry Opportunity Ready! {opportunity.side} @ ${current_price:,.2f}",
                priority=AlertPriority.HIGH,
                details={
                    'action': 'STOCK_ENTRY_READY',
                    'symbol': opportunity.symbol,
                    'side': opportunity.side,
                    'current_price': current_price,
                    'position_size': opportunity.position_size,
                    'rsi': technical_data.get('rsi', 0) if technical_data else 0,
                    'trend': technical_data.get('trend', 'N/A') if technical_data else 'N/A',
                    'reasoning': opportunity.original_analysis.reasoning
                }
            )
            send_discord_alert(alert)
            logger.info(f"📢 Entry notification sent for {opportunity.symbol}")
            
        except Exception as e:
            logger.warning(f"Failed to send entry notification: {e}")
    
    def remove_opportunity(self, opportunity_id: str) -> bool:
        """Remove opportunity from monitoring"""
        if opportunity_id in self.opportunities:
            del self.opportunities[opportunity_id]
            logger.info(f"🗑️ Removed opportunity: {opportunity_id}")
            # Save state after removal
            self._save_state()
            return True
        return False
    
    def get_monitored_opportunities(self) -> List[MonitoredEntryOpportunity]:
        """Get all monitored opportunities"""
        return list(self.opportunities.values())
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Save state before stopping
        self._save_state()
        logger.info("🛑 Stock entry opportunity monitoring stopped")
    
    def _save_state(self):
        """Save monitored opportunities to file"""
        try:
            import os
            
            if not self.opportunities:
                # No opportunities to save, remove file if exists
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
                return
            
            # Convert opportunities to serializable format
            state = {}
            for opp_id, opp in self.opportunities.items():
                state[opp_id] = {
                    'symbol': opp.symbol,
                    'side': opp.side,
                    'target_price': opp.target_price,
                    'target_rsi': opp.target_rsi,
                    'target_conditions': opp.target_conditions,
                    'position_size': opp.position_size,
                    'risk_pct': opp.risk_pct,
                    'take_profit_pct': opp.take_profit_pct,
                    'created_time': opp.created_time.isoformat(),
                    'last_check_time': opp.last_check_time.isoformat(),
                    'current_price': opp.current_price,
                    'notification_sent': opp.notification_sent,
                    'auto_execute': opp.auto_execute,
                    # Save original analysis
                    'original_analysis': {
                        'symbol': opp.original_analysis.symbol,
                        'action': opp.original_analysis.action,
                        'confidence': opp.original_analysis.confidence,
                        'reasoning': opp.original_analysis.reasoning,
                        'urgency': opp.original_analysis.urgency,
                        'current_price': opp.original_analysis.current_price,
                        'suggested_entry': opp.original_analysis.suggested_entry,
                        'suggested_stop': opp.original_analysis.suggested_stop,
                        'suggested_target': opp.original_analysis.suggested_target,
                        'risk_reward_ratio': opp.original_analysis.risk_reward_ratio,
                        'technical_score': opp.original_analysis.technical_score,
                        'trend_score': opp.original_analysis.trend_score,
                        'timing_score': opp.original_analysis.timing_score,
                        'risk_score': opp.original_analysis.risk_score,
                        'wait_for_price': opp.original_analysis.wait_for_price,
                        'wait_for_rsi': opp.original_analysis.wait_for_rsi,
                        'analysis_time': opp.original_analysis.analysis_time.isoformat() if opp.original_analysis.analysis_time else None
                    }
                }
            
            # Write to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"💾 Saved {len(state)} monitored stock opportunities to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error saving stock entry assistant state: {e}", exc_info=True)
    
    def _load_state(self):
        """Load monitored opportunities from file"""
        try:
            import os
            
            if not os.path.exists(self.state_file):
                logger.debug("No saved stock monitors found")
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            if not state:
                return
            
            # Restore opportunities
            for opp_id, opp_data in state.items():
                try:
                    # Reconstruct EntryAnalysis
                    analysis_data = opp_data['original_analysis']
                    analysis = EntryAnalysis(
                        symbol=analysis_data['symbol'],
                        action=analysis_data['action'],
                        confidence=analysis_data['confidence'],
                        reasoning=analysis_data['reasoning'],
                        urgency=analysis_data['urgency'],
                        current_price=analysis_data['current_price'],
                        suggested_entry=analysis_data.get('suggested_entry'),
                        suggested_stop=analysis_data.get('suggested_stop'),
                        suggested_target=analysis_data.get('suggested_target'),
                        risk_reward_ratio=analysis_data['risk_reward_ratio'],
                        technical_score=analysis_data['technical_score'],
                        trend_score=analysis_data['trend_score'],
                        timing_score=analysis_data['timing_score'],
                        risk_score=analysis_data['risk_score'],
                        wait_for_price=analysis_data.get('wait_for_price'),
                        wait_for_rsi=analysis_data.get('wait_for_rsi'),
                        analysis_time=datetime.fromisoformat(analysis_data['analysis_time'])
                    )
                    
                    # Reconstruct MonitoredEntryOpportunity
                    opportunity = MonitoredEntryOpportunity(
                        symbol=opp_data['symbol'],
                        side=opp_data['side'],
                        target_price=opp_data.get('target_price'),
                        target_rsi=opp_data.get('target_rsi'),
                        target_conditions=opp_data['target_conditions'],
                        position_size=opp_data['position_size'],
                        risk_pct=opp_data['risk_pct'],
                        take_profit_pct=opp_data['take_profit_pct'],
                        original_analysis=analysis,
                        created_time=datetime.fromisoformat(opp_data['created_time']),
                        last_check_time=datetime.fromisoformat(opp_data['last_check_time']),
                        current_price=opp_data['current_price'],
                        notification_sent=opp_data['notification_sent'],
                        auto_execute=opp_data['auto_execute']
                    )
                    
                    self.opportunities[opp_id] = opportunity
                    logger.info(f"📂 Restored stock monitor: {opportunity.symbol} (target: ${opportunity.target_price if opportunity.target_price else 'breakout'})")
                    
                except Exception as e:
                    logger.warning(f"Failed to restore opportunity {opp_id}: {e}")
                    continue
            
            if self.opportunities:
                logger.info(f"✅ Loaded {len(self.opportunities)} monitored stock opportunities from previous session")
                # Auto-start monitoring if we have opportunities
                if not self.is_running:
                    self.start_monitoring()
            
        except Exception as e:
            logger.error(f"Error loading stock entry assistant state: {e}", exc_info=True)


# Singleton instance
_stock_entry_assistant_instance = None

def get_ai_stock_entry_assistant(
    broker_client,
    llm_analyzer=None,
    check_interval_seconds: int = 60,
    enable_auto_entry: bool = False
):
    """Get or create singleton Stock AI Entry Assistant instance"""
    global _stock_entry_assistant_instance
    
    if _stock_entry_assistant_instance is None:
        _stock_entry_assistant_instance = AIStockEntryAssistant(
            broker_client=broker_client,
            llm_analyzer=llm_analyzer,
            check_interval_seconds=check_interval_seconds,
            enable_auto_entry=enable_auto_entry
        )
    
    return _stock_entry_assistant_instance


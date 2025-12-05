"""
AI Entry Assistant - Intelligent Trade Entry Timing
Analyzes market conditions BEFORE entry to optimize timing and reduce bad trades
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, List
from loguru import logger
import json
import time
import pandas as pd
import threading
from clients.kraken_client import OrderSide, OrderType


class EntryAction(Enum):
    """Recommended entry actions"""
    ENTER_NOW = "ENTER_NOW"
    WAIT_FOR_PULLBACK = "WAIT_FOR_PULLBACK"
    WAIT_FOR_BREAKOUT = "WAIT_FOR_BREAKOUT"
    DO_NOT_ENTER = "DO_NOT_ENTER"
    PLACE_LIMIT_ORDER = "PLACE_LIMIT_ORDER"


@dataclass
class EntryAnalysis:
    """AI entry recommendation"""
    pair: str
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
    bear_case: Optional[str] = None  # Devil's advocate: why this trade could fail
    analysis_time: datetime = None
    
    def __post_init__(self):
        if self.analysis_time is None:
            self.analysis_time = datetime.now()


@dataclass
class MonitoredEntryOpportunity:
    """Entry opportunity being monitored"""
    pair: str
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


class AIEntryAssistant:
    """
    AI-powered entry assistant that analyzes market conditions
    before entering trades to optimize timing
    """
    
    def __init__(
        self,
        kraken_client,
        llm_analyzer=None,
        check_interval_seconds: int = 60,
        enable_auto_entry: bool = False,
        min_confidence_for_entry: float = 85.0,
        min_confidence_for_auto: float = 90.0
    ):
        """
        Initialize AI Entry Assistant
        
        Args:
            kraken_client: Kraken API client
            llm_analyzer: LLM strategy analyzer for AI decisions
            check_interval_seconds: How often to check monitored opportunities
            enable_auto_entry: Allow auto-execution on high confidence
            min_confidence_for_entry: Minimum confidence to recommend entry
            min_confidence_for_auto: Minimum confidence for auto-execution
        """
        self.kraken_client = kraken_client
        self.llm_analyzer = llm_analyzer
        self.check_interval = check_interval_seconds
        self.enable_auto_entry = enable_auto_entry
        self.min_confidence_entry = min_confidence_for_entry
        self.min_confidence_auto = min_confidence_for_auto
        
        # Monitored opportunities
        self.opportunities: Dict[str, MonitoredEntryOpportunity] = {}
        
        # State persistence
        self.state_file = "ai_entry_monitors.json"
        
        # Monitoring thread
        self.monitoring_thread = None
        self.is_running = False
        self.stop_event = threading.Event()
        
        logger.info("🎯 AI Entry Assistant initialized")
        logger.info(f"   Check interval: {check_interval_seconds}s")
        logger.info(f"   Auto-entry: {'Enabled' if enable_auto_entry else 'Disabled'}")
        logger.info(f"   Min confidence (entry): {min_confidence_for_entry}%")
        logger.info(f"   Min confidence (auto): {min_confidence_for_auto}%")
        
        # Load saved monitors from previous session
        self._load_state()
    
    def analyze_entry(
        self,
        pair: str,
        side: str,
        position_size: float,
        risk_pct: float,
        take_profit_pct: float,
        additional_context: str = None
    ) -> EntryAnalysis:
        """
        Analyze if NOW is a good time to enter this trade
        
        Args:
            pair: Trading pair (e.g., "HIPPO/USD")
            side: "BUY" or "SELL"
            position_size: Position size in USD
            risk_pct: Risk percentage for stop loss
            take_profit_pct: Take profit percentage
            additional_context: Optional context (e.g., scanner analysis) to include
        
        Returns:
            EntryAnalysis with recommendation
        """
        try:
            logger.info(f"🎯 Analyzing entry for {pair} ({side})")
            
            # Get current market data
            ticker_info = self.kraken_client.get_ticker_info(pair)
            if not ticker_info:
                return self._create_error_analysis(pair, "Failed to fetch ticker data")
            
            current_price = float(ticker_info.get('c', [0])[0])
            if current_price <= 0:
                return self._create_error_analysis(pair, "Invalid price data")
            
            # Get technical indicators
            technical_data = self._get_technical_indicators(pair)
            if not technical_data:
                logger.warning(f"⚠️ Limited technical data for {pair}, using price action only")
            
            # Build AI prompt for entry analysis
            prompt = self._build_entry_analysis_prompt(
                pair=pair,
                side=side,
                current_price=current_price,
                position_size=position_size,
                risk_pct=risk_pct,
                take_profit_pct=take_profit_pct,
                technical_data=technical_data,
                additional_context=additional_context
            )
            
            # Get AI recommendation - use proper routing to handle both Ollama and OpenRouter
            response = self.llm_analyzer._call_llm_api(prompt)
            
            if not response:
                logger.error("Failed to get AI response")
                # If we have scanner context (any scanner context means it was approved by multi-config), use fallback
                has_scanner_context = additional_context and (
                    "SCANNER" in additional_context.upper() or 
                    "APPROVED" in additional_context.upper() or 
                    "RECOMMENDED" in additional_context.upper() or 
                    "SCORE" in additional_context.upper() or
                    "STRATEGY" in additional_context.upper()
                )
                
                if has_scanner_context:
                    logger.warning(f"⚠️ AI call failed but scanner context present for {pair} - using technical fallback analysis")
                    logger.debug(f"Scanner context preview: {additional_context[:200] if additional_context else 'None'}...")
                    return self._create_scanner_fallback_analysis(
                        pair=pair,
                        side=side,
                        current_price=current_price,
                        risk_pct=risk_pct,
                        take_profit_pct=take_profit_pct,
                        technical_data=technical_data,
                        scanner_context=additional_context
                    )
                else:
                    logger.warning(f"⚠️ AI call failed for {pair} with no scanner context - returning error analysis")
                    return self._create_error_analysis(pair, "LLM API call failed (insufficient credits or API error)")
            
            decision_data = self._parse_ai_response(response)
            
            if decision_data:
                # Safe float conversion helper
                def safe_float(value, default=0.0):
                    """Convert value to float, return default if None or invalid"""
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                
                # Safe optional float conversion (returns None if value is missing)
                def safe_optional_float(value):
                    """Convert value to float or None if missing/invalid"""
                    if value is None or value == '':
                        return None
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return None
                
                analysis = EntryAnalysis(
                    pair=pair,
                    action=decision_data.get('action', 'DO_NOT_ENTER'),
                    confidence=safe_float(decision_data.get('confidence'), 0.0),
                    reasoning=decision_data.get('reasoning', 'No reasoning provided'),
                    urgency=decision_data.get('urgency', 'LOW'),
                    current_price=current_price,
                    suggested_entry=safe_optional_float(decision_data.get('suggested_entry')),
                    suggested_stop=safe_optional_float(decision_data.get('suggested_stop')),
                    suggested_target=safe_optional_float(decision_data.get('suggested_target')),
                    risk_reward_ratio=safe_float(decision_data.get('risk_reward_ratio'), 0.0),
                    technical_score=safe_float(decision_data.get('technical_score'), 0.0),
                    trend_score=safe_float(decision_data.get('trend_score'), 0.0),
                    timing_score=safe_float(decision_data.get('timing_score'), 0.0),
                    risk_score=safe_float(decision_data.get('risk_score'), 100.0),
                    wait_for_price=safe_optional_float(decision_data.get('wait_for_price')),
                    wait_for_rsi=safe_optional_float(decision_data.get('wait_for_rsi')),
                    bear_case=decision_data.get('bear_case')
                )
                
                logger.info(f"✅ Entry analysis complete: {analysis.action} (Confidence: {analysis.confidence:.1f}%)")
                logger.info(f"   Reasoning: {analysis.reasoning}")
                if analysis.bear_case:
                    logger.info(f"   Bear Case: {analysis.bear_case}")
                
                return analysis
            else:
                return self._create_error_analysis(pair, "Failed to parse AI response")
                
        except Exception as e:
            logger.error("Error analyzing entry for {pair}: {}", str(e), exc_info=True)
            return self._create_error_analysis(pair, str(e))
    
    def _get_technical_indicators(self, pair: str) -> Dict:
        """Get technical indicators for entry analysis"""
        try:
            # Get OHLC data (1 hour candles, last 100)
            ohlc_data = self.kraken_client.get_ohlc_data(pair, interval=60, since=None)
            
            if not ohlc_data or len(ohlc_data) < 20:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlc_data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
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
            delta = df['close'].diff()
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
            indicators['volume_change_pct'] = ((indicators['volume_current'] / indicators['volume_avg']) - 1) * 100
            
            # Support/Resistance (recent highs/lows)
            indicators['support'] = df['low'].tail(20).min()
            indicators['resistance'] = df['high'].tail(20).max()
            indicators['distance_to_support'] = ((df['close'].iloc[-1] - indicators['support']) / df['close'].iloc[-1]) * 100
            indicators['distance_to_resistance'] = ((indicators['resistance'] - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
            
            # Recent momentum (last 5 candles)
            indicators['recent_move_pct'] = ((df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]) * 100 if len(df) >= 5 else 0
            
            # Trend
            if indicators['ema_20'] > indicators['ema_50']:
                indicators['trend'] = 'BULLISH'
            elif indicators['ema_20'] < indicators['ema_50']:
                indicators['trend'] = 'BEARISH'
            else:
                indicators['trend'] = 'NEUTRAL'
            
            # Volatility (recent price swings)
            indicators['volatility_5h'] = (df['high'].tail(5).max() - df['low'].tail(5).min()) / df['close'].iloc[-1] * 100
            
            return indicators
            
        except Exception as e:
            logger.error("Error calculating indicators for {pair}: {}", str(e), exc_info=True)
            return {}
    
    def _build_entry_analysis_prompt(
        self,
        pair: str,
        side: str,
        current_price: float,
        position_size: float,
        risk_pct: float,
        take_profit_pct: float,
        technical_data: Dict,
        additional_context: str = None
    ) -> str:
        """Build AI prompt for entry analysis"""
        
        prompt = f"""
Analyze this POTENTIAL trade setup and determine if NOW is a good time to ENTER or if we should WAIT for better conditions.

**Trade Request:**
- Asset: {pair} ({side} position)
- Current Price: ${current_price:,.6f}
- Position Size: ${position_size:,.2f}
- Planned Risk: {risk_pct}% (stop loss)
- Planned Target: {take_profit_pct}% (take profit)
"""
        
        if technical_data:
            prompt += f"""
**Current Technical Indicators:**
- RSI: {technical_data.get('rsi', 0):.2f} {'(OVERSOLD ✓)' if technical_data.get('rsi', 50) < 30 else '(OVERBOUGHT ⚠️)' if technical_data.get('rsi', 50) > 70 else '(Neutral)'}
- MACD: {technical_data.get('macd', 0):.6f} (Signal: {technical_data.get('macd_signal', 0):.6f})
- MACD Histogram: {technical_data.get('macd_histogram', 0):.6f} {'(Bullish ✓)' if technical_data.get('macd_histogram', 0) > 0 else '(Bearish ⚠️)'}
- EMA 20: ${technical_data.get('ema_20', 0):,.6f} | EMA 50: ${technical_data.get('ema_50', 0):,.6f}
- Price vs EMA20: {technical_data.get('price_vs_ema20', 0):+.2f}%
- Trend: {technical_data.get('trend', 'NEUTRAL')}
- Volume Change: {technical_data.get('volume_change_pct', 0):+.1f}% {'(High volume ✓)' if technical_data.get('volume_change_pct', 0) > 50 else '(Low volume ⚠️)' if technical_data.get('volume_change_pct', 0) < -30 else ''}
- Recent Move (5h): {technical_data.get('recent_move_pct', 0):+.2f}%
- Support: ${technical_data.get('support', 0):,.6f} (Distance: {technical_data.get('distance_to_support', 0):.2f}%)
- Resistance: ${technical_data.get('resistance', 0):,.6f} (Distance: {technical_data.get('distance_to_resistance', 0):.2f}%)
- Volatility (5h): {technical_data.get('volatility_5h', 0):.2f}%
"""
        else:
            prompt += "\n**Technical Indicators:** Unavailable (analyze price action)\n"
        
        # Add scanner context if provided - but encourage INDEPENDENT evaluation
        if additional_context:
            prompt += f"\n{additional_context}\n"
            prompt += """
**SCANNER CONTEXT AVAILABLE - BUT EVALUATE INDEPENDENTLY**
Scanner context is provided for reference only. Your job is to CRITICALLY evaluate the setup.
Do NOT blindly trust scanner recommendations. Form your own opinion FIRST, then compare.

**INDEPENDENT EVALUATION REQUIRED:**
1. FIRST: Analyze the technical indicators above objectively (ignore scanner opinion)
2. SECOND: Form your preliminary conclusion about entry timing
3. THIRD: Compare your analysis with scanner - note any disagreements
4. FOURTH: Provide final recommendation with reasoning for any divergence from scanner

If your independent analysis disagrees with scanner, explain WHY in your reasoning.
"""
        
        prompt += f"""
**Entry Timing Analysis Framework:**
Evaluate these critical factors OBJECTIVELY (use full 0-100 range):

1. **Independent Technical Analysis** (REQUIRED BEFORE considering scanner):
   - What do the raw indicators above tell you?
   - Form your own opinion first, THEN compare to scanner if provided
   - Note any conflicts between your analysis and scanner

2. **Trend Alignment** (0-100 - use FULL range):
   - 0-30: Counter-trend, momentum weakening, likely reversal imminent
   - 31-50: Weak trend, mixed signals, wait for confirmation
   - 51-70: Moderate trend, some confirmation but risks present
   - 71-85: Strong trend with good confirmation
   - 86-100: Exceptional setup (RARE - only 5-10% of trades)

3. **Entry Price Quality** (0-100):
   - 0-30: Chasing a pump, resistance nearby, poor R:R
   - 31-50: Suboptimal entry, could wait for better price
   - 51-70: Acceptable entry, moderate setup
   - 71-85: Good entry near support with room to run
   - 86-100: Excellent entry (pullback to key level, rare)

4. **Technical Setup** (0-100):
   - RSI overbought (>70) = score 30-40 max
   - RSI neutral (40-60) = score 50-70
   - RSI oversold (<30) = score 70-90
   - MACD divergence = subtract 15-20 points

5. **Risk/Reward Assessment** (0-100):
   - R:R < 1.5 = score 0-40
   - R:R 1.5-2.0 = score 40-60
   - R:R 2.0-3.0 = score 60-80
   - R:R > 3.0 = score 80-100

**DEVIL'S ADVOCATE SECTION (REQUIRED):**
Before recommending entry, you MUST consider:
- Why could this trade FAIL? What's the bear case?
- What contradictory signals exist?
- What could go wrong in the next 24 hours?
- Is there something I'm missing or being too optimistic about?

**Available Actions:**
1. **ENTER_NOW** - Excellent setup after critical evaluation (80%+ confidence REQUIRED)
2. **WAIT_FOR_PULLBACK** - Decent coin but entry price suboptimal (provide wait_for_price and wait_for_rsi)
3. **WAIT_FOR_BREAKOUT** - Consolidating, wait for confirmed move (provide wait_for_price)
4. **PLACE_LIMIT_ORDER** - Set limit at better price (provide suggested_entry)
5. **DO_NOT_ENTER** - Poor setup, high risk, or low conviction (<60% confidence)

**CALIBRATION GUIDELINES - USE THE FULL RANGE:**
- **90-100**: Exceptional (should be < 5% of all signals) - perfect storm conditions
- **80-89**: Strong entry (good setup, most factors aligned, minor concerns)
- **70-79**: Moderate (acceptable but not ideal, some yellow flags)
- **60-69**: Weak (proceed with caution, significant concerns)
- **50-59**: Poor (more reasons to skip than enter)
- **0-49**: Do not enter (high risk, unclear setup, red flags)

**CRITICAL RULES:**
- Average confidence across all trades should be ~55-65 (not 75+)
- ENTER_NOW requires 80%+ confidence (higher bar than before)
- DO_NOT_ENTER for anything <60% confidence
- RSI > 70 should almost never be ENTER_NOW
- Low volume = subtract 10-15 points minimum
- Counter-trend entries = subtract 20 points minimum
- Use WAIT actions more frequently - timing matters

**Respond ONLY with valid JSON (no other text):**
{{
    "action": "ENTER_NOW|WAIT_FOR_PULLBACK|WAIT_FOR_BREAKOUT|PLACE_LIMIT_ORDER|DO_NOT_ENTER",
    "confidence": 55,
    "reasoning": "2-3 sentences: First state your independent analysis. Then note any scanner agreement/disagreement. Include bear case consideration.",
    "bear_case": "1-2 sentences explaining why this trade could fail",
    "urgency": "LOW|MEDIUM|HIGH",
    "suggested_entry": {current_price},
    "suggested_stop": "{current_price * (1 - risk_pct/100):.6f}",
    "suggested_target": "{current_price * (1 + take_profit_pct/100):.6f}",
    "risk_reward_ratio": {take_profit_pct / risk_pct:.2f},
    "wait_for_price": null,
    "wait_for_rsi": null,
    "technical_score": 55,
    "trend_score": 55,
    "timing_score": 55,
    "risk_score": 50
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
                logger.error("No JSON found in response")
                return None
            
            json_str = response[start_idx:end_idx]
            
            # Pre-process to fix common LLM mistakes before JSON parsing
            # Fix bare identifiers like "current_or_better_price" -> null
            json_str = json_str.replace(': current_or_better_price', ': null')
            json_str = json_str.replace(': price_for_stop_loss', ': null')
            json_str = json_str.replace(': price_for_take_profit', ': null')
            json_str = json_str.replace(': calculated_rr_ratio', ': null')
            json_str = json_str.replace(': price_to_wait_for_or_null', ': null')
            json_str = json_str.replace(': rsi_level_to_wait_for_or_null', ': null')
            
            data = json.loads(json_str)
            
            # Post-process: Clean string prices like "$0.001518" -> 0.001518
            for field in ['suggested_entry', 'suggested_stop', 'suggested_target', 'wait_for_price']:
                if field in data and isinstance(data[field], str):
                    try:
                        # Remove $ and convert to float
                        data[field] = float(data[field].replace('$', '').replace(',', ''))
                    except (ValueError, AttributeError):
                        data[field] = None
            
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response JSON: {e}")
            logger.debug(f"Response: {response}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing AI response: {e}")
            return None
    
    def _create_error_analysis(self, pair: str, error_msg: str) -> EntryAnalysis:
        """Create error analysis response"""
        return EntryAnalysis(
            pair=pair,
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
    
    def _create_scanner_fallback_analysis(
        self,
        pair: str,
        side: str,
        current_price: float,
        risk_pct: float,
        take_profit_pct: float,
        technical_data: Dict,
        scanner_context: str
    ) -> EntryAnalysis:
        """
        Create fallback analysis when AI call fails but scanner approved the trade.
        Uses technical indicators and scanner context to provide a reasonable recommendation.
        """
        # Extract scanner confidence from context if available
        scanner_confidence = 70.0  # Default moderate confidence
        if "CONFIDENCE" in scanner_context.upper():
            import re
            conf_match = re.search(r'confidence[:\s]+(\d+)', scanner_context, re.IGNORECASE)
            if conf_match:
                scanner_confidence = float(conf_match.group(1))
        
        # Calculate technical scores from available data
        technical_score = 50.0  # Neutral default
        trend_score = 50.0
        timing_score = 50.0
        risk_score = 50.0
        
        if technical_data:
            # RSI-based timing score
            rsi = technical_data.get('rsi', 50)
            if side == "BUY":
                if rsi < 30:
                    timing_score = 80.0  # Oversold = good entry
                elif rsi < 50:
                    timing_score = 65.0  # Below neutral = decent
                elif rsi > 70:
                    timing_score = 30.0  # Overbought = wait
                else:
                    timing_score = 50.0
            else:  # SELL
                if rsi > 70:
                    timing_score = 80.0  # Overbought = good short entry
                elif rsi > 50:
                    timing_score = 65.0  # Above neutral = decent
                elif rsi < 30:
                    timing_score = 30.0  # Oversold = wait
                else:
                    timing_score = 50.0
            
            # Trend score
            trend = technical_data.get('trend', 'NEUTRAL')
            if (side == "BUY" and trend == "BULLISH") or (side == "SELL" and trend == "BEARISH"):
                trend_score = 75.0
            elif trend == "NEUTRAL":
                trend_score = 50.0
            else:
                trend_score = 35.0  # Counter-trend
            
            # MACD confirmation
            macd_hist = technical_data.get('macd_histogram', 0)
            if (side == "BUY" and macd_hist > 0) or (side == "SELL" and macd_hist < 0):
                technical_score += 10.0
            
            # Volume confirmation
            volume_change = technical_data.get('volume_change_pct', 0)
            if volume_change > 20:
                technical_score += 10.0
            elif volume_change < -30:
                technical_score -= 10.0
            
            technical_score = max(30.0, min(90.0, technical_score))  # Clamp to reasonable range
            
            # Risk score (lower is better)
            volatility = technical_data.get('volatility_5h', 0)
            if volatility > 10:
                risk_score = 70.0  # High volatility = higher risk
            elif volatility < 3:
                risk_score = 30.0  # Low volatility = lower risk
            else:
                risk_score = 50.0
        
        # Determine action based on scores
        avg_score = (technical_score + trend_score + timing_score) / 3
        if avg_score >= 65:
            action = EntryAction.ENTER_NOW.value
            urgency = "MEDIUM"
        elif avg_score >= 50:
            action = EntryAction.WAIT_FOR_PULLBACK.value
            urgency = "LOW"
        else:
            action = EntryAction.WAIT_FOR_PULLBACK.value
            urgency = "LOW"
        
        # Calculate stop and target
        if side == "BUY":
            suggested_stop = current_price * (1 - risk_pct / 100)
            suggested_target = current_price * (1 + take_profit_pct / 100)
        else:  # SELL
            suggested_stop = current_price * (1 + risk_pct / 100)
            suggested_target = current_price * (1 - take_profit_pct / 100)
        
        rsi_val = technical_data.get('rsi', 50)
        trend_val = technical_data.get('trend', 'NEUTRAL')
        volume_change = technical_data.get('volume_change_pct', 0)
        
        reasoning = f"Scanner approved this trade but AI analysis unavailable. "
        reasoning += f"Technical indicators: RSI={rsi_val:.1f}, "
        reasoning += f"Trend={trend_val}, "
        reasoning += f"Volume change={volume_change:+.1f}%. "
        reasoning += f"Recommendation based on technical analysis only."
        
        return EntryAnalysis(
            pair=pair,
            action=action,
            confidence=min(scanner_confidence, avg_score),  # Use scanner confidence or technical score, whichever is lower
            reasoning=reasoning,
            urgency=urgency,
            current_price=current_price,
            suggested_stop=suggested_stop,
            suggested_target=suggested_target,
            risk_reward_ratio=take_profit_pct / risk_pct,
            technical_score=technical_score,
            trend_score=trend_score,
            timing_score=timing_score,
            risk_score=risk_score
        )
    
    def monitor_entry_opportunity(
        self,
        pair: str,
        side: str,
        position_size: float,
        risk_pct: float,
        take_profit_pct: float,
        analysis: EntryAnalysis,
        auto_execute: bool = False
    ) -> str:
        """
        Start monitoring for entry opportunity
        
        Args:
            pair: Trading pair
            side: BUY or SELL
            position_size: Position size in USD
            risk_pct: Risk percentage
            take_profit_pct: Take profit percentage
            analysis: Original entry analysis
            auto_execute: Auto-execute when conditions met
        
        Returns:
            Opportunity ID
        """
        opportunity_id = f"{pair}_{int(time.time())}"
        
        # Extract target conditions from analysis
        target_conditions = {
            'wait_for_price': analysis.wait_for_price,
            'wait_for_rsi': analysis.wait_for_rsi,
            'action': analysis.action
        }
        
        opportunity = MonitoredEntryOpportunity(
            pair=pair,
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
        
        logger.info(f"📊 Monitoring entry opportunity: {pair} (ID: {opportunity_id})")
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
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🎯 Entry opportunity monitoring started")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running and not self.stop_event.is_set():
            try:
                if self.opportunities:
                    self._check_opportunities()
                
                # Wait for next check
                self.stop_event.wait(self.check_interval)
                
            except Exception as e:
                logger.error("Error in entry monitoring loop: {}", str(e), exc_info=True)
                time.sleep(self.check_interval)
    
    def _check_opportunities(self):
        """Check all monitored opportunities"""
        for opp_id, opportunity in list(self.opportunities.items()):
            try:
                # Get current market data
                ticker_info = self.kraken_client.get_ticker_info(opportunity.pair)
                if not ticker_info:
                    continue
                
                current_price = float(ticker_info.get('c', [0])[0])
                opportunity.current_price = current_price
                opportunity.last_check_time = datetime.now()
                
                # Get technical indicators
                technical_data = self._get_technical_indicators(opportunity.pair)
                
                # Check if conditions are met
                conditions_met = self._check_entry_conditions(opportunity, current_price, technical_data)
                
                if conditions_met:
                    logger.info(f"✅ Entry conditions met for {opportunity.pair}!")
                    logger.info(f"   Price: ${current_price:,.6f}")
                    if technical_data:
                        pass  # logger.info("   RSI: {}", str(technical_data.get('rsi', 0):.2f))
                    
                    # Send notification
                    if not opportunity.notification_sent:
                        self._send_entry_notification(opportunity, current_price, technical_data)
                        opportunity.notification_sent = True
                    
                    # Auto-execute if enabled
                    if opportunity.auto_execute and self.enable_auto_entry:
                        logger.info(f"🤖 Auto-executing entry for {opportunity.pair}")
                        # TODO: Integrate with trade execution
                        # This would call execute_trade() and add to position manager
                        self.remove_opportunity(opp_id)
                
            except Exception as e:
                logger.error("Error checking opportunity {opp_id}: {}", str(e), exc_info=True)
    
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
        """Send notification that entry conditions are met with approval request"""
        try:
            # Send Discord approval request
            try:
                from services.discord_trade_approval import get_discord_approval_manager
                
                approval_manager = get_discord_approval_manager(
                    approval_callback=self._handle_approval_response
                )
                
                if approval_manager and approval_manager.is_running():
                    # Calculate stop and target prices
                    if opportunity.side == "BUY":
                        stop_loss = current_price * (1 - opportunity.risk_pct / 100)
                        take_profit = current_price * (1 + opportunity.take_profit_pct / 100)
                    else:  # SELL
                        stop_loss = current_price * (1 + opportunity.risk_pct / 100)
                        take_profit = current_price * (1 - opportunity.take_profit_pct / 100)
                    
                    # Build additional info
                    additional_info = ""
                    if technical_data:
                        rsi = technical_data.get('rsi', 0)
                        trend = technical_data.get('trend', 'N/A')
                        volume_change = technical_data.get('volume_change_pct', 0)
                        additional_info = (
                            f"**Technical Indicators:**\n"
                            f"• RSI: {rsi:.2f}\n"
                            f"• Trend: {trend}\n"
                            f"• Volume Change: {volume_change:+.1f}%\n\n"
                            f"**Original Analysis:**\n"
                            f"• Action: {opportunity.original_analysis.action}\n"
                            f"• Technical Score: {opportunity.original_analysis.technical_score:.0f}/100\n"
                            f"• Timing Score: {opportunity.original_analysis.timing_score:.0f}/100"
                        )
                    
                    # Generate approval ID
                    import time
                    approval_id = f"{opportunity.pair}_{int(time.time())}"
                    
                    # Store approval ID in opportunity
                    opportunity.target_conditions['approval_id'] = approval_id
                    
                    # Send approval request
                    success = approval_manager.send_approval_request(
                        approval_id=approval_id,
                        pair=opportunity.pair,
                        side=opportunity.side,
                        entry_price=current_price,
                        position_size=opportunity.position_size,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        strategy="AI_ENTRY",
                        confidence=opportunity.original_analysis.confidence,
                        reasoning=opportunity.original_analysis.reasoning,
                        additional_info=additional_info
                    )
                    
                    if success:
                        logger.info(f"✅ Discord approval request sent for {opportunity.pair}")
                        logger.info(f"   Approval ID: {approval_id}")
                        return
                    else:
                        logger.warning(f"⚠️ Failed to send Discord approval request for {opportunity.pair}")
                else:
                    logger.debug("Discord approval manager not available")
            
            except Exception as e:
                logger.debug(f"Discord approval manager error: {e}")
            
            # Fallback: Send simple webhook notification
            try:
                from src.integrations.discord_webhook import send_discord_alert
                from models.alerts import TradingAlert, AlertType, AlertPriority
            except ImportError:
                logger.debug("Discord webhook not available, skipping notification")
                return
            
            rsi_text = f"RSI: {technical_data.get('rsi', 0):.2f}" if technical_data else ""
            trend_text = f"Trend: {technical_data.get('trend', 'N/A')}" if technical_data else ""
            
            alert = TradingAlert(
                ticker=opportunity.pair,
                alert_type=AlertType.AI_SIGNAL,
                message=f"🎯 Entry Opportunity Ready! {opportunity.side} @ ${current_price:,.6f}",
                priority=AlertPriority.HIGH,
                details={
                    'action': 'ENTRY_READY',
                    'pair': opportunity.pair,
                    'side': opportunity.side,
                    'current_price': current_price,
                    'position_size': opportunity.position_size,
                    'rsi': technical_data.get('rsi', 0) if technical_data else 0,
                    'trend': technical_data.get('trend', 'N/A') if technical_data else 'N/A',
                    'reasoning': opportunity.original_analysis.reasoning
                }
            )
            send_discord_alert(alert)
            logger.info(f"📢 Entry notification sent for {opportunity.pair}")
            
        except ImportError:
            # Already handled above, but catch here too for safety
            logger.debug("Discord webhook not available, skipping notification")
        except Exception as e:
            logger.warning(f"Failed to send entry notification for {opportunity.pair}: {type(e).__name__}: {e}", exc_info=True)
    
    def _handle_approval_response(self, approval_id: str, approved: bool):
        """Handle approval response from Discord"""
        try:
            logger.info(f"📬 Received approval response: {approval_id} -> {'APPROVED' if approved else 'REJECTED'}")
            
            # Find the opportunity with this approval ID
            opportunity_found = None
            opp_id_found = None
            
            for opp_id, opp in self.opportunities.items():
                if opp.target_conditions.get('approval_id') == approval_id:
                    opportunity_found = opp
                    opp_id_found = opp_id
                    break
            
            if not opportunity_found:
                logger.warning(f"⚠️ Could not find opportunity for approval ID: {approval_id}")
                return
            
            if approved:
                logger.info(f"✅ User approved trade for {opportunity_found.pair}")
                logger.info(f"   Side: {opportunity_found.side}")
                logger.info(f"   Position Size: ${opportunity_found.position_size:,.2f}")
                
                # Execute the trade
                if opp_id_found:
                    self._execute_approved_trade(opportunity_found, opp_id_found)
            else:
                logger.info(f"❌ User rejected trade for {opportunity_found.pair}")
                # Remove from monitoring
                if opp_id_found:
                    self.remove_opportunity(opp_id_found)
        
        except Exception as e:
            logger.error("Error handling approval response: {}", str(e), exc_info=True)
    
    def _execute_approved_trade(self, opportunity: MonitoredEntryOpportunity, opp_id: str):
        """Execute approved trade and add to position manager"""
        try:
            logger.info(f"🚀 Executing approved trade: {opportunity.pair}")
            
            # Calculate prices
            if opportunity.side == "BUY":
                stop_loss = opportunity.current_price * (1 - opportunity.risk_pct / 100)
                take_profit = opportunity.current_price * (1 + opportunity.take_profit_pct / 100)
            else:  # SELL
                stop_loss = opportunity.current_price * (1 + opportunity.risk_pct / 100)
                take_profit = opportunity.current_price * (1 - opportunity.take_profit_pct / 100)
            
            # Execute trade via Kraken - Convert strings to enums
            # Convert side string to OrderSide enum
            side_enum = OrderSide.BUY if opportunity.side == "BUY" else OrderSide.SELL
            
            executed_order = self.kraken_client.place_order(
                pair=opportunity.pair,
                side=side_enum,  # Use enum instead of string
                order_type=OrderType.MARKET,  # Use enum instead of string
                volume=opportunity.position_size / opportunity.current_price,
                price=None  # Market order
            )
            
            if executed_order:
                order_id = executed_order.order_id
                logger.info(f"✅ Trade executed successfully!")
                logger.info(f"   Order ID: {order_id}")
                logger.info(f"   Pair: {opportunity.pair}")
                logger.info(f"   Side: {opportunity.side}")
                logger.info(f"   Price: ${opportunity.current_price:,.6f}")
                
                # Add to position manager
                try:
                    from services.ai_crypto_position_manager import get_ai_crypto_position_manager
                    
                    position_manager = get_ai_crypto_position_manager(self.kraken_client, self.llm_analyzer)
                    
                    if position_manager:
                        # Generate trade ID
                        trade_id = f"{opportunity.pair}_{int(time.time())}"
                        
                        # Calculate volume
                        volume = opportunity.position_size / opportunity.current_price
                        
                        # Add to monitoring
                        position_manager.add_position(
                            trade_id=trade_id,
                            pair=opportunity.pair,
                            side=opportunity.side,
                            volume=volume,
                            entry_price=opportunity.current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            strategy="AI_ENTRY",
                            entry_order_id=order_id
                        )
                        
                        logger.info(f"✅ Position added to AI monitoring: {trade_id}")
                except Exception as e:
                    logger.warning(f"Could not add to position manager: {e}")
                
                # Remove from entry monitoring
                self.remove_opportunity(opp_id)
            else:
                logger.error(f"❌ Trade execution failed: {result}")
        
        except Exception as e:
            logger.error("Error executing approved trade: {}", str(e), exc_info=True)
    
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
        logger.info("🛑 Entry opportunity monitoring stopped")
    
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
                    'pair': opp.pair,
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
                    'original_analysis': {
                        'pair': opp.original_analysis.pair,
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
                        'analysis_time': opp.original_analysis.analysis_time.isoformat()
                    }
                }
            
            # Write to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug(f"💾 Saved {len(state)} monitored opportunities to {self.state_file}")
            
        except Exception as e:
            logger.error("Error saving entry assistant state: {}", str(e), exc_info=True)
    
    def _load_state(self):
        """Load monitored opportunities from file"""
        try:
            import os
            
            if not os.path.exists(self.state_file):
                logger.debug("No saved monitors found")
                return
            
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            if not state:
                return
            
            # Restore opportunities
            for opp_id, opp_data in state.items():
                try:
                    # Check if this is new format (with nested original_analysis) or old format (flat)
                    if 'original_analysis' in opp_data:
                        # New format - properly nested
                        analysis_data = opp_data['original_analysis']
                    else:
                        # Old format - fields are at top level
                        # Migrate to new structure
                        logger.info(f"🔄 Migrating old format for {opp_id}")
                        analysis_data = opp_data  # Use top-level as analysis data
                    
                    # Reconstruct EntryAnalysis
                    analysis = EntryAnalysis(
                        pair=analysis_data.get('pair', opp_data.get('pair', 'UNKNOWN')),
                        action=analysis_data.get('action', 'WAIT_FOR_PULLBACK'),  # Default for old format
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
                    
                    # Reconstruct MonitoredEntryOpportunity (with defaults for missing fields in old format)
                    opportunity = MonitoredEntryOpportunity(
                        pair=opp_data['pair'],
                        side=opp_data['side'],
                        target_price=opp_data.get('target_price'),
                        target_rsi=opp_data.get('target_rsi'),
                        target_conditions=opp_data.get('target_conditions', []),
                        position_size=opp_data.get('position_size', 100.0),  # Default for old format
                        risk_pct=opp_data.get('risk_pct', 2.0),  # Default for old format
                        take_profit_pct=opp_data.get('take_profit_pct', 5.0),  # Default for old format
                        original_analysis=analysis,
                        created_time=datetime.fromisoformat(opp_data['created_time']) if 'created_time' in opp_data else datetime.now(),
                        last_check_time=datetime.fromisoformat(opp_data['last_check_time']) if 'last_check_time' in opp_data else datetime.now(),
                        current_price=opp_data.get('current_price', analysis_data['current_price']),
                        notification_sent=opp_data.get('notification_sent', False),
                        auto_execute=opp_data.get('auto_execute', False)
                    )
                    
                    self.opportunities[opp_id] = opportunity
                    logger.info("📂 Restored monitor: {} (target: ${opportunity.target_price if opportunity.target_price else 'breakout'})", str(opportunity.pair))
                    
                except Exception as e:
                    logger.warning(f"Failed to restore opportunity {opp_id}: {e}")
                    continue
            
            if self.opportunities:
                logger.info(f"✅ Loaded {len(self.opportunities)} monitored opportunities from previous session")
                
                # Save to convert old format to new format
                self._save_state()
                logger.info("💾 Migrated and saved monitors to new format")
                
                # Auto-start monitoring if we have opportunities
                if not self.is_running:
                    self.start_monitoring()
            
        except Exception as e:
            logger.error("Error loading entry assistant state: {}", str(e), exc_info=True)


# Singleton instance
_entry_assistant_instance = None

def get_ai_entry_assistant(
    kraken_client,
    llm_analyzer=None,
    check_interval_seconds: int = 60,
    enable_auto_entry: bool = False
):
    """Get or create singleton AI Entry Assistant instance"""
    global _entry_assistant_instance
    
    if _entry_assistant_instance is None:
        _entry_assistant_instance = AIEntryAssistant(
            kraken_client=kraken_client,
            llm_analyzer=llm_analyzer,
            check_interval_seconds=check_interval_seconds,
            enable_auto_entry=enable_auto_entry
        )
    
    return _entry_assistant_instance


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
        take_profit_pct: float
    ) -> EntryAnalysis:
        """
        Analyze if NOW is a good time to enter this trade
        
        Args:
            pair: Trading pair (e.g., "HIPPO/USD")
            side: "BUY" or "SELL"
            position_size: Position size in USD
            risk_pct: Risk percentage for stop loss
            take_profit_pct: Take profit percentage
        
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
                technical_data=technical_data
            )
            
            # Get AI recommendation
            response = self.llm_analyzer._call_openrouter(prompt, max_retries=2, try_fallbacks=True)
            
            if not response:
                logger.error("Failed to get AI response")
                return self._create_error_analysis(pair, "LLM API call failed")
            
            decision_data = self._parse_ai_response(response)
            
            if decision_data:
                analysis = EntryAnalysis(
                    pair=pair,
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
                return self._create_error_analysis(pair, "Failed to parse AI response")
                
        except Exception as e:
            logger.error(f"Error analyzing entry for {pair}: {e}", exc_info=True)
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
            logger.error(f"Error calculating indicators for {pair}: {e}", exc_info=True)
            return {}
    
    def _build_entry_analysis_prompt(
        self,
        pair: str,
        side: str,
        current_price: float,
        position_size: float,
        risk_pct: float,
        take_profit_pct: float,
        technical_data: Dict
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
        
        prompt += f"""
**Entry Timing Analysis Framework:**
Evaluate these critical factors:

1. **Trend Alignment** (0-100):
   - Is the trend strong and confirmed?
   - Are we entering WITH the trend or against it?
   - Is momentum building or weakening?

2. **Entry Price Quality** (0-100):
   - Are we buying near support or chasing resistance?
   - Is this a good price or are we FOMOing into a pump?
   - Is there room to the upside or are we at resistance?

3. **Technical Setup** (0-100):
   - Are indicators aligned for entry?
   - Is RSI in a good zone (not overbought)?
   - Is MACD confirming or diverging?

4. **Risk/Reward at Current Price** (0-100):
   - Is the R:R favorable at this price?
   - Would waiting for pullback improve R:R?
   - Is the stop loss too tight for current volatility?

5. **Volume & Conviction** (0-100):
   - Is volume confirming the move?
   - Is this a "hot coin" with genuine interest?
   - Or is volume dying (low conviction)?

**Available Actions:**
1. **ENTER_NOW** - Excellent setup, execute immediately (85%+ confidence)
2. **WAIT_FOR_PULLBACK** - Good coin but overbought, wait for better entry (provide wait_for_price and wait_for_rsi)
3. **WAIT_FOR_BREAKOUT** - Consolidating, wait for confirmed move (provide wait_for_price)
4. **PLACE_LIMIT_ORDER** - Set limit at better price (provide suggested_entry)
5. **DO_NOT_ENTER** - Poor setup, avoid this trade (<50% confidence)

**Critical Rules for Entry Timing:**
- Only ENTER_NOW if 85%+ confidence AND good technical setup
- WAIT_FOR_PULLBACK if RSI > 70 (overbought) even if trend is good
- WAIT_FOR_PULLBACK if price just moved >10% in last few hours (let it reset)
- WAIT_FOR_BREAKOUT if price is consolidating near resistance
- DO_NOT_ENTER if trend is against you or setup is poor
- Consider volatility - wider stops needed for volatile assets

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
                logger.error(f"Error in entry monitoring loop: {e}", exc_info=True)
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
                        logger.info(f"   RSI: {technical_data.get('rsi', 0):.2f}")
                    
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
            try:
                from services.discord_notifier import send_discord_alert, TradingAlert, AlertType, AlertPriority
            except ImportError:
                logger.debug("Discord notifier not available, skipping notification")
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
                    # Save original analysis
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
            logger.error(f"Error saving entry assistant state: {e}", exc_info=True)
    
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
                    # Reconstruct EntryAnalysis
                    analysis_data = opp_data['original_analysis']
                    analysis = EntryAnalysis(
                        pair=analysis_data['pair'],
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
                        pair=opp_data['pair'],
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
                    logger.info(f"📂 Restored monitor: {opportunity.pair} (target: ${opportunity.target_price if opportunity.target_price else 'breakout'})")
                    
                except Exception as e:
                    logger.warning(f"Failed to restore opportunity {opp_id}: {e}")
                    continue
            
            if self.opportunities:
                logger.info(f"✅ Loaded {len(self.opportunities)} monitored opportunities from previous session")
                # Auto-start monitoring if we have opportunities
                if not self.is_running:
                    self.start_monitoring()
            
        except Exception as e:
            logger.error(f"Error loading entry assistant state: {e}", exc_info=True)


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


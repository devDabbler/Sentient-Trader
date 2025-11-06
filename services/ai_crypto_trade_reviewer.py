"""
AI Crypto Trade Reviewer
Pre-trade validation and ongoing trade monitoring with AI analysis
"""

from loguru import logger
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timezone
import json


class AICryptoTradeReviewer:
    """AI-powered pre-trade validation and monitoring for crypto trades"""
    
    def __init__(self, llm_analyzer=None, active_monitors=None, supabase_client=None):
        """
        Initialize AI trade reviewer
        
        Args:
            llm_analyzer: LLM analyzer instance for AI analysis
            active_monitors: Optional dict of existing monitors (for session state restoration)
            supabase_client: Optional Supabase client for database persistence
        """
        self.llm_analyzer = llm_analyzer
        self.supabase = supabase_client
        
        # Restore monitors from session state if provided, otherwise load from database
        if active_monitors is not None:
            self.active_monitors = active_monitors
        else:
            self.active_monitors = self._load_monitors_from_db() if self.supabase else {}
        
        logger.info(f"🔧 AI Trade Reviewer initialized with {len(self.active_monitors)} monitors (DB: {'enabled' if self.supabase else 'disabled'})")
        
    def pre_trade_review(
        self,
        pair: str,
        side: str,
        entry_price: float,
        position_size_usd: float,
        stop_loss_price: float,
        take_profit_price: float,
        strategy: str,
        market_data: Optional[Dict] = None,
        total_capital: Optional[float] = None,
        actual_balance: Optional[float] = None
    ) -> Tuple[bool, float, str, Dict]:
        """
        AI pre-trade validation with detailed risk assessment
        
        Args:
            pair: Trading pair (e.g., BTC/USD)
            side: BUY or SELL
            entry_price: Entry price
            position_size_usd: Position size in USD
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            strategy: Trading strategy (SCALP, MOMENTUM, SWING)
            market_data: Optional market data dict
            
        Returns:
            Tuple of (approved: bool, confidence: float, reasoning: str, recommendations: dict)
        """
        logger.info(f"🤖 AI reviewing {side} trade for {pair} @ ${entry_price:,.2f}")
        
        # Calculate trade metrics
        risk_amount = abs(entry_price - stop_loss_price) / entry_price * position_size_usd
        reward_amount = abs(take_profit_price - entry_price) / entry_price * position_size_usd
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Calculate capital utilization
        capital_info = ""
        if total_capital:
            capital_pct = (position_size_usd / total_capital) * 100
            risk_pct = (risk_amount / total_capital) * 100
            capital_info = f"\n- Investment Capital: ${total_capital:,.2f}\n- Position as % of Capital: {capital_pct:.1f}%\n- Risk as % of Capital: {risk_pct:.2f}%"
        
        if actual_balance:
            if actual_balance < position_size_usd:
                capital_info += f"\n- ⚠️ WARNING: Position size (${position_size_usd:,.2f}) exceeds Kraken balance (${actual_balance:,.2f})"
            else:
                remaining = actual_balance - position_size_usd
                capital_info += f"\n- Kraken Balance: ${actual_balance:,.2f}\n- Remaining After Trade: ${remaining:,.2f}"
        
        # Build analysis prompt
        prompt = f"""
        As an expert cryptocurrency trader, analyze this proposed trade for risk and opportunity:
        
        **Trade Details:**
        - Asset: {pair}
        - Direction: {side}
        - Strategy: {strategy}
        - Entry Price: ${entry_price:,.2f}
        - Position Size: ${position_size_usd:,.2f}
        - Stop Loss: ${stop_loss_price:,.2f} ({abs(entry_price - stop_loss_price) / entry_price * 100:.2f}%)
        - Take Profit: ${take_profit_price:,.2f} ({abs(take_profit_price - entry_price) / entry_price * 100:.2f}%)
        - Risk Amount: ${risk_amount:,.2f}
        - Reward Amount: ${reward_amount:,.2f}
        - Risk:Reward Ratio: {risk_reward_ratio:.2f}:1
        
        **Capital & Account Context:**{capital_info}
        
        **Market Context:**
        {json.dumps(market_data, indent=2) if market_data else "No additional market data available"}
        
        **Your Analysis Must Include:**
        1. **APPROVE or REJECT** (first line)
        2. **Confidence Score** (0-100)
        3. **Key Risks** (top 3)
        4. **Entry Timing** (optimal, acceptable, poor)
        5. **Position Size Assessment** (too large, appropriate, too small)
        6. **Specific Recommendations** (adjustments if needed)
        7. **Exit Strategy Review** (stop loss and take profit levels)
        8. **Market Conditions** (favorable, neutral, unfavorable)
        
        **Rejection Criteria:**
        - Risk:Reward ratio < 1.5:1
        - Position size > 15% of capital (without strong justification)
        - Extremely unfavorable market conditions
        - Entry during high volatility without proper protection
        - Conflicting technical/fundamental signals
        
        Provide concise, actionable analysis focusing on risk management.
        """
        
        # Get AI analysis if available
        if self.llm_analyzer and hasattr(self.llm_analyzer, 'analyze_with_llm'):
            try:
                ai_response = self.llm_analyzer.analyze_with_llm(prompt)
                
                # Parse AI response
                approved = self._parse_approval(ai_response)
                confidence = self._parse_confidence(ai_response)
                reasoning = ai_response[:500]  # First 500 chars as summary
                
                recommendations = {
                    'entry_timing': self._extract_field(ai_response, 'Entry Timing'),
                    'position_size': self._extract_field(ai_response, 'Position Size'),
                    'risks': self._extract_risks(ai_response),
                    'recommendations': self._extract_recommendations(ai_response),
                    'market_conditions': self._extract_field(ai_response, 'Market Conditions')
                }
                
                logger.info(f"✅ AI Review: {'APPROVED' if approved else 'REJECTED'} ({confidence}% confidence)")
                
                return approved, confidence, reasoning, recommendations
                
            except Exception as e:
                logger.error(f"AI analysis error: {e}")
                # Fall through to rule-based backup
        
        # Rule-based backup if AI unavailable
        return self._rule_based_review(
            risk_reward_ratio, position_size_usd, strategy, side, pair
        )
    
    def _rule_based_review(
        self, 
        risk_reward_ratio: float, 
        position_size_usd: float, 
        strategy: str,
        side: str,
        pair: str
    ) -> Tuple[bool, float, str, Dict]:
        """Fallback rule-based review"""
        
        approved = True
        confidence = 70.0
        issues = []
        
        # Check R:R ratio
        if risk_reward_ratio < 1.5:
            issues.append(f"Low R:R ratio ({risk_reward_ratio:.2f}:1)")
            confidence -= 20
            if risk_reward_ratio < 1.0:
                approved = False
        
        # Check position size (assuming $1000 capital)
        if position_size_usd > 150:
            issues.append(f"Large position size (${position_size_usd:,.2f})")
            confidence -= 15
            if position_size_usd > 200:
                approved = False
        
        # Strategy-specific checks
        if strategy == "SCALP" and risk_reward_ratio < 2.0:
            issues.append("Scalping requires minimum 2:1 R:R")
            confidence -= 10
        
        reasoning = " | ".join(issues) if issues else "Rule-based approval"
        
        recommendations = {
            'entry_timing': 'acceptable',
            'position_size': 'appropriate' if position_size_usd <= 100 else 'review',
            'risks': issues[:3],
            'recommendations': ['Monitor closely', 'Use limit orders', 'Set alerts'],
            'market_conditions': 'neutral'
        }
        
        return approved, confidence, reasoning, recommendations
    
    def start_trade_monitoring(
        self,
        trade_id: str,
        pair: str,
        side: str,
        entry_price: float,
        current_price: float,
        volume: float,
        stop_loss: float,
        take_profit: float,
        strategy: str
    ) -> None:
        """
        Start AI monitoring for an active trade
        
        Args:
            trade_id: Unique trade identifier
            pair: Trading pair
            side: BUY or SELL
            entry_price: Entry price
            current_price: Current market price
            volume: Trade volume
            stop_loss: Stop loss price
            take_profit: Take profit price
            strategy: Trading strategy
        """
        monitor_data = {
            'pair': pair,
            'side': side,
            'entry_price': entry_price,
            'current_price': current_price,
            'entry_time': datetime.now(),
            'volume': volume,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': strategy,
            'highest_price': current_price if side == 'BUY' else entry_price,
            'lowest_price': current_price if side == 'SELL' else entry_price,
            'adjustments': []
        }
        
        self.active_monitors[trade_id] = monitor_data
        
        # Save to database for persistence
        self._save_monitor_to_db(trade_id, monitor_data)
        
        logger.info(f"📊 Started monitoring trade {trade_id} for {pair}")
    
    def check_trade_status(
        self,
        trade_id: str,
        current_price: float,
        market_data: Optional[Dict] = None
    ) -> Dict:
        """
        Check trade status and get AI recommendations for adjustments
        
        Args:
            trade_id: Trade identifier
            current_price: Current market price
            market_data: Optional market data
            
        Returns:
            Dict with action, reasoning, and parameters
        """
        if trade_id not in self.active_monitors:
            return {'action': 'NO_MONITOR', 'reasoning': 'Trade not being monitored'}
        
        monitor = self.active_monitors[trade_id]
        pair = monitor['pair']
        side = monitor['side']
        entry_price = monitor['entry_price']
        
        # Update price extremes
        if side == 'BUY':
            monitor['highest_price'] = max(monitor['highest_price'], current_price)
        else:
            monitor['lowest_price'] = min(monitor['lowest_price'], current_price)
        
        # Calculate P&L
        if side == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        # Calculate time in trade
        time_in_trade = (datetime.now() - monitor['entry_time']).total_seconds() / 60
        
        logger.info(f"📈 {pair} P&L: {pnl_pct:+.2f}% | Time: {time_in_trade:.1f}min")
        
        # Build analysis prompt for AI
        prompt = f"""
        As a cryptocurrency trader, analyze this active trade and recommend actions:
        
        **Trade Status:**
        - Asset: {pair}
        - Side: {side}
        - Entry: ${entry_price:,.2f}
        - Current: ${current_price:,.2f}
        - P&L: {pnl_pct:+.2f}%
        - Time in Trade: {time_in_trade:.1f} minutes
        - Strategy: {monitor['strategy']}
        - Stop Loss: ${monitor['stop_loss']:,.2f}
        - Take Profit: ${monitor['take_profit']:,.2f}
        - Highest/Lowest: ${monitor.get('highest_price', current_price):,.2f} / ${monitor.get('lowest_price', current_price):,.2f}
        
        **Recommend ONE action:**
        1. **HOLD** - Continue monitoring, no changes
        2. **TAKE_PARTIAL** - Take partial profits (suggest %)
        3. **ADD_POSITION** - Add to position if trend strengthening
        4. **TIGHTEN_STOP** - Move stop loss to protect profits
        5. **CLOSE_NOW** - Exit immediately (explain urgency)
        6. **ADJUST_TP** - Modify take profit target
        
        **Provide:**
        - Action (one of above)
        - Confidence (0-100)
        - Reasoning (1-2 sentences)
        - Parameters (e.g., partial_pct: 50, new_stop: 45000)
        
        Focus on maximizing profit while protecting capital.
        """
        
        # Get AI recommendation
        if self.llm_analyzer and hasattr(self.llm_analyzer, 'analyze_with_llm'):
            try:
                ai_response = self.llm_analyzer.analyze_with_llm(prompt)
                
                action = self._extract_action(ai_response)
                confidence = self._parse_confidence(ai_response)
                reasoning = ai_response[:300]
                parameters = self._extract_parameters(ai_response)
                
                result = {
                    'action': action,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'parameters': parameters,
                    'current_pnl_pct': pnl_pct,
                    'time_in_trade_min': time_in_trade
                }
                
                # Log adjustments
                if action != 'HOLD':
                    monitor['adjustments'].append({
                        'timestamp': datetime.now(),
                        'action': action,
                        'price': current_price,
                        'pnl_pct': pnl_pct
                    })
                
                return result
                
            except Exception as e:
                logger.error(f"AI monitoring error: {e}")
        
        # Rule-based fallback
        return self._rule_based_monitoring(monitor, current_price, pnl_pct, time_in_trade)
    
    def _rule_based_monitoring(
        self, 
        monitor: Dict, 
        current_price: float, 
        pnl_pct: float, 
        time_in_trade: float
    ) -> Dict:
        """Rule-based trade monitoring fallback"""
        
        action = 'HOLD'
        reasoning = 'Monitoring trade within parameters'
        parameters = {}
        
        # Take partial profits if up 5%+
        if pnl_pct >= 5.0:
            action = 'TAKE_PARTIAL'
            parameters = {'partial_pct': 50}
            reasoning = f'Up {pnl_pct:.1f}%, secure 50% profits'
        
        # Tighten stop if up 3%+
        elif pnl_pct >= 3.0:
            action = 'TIGHTEN_STOP'
            new_stop = monitor['entry_price'] * 1.01  # Move to +1% breakeven
            parameters = {'new_stop': new_stop}
            reasoning = 'Move stop to breakeven'
        
        # Consider exit if losing 2%+ and time > 30min
        elif pnl_pct <= -2.0 and time_in_trade > 30:
            action = 'CLOSE_NOW'
            reasoning = 'Cutting losses after 30min'
        
        return {
            'action': action,
            'confidence': 65.0,
            'reasoning': reasoning,
            'parameters': parameters,
            'current_pnl_pct': pnl_pct,
            'time_in_trade_min': time_in_trade
        }
    
    def stop_monitoring(self, trade_id: str, reason: str = None) -> Dict:
        """Stop monitoring a trade and return summary"""
        if trade_id in self.active_monitors:
            summary = self.active_monitors[trade_id].copy()
            del self.active_monitors[trade_id]
            
            # Close in database
            self._close_monitor_in_db(trade_id, reason)
            
            logger.info(f"🛑 Stopped monitoring trade {trade_id}")
            return summary
        return {}
    
    # Helper methods for parsing AI responses
    def _parse_approval(self, response: str) -> bool:
        """Parse approval decision from AI response"""
        response_lower = response.lower()
        if 'approve' in response_lower and 'reject' not in response_lower:
            return True
        return False
    
    def _parse_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        import re
        matches = re.findall(r'confidence[:\s]+(\d+)', response.lower())
        if matches:
            return float(matches[0])
        return 70.0
    
    def _extract_field(self, response: str, field: str) -> str:
        """Extract specific field from response"""
        import re
        pattern = f"{field}[:\s]+([^\n]+)"
        matches = re.findall(pattern, response, re.IGNORECASE)
        return matches[0].strip() if matches else 'unknown'
    
    def _extract_risks(self, response: str) -> List[str]:
        """Extract risk items from response"""
        risks = []
        lines = response.split('\n')
        in_risks = False
        for line in lines:
            if 'key risk' in line.lower():
                in_risks = True
                continue
            if in_risks and line.strip().startswith('-'):
                risks.append(line.strip('- ').strip())
            if len(risks) >= 3:
                break
        return risks if risks else ['Volatility', 'Market uncertainty', 'Timing risk']
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response"""
        recs = []
        lines = response.split('\n')
        in_recs = False
        for line in lines:
            if 'recommendation' in line.lower():
                in_recs = True
                continue
            if in_recs and line.strip().startswith('-'):
                recs.append(line.strip('- ').strip())
            if len(recs) >= 3:
                break
        return recs if recs else ['Monitor closely', 'Use limit orders']
    
    def _extract_action(self, response: str) -> str:
        """Extract action from monitoring response"""
        actions = ['HOLD', 'TAKE_PARTIAL', 'ADD_POSITION', 'TIGHTEN_STOP', 'CLOSE_NOW', 'ADJUST_TP']
        response_upper = response.upper()
        for action in actions:
            if action in response_upper:
                return action
        return 'HOLD'
    
    def _extract_parameters(self, response: str) -> Dict:
        """Extract action parameters from response"""
        import re
        params = {}
        
        # Look for partial_pct
        partial_match = re.search(r'partial[_\s]?pct[:\s]+(\d+)', response.lower())
        if partial_match:
            params['partial_pct'] = int(partial_match.group(1))
        
        # Look for new_stop
        stop_match = re.search(r'new[_\s]?stop[:\s]+(\d+\.?\d*)', response.lower())
        if stop_match:
            params['new_stop'] = float(stop_match.group(1))
        
        # Look for new_tp
        tp_match = re.search(r'new[_\s]?(?:take[_\s]?profit|tp)[:\s]+(\d+\.?\d*)', response.lower())
        if tp_match:
            params['new_tp'] = float(tp_match.group(1))
        
        return params
    
    # =========================================================================
    # DATABASE PERSISTENCE METHODS
    # =========================================================================
    
    def _load_monitors_from_db(self) -> Dict:
        """Load active monitors from database"""
        if not self.supabase:
            return {}
        
        try:
            response = self.supabase.table('crypto_trade_monitors')\
                .select('*')\
                .eq('status', 'active')\
                .execute()
            
            monitors = {}
            if response.data:
                for row in response.data:
                    monitors[row['trade_id']] = {
                        'pair': row['pair'],
                        'side': row['side'],
                        'entry_price': float(row['entry_price']),
                        'current_price': float(row['current_price']) if row['current_price'] else float(row['entry_price']),
                        'volume': float(row['volume']),
                        'stop_loss': float(row['stop_loss']) if row['stop_loss'] else None,
                        'take_profit': float(row['take_profit']) if row['take_profit'] else None,
                        'strategy': row['strategy'],
                        'start_time': row['created_at']
                    }
                logger.info(f"📥 Loaded {len(monitors)} active monitors from database")
            return monitors
        except Exception as e:
            logger.error(f"❌ Error loading monitors from database: {e}")
            return {}
    
    def _save_monitor_to_db(self, trade_id: str, monitor_data: Dict):
        """Save a monitor to database"""
        if not self.supabase:
            return
        
        try:
            data = {
                'trade_id': trade_id,
                'pair': monitor_data['pair'],
                'side': monitor_data['side'],
                'entry_price': monitor_data['entry_price'],
                'current_price': monitor_data.get('current_price', monitor_data['entry_price']),
                'volume': monitor_data['volume'],
                'stop_loss': monitor_data.get('stop_loss'),
                'take_profit': monitor_data.get('take_profit'),
                'strategy': monitor_data.get('strategy'),
                'status': 'active',
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase.table('crypto_trade_monitors').upsert(data, on_conflict='trade_id').execute()
            logger.info(f"💾 Saved monitor {trade_id} to database")
        except Exception as e:
            logger.error(f"❌ Error saving monitor to database: {e}")
    
    def _update_monitor_in_db(self, trade_id: str, updates: Dict):
        """Update a monitor in database"""
        if not self.supabase:
            return
        
        try:
            updates['updated_at'] = datetime.now(timezone.utc).isoformat()
            self.supabase.table('crypto_trade_monitors')\
                .update(updates)\
                .eq('trade_id', trade_id)\
                .execute()
        except Exception as e:
            logger.error(f"❌ Error updating monitor in database: {e}")
    
    def _close_monitor_in_db(self, trade_id: str, reason: str = None):
        """Mark a monitor as closed in database"""
        if not self.supabase:
            return
        
        try:
            self.supabase.table('crypto_trade_monitors')\
                .update({
                    'status': 'closed',
                    'closed_at': datetime.now(timezone.utc).isoformat(),
                    'close_reason': reason,
                    'updated_at': datetime.now(timezone.utc).isoformat()
                })\
                .eq('trade_id', trade_id)\
                .execute()
            logger.info(f"🔒 Closed monitor {trade_id} in database")
        except Exception as e:
            logger.error(f"❌ Error closing monitor in database: {e}")

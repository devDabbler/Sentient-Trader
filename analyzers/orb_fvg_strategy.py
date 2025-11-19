"""
Opening Range Breakout (ORB) + Fair Value Gap (FVG) Strategy
Based on Reddit trading strategy for NQ, stocks, and crypto
"""
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
from loguru import logger


class ORBFVGAnalyzer:
    """
    Analyzes Opening Range Breakout with Fair Value Gap confirmation
    
    Strategy Rules:
    1. Define first 15-minute opening range (ORH/ORL)
    2. Wait for clean breakout (no front-running)
    3. Confirm with FVG in direction of break
    4. Enter on break + close of FVG on 1min
    5. Target: 1-2R based on stop loss size
    6. Risk: 0.5-2% per trade
    """
    
    def __init__(self):
        self.market_open = time(9, 30)
        self.orb_window_minutes = 15
        
    def detect_fair_value_gap(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Detect Fair Value Gaps (FVG) in price action
        
        FVG = Gap between candle wicks indicating imbalance:
        - Bullish FVG: Current candle low > 2 candles ago high
        - Bearish FVG: Current candle high < 2 candles ago low
        """
        fvgs = {
            'bullish': [],
            'bearish': [],
            'current_signal': None,
            'strength': 0
        }
        
        if len(df) < 3:
            return fvgs
            
        # Check recent candles for FVG patterns
        for i in range(2, min(lookback, len(df))):
            # Bullish FVG: Gap up (current low > 2 candles ago high)
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                fvgs['bullish'].append({
                    'index': i,
                    'timestamp': df.index[i],
                    'gap_size': gap_size,
                    'gap_low': df['high'].iloc[i-2],
                    'gap_high': df['low'].iloc[i]
                })
                
            # Bearish FVG: Gap down (current high < 2 candles ago low)
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                fvgs['bearish'].append({
                    'index': i,
                    'timestamp': df.index[i],
                    'gap_size': gap_size,
                    'gap_high': df['low'].iloc[i-2],
                    'gap_low': df['high'].iloc[i]
                })
        
        # Determine most recent signal
        if fvgs['bullish'] and (not fvgs['bearish'] or 
                                fvgs['bullish'][-1]['index'] > fvgs['bearish'][-1]['index']):
            fvgs['current_signal'] = 'BULLISH'
            fvgs['strength'] = len(fvgs['bullish'][-3:])  # Recent FVG count
        elif fvgs['bearish']:
            fvgs['current_signal'] = 'BEARISH'
            fvgs['strength'] = len(fvgs['bearish'][-3:])
        else:
            fvgs['current_signal'] = 'NEUTRAL'
            
        return fvgs
    
    def calculate_opening_range(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate Opening Range High/Low from first 15 minutes
        
        Returns:
            Dict with ORH, ORL, range_size, breakout_level
        """
        try:
            # Get today's date
            today = datetime.now().date()
            
            # Filter for today's data starting from market open
            market_open_time = datetime.combine(today, self.market_open)
            orb_end_time = market_open_time + timedelta(minutes=self.orb_window_minutes)
            
            # Filter data for opening range period
            if isinstance(df.index, pd.DatetimeIndex):
                or_data = df[(df.index >= market_open_time) & (df.index < orb_end_time)]
            else:
                # If no datetime index, use last 15 bars as approximation
                or_data = df.tail(15) if len(df) >= 15 else df
            
            if len(or_data) == 0:
                logger.warning("No data found for opening range period")
                return None
                
            orh = or_data['high'].max()
            orl = or_data['low'].min()
            range_size = orh - orl
            
            return {
                'orh': orh,
                'orl': orl,
                'range_size': range_size,
                'range_pct': (range_size / orl * 100) if orl > 0 else 0,
                'start_time': or_data.index[0] if len(or_data) > 0 else None,
                'end_time': or_data.index[-1] if len(or_data) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error calculating opening range: {e}")
            return None
    
    def check_breakout(self, current_price: float, opening_range: Optional[Dict], 
                       fvg: Dict) -> Dict:
        """
        Check if price has broken out of opening range with FVG confirmation
        
        Returns:
            Signal dict with direction, entry, stop, targets
        """
        if not opening_range:
            return {'signal': 'HOLD', 'reason': 'No opening range data'}
            
        orh = opening_range['orh']
        orl = opening_range['orl']
        
        # Check for bullish breakout (price > ORH + Bullish FVG)
        if current_price > orh and fvg['current_signal'] == 'BULLISH':
            stop_loss = fvg['bullish'][-1]['gap_low'] if fvg['bullish'] else orl
            risk = current_price - stop_loss
            
            # Determine target based on risk size (strategy rules)
            if risk <= 30:  # Small stop = 2R target
                target_multiplier = 2.0
            else:  # Large stop = 1R target
                target_multiplier = 1.0
                
            target = current_price + (risk * target_multiplier)
            
            return {
                'signal': 'BUY',
                'direction': 'LONG',
                'entry': current_price,
                'stop_loss': stop_loss,
                'target': target,
                'risk': risk,
                'reward': risk * target_multiplier,
                'risk_reward_ratio': target_multiplier,
                'orh': orh,
                'orl': orl,
                'fvg_type': 'BULLISH',
                'fvg_strength': fvg['strength'],
                'reason': f'Bullish breakout above ORH ${orh:.2f} with {fvg["strength"]} bullish FVG(s)'
            }
        
        # Check for bearish breakout (price < ORL + Bearish FVG)
        elif current_price < orl and fvg['current_signal'] == 'BEARISH':
            stop_loss = fvg['bearish'][-1]['gap_high'] if fvg['bearish'] else orh
            risk = stop_loss - current_price
            
            # Determine target based on risk size
            if risk <= 30:
                target_multiplier = 2.0
            else:
                target_multiplier = 1.0
                
            target = current_price - (risk * target_multiplier)
            
            return {
                'signal': 'SELL',
                'direction': 'SHORT',
                'entry': current_price,
                'stop_loss': stop_loss,
                'target': target,
                'risk': risk,
                'reward': risk * target_multiplier,
                'risk_reward_ratio': target_multiplier,
                'orh': orh,
                'orl': orl,
                'fvg_type': 'BEARISH',
                'fvg_strength': fvg['strength'],
                'reason': f'Bearish breakout below ORL ${orl:.2f} with {fvg["strength"]} bearish FVG(s)'
            }
        
        # No breakout or no FVG confirmation
        else:
            if current_price > orl and current_price < orh:
                return {
                    'signal': 'HOLD',
                    'reason': f'Price ${current_price:.2f} within opening range (${orl:.2f} - ${orh:.2f})'
                }
            elif fvg['current_signal'] == 'NEUTRAL':
                return {
                    'signal': 'HOLD',
                    'reason': 'No Fair Value Gap confirmation'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'reason': 'Breakout direction does not match FVG direction'
                }
    
    def calculate_confidence_score(self, signal: Dict, historical_data: pd.DataFrame) -> float:
        """
        Calculate AI confidence score for the setup
        
        Factors:
        - FVG strength (more gaps = higher confidence)
        - Range size (clean range = higher confidence)
        - Volume confirmation
        - Trend alignment
        """
        if signal['signal'] == 'HOLD':
            return 0.0
            
        confidence = 50.0  # Base confidence
        
        # FVG strength factor (+0-20 points)
        fvg_strength = signal.get('fvg_strength', 0)
        confidence += min(fvg_strength * 10, 20)
        
        # Risk/reward ratio factor (+0-15 points)
        rr_ratio = signal.get('risk_reward_ratio', 1.0)
        if rr_ratio >= 2.0:
            confidence += 15
        elif rr_ratio >= 1.5:
            confidence += 10
        else:
            confidence += 5
            
        # Range size factor (+0-10 points)
        # Clean, defined ranges are better
        if 'orh' in signal and 'orl' in signal and signal.get('orh') and signal.get('orl'):
            range_pct = ((signal['orh'] - signal['orl']) / signal['orl']) * 100
            if 0.5 <= range_pct <= 2.0:  # Ideal range
                confidence += 10
            elif range_pct < 0.5:  # Too tight
                confidence += 3
            else:  # Too wide
                confidence += 5
        
        # Volume analysis (+0-15 points)
        if len(historical_data) > 20 and 'volume' in historical_data.columns:
            recent_volume = historical_data['volume'].iloc[-5:].mean()
            avg_volume = historical_data['volume'].iloc[-20:].mean()
            
            if recent_volume > avg_volume * 1.5:
                confidence += 15
            elif recent_volume > avg_volume * 1.2:
                confidence += 10
            else:
                confidence += 5
        
        # Trend alignment (+0-10 points)
        if len(historical_data) >= 20:
            sma_20 = historical_data['close'].rolling(20).mean().iloc[-1]
            current_price = historical_data['close'].iloc[-1]
            
            if signal['direction'] == 'LONG' and current_price > sma_20:
                confidence += 10
            elif signal['direction'] == 'SHORT' and current_price < sma_20:
                confidence += 10
            else:
                confidence += 3
        
        return min(confidence, 95.0)  # Cap at 95%
    
    def analyze(self, ticker: str, historical_data: pd.DataFrame, 
                current_price: Optional[float] = None) -> Dict:
        """
        Main analysis function for ORB+FVG strategy
        
        Args:
            ticker: Stock/crypto symbol
            historical_data: OHLCV data (should include today's data)
            current_price: Current market price (optional, will use last close if not provided)
            
        Returns:
            Comprehensive analysis dict with signal, confidence, targets, and reasoning
        """
        try:
            if current_price is None:
                current_price = float(historical_data['close'].iloc[-1])
            
            # Step 1: Calculate opening range
            opening_range = self.calculate_opening_range(historical_data)
            
            if not opening_range:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'reason': 'Insufficient data for opening range calculation',
                    'ticker': ticker
                }
            
            # Step 2: Detect Fair Value Gaps
            fvg = self.detect_fair_value_gap(historical_data)
            
            # Step 3: Check for breakout with FVG confirmation
            signal = self.check_breakout(current_price, opening_range, fvg)
            
            # Step 4: Calculate confidence score
            confidence = self.calculate_confidence_score(signal, historical_data)
            
            # Step 5: Generate key signals and recommendations
            key_signals = self._generate_key_signals(signal, opening_range, fvg, historical_data)
            recommendations = self._generate_recommendations(signal, confidence)
            
            # Build comprehensive result
            result = {
                'ticker': ticker,
                'strategy': 'ORB+FVG',
                'signal': signal['signal'],
                'direction': signal.get('direction', 'NEUTRAL'),
                'confidence': confidence,
                'current_price': current_price,
                'entry': signal.get('entry', current_price),
                'stop_loss': signal.get('stop_loss'),
                'target': signal.get('target'),
                'risk': signal.get('risk'),
                'reward': signal.get('reward'),
                'risk_reward_ratio': signal.get('risk_reward_ratio'),
                'opening_range': opening_range,
                'fvg_signal': fvg['current_signal'],
                'fvg_strength': fvg['strength'],
                'bullish_fvgs': len(fvg['bullish']),
                'bearish_fvgs': len(fvg['bearish']),
                'key_signals': key_signals,
                'recommendations': recommendations,
                'reason': signal.get('reason', ''),
                'risk_level': self._calculate_risk_level(confidence, signal),
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker} with ORB+FVG strategy: {e}", exc_info=True)
            return {
                'ticker': ticker,
                'strategy': 'ORB+FVG',
                'signal': 'HOLD',
                'confidence': 0.0,
                'reason': f'Analysis error: {str(e)}',
                'error': str(e)
            }
    
    def _generate_key_signals(self, signal: Dict, opening_range: Dict, 
                             fvg: Dict, historical_data: pd.DataFrame) -> List[str]:
        """Generate list of key technical signals"""
        signals = []
        
        # Opening range info
        if opening_range:
            signals.append(f"📊 Opening Range: ${opening_range['orl']:.2f} - ${opening_range['orh']:.2f} ({opening_range['range_pct']:.1f}%)")
        
        # FVG signals
        if fvg['current_signal'] != 'NEUTRAL':
            signals.append(f"🎯 {fvg['current_signal']} FVG detected (Strength: {fvg['strength']})")
        
        # Breakout info
        if signal['signal'] in ['BUY', 'SELL']:
            signals.append(f"🚀 {signal['direction']} breakout confirmed")
            signals.append(f"📈 R:R Ratio: 1:{signal['risk_reward_ratio']:.1f}")
            signals.append(f"🛡️ Stop Loss: ${signal['stop_loss']:.2f} (Risk: ${signal['risk']:.2f})")
            signals.append(f"🎯 Target: ${signal['target']:.2f} (Reward: ${signal['reward']:.2f})")
        
        # Volume analysis
        if len(historical_data) > 20 and 'volume' in historical_data.columns:
            recent_vol = historical_data['volume'].iloc[-5:].mean()
            avg_vol = historical_data['volume'].iloc[-20:].mean()
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
            signals.append(f"📊 Volume: {vol_ratio:.1f}x average")
        
        return signals[:7]  # Limit to top 7 signals
    
    def _generate_recommendations(self, signal: Dict, confidence: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if signal['signal'] == 'BUY':
            recommendations.append(f"✅ LONG entry at ${signal['entry']:.2f}")
            recommendations.append(f"🛡️ Set stop loss at ${signal['stop_loss']:.2f}")
            recommendations.append(f"🎯 Take profit at ${signal['target']:.2f}")
            recommendations.append("📊 Move stop to breakeven after clearing structure")
            
            if confidence >= 70:
                recommendations.append("🔥 HIGH confidence setup - consider full position")
            elif confidence >= 50:
                recommendations.append("⚠️ MEDIUM confidence - consider reduced position")
            else:
                recommendations.append("🚨 LOW confidence - wait for better setup")
                
        elif signal['signal'] == 'SELL':
            recommendations.append(f"✅ SHORT entry at ${signal['entry']:.2f}")
            recommendations.append(f"🛡️ Set stop loss at ${signal['stop_loss']:.2f}")
            recommendations.append(f"🎯 Take profit at ${signal['target']:.2f}")
            recommendations.append("📊 Move stop to breakeven after clearing structure")
            
            if confidence >= 70:
                recommendations.append("🔥 HIGH confidence setup - consider full position")
            elif confidence >= 50:
                recommendations.append("⚠️ MEDIUM confidence - consider reduced position")
            else:
                recommendations.append("🚨 LOW confidence - wait for better setup")
        else:
            recommendations.append("⏳ Wait for clean breakout + FVG confirmation")
            recommendations.append("📊 Monitor opening range levels")
            recommendations.append("🎯 Prepare if-then scenarios for both directions")
        
        # Risk management
        recommendations.append("💰 Risk 0.5-2% of account per trade")
        recommendations.append("📈 Max 2 trades per day (1 if first wins)")
        
        return recommendations[:7]  # Limit to top 7
    
    def _calculate_risk_level(self, confidence: float, signal: Dict) -> str:
        """Calculate overall risk level"""
        if signal['signal'] == 'HOLD':
            return 'NONE'
        
        if confidence >= 70:
            return 'LOW'
        elif confidence >= 50:
            return 'MEDIUM'
        else:
            return 'HIGH'

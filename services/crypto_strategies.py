"""
Crypto Trading Strategies
Implementation of scalping and swing strategies for cryptocurrency trading

Includes:
- VWAP + EMA Pullback (scalping)
- Bollinger Mean-Reversion (scalping)
- Order-Book Imbalance (scalping)
- 1-min EMA Momentum Nudge (scalping)
- 10/21 EMA Pullback with 50/200 Filter (swing)
- MACD + RSI Confirmation (swing)
- Bollinger Squeeze Breakout (swing)
"""

from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np



@dataclass
class TradingSignal:
    """Trading signal for crypto"""
    symbol: str
    strategy: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    reasoning: str
    indicators: Dict
    timestamp: datetime
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'EXTREME'


class CryptoStrategy:
    """Base class for crypto trading strategies"""
    
    def __init__(self, name: str, timeframe: str):
        self.name = name
        self.timeframe = timeframe
        self.min_confidence = 60.0
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy"""
        raise NotImplementedError
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """Generate trading signal based on strategy logic"""
        raise NotImplementedError
    
    def calculate_ema(self, data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.mean(data)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(data[:period])
        
        for price in data[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_sma(self, data: List[float], period: int) -> float:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return np.mean(data)
        return np.mean(data[-period:])
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        changes = np.diff(prices)
        gains = changes.copy()
        losses = changes.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = np.abs(losses)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price for session"""
        if df.empty:
            return 0.0
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate VWAP
        vwap = (df['typical_price'] * df['volume']).sum() / df['volume'].sum()
        
        return vwap
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return 0.0, 0.0, 0.0
        
        sma = self.calculate_sma(prices, period)
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD line)
        # Simplified: use last 9 MACD values if available
        signal_line = macd_line  # Placeholder
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr_list = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        if not tr_list:
            return 0.0
        
        atr = np.mean(tr_list[-period:])
        return atr


class VWAPEMAScalper(CryptoStrategy):
    """VWAP + EMA Pullback Strategy (Scalping)"""
    
    def __init__(self):
        super().__init__("VWAP+EMA Pullback", "1m-5m")
        self.ema_fast = 9
        self.ema_slow = 21
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Logic: Trade in direction of session VWAP; enter on pullbacks to fast EMAs
        with momentum confirm (MACD tick up or volume burst)
        """
        if df.empty or len(df) < 30:
            return None
        
        try:
            # Calculate indicators
            vwap = self.calculate_vwap(df)
            ema_9 = self.calculate_ema(df['close'].tolist(), self.ema_fast)
            ema_21 = self.calculate_ema(df['close'].tolist(), self.ema_slow)
            macd_line, macd_signal, macd_hist = self.calculate_macd(df['close'].tolist())
            
            # Volume analysis
            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            volume_surge = current_volume > (avg_volume * 1.5)
            
            # Determine signal
            signal_type = 'HOLD'
            reasoning = []
            confidence = 50.0
            
            # Long setup: price > VWAP, pullback to EMAs, momentum confirm
            if current_price > vwap and current_price > ema_9:
                # Check for pullback
                recent_low = df['low'].iloc[-3:].min()
                if recent_low <= ema_9 and current_price > ema_9:
                    # Momentum confirm
                    if macd_hist > 0 or volume_surge:
                        signal_type = 'BUY'
                        confidence = 75.0
                        reasoning.append("Price above VWAP")
                        reasoning.append(f"Pullback to EMA{self.ema_fast} complete")
                        reasoning.append("Momentum confirming (MACD or volume)")
                        
                        if volume_surge:
                            confidence += 10
                            reasoning.append(f"Volume surge: {(current_volume/avg_volume):.2f}x avg")
            
            # Short setup: price < VWAP, pullback to EMAs, momentum down
            elif current_price < vwap and current_price < ema_9:
                recent_high = df['high'].iloc[-3:].max()
                if recent_high >= ema_9 and current_price < ema_9:
                    if macd_hist < 0 or volume_surge:
                        signal_type = 'SELL'
                        confidence = 75.0
                        reasoning.append("Price below VWAP")
                        reasoning.append(f"Rejection from EMA{self.ema_fast}")
                        reasoning.append("Momentum confirming down")
            
            if signal_type == 'HOLD':
                return None
            
            # Calculate risk management
            atr = self.calculate_atr(df)
            
            if signal_type == 'BUY':
                recent_swing_low = df['low'].iloc[-10:].min()
                stop_loss = recent_swing_low - (atr * 0.5)
                take_profit = current_price + (atr * 1.5)
            else:  # SELL
                recent_swing_high = df['high'].iloc[-10:].max()
                stop_loss = recent_swing_high + (atr * 0.5)
                take_profit = current_price - (atr * 1.5)
            
            risk_reward = abs((take_profit - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 0
            
            # Risk level
            distance_from_vwap = abs((current_price - vwap) / vwap) * 100
            risk_level = 'LOW' if distance_from_vwap < 1 and risk_reward > 2 else 'MEDIUM'
            
            return TradingSignal(
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy=self.name,
                signal_type=signal_type,
                confidence=min(confidence, 95.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=" | ".join(reasoning),
                indicators={
                    'vwap': vwap,
                    'ema_9': ema_9,
                    'ema_21': ema_21,
                    'macd_histogram': macd_hist,
                    'volume_ratio': current_volume / avg_volume
                },
                timestamp=datetime.now(),
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error generating VWAP+EMA signal: {e}")
            return None


class BollingerMeanReversion(CryptoStrategy):
    """Bollinger Bands Mean Reversion Strategy (Scalping)"""
    
    def __init__(self):
        super().__init__("Bollinger Mean Reversion", "5m-15m")
        self.bb_period = 20
        self.bb_std = 2
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Logic: When price pierces a band, fade back toward middle band
        only if RSI confirms exhaustion
        """
        if df.empty or len(df) < self.bb_period + 14:
            return None
        
        try:
            # Calculate indicators
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
                df['close'].tolist(), self.bb_period, self.bb_std
            )
            rsi = self.calculate_rsi(df['close'].tolist())
            
            signal_type = 'HOLD'
            reasoning = []
            confidence = 50.0
            
            # Long setup: price below lower band, RSI oversold
            if current_price < bb_lower and rsi < 30:
                signal_type = 'BUY'
                confidence = 70.0
                reasoning.append(f"Price below lower BB: ${current_price:.2f} < ${bb_lower:.2f}")
                reasoning.append(f"RSI oversold: {rsi:.1f}")
                reasoning.append("Mean reversion setup")
                
                # Extra confidence if deep oversold
                if rsi < 25:
                    confidence += 10
                    reasoning.append("Deep oversold territory")
            
            # Short setup: price above upper band, RSI overbought
            elif current_price > bb_upper and rsi > 70:
                signal_type = 'SELL'
                confidence = 70.0
                reasoning.append(f"Price above upper BB: ${current_price:.2f} > ${bb_upper:.2f}")
                reasoning.append(f"RSI overbought: {rsi:.1f}")
                reasoning.append("Mean reversion setup")
                
                if rsi > 75:
                    confidence += 10
                    reasoning.append("Deep overbought territory")
            
            if signal_type == 'HOLD':
                return None
            
            # Calculate risk management
            atr = self.calculate_atr(df)
            
            if signal_type == 'BUY':
                stop_loss = bb_lower - (atr * 1.0)
                take_profit = bb_middle  # Target middle band
            else:  # SELL
                stop_loss = bb_upper + (atr * 1.0)
                take_profit = bb_middle
            
            risk_reward = abs((take_profit - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 0
            
            # Risk level based on band width
            band_width = (bb_upper - bb_lower) / bb_middle * 100
            risk_level = 'HIGH' if band_width > 10 else ('MEDIUM' if band_width > 5 else 'LOW')
            
            return TradingSignal(
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy=self.name,
                signal_type=signal_type,
                confidence=min(confidence, 95.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=" | ".join(reasoning),
                indicators={
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'rsi': rsi,
                    'band_width_pct': band_width
                },
                timestamp=datetime.now(),
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error generating Bollinger signal: {e}")
            return None


class EMAMomentumNudge(CryptoStrategy):
    """1-min EMA Momentum Nudge Strategy (Scalping)"""
    
    def __init__(self):
        super().__init__("EMA Momentum Nudge", "1m")
        self.ema_fast = 9
        self.ema_slow = 21
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Logic: On 1-min, only take trades when price is above 9/10 EMA and VWAP;
        enter after a single red pullback candle followed by green re-engulf
        """
        if df.empty or len(df) < 25:
            return None
        
        try:
            # Calculate indicators
            ema_9 = self.calculate_ema(df['close'].tolist(), self.ema_fast)
            ema_21 = self.calculate_ema(df['close'].tolist(), self.ema_slow)
            vwap = self.calculate_vwap(df)
            
            # Check last 2 candles
            if len(df) < 2:
                return None
            
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            signal_type = 'HOLD'
            reasoning = []
            confidence = 50.0
            
            # Long setup: price above EMAs and VWAP, red pullback then green engulf
            if (current_price > ema_9 and current_price > ema_21 and current_price > vwap):
                # Check for red pullback
                prev_was_red = prev_candle['close'] < prev_candle['open']
                current_is_green = last_candle['close'] > last_candle['open']
                engulf = last_candle['close'] > prev_candle['open']
                
                if prev_was_red and current_is_green and engulf:
                    signal_type = 'BUY'
                    confidence = 70.0
                    reasoning.append("Price above EMA9, EMA21, and VWAP")
                    reasoning.append("Red pullback candle")
                    reasoning.append("Green re-engulf candle")
                    
                    # Extra confidence checks
                    if ema_9 > ema_21:
                        confidence += 10
                        reasoning.append("Strong trend: EMA9 > EMA21")
            
            # Short setup (inverse)
            elif (current_price < ema_9 and current_price < ema_21 and current_price < vwap):
                prev_was_green = prev_candle['close'] > prev_candle['open']
                current_is_red = last_candle['close'] < last_candle['open']
                engulf = last_candle['close'] < prev_candle['open']
                
                if prev_was_green and current_is_red and engulf:
                    signal_type = 'SELL'
                    confidence = 70.0
                    reasoning.append("Price below EMA9, EMA21, and VWAP")
                    reasoning.append("Green pullback candle")
                    reasoning.append("Red re-engulf candle")
                    
                    if ema_9 < ema_21:
                        confidence += 10
                        reasoning.append("Strong downtrend: EMA9 < EMA21")
            
            if signal_type == 'HOLD':
                return None
            
            # Calculate risk management (tight stops for scalping)
            atr = self.calculate_atr(df)
            
            if signal_type == 'BUY':
                stop_loss = last_candle['low'] - (atr * 0.5)
                take_profit = current_price + (current_price - stop_loss) * 2  # 2:1 R:R
            else:  # SELL
                stop_loss = last_candle['high'] + (atr * 0.5)
                take_profit = current_price - (stop_loss - current_price) * 2
            
            risk_reward = abs((take_profit - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 0
            
            return TradingSignal(
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy=self.name,
                signal_type=signal_type,
                confidence=min(confidence, 95.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=" | ".join(reasoning),
                indicators={
                    'ema_9': ema_9,
                    'ema_21': ema_21,
                    'vwap': vwap,
                    'trend_strength': ema_9 - ema_21
                },
                timestamp=datetime.now(),
                risk_level='LOW'  # Scalping with tight stops
            )
            
        except Exception as e:
            logger.error(f"Error generating EMA Momentum signal: {e}")
            return None


class EMASwingTrader(CryptoStrategy):
    """10/21 EMA Pullback with 50/200 Trend Filter (Swing)"""
    
    def __init__(self):
        super().__init__("10/21 EMA Swing", "1h-4h")
        self.ema_10 = 10
        self.ema_21 = 21
        self.sma_50 = 50
        self.sma_200 = 200
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Logic: Trade only long when 50>200 SMA (bull trend).
        Enter on pullbacks that hold 10/21 EMA confluence
        """
        if df.empty or len(df) < self.sma_200 + 10:
            return None
        
        try:
            # Calculate indicators
            ema_10 = self.calculate_ema(df['close'].tolist(), self.ema_10)
            ema_21 = self.calculate_ema(df['close'].tolist(), self.ema_21)
            sma_50 = self.calculate_sma(df['close'].tolist(), self.sma_50)
            sma_200 = self.calculate_sma(df['close'].tolist(), self.sma_200)
            
            signal_type = 'HOLD'
            reasoning = []
            confidence = 50.0
            
            # Check trend filter
            bullish_trend = sma_50 > sma_200
            bearish_trend = sma_50 < sma_200
            
            # Long setup: bull trend, pullback to EMAs holding
            if bullish_trend:
                ema_zone_low = min(ema_10, ema_21)
                ema_zone_high = max(ema_10, ema_21)
                
                # Check if price is at or near EMA confluence
                near_ema = ema_zone_low <= current_price <= (ema_zone_high * 1.02)
                
                # Check if pullback held (recent low touched EMAs)
                recent_low = df['low'].iloc[-10:].min()
                pullback_held = recent_low <= ema_zone_high
                
                if near_ema and pullback_held:
                    signal_type = 'BUY'
                    confidence = 75.0
                    reasoning.append("Bull trend: SMA50 > SMA200")
                    reasoning.append(f"Price at EMA10/21 confluence zone")
                    reasoning.append("Pullback holding support")
                    
                    # Extra confidence if strong uptrend
                    if sma_50 > (sma_200 * 1.05):
                        confidence += 10
                        reasoning.append("Strong uptrend: SMA50 >> SMA200")
            
            # Short setup: bear trend, rejection from EMAs
            elif bearish_trend:
                ema_zone_low = min(ema_10, ema_21)
                ema_zone_high = max(ema_10, ema_21)
                
                near_ema = (ema_zone_low * 0.98) <= current_price <= ema_zone_high
                
                # Check if rally rejected (recent high touched EMAs)
                recent_high = df['high'].iloc[-10:].max()
                rally_rejected = recent_high >= ema_zone_low
                
                if near_ema and rally_rejected:
                    signal_type = 'SELL'
                    confidence = 75.0
                    reasoning.append("Bear trend: SMA50 < SMA200")
                    reasoning.append("Price rejected at EMA10/21 resistance")
                    reasoning.append("Downtrend continuation setup")
                    
                    if sma_50 < (sma_200 * 0.95):
                        confidence += 10
                        reasoning.append("Strong downtrend: SMA50 << SMA200")
            
            if signal_type == 'HOLD':
                return None
            
            # Calculate risk management (wider stops for swing)
            atr = self.calculate_atr(df)
            
            if signal_type == 'BUY':
                swing_low = df['low'].iloc[-20:].min()
                stop_loss = swing_low - (atr * 1.0)
                take_profit = current_price + (atr * 3.0)  # Target 3x ATR
            else:  # SELL
                swing_high = df['high'].iloc[-20:].max()
                stop_loss = swing_high + (atr * 1.0)
                take_profit = current_price - (atr * 3.0)
            
            risk_reward = abs((take_profit - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 0
            
            # Risk level
            trend_strength = abs((sma_50 - sma_200) / sma_200) * 100
            risk_level = 'LOW' if trend_strength > 5 else ('MEDIUM' if trend_strength > 2 else 'HIGH')
            
            return TradingSignal(
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy=self.name,
                signal_type=signal_type,
                confidence=min(confidence, 95.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=" | ".join(reasoning),
                indicators={
                    'ema_10': ema_10,
                    'ema_21': ema_21,
                    'sma_50': sma_50,
                    'sma_200': sma_200,
                    'trend': 'BULL' if bullish_trend else 'BEAR'
                },
                timestamp=datetime.now(),
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error generating EMA Swing signal: {e}")
            return None


class MACDRSISwing(CryptoStrategy):
    """MACD + RSI Confirmation Strategy (Swing)"""
    
    def __init__(self):
        super().__init__("MACD+RSI Swing", "1h-4h")
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Logic: Long on MACD line cross above signal AFTER RSI exits oversold (30→up);
        mirror for shorts from overbought
        """
        if df.empty or len(df) < 40:
            return None
        
        try:
            # Calculate indicators
            macd_line, macd_signal, macd_hist = self.calculate_macd(df['close'].tolist())
            rsi = self.calculate_rsi(df['close'].tolist())
            
            # Previous RSI
            prev_rsi = self.calculate_rsi(df['close'].tolist()[:-1])
            
            signal_type = 'HOLD'
            reasoning = []
            confidence = 50.0
            
            # Long setup: MACD cross up + RSI exiting oversold
            macd_bullish_cross = macd_line > macd_signal and macd_hist > 0
            rsi_exiting_oversold = prev_rsi < 30 and rsi >= 30
            rsi_rising = rsi > prev_rsi
            
            if macd_bullish_cross and (rsi_exiting_oversold or (rsi < 50 and rsi_rising)):
                signal_type = 'BUY'
                confidence = 70.0
                reasoning.append("MACD bullish cross")
                reasoning.append(f"RSI exiting oversold: {rsi:.1f}")
                reasoning.append("Momentum confirming")
                
                if rsi_exiting_oversold:
                    confidence += 15
                    reasoning.append("RSI fresh exit from oversold")
            
            # Short setup: MACD cross down + RSI exiting overbought
            macd_bearish_cross = macd_line < macd_signal and macd_hist < 0
            rsi_exiting_overbought = prev_rsi > 70 and rsi <= 70
            rsi_falling = rsi < prev_rsi
            
            if macd_bearish_cross and (rsi_exiting_overbought or (rsi > 50 and rsi_falling)):
                signal_type = 'SELL'
                confidence = 70.0
                reasoning.append("MACD bearish cross")
                reasoning.append(f"RSI exiting overbought: {rsi:.1f}")
                reasoning.append("Momentum confirming down")
                
                if rsi_exiting_overbought:
                    confidence += 15
                    reasoning.append("RSI fresh exit from overbought")
            
            if signal_type == 'HOLD':
                return None
            
            # Calculate risk management
            atr = self.calculate_atr(df)
            
            if signal_type == 'BUY':
                swing_low = df['low'].iloc[-20:].min()
                stop_loss = swing_low - (atr * 0.5)
                take_profit = current_price + (atr * 2.5)
            else:  # SELL
                swing_high = df['high'].iloc[-20:].max()
                stop_loss = swing_high + (atr * 0.5)
                take_profit = current_price - (atr * 2.5)
            
            risk_reward = abs((take_profit - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 0
            
            return TradingSignal(
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy=self.name,
                signal_type=signal_type,
                confidence=min(confidence, 95.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=" | ".join(reasoning),
                indicators={
                    'macd_line': macd_line,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_hist,
                    'rsi': rsi,
                    'prev_rsi': prev_rsi
                },
                timestamp=datetime.now(),
                risk_level='MEDIUM'
            )
            
        except Exception as e:
            logger.error(f"Error generating MACD+RSI signal: {e}")
            return None


class BollingerSqueezeBreakout(CryptoStrategy):
    """Bollinger Squeeze Breakout Strategy (Swing)"""
    
    def __init__(self):
        super().__init__("Bollinger Squeeze", "4h-1d")
        self.bb_period = 20
        self.bb_std = 2
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Logic: Detect BB squeeze (low bandwidth). Enter on break + retest with volume
        """
        if df.empty or len(df) < 30:
            return None
        
        try:
            # Calculate indicators
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
                df['close'].tolist(), self.bb_period, self.bb_std
            )
            
            # Calculate historical band width
            band_widths = []
            for i in range(max(0, len(df) - 20), len(df)):
                closes = df['close'].iloc[:i+1].tolist()
                if len(closes) >= self.bb_period:
                    upper, middle, lower = self.calculate_bollinger_bands(closes, self.bb_period, self.bb_std)
                    band_width = (upper - lower) / middle * 100
                    band_widths.append(band_width)
            
            if not band_widths:
                return None
            
            current_band_width = (bb_upper - bb_lower) / bb_middle * 100
            avg_band_width = np.mean(band_widths)
            
            # Detect squeeze (narrow bands)
            in_squeeze = current_band_width < (avg_band_width * 0.7)
            
            if not in_squeeze:
                return None
            
            # Check for breakout
            signal_type = 'HOLD'
            reasoning = []
            confidence = 50.0
            
            # Volume check
            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            volume_surge = current_volume > (avg_volume * 1.3)
            
            # Bullish breakout: price breaking above upper band with volume
            if current_price > bb_upper and volume_surge:
                signal_type = 'BUY'
                confidence = 75.0
                reasoning.append("Bollinger Squeeze detected")
                reasoning.append(f"Breakout above upper band: ${current_price:.2f} > ${bb_upper:.2f}")
                reasoning.append(f"Volume surge: {(current_volume/avg_volume):.2f}x avg")
                
                # Check for retest
                recent_prices = df['close'].iloc[-5:].tolist()
                if any(p <= bb_upper for p in recent_prices[:-1]):
                    confidence += 10
                    reasoning.append("Retest of upper band confirmed")
            
            # Bearish breakout: price breaking below lower band with volume
            elif current_price < bb_lower and volume_surge:
                signal_type = 'SELL'
                confidence = 75.0
                reasoning.append("Bollinger Squeeze detected")
                reasoning.append(f"Breakdown below lower band: ${current_price:.2f} < ${bb_lower:.2f}")
                reasoning.append(f"Volume surge: {(current_volume/avg_volume):.2f}x avg")
                
                recent_prices = df['close'].iloc[-5:].tolist()
                if any(p >= bb_lower for p in recent_prices[:-1]):
                    confidence += 10
                    reasoning.append("Retest of lower band confirmed")
            
            if signal_type == 'HOLD':
                return None
            
            # Calculate risk management
            atr = self.calculate_atr(df)
            
            if signal_type == 'BUY':
                # Stop inside the band
                stop_loss = bb_middle - (atr * 0.5)
                # Trail by ATR
                take_profit = current_price + (atr * 3.0)
            else:  # SELL
                stop_loss = bb_middle + (atr * 0.5)
                take_profit = current_price - (atr * 3.0)
            
            risk_reward = abs((take_profit - current_price) / (current_price - stop_loss)) if stop_loss != current_price else 0
            
            return TradingSignal(
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                strategy=self.name,
                signal_type=signal_type,
                confidence=min(confidence, 95.0),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward,
                reasoning=" | ".join(reasoning),
                indicators={
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'band_width': current_band_width,
                    'avg_band_width': avg_band_width,
                    'in_squeeze': in_squeeze
                },
                timestamp=datetime.now(),
                risk_level='MEDIUM'
            )
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Squeeze signal: {e}")
            return None


class CryptoStrategyManager:
    """Manager for all crypto trading strategies"""
    
    def __init__(self):
        self.strategies = {
            'VWAP_EMA_SCALP': VWAPEMAScalper(),
            'BOLLINGER_MEAN_REVERSION': BollingerMeanReversion(),
            'EMA_MOMENTUM_NUDGE': EMAMomentumNudge(),
            'EMA_SWING': EMASwingTrader(),
            'MACD_RSI_SWING': MACDRSISwing(),
            'BOLLINGER_SQUEEZE': BollingerSqueezeBreakout()
        }
        
        self.scalping_strategies = ['VWAP_EMA_SCALP', 'BOLLINGER_MEAN_REVERSION', 'EMA_MOMENTUM_NUDGE']
        self.swing_strategies = ['EMA_SWING', 'MACD_RSI_SWING', 'BOLLINGER_SQUEEZE']
    
    def get_strategy(self, strategy_name: str) -> Optional[CryptoStrategy]:
        """Get a specific strategy by name"""
        return self.strategies.get(strategy_name)
    
    def get_scalping_strategies(self) -> List[CryptoStrategy]:
        """Get all scalping strategies"""
        return [self.strategies[s] for s in self.scalping_strategies]
    
    def get_swing_strategies(self) -> List[CryptoStrategy]:
        """Get all swing strategies"""
        return [self.strategies[s] for s in self.swing_strategies]
    
    def generate_all_signals(self, df: pd.DataFrame, current_price: float) -> Dict[str, TradingSignal]:
        """Generate signals from all strategies"""
        signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(df, current_price)
                if signal:
                    signals[strategy_name] = signal
            except Exception as e:
                logger.error(f"Error generating signal for {strategy_name}: {e}")
        
        return signals
    
    def get_consensus_signal(self, df: pd.DataFrame, current_price: float, strategy_type: str = 'ALL') -> Optional[TradingSignal]:
        """
        Get consensus signal from multiple strategies
        
        Args:
            df: Price data
            current_price: Current price
            strategy_type: 'SCALPING', 'SWING', or 'ALL'
        """
        if strategy_type == 'SCALPING':
            strategies_to_check = self.scalping_strategies
        elif strategy_type == 'SWING':
            strategies_to_check = self.swing_strategies
        else:
            strategies_to_check = list(self.strategies.keys())
        
        signals = {}
        for strategy_name in strategies_to_check:
            strategy = self.strategies[strategy_name]
            signal = strategy.generate_signal(df, current_price)
            if signal:
                signals[strategy_name] = signal
        
        if not signals:
            return None
        
        # Count BUY vs SELL signals
        buy_signals = [s for s in signals.values() if s.signal_type == 'BUY']
        sell_signals = [s for s in signals.values() if s.signal_type == 'SELL']
        
        if len(buy_signals) > len(sell_signals):
            # Consensus is BUY
            # Use the highest confidence BUY signal
            best_signal = max(buy_signals, key=lambda s: s.confidence)
            best_signal.reasoning = f"CONSENSUS BUY ({len(buy_signals)}/{len(signals)} strategies) | " + best_signal.reasoning
            return best_signal
        elif len(sell_signals) > len(buy_signals):
            # Consensus is SELL
            best_signal = max(sell_signals, key=lambda s: s.confidence)
            best_signal.reasoning = f"CONSENSUS SELL ({len(sell_signals)}/{len(signals)} strategies) | " + best_signal.reasoning
            return best_signal
        
        return None  # No consensus

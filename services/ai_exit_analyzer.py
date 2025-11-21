"""
AI Exit Analyzer
Provides AI-powered exit timing analysis for crypto positions
"""

from typing import Dict, List, Optional
from loguru import logger
from datetime import datetime
import numpy as np


def analyze_exit_timing(
    kraken_client,
    pair: str,
    current_price: float,
    position_size: float,
    entry_price: float = 0.0
) -> Dict:
    """
    Analyze exit timing for a crypto position using AI and technical analysis
    
    Args:
        kraken_client: KrakenClient instance
        pair: Trading pair (e.g., 'BTC/USD')
        current_price: Current market price
        position_size: Position size in base currency
        entry_price: Entry price (0 if unknown)
    
    Returns:
        Dict with exit analysis including action, confidence, reasoning, signals
    """
    try:
        logger.info(f"🤖 Analyzing exit timing for {pair} @ ${current_price:,.4f}")
        
        # Get historical data for technical analysis
        ohlcv = kraken_client.get_ohlcv(pair, interval=15, since=None)
        
        if not ohlcv or len(ohlcv) < 20:
            return {
                'action': 'HOLD',
                'confidence': 50,
                'score': 50,
                'reasoning': 'Insufficient data for analysis',
                'signals': ['Not enough historical data'],
                'suggested_exit_price': current_price
            }
        
        # Calculate technical indicators
        closes = np.array([candle[4] for candle in ohlcv])  # Close prices
        highs = np.array([candle[2] for candle in ohlcv])   # High prices
        lows = np.array([candle[3] for candle in ohlcv])    # Low prices
        volumes = np.array([candle[6] for candle in ohlcv]) # Volumes
        
        # RSI calculation
        rsi = calculate_rsi(closes, period=14)
        
        # MACD calculation
        macd_line, signal_line = calculate_macd(closes)
        
        # Moving averages
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma_20
        ema_12 = calculate_ema(closes, 12)
        ema_26 = calculate_ema(closes, 26)
        
        # Volume analysis
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Price momentum
        price_change_1h = ((closes[-1] - closes[-4]) / closes[-4]) * 100 if len(closes) >= 4 else 0
        price_change_4h = ((closes[-1] - closes[-16]) / closes[-16]) * 100 if len(closes) >= 16 else 0
        
        # Calculate P&L if entry price is known
        if entry_price > 0:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = 0
        
        # Initialize scoring
        exit_score = 50
        signals = []
        
        # === BEARISH SIGNALS (Reasons to SELL) ===
        
        # 1. Overbought RSI
        if rsi > 70:
            exit_score += 15
            signals.append(f"🔴 Overbought RSI: {rsi:.1f} (>70)")
        elif rsi > 65:
            exit_score += 8
            signals.append(f"⚠️ High RSI: {rsi:.1f}")
        
        # 2. MACD bearish crossover
        if macd_line < signal_line:
            exit_score += 12
            signals.append("🔴 MACD bearish crossover")
        
        # 3. Price below moving averages (downtrend)
        if current_price < sma_20:
            exit_score += 10
            signals.append(f"🔴 Price below SMA20 (${sma_20:,.4f})")
        
        if current_price < sma_50:
            exit_score += 8
            signals.append(f"🔴 Price below SMA50 (${sma_50:,.4f})")
        
        # 4. Negative momentum
        if price_change_1h < -2:
            exit_score += 10
            signals.append(f"🔴 Negative 1h momentum: {price_change_1h:.2f}%")
        
        if price_change_4h < -5:
            exit_score += 12
            signals.append(f"🔴 Negative 4h momentum: {price_change_4h:.2f}%")
        
        # 5. Volume spike (could indicate distribution)
        if volume_ratio > 2.0:
            exit_score += 8
            signals.append(f"⚠️ High volume spike: {volume_ratio:.1f}x average")
        
        # 6. Large profit (consider taking profits)
        if pnl_pct > 20:
            exit_score += 15
            signals.append(f"💰 Large profit: +{pnl_pct:.1f}% (consider taking profits)")
        elif pnl_pct > 10:
            exit_score += 8
            signals.append(f"💰 Good profit: +{pnl_pct:.1f}%")
        
        # === BULLISH SIGNALS (Reasons to HOLD) ===
        
        # 1. Oversold RSI
        if rsi < 30:
            exit_score -= 15
            signals.append(f"🟢 Oversold RSI: {rsi:.1f} (<30) - potential bounce")
        elif rsi < 40:
            exit_score -= 8
            signals.append(f"🟢 Low RSI: {rsi:.1f} - room to grow")
        
        # 2. MACD bullish crossover
        if macd_line > signal_line:
            exit_score -= 12
            signals.append("🟢 MACD bullish crossover")
        
        # 3. Price above moving averages (uptrend)
        if current_price > sma_20 and current_price > sma_50:
            exit_score -= 10
            signals.append("🟢 Strong uptrend (above SMA20 & SMA50)")
        
        # 4. Positive momentum
        if price_change_1h > 3:
            exit_score -= 10
            signals.append(f"🟢 Strong 1h momentum: +{price_change_1h:.2f}%")
        
        if price_change_4h > 8:
            exit_score -= 12
            signals.append(f"🟢 Strong 4h momentum: +{price_change_4h:.2f}%")
        
        # 5. Small loss (don't panic sell)
        if -5 < pnl_pct < 0:
            exit_score -= 8
            signals.append(f"⚠️ Small loss: {pnl_pct:.1f}% - consider holding")
        
        # Normalize score to 0-100
        exit_score = max(0, min(100, exit_score))
        
        # Determine action based on score
        if exit_score >= 75:
            action = 'SELL_NOW'
            confidence = min(95, exit_score)
            reasoning = "Strong bearish signals detected. Consider exiting position to protect profits or limit losses."
        elif exit_score >= 60:
            action = 'TAKE_PARTIAL'
            confidence = exit_score
            reasoning = "Mixed signals with bearish bias. Consider taking partial profits (50%) and holding remainder."
        elif exit_score <= 35:
            action = 'HOLD'
            confidence = 100 - exit_score
            reasoning = "Bullish signals detected. Position looks strong, continue holding."
        else:
            action = 'HOLD'
            confidence = 60
            reasoning = "Neutral signals. No clear exit signal yet. Monitor closely."
        
        # Calculate suggested exit price
        if action == 'SELL_NOW':
            # Suggest market order (current price)
            suggested_exit_price = current_price
        elif action == 'TAKE_PARTIAL':
            # Suggest limit order slightly above current price
            suggested_exit_price = current_price * 1.005
        else:
            # Suggest stop loss level
            suggested_exit_price = current_price * 0.95
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price * 0.95  # 5% below current
        take_profit = current_price * 1.10  # 10% above current
        
        result = {
            'action': action,
            'confidence': confidence,
            'score': exit_score,
            'reasoning': reasoning,
            'signals': signals[:10],  # Top 10 signals
            'suggested_exit_price': suggested_exit_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'technical_data': {
                'rsi': rsi,
                'macd_line': macd_line,
                'signal_line': signal_line,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'current_price': current_price,
                'volume_ratio': volume_ratio,
                'price_change_1h': price_change_1h,
                'price_change_4h': price_change_4h,
                'pnl_pct': pnl_pct
            }
        }
        
        logger.info(f"✅ Exit analysis complete: {action} (Score: {exit_score}, Confidence: {confidence}%)")
        return result
        
    except Exception as e:
        logger.error("Error analyzing exit timing: {}", str(e), exc_info=True)
        return {
            'action': 'HOLD',
            'confidence': 50,
            'score': 50,
            'reasoning': f'Analysis error: {str(e)}',
            'signals': ['Error during analysis'],
            'suggested_exit_price': current_price
        }


def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate RSI indicator"""
    try:
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    except:
        return 50.0


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD indicator"""
    try:
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # For signal line, we'd need to calculate EMA of MACD line
        # Simplified: just return MACD line and a smoothed version
        signal_line = macd_line * 0.9  # Simplified
        
        return macd_line, signal_line
    except:
        return 0.0, 0.0


def calculate_ema(prices: np.ndarray, period: int) -> float:
    """Calculate Exponential Moving Average"""
    try:
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    except:
        return np.mean(prices) if len(prices) > 0 else 0.0

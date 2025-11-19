"""
Freqtrade Signal Adapter
Converts Freqtrade strategy outputs to TradingSignal format for signal generator UI
"""

from loguru import logger
from typing import Optional, Dict
import pandas as pd
from datetime import datetime

from services.freqtrade_strategies import FreqtradeStrategyAdapter
from services.crypto_strategies import TradingSignal


class FreqtradeSignalWrapper:
    """
    Wraps Freqtrade strategies to work with the Signal Generator UI
    Converts freqtrade analysis results to TradingSignal objects
    """
    
    def __init__(self, kraken_client, strategy_id: str, strategy_name: str, timeframe: str):
        """
        Args:
            kraken_client: KrakenClient instance
            strategy_id: ID of the freqtrade strategy (e.g., 'ema_crossover')
            strategy_name: Display name (e.g., 'EMA Crossover + Heikin Ashi')
            timeframe: Timeframe string (e.g., '5m', '15m')
        """
        self.kraken_client = kraken_client
        self.strategy_id = strategy_id
        self.name = strategy_name
        self.timeframe = timeframe
        self.adapter = FreqtradeStrategyAdapter(kraken_client)
    
    def generate_signal(self, df: pd.DataFrame, current_price: float) -> Optional[TradingSignal]:
        """
        Generate trading signal from freqtrade strategy
        
        Args:
            df: OHLCV DataFrame (already fetched by signal generator)
            current_price: Current market price
        
        Returns:
            TradingSignal object or None
        """
        try:
            # Freqtrade strategies need specific columns
            if df.empty or len(df) < 100:
                logger.debug(f"Insufficient data for {self.strategy_id}: {len(df)} candles")
                return None
            
            # Calculate indicators
            df_with_indicators = self.adapter.calculate_indicators(df)
            
            if df_with_indicators.empty:
                logger.warning(f"Failed to calculate indicators for {self.strategy_id}")
                return None
            
            # Get strategy function
            strategy_func = getattr(self.adapter, f'strategy_{self.strategy_id}', None)
            if not strategy_func:
                logger.error(f"Strategy function not found: strategy_{self.strategy_id}")
                return None
            
            # Run strategy
            entry_signal, exit_signal, signals = strategy_func(df_with_indicators)
            
            # Convert to TradingSignal
            if entry_signal:
                signal_type = 'BUY'
            elif exit_signal:
                signal_type = 'SELL'
            else:
                # For HOLD signals, check if we're near entry or exit
                if signals.get('near_entry'):
                    signal_type = 'BUY'  # Weak buy signal
                elif signals.get('near_exit'):
                    signal_type = 'SELL'  # Weak sell signal
                else:
                    # No clear signal - don't generate
                    return None
            
            # Get confidence score (already calculated dynamically in freqtrade strategies)
            confidence = signals.get('confidence_score', 50)
            if 'confidence_factors' in signals:
                # Already has dynamic confidence from _calculate_dynamic_confidence
                confidence = self.adapter._calculate_dynamic_confidence(
                    entry_signal=entry_signal,
                    exit_signal=exit_signal,
                    signals=signals,
                    strategy=self.strategy_id,
                    df=df_with_indicators
                )
            
            # Calculate entry/stop/target based on freqtrade ROI settings
            strategy_config = self.adapter.strategies.get(self.strategy_id, {})
            stoploss_pct = abs(strategy_config.get('stoploss', -0.05))
            
            # Get ROI targets (use first non-immediate target)
            roi_config = strategy_config.get('minimal_roi', {"0": 0.03})
            roi_targets = sorted([(int(k), v) for k, v in roi_config.items()], reverse=True)
            
            # Use the best ROI target (immediate or first time target)
            if roi_targets:
                target_roi = roi_targets[-1][1]  # Highest profit target
            else:
                target_roi = 0.03  # Default 3%
            
            # Calculate prices based on signal type
            if signal_type == 'BUY':
                entry_price = current_price
                stop_loss = entry_price * (1 - stoploss_pct)
                take_profit = entry_price * (1 + target_roi)
            else:  # SELL
                entry_price = current_price
                stop_loss = entry_price * (1 + stoploss_pct)
                take_profit = entry_price * (1 - target_roi)
            
            # Calculate risk/reward
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 1.0
            
            # Determine risk level
            if stoploss_pct < 0.03:
                risk_level = 'LOW'
            elif stoploss_pct < 0.06:
                risk_level = 'MEDIUM'
            elif stoploss_pct < 0.10:
                risk_level = 'HIGH'
            else:
                risk_level = 'EXTREME'
            
            # Build reasoning from signals
            reasoning_parts = []
            
            # Add strategy-specific reasoning
            if self.strategy_id == 'ema_crossover':
                reasoning_parts.append(f"EMA20/50/100 alignment: {'✓' if signals.get('ema_aligned') else '✗'}")
                reasoning_parts.append(f"Heikin Ashi: {signals.get('ha_type', 'unknown')}")
                reasoning_parts.append(f"RSI: {signals.get('rsi', 0):.1f}")
                
            elif self.strategy_id == 'rsi_stoch_hammer':
                reasoning_parts.append(f"RSI: {signals.get('rsi', 0):.1f} (target <30)")
                reasoning_parts.append(f"Stochastic: {signals.get('slowk', 0):.1f} (target <20)")
                reasoning_parts.append(f"Hammer pattern: {'✓' if signals.get('hammer') else '✗'}")
                
            elif self.strategy_id == 'fisher_rsi_multi':
                reasoning_parts.append(f"Fisher RSI: {signals.get('fisher_rsi', 0):.3f}")
                reasoning_parts.append(f"MFI: {signals.get('mfi', 0):.1f}")
                reasoning_parts.append(f"EMA aligned: {'✓' if signals.get('ema_aligned') else '✗'}")
                
            elif self.strategy_id == 'macd_volume':
                reasoning_parts.append(f"MACD: {signals.get('macd', 0):.4f} vs Signal: {signals.get('macdsignal', 0):.4f}")
                reasoning_parts.append(f"Volume spike: {'✓' if signals.get('volume_spike') else '✗'} ({signals.get('volume_ratio', 1):.1f}x)")
                reasoning_parts.append(f"Fisher RSI Norma: {signals.get('fisher_rsi_norma', 50):.1f}")
                
            elif self.strategy_id == 'aggressive_scalp':
                reasoning_parts.append(f"EMA5/10 cross: {signals.get('ema_cross_strength', 0):.2f}% difference")
                reasoning_parts.append(f"Volume spike: {'✓' if signals.get('volume_spike') else '✗'}")
                reasoning_parts.append(f"ADX: {signals.get('adx', 0):.1f} (trend strength)")
                
            elif self.strategy_id == 'orb_fvg':
                reasoning_parts.append(f"ORB Signal: {signals.get('orb_signal', 'NEUTRAL')}")
                reasoning_parts.append(f"FVG Strength: {signals.get('fvg_strength', 0):.1f}")
                reasoning_parts.append(f"Risk/Reward: {signals.get('risk_reward_ratio', 0):.2f}")
            
            # Add confidence factors if available
            if 'confidence_factors' in signals:
                reasoning_parts.append("\nConfidence Factors:")
                for factor in signals['confidence_factors'][:5]:  # Top 5 factors
                    reasoning_parts.append(f"  • {factor}")
            
            reasoning = " | ".join(reasoning_parts) if reasoning_parts else f"{self.name} conditions met"
            
            # Create TradingSignal object
            return TradingSignal(
                symbol="",  # Will be set by caller
                strategy=self.name,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                reasoning=reasoning,
                indicators=signals,
                timestamp=datetime.now(),
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error generating signal from {self.strategy_id}: {e}", exc_info=True)
            return None


def get_freqtrade_strategy_wrappers(kraken_client) -> Dict[str, FreqtradeSignalWrapper]:
    """
    Get all available Freqtrade strategies wrapped for signal generation
    
    Returns:
        Dict mapping strategy display name to FreqtradeSignalWrapper instance
    """
    strategies = {
        '📈 EMA Crossover + Heikin Ashi (Freqtrade)': FreqtradeSignalWrapper(
            kraken_client, 'ema_crossover', 'EMA Crossover + Heikin Ashi', '5m'
        ),
        '📊 RSI + Stochastic + Hammer (Freqtrade)': FreqtradeSignalWrapper(
            kraken_client, 'rsi_stoch_hammer', 'RSI + Stochastic + Hammer', '5m'
        ),
        '🎯 Fisher RSI Multi-Indicator (Freqtrade)': FreqtradeSignalWrapper(
            kraken_client, 'fisher_rsi_multi', 'Fisher RSI Multi-Indicator', '5m'
        ),
        '📉 MACD + Volume + RSI (Freqtrade)': FreqtradeSignalWrapper(
            kraken_client, 'macd_volume', 'MACD + Volume + RSI', '5m'
        ),
        '🔥 Aggressive Scalping (Freqtrade)': FreqtradeSignalWrapper(
            kraken_client, 'aggressive_scalp', 'Aggressive Scalping', '1m'
        ),
        '🎪 ORB + Fair Value Gap (Freqtrade)': FreqtradeSignalWrapper(
            kraken_client, 'orb_fvg', 'Opening Range Breakout + FVG', '1m'
        ),
    }
    
    return strategies

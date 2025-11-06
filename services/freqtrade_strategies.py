"""
Freqtrade Strategy Integration for Sentient Trader
Adapted from: https://github.com/freqtrade/freqtrade-strategies

Provides 5 proven crypto trading strategies optimized for different market conditions.
"""

from loguru import logger
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import talib
    ta = talib
except ImportError:
    logger.warning("TA-Lib not found. Install with: pip install TA-Lib")
    ta = None


class FreqtradeStrategyAdapter:
    """
    Adapter to translate Freqtrade strategies for use in Sentient Trader.
    Implements 5 battle-tested crypto strategies.
    """
    
    def __init__(self, kraken_client):
        self.kraken_client = kraken_client
        
        # Strategy configurations
        self.strategies = {
            'ema_crossover': {
                'name': 'EMA Crossover + Heikin Ashi',
                'timeframe': '5m',
                'minimal_roi': {"60": 0.01, "30": 0.03, "20": 0.04, "0": 0.05},
                'stoploss': -0.10,
                'description': 'EMA 20/50/100 crossover with Heikin Ashi confirmation'
            },
            'rsi_stoch_hammer': {
                'name': 'RSI + Stochastic + Hammer',
                'timeframe': '5m',
                'minimal_roi': {"60": 0.01, "30": 0.03, "20": 0.04, "0": 0.05},
                'stoploss': -0.10,
                'description': 'Oversold RSI/Stoch + Bollinger Band + Hammer candlestick'
            },
            'fisher_rsi_multi': {
                'name': 'Fisher RSI Multi-Indicator',
                'timeframe': '5m',
                'minimal_roi': {"60": 0.01, "30": 0.03, "20": 0.04, "0": 0.05},
                'stoploss': -0.10,
                'description': 'Fisher RSI + MFI + Stoch + EMA confirmation'
            },
            'macd_volume': {
                'name': 'MACD + Volume + RSI',
                'timeframe': '5m',
                'minimal_roi': {"1440": 0.01, "80": 0.02, "40": 0.03, "20": 0.04, "0": 0.05},
                'stoploss': -0.10,
                'description': 'MACD momentum with volume surge and Fisher RSI'
            },
            'aggressive_scalp': {
                'name': 'Aggressive Scalping',
                'timeframe': '1m',
                'minimal_roi': {"10": 0.01, "5": 0.02, "0": 0.03},
                'stoploss': -0.05,
                'description': 'Fast scalping with tight stops and quick exits'
            }
        }
    
    def get_available_strategies(self) -> List[Dict]:
        """Get list of available strategies"""
        return [
            {
                'id': key,
                **config
            }
            for key, config in self.strategies.items()
        ]
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for strategies"""
        if df.empty or len(df) < 100:
            logger.warning(f"Insufficient data: {len(df)} candles")
            return df
        
        try:
            # Validate DataFrame structure
            logger.debug(f"DataFrame type: {type(df)}, shape: {df.shape}")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            logger.debug(f"First row dtypes: {df.dtypes.to_dict()}")
            
            # Ensure we're working with a proper DataFrame, not a list
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Expected DataFrame, got {type(df)}")
                return df
            # EMAs
            df['ema5'] = ta.EMA(df['close'].values, timeperiod=5)
            df['ema10'] = ta.EMA(df['close'].values, timeperiod=10)
            df['ema20'] = ta.EMA(df['close'].values, timeperiod=20)
            df['ema50'] = ta.EMA(df['close'].values, timeperiod=50)
            df['ema100'] = ta.EMA(df['close'].values, timeperiod=100)
            
            # RSI
            df['rsi'] = ta.RSI(df['close'].values, timeperiod=14)
            
            # Fisher RSI
            rsi_normalized = 0.1 * (df['rsi'] - 50)
            df['fisher_rsi'] = (np.exp(2 * rsi_normalized) - 1) / (np.exp(2 * rsi_normalized) + 1)
            df['fisher_rsi_norma'] = 50 * (df['fisher_rsi'] + 1)
            
            # Stochastic (returns tuple: slowk, slowd)
            slowk, slowd = ta.STOCH(df['high'].values, df['low'].values, df['close'].values)
            df['slowk'] = slowk
            df['slowd'] = slowd
            
            # Stochastic Fast (returns tuple: fastk, fastd)
            fastk, fastd = ta.STOCHF(df['high'].values, df['low'].values, df['close'].values)
            df['fastk'] = fastk
            df['fastd'] = fastd
            
            # MACD (returns tuple: macd, macdsignal, macdhist)
            macd, macdsignal, macdhist = ta.MACD(df['close'].values)
            df['macd'] = macd
            df['macdsignal'] = macdsignal
            df['macdhist'] = macdhist
            
            # Bollinger Bands (returns tuple: upper, middle, lower)
            bb_upper, bb_middle, bb_lower = ta.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # SAR
            df['sar'] = ta.SAR(df['high'].values, df['low'].values)
            
            # MFI
            df['mfi'] = ta.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14)
            
            # Minus DI
            df['minus_di'] = ta.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # ADX
            df['adx'] = ta.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            # Candlestick patterns
            df['hammer'] = ta.CDLHAMMER(df['open'].values, df['high'].values, df['low'].values, df['close'].values)
            
            # Heikin Ashi
            try:
                logger.debug("Calculating Heikin Ashi candles...")
                df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
                logger.debug(f"ha_close calculated, sample: {df['ha_close'].iloc[-1]}")
                
                df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
                logger.debug(f"ha_open calculated, sample: {df['ha_open'].iloc[-1]}")
                
                # Use explicit column selection
                df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
                df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
                logger.debug("Heikin Ashi candles completed")
            except Exception as ha_error:
                logger.error(f"Error calculating Heikin Ashi: {ha_error}", exc_info=True)
                # Set default values if HA calculation fails
                df['ha_close'] = df['close']
                df['ha_open'] = df['open']
                df['ha_high'] = df['high']
                df['ha_low'] = df['low']
            
            # SMA
            df['sma40'] = ta.SMA(df['close'].values, timeperiod=40)
            
            # Volume MA
            df['volume_ma'] = ta.SMA(df['volume'].values, timeperiod=20)
            
            logger.debug(f"Calculated indicators for {len(df)} candles")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return df
    
    def strategy_ema_crossover(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Strategy 001: EMA Crossover + Heikin Ashi
        Entry: EMA20 crosses above EMA50, HA close > EMA20, green HA bar
        Exit: EMA50 crosses above EMA100, HA close < EMA20, red HA bar
        """
        if len(df) < 2:
            return False, False, {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Entry signal
        ema20_cross_up = (previous['ema20'] <= previous['ema50']) and (current['ema20'] > current['ema50'])
        ha_green = current['ha_close'] > current['ha_open']
        entry = ema20_cross_up and (current['ha_close'] > current['ema20']) and ha_green
        
        # Exit signal
        ema50_cross_up = (previous['ema50'] <= previous['ema100']) and (current['ema50'] > current['ema100'])
        ha_red = current['ha_close'] < current['ha_open']
        exit_signal = ema50_cross_up and (current['ha_close'] < current['ema20']) and ha_red
        
        signals = {
            'ema20': current['ema20'],
            'ema50': current['ema50'],
            'ema100': current['ema100'],
            'ha_close': current['ha_close'],
            'ha_type': 'green' if ha_green else 'red'
        }
        
        return entry, exit_signal, signals
    
    def strategy_rsi_stoch_hammer(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Strategy 002: RSI + Stochastic + Hammer
        Entry: RSI<30, Stoch<20, Close below BB lower, Hammer pattern
        Exit: SAR above close, Fisher RSI > 0.3
        """
        if len(df) < 2:
            return False, False, {}
        
        current = df.iloc[-1]
        
        # Entry signal
        entry = (
            current['rsi'] < 30 and
            current['slowk'] < 20 and
            current['close'] < current['bb_lower'] and
            current['hammer'] == 100
        )
        
        # Exit signal
        exit_signal = (
            current['sar'] > current['close'] and
            current['fisher_rsi'] > 0.3
        )
        
        signals = {
            'rsi': current['rsi'],
            'slowk': current['slowk'],
            'bb_lower': current['bb_lower'],
            'hammer': current['hammer'] == 100,
            'sar': current['sar'],
            'fisher_rsi': current['fisher_rsi']
        }
        
        return entry, exit_signal, signals
    
    def strategy_fisher_rsi_multi(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Strategy 003: Fisher RSI Multi-Indicator
        Entry: RSI<28, Fisher RSI<-0.94, MFI<16, EMA50>EMA100 or EMA5 crosses EMA10, fastd>fastk
        Exit: SAR above close, Fisher RSI>0.3
        """
        if len(df) < 2:
            return False, False, {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Entry signal
        ema_condition = (current['ema50'] > current['ema100']) or (
            (previous['ema5'] <= previous['ema10']) and (current['ema5'] > current['ema10'])
        )
        
        entry = (
            current['rsi'] < 28 and
            current['rsi'] > 0 and
            current['close'] < current['sma40'] and
            current['fisher_rsi'] < -0.94 and
            current['mfi'] < 16.0 and
            ema_condition and
            current['fastd'] > current['fastk'] and
            current['fastd'] > 0
        )
        
        # Exit signal
        exit_signal = (
            current['sar'] > current['close'] and
            current['fisher_rsi'] > 0.3
        )
        
        signals = {
            'rsi': current['rsi'],
            'fisher_rsi': current['fisher_rsi'],
            'mfi': current['mfi'],
            'fastd': current['fastd'],
            'fastk': current['fastk'],
            'ema_aligned': ema_condition
        }
        
        return entry, exit_signal, signals
    
    def strategy_macd_volume(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Strategy 005: MACD + Volume + Fisher RSI
        Entry: MACD crossover, Volume > avg, RSI<30, FastD<30, Fisher RSI norma<30
        Exit: RSI>70, Minus DI>50, Fisher RSI norma>50
        """
        if len(df) < 2:
            return False, False, {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # MACD crossover
        macd_cross = (previous['macd'] <= previous['macdsignal']) and (current['macd'] > current['macdsignal'])
        
        # Entry signal
        entry = (
            macd_cross and
            current['volume'] > current['volume_ma'] * 1.5 and
            current['rsi'] < 30 and
            current['fastd'] < 30 and
            current['fisher_rsi_norma'] < 30
        )
        
        # Exit signal
        exit_signal = (
            current['rsi'] > 70 and
            current['minus_di'] > 50 and
            current['fisher_rsi_norma'] > 50
        )
        
        signals = {
            'macd': current['macd'],
            'macdsignal': current['macdsignal'],
            'volume_ratio': current['volume'] / current['volume_ma'] if current['volume_ma'] > 0 else 0,
            'rsi': current['rsi'],
            'fastd': current['fastd'],
            'fisher_rsi_norma': current['fisher_rsi_norma'],
            'minus_di': current['minus_di']
        }
        
        return entry, exit_signal, signals
    
    def strategy_aggressive_scalp(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Aggressive Scalping Strategy
        Entry: Fast EMA cross, RSI<35, Volume spike, ADX>20
        Exit: Quick profit (1-3%), RSI>65, or stop loss
        """
        if len(df) < 2:
            return False, False, {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Fast EMA crossover
        ema_cross = (previous['ema5'] <= previous['ema10']) and (current['ema5'] > current['ema10'])
        
        # Volume spike
        volume_spike = current['volume'] > current['volume_ma'] * 2
        
        # Entry signal
        entry = (
            ema_cross and
            current['rsi'] < 35 and
            volume_spike and
            current['adx'] > 20 and
            current['close'] > current['ema20']
        )
        
        # Exit signal (quick profit or reversal)
        exit_signal = (
            current['rsi'] > 65 or
            current['ema5'] < current['ema10'] or
            current['close'] < current['ema10']
        )
        
        signals = {
            'ema5': current['ema5'],
            'ema10': current['ema10'],
            'rsi': current['rsi'],
            'adx': current['adx'],
            'volume_spike': volume_spike
        }
        
        return entry, exit_signal, signals
    
    def analyze_crypto(self, symbol: str, strategy: str = 'ema_crossover', interval: str = '5') -> Dict:
        """
        Analyze a crypto pair using the specified freqtrade strategy
        
        Args:
            symbol: Crypto pair (e.g., 'BTCUSD', 'ETHUSD')
            strategy: Strategy ID ('ema_crossover', 'rsi_stoch_hammer', etc.)
            interval: Timeframe in minutes (1, 5, 15, 60, etc.)
        
        Returns:
            Dict with analysis results
        """
        if strategy not in self.strategies:
            logger.error(f"Unknown strategy: {strategy}")
            return {'error': f'Unknown strategy: {strategy}'}
        
        if ta is None:
            return {'error': 'TA-Lib not installed. Run: pip install TA-Lib'}
        
        try:
            # Normalize symbol format (crvusd -> CRV/USD)
            if '/' not in symbol:
                # Assume format is like 'crvusd', 'btcusd', etc.
                symbol_upper = symbol.upper()
                if symbol_upper.endswith('USD'):
                    base = symbol_upper[:-3]
                    symbol = f"{base}/USD"
                elif symbol_upper.endswith('USDT'):
                    base = symbol_upper[:-4]
                    symbol = f"{base}/USDT"
                else:
                    symbol = symbol_upper
            
            # Get OHLCV data
            logger.info(f"Fetching {symbol} data for {strategy} strategy...")
            ohlcv_list = self.kraken_client.get_ohlc_data(symbol, interval=interval)
            
            # Convert list to DataFrame
            if not ohlcv_list:
                logger.warning(f"No OHLC data returned for {symbol}")
                return {'error': f'Failed to fetch data for {symbol}'}
            
            import pandas as pd
            ohlcv = pd.DataFrame(ohlcv_list)
            
            if ohlcv.empty:
                return {'error': f'Failed to fetch data for {symbol}'}
            
            # Ensure required columns exist and are numeric
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in ohlcv.columns]
            if missing_cols:
                logger.error(f"Missing columns in OHLC data: {missing_cols}. Available: {ohlcv.columns.tolist()}")
                return {'error': f'Invalid OHLC data structure for {symbol}'}
            
            # Convert to numeric types (TA-Lib requires numpy arrays)
            for col in required_cols:
                ohlcv[col] = pd.to_numeric(ohlcv[col], errors='coerce')
            
            logger.debug(f"DataFrame created with {len(ohlcv)} rows and columns: {ohlcv.columns.tolist()}")
            
            # Calculate indicators
            df = self.calculate_indicators(ohlcv)
            
            # Run strategy
            strategy_func = getattr(self, f'strategy_{strategy}', None)
            if not strategy_func:
                return {'error': f'Strategy function not found: {strategy}'}
            
            entry_signal, exit_signal, signals = strategy_func(df)
            
            # Calculate targets based on ROI settings
            current_price = float(df.iloc[-1]['close'])
            roi_config = self.strategies[strategy]['minimal_roi']
            stoploss = self.strategies[strategy]['stoploss']
            
            # Get ROI targets
            roi_targets = []
            for time_minutes, roi_percent in sorted(roi_config.items(), key=lambda x: int(x[0]), reverse=True):
                target_price = current_price * (1 + roi_percent)
                roi_targets.append({
                    'time': f"{time_minutes} min" if time_minutes != "0" else "immediate",
                    'price': round(target_price, 8),
                    'gain_percent': roi_percent * 100
                })
            
            # Calculate stop loss
            stop_price = current_price * (1 + stoploss)
            
            # Risk assessment
            risk_level = 'LOW'
            if stoploss < -0.08:
                risk_level = 'HIGH'
            elif stoploss < -0.05:
                risk_level = 'MEDIUM'
            
            # Confidence score - DYNAMIC based on indicator alignment
            confidence = self._calculate_dynamic_confidence(
                entry_signal=entry_signal,
                exit_signal=exit_signal,
                signals=signals,
                strategy=strategy,
                df=df
            )
            
            result = {
                'symbol': symbol,
                'strategy': self.strategies[strategy]['name'],
                'strategy_id': strategy,
                'timeframe': f"{interval} min",
                'timestamp': datetime.now().isoformat(),
                'current_price': round(current_price, 8),
                'entry_signal': entry_signal,
                'exit_signal': exit_signal,
                'signals': signals,
                'roi_targets': roi_targets,
                'stop_loss': round(stop_price, 8),
                'risk_level': risk_level,
                'confidence_score': confidence,
                'description': self.strategies[strategy]['description'],
                'recommendation': 'BUY' if entry_signal else ('SELL' if exit_signal else 'HOLD')
            }
            
            logger.info(f"{symbol} {strategy}: {result['recommendation']} (confidence: {confidence}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} with {strategy}: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _calculate_dynamic_confidence(self, entry_signal: bool, exit_signal: bool, signals: Dict, strategy: str, df: pd.DataFrame) -> int:
        """
        Calculate dynamic confidence score based on indicator alignment and market conditions.
        This ensures HOLD confidence is truly dynamic and reflects current market state.
        
        Returns: Confidence score 0-100
        """
        confidence = 50  # Base confidence
        
        try:
            current = df.iloc[-1] if len(df) > 0 else None
            if current is None:
                return confidence
            
            # ENTRY SIGNAL CONFIDENCE (75-95%)
            if entry_signal:
                confidence = 75
                
                # Boost for strong RSI conditions
                if 'rsi' in signals:
                    rsi = signals['rsi']
                    if rsi < 20:  # Very oversold
                        confidence += 8
                    elif rsi < 30:  # Oversold
                        confidence += 5
                
                # Boost for volume confirmation
                if 'volume_ratio' in signals and signals['volume_ratio'] > 2.0:
                    confidence += 8
                elif 'volume_spike' in signals and signals['volume_spike']:
                    confidence += 5
                
                # Boost for EMA alignment
                if 'ema_aligned' in signals and signals['ema_aligned']:
                    confidence += 5
                
                # Boost for MACD alignment
                if 'macd' in signals and 'macdsignal' in signals:
                    if signals['macd'] > signals['macdsignal']:
                        confidence += 3
                
                confidence = min(confidence, 95)
            
            # EXIT SIGNAL CONFIDENCE (55-75%)
            elif exit_signal:
                confidence = 60
                
                # Boost for strong overbought conditions
                if 'rsi' in signals and signals['rsi'] > 70:
                    confidence += 8
                elif 'rsi' in signals and signals['rsi'] > 65:
                    confidence += 5
                
                # Boost for Fisher RSI extreme
                if 'fisher_rsi' in signals and signals['fisher_rsi'] > 0.5:
                    confidence += 5
                
                confidence = min(confidence, 75)
            
            # HOLD SIGNAL CONFIDENCE (25-55%)
            else:
                confidence = 40
                
                # Adjust based on RSI position
                if 'rsi' in signals:
                    rsi = signals['rsi']
                    if 40 < rsi < 60:
                        confidence += 10  # Neutral zone is good for HOLD
                    elif 30 < rsi < 70:
                        confidence += 5   # Tradeable zone
                    elif rsi < 30 or rsi > 70:
                        confidence -= 10  # Extreme conditions = lower hold confidence
                
                # Adjust based on trend alignment
                if 'ema_aligned' in signals:
                    if signals['ema_aligned']:
                        confidence += 5
                    else:
                        confidence -= 5
                
                # Adjust based on volume
                if 'volume_ratio' in signals:
                    vol_ratio = signals['volume_ratio']
                    if 0.8 < vol_ratio < 1.5:
                        confidence += 3  # Normal volume = good for hold
                    elif vol_ratio > 2.0:
                        confidence -= 5  # Spike = less likely to hold
                
                # Adjust based on MACD
                if 'macd' in signals and 'macdsignal' in signals:
                    macd_diff = signals['macd'] - signals['macdsignal']
                    if abs(macd_diff) < 0.001:  # Converging = uncertain
                        confidence += 5
                
                confidence = max(25, min(55, confidence))  # Clamp between 25-55 for HOLD
            
            return int(confidence)
        
        except Exception as e:
            logger.error(f"Error calculating dynamic confidence: {e}", exc_info=True)
            return confidence
    
    def bulk_analyze(self, symbols: List[str], strategy: str = 'ema_crossover', interval: str = '5') -> List[Dict]:
        """
        Analyze multiple crypto pairs with the same strategy
        
        Args:
            symbols: List of crypto pairs
            strategy: Strategy ID
            interval: Timeframe in minutes
        
        Returns:
            List of analysis results, sorted by confidence
        """
        results = []
        
        # Ensure symbols is a list, not a string
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        
        logger.info(f"Bulk analyzing {len(symbols)} symbols: {symbols}")
        
        for symbol in symbols:
            # Normalize symbol format
            symbol = str(symbol).strip().upper()
            
            # Ensure forward slash format
            if '/' not in symbol and len(symbol) > 3:
                # Try to split common formats like BTCUSD -> BTC/USD
                if symbol.endswith('USD'):
                    symbol = symbol[:-3] + '/USD'
                elif symbol.endswith('USDT'):
                    symbol = symbol[:-4] + '/USDT'
            
            logger.info(f"Analyzing {symbol} with {strategy}...")
            
            try:
                result = self.analyze_crypto(symbol, strategy, interval)
                if 'error' not in result:
                    results.append(result)
                else:
                    logger.warning(f"Skipping {symbol}: {result.get('error')}")
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
                continue
        
        # Sort by entry signals first, then confidence
        results.sort(key=lambda x: (not x['entry_signal'], -x['confidence_score']))
        
        logger.info(f"Bulk analysis complete: {len(results)} successful results")
        return results

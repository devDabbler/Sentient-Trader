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
            'orb_fvg': {
                'name': 'ORB+FVG (15min)',
                'timeframe': '1m',  # 1-minute bars for FVG detection
                'minimal_roi': {"30": 0.01, "15": 0.02, "0": 0.03},
                'stoploss': -0.02,  # Dynamic stop based on FVG
                'description': 'Opening Range Breakout with Fair Value Gap confirmation'
            },
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
            pass  # logger.debug(f"DataFrame columns: {df.columns.tolist(}")
            pass  # logger.debug(f"First row dtypes: {df.dtypes.to_dict(}")
            
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
                logger.error("Error calculating Heikin Ashi: {}", str(ha_error), exc_info=True)
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
            logger.error("Error calculating indicators: {}", str(e), exc_info=True)
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
        
        # Near-entry condition
        near_entry = (
            not entry and
            current['ema20'] > current['ema50'] * 0.995 and  # EMAs converging
            current['ema20'] < current['ema50'] * 1.005 and
            ha_green and
            current['ha_close'] > current['ema20']
        )
        
        # Exit signal
        ema50_cross_up = (previous['ema50'] <= previous['ema100']) and (current['ema50'] > current['ema100'])
        ha_red = current['ha_close'] < current['ha_open']
        exit_signal = ema50_cross_up and (current['ha_close'] < current['ema20']) and ha_red
        
        # Near-exit condition
        near_exit = (
            not exit_signal and
            (ha_red or current['ha_close'] < current['ema20'])
        )
        
        # EMA alignment
        ema_aligned = current['ema20'] > current['ema50'] > current['ema100']
        
        signals = {
            'ema20': current['ema20'],
            'ema50': current['ema50'],
            'ema100': current['ema100'],
            'ha_close': current['ha_close'],
            'ha_type': 'green' if ha_green else 'red',
            'ema_aligned': ema_aligned,
            'near_entry': near_entry,
            'near_exit': near_exit,
            'rsi': current.get('rsi', 50),  # Include RSI for confidence calc
            'volume_ratio': current['volume'] / current['volume_ma'] if current.get('volume_ma', 0) > 0 else 1.0,
            'adx': current.get('adx', 0)
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
        
        # Volume ratio
        volume_ratio = current['volume'] / current['volume_ma'] if current['volume_ma'] > 0 else 1.0
        volume_spike = volume_ratio > 1.5
        
        # Entry signal
        entry = (
            macd_cross and
            volume_spike and
            current['rsi'] < 30 and
            current['fastd'] < 30 and
            current['fisher_rsi_norma'] < 30
        )
        
        # Near-entry condition
        near_entry = (
            not entry and
            current['macd'] > current['macdsignal'] * 0.95 and  # MACD converging
            volume_ratio > 1.2 and
            current['rsi'] < 35 and
            current['fastd'] < 35
        )
        
        # Exit signal
        exit_signal = (
            current['rsi'] > 70 and
            current['minus_di'] > 50 and
            current['fisher_rsi_norma'] > 50
        )
        
        # Near-exit condition
        near_exit = (
            not exit_signal and
            (current['rsi'] > 65 or current['fisher_rsi_norma'] > 45)
        )
        
        # EMA alignment for trend strength
        ema_aligned = current.get('ema5', 0) > current.get('ema10', 0) > current.get('ema20', 0)
        
        signals = {
            'macd': current['macd'],
            'macdsignal': current['macdsignal'],
            'volume_ratio': volume_ratio,
            'volume_spike': volume_spike,
            'rsi': current['rsi'],
            'fastd': current['fastd'],
            'fisher_rsi_norma': current['fisher_rsi_norma'],
            'minus_di': current['minus_di'],
            'ema_aligned': ema_aligned,
            'near_entry': near_entry,
            'near_exit': near_exit,
            'adx': current.get('adx', 0)
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
        
        # Volume spike and volume ratio for confidence calc
        volume_spike = current['volume'] > current['volume_ma'] * 2
        volume_ratio = current['volume'] / current['volume_ma'] if current['volume_ma'] > 0 else 1.0
        
        # EMA alignment for trend strength
        ema_aligned = (current['ema5'] > current['ema10'] > current['ema20'])
        
        # Calculate MACD if available for additional confirmation
        macd = current.get('macd', 0)
        macdsignal = current.get('macdsignal', 0)
        
        # Entry signal (strict conditions)
        entry = (
            ema_cross and
            current['rsi'] < 35 and
            volume_spike and
            current['adx'] > 20 and
            current['close'] > current['ema20']
        )
        
        # Near-entry condition (almost ready to buy - helps with confidence scoring)
        near_entry = (
            not entry and  # Not already an entry signal
            current['rsi'] < 40 and  # Oversold but not extreme
            current['adx'] > 15 and  # Some trend strength
            current['ema5'] > current['ema10'] * 0.995 and  # EMAs converging for possible cross
            volume_ratio > 1.2  # Some volume increase
        )
        
        # Exit signal (quick profit or reversal)
        exit_signal = (
            current['rsi'] > 65 or
            current['ema5'] < current['ema10'] or
            current['close'] < current['ema10']
        )
        
        # Near-exit condition (approaching exit)
        near_exit = (
            not exit_signal and  # Not already an exit signal
            (current['rsi'] > 60 or  # Getting overbought
             current['ema5'] < current['ema10'] * 1.005)  # EMAs converging for possible cross down
        )
        
        signals = {
            'ema5': current['ema5'],
            'ema10': current['ema10'],
            'ema20': current['ema20'],
            'rsi': current['rsi'],
            'adx': current['adx'],
            'volume_spike': volume_spike,
            'volume_ratio': volume_ratio,
            'ema_aligned': ema_aligned,
            'macd': macd,
            'macdsignal': macdsignal,
            'near_entry': near_entry,
            'near_exit': near_exit,
            'ema_cross_strength': abs(current['ema5'] - current['ema10']) / current['ema10'] * 100  # % difference
        }
        
        return entry, exit_signal, signals
    
    def strategy_orb_fvg(self, df: pd.DataFrame) -> Tuple[bool, bool, Dict]:
        """
        Opening Range Breakout + Fair Value Gap Strategy
        
        Detects:
        - Opening Range High/Low from first 15 minutes
        - Fair Value Gaps in price action
        - Breakout confirmation with FVG alignment
        
        Returns: (entry_signal, exit_signal, signals_dict)
        """
        try:
            from analyzers.orb_fvg_strategy import ORBFVGAnalyzer
            
            if len(df) < 20:
                logger.warning("Insufficient data for ORB+FVG analysis")
                return False, False, {}
            
            # Initialize ORB+FVG analyzer
            orb_analyzer = ORBFVGAnalyzer()
            
            # Get current price
            current_price = float(df.iloc[-1]['close'])
            
            # Run ORB+FVG analysis (pass empty ticker since we already have the data)
            orb_results = orb_analyzer.analyze("CRYPTO", df, current_price)
            
            # Convert ORB+FVG results to entry/exit signals
            entry_signal = orb_results['signal'] == 'BUY'
            exit_signal = orb_results['signal'] == 'SELL'
            
            # Build signals dict with ORB+FVG-specific data
            signals = {
                'orb_signal': orb_results['signal'],
                'confidence': orb_results['confidence'],
                'fvg_signal': orb_results.get('fvg_signal', 'NEUTRAL'),
                'fvg_strength': orb_results.get('fvg_strength', 0),
                'risk_level': orb_results.get('risk_level', 'MEDIUM'),
                'entry_price': orb_results.get('entry', current_price),
                'stop_loss': orb_results.get('stop_loss', current_price * 0.98),
                'target': orb_results.get('target', current_price * 1.02),
                'risk_reward_ratio': orb_results.get('risk_reward_ratio', 1.0)
            }
            
            # Add opening range data if available
            if orb_results.get('opening_range'):
                orb = orb_results['opening_range']
                signals['orh'] = orb.get('orh', 0)
                signals['orl'] = orb.get('orl', 0)
                signals['range_pct'] = orb.get('range_pct', 0)
            
            # Add key signals if available
            if orb_results.get('key_signals'):
                signals['key_signals'] = orb_results['key_signals'][:5]
            
            # Add recommendations if available
            if orb_results.get('recommendations'):
                signals['recommendations'] = orb_results['recommendations'][:5]
            
            return entry_signal, exit_signal, signals
            
        except Exception as e:
            logger.error("Error in ORB+FVG strategy: {}", str(e), exc_info=True)
            return False, False, {'error': str(e)}
    
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
                pass  # logger.error(f"Missing columns in OHLC data: {missing_cols}. Available: {ohlcv.columns.tolist(}")
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
            
            logger.info("{} {strategy}: {result['recommendation']} (confidence: {confidence}%)", str(symbol))
            return result
            
        except Exception as e:
            logger.error("Error analyzing {symbol} with {strategy}: {}", str(e), exc_info=True)
            return {'error': str(e)}
    
    def _calculate_dynamic_confidence(self, entry_signal: bool, exit_signal: bool, signals: Dict, strategy: str, df: pd.DataFrame) -> int:
        """
        Calculate dynamic confidence score based on indicator alignment and market conditions.
        This ensures HOLD confidence is truly dynamic and reflects current market state.
        
        Returns: Confidence score 0-100
        """
        confidence = 50  # Base confidence
        confidence_factors = []  # Track what drives the confidence score
        
        try:
            current = df.iloc[-1] if len(df) > 0 else None
            if current is None:
                return confidence
            
            # ENTRY SIGNAL CONFIDENCE (70-95%)
            if entry_signal:
                confidence = 70
                confidence_factors.append("Entry signal detected (+70)")
                
                # Boost for strong RSI conditions
                if 'rsi' in signals:
                    rsi = signals['rsi']
                    if rsi < 20:  # Very oversold
                        confidence += 10
                        confidence_factors.append(f"Very oversold RSI {rsi:.1f} (+10)")
                    elif rsi < 30:  # Oversold
                        confidence += 7
                        confidence_factors.append(f"Oversold RSI {rsi:.1f} (+7)")
                    elif rsi < 35:
                        confidence += 4
                        confidence_factors.append(f"Low RSI {rsi:.1f} (+4)")
                
                # Boost for volume confirmation
                if 'volume_ratio' in signals:
                    vol_ratio = signals['volume_ratio']
                    if vol_ratio > 3.0:
                        confidence += 10
                        confidence_factors.append(f"Huge volume {vol_ratio:.1f}x (+10)")
                    elif vol_ratio > 2.0:
                        confidence += 7
                        confidence_factors.append(f"High volume {vol_ratio:.1f}x (+7)")
                    elif vol_ratio > 1.5:
                        confidence += 4
                        confidence_factors.append(f"Elevated volume {vol_ratio:.1f}x (+4)")
                
                # Boost for EMA alignment
                if 'ema_aligned' in signals and signals['ema_aligned']:
                    confidence += 6
                    confidence_factors.append("EMA trend aligned (+6)")
                
                # Boost for MACD alignment
                if 'macd' in signals and 'macdsignal' in signals:
                    macd_diff = signals['macd'] - signals['macdsignal']
                    if macd_diff > 0:
                        confidence += 4
                        confidence_factors.append(f"MACD bullish (+4)")
                
                # Boost for strong ADX
                if 'adx' in signals:
                    adx = signals['adx']
                    if adx > 30:
                        confidence += 5
                        confidence_factors.append(f"Strong trend ADX {adx:.1f} (+5)")
                    elif adx > 25:
                        confidence += 3
                        confidence_factors.append(f"Decent trend ADX {adx:.1f} (+3)")
                
                confidence = min(confidence, 95)
            
            # EXIT SIGNAL CONFIDENCE (60-80%)
            elif exit_signal:
                confidence = 60
                confidence_factors.append("Exit signal detected (+60)")
                
                # Boost for strong overbought conditions
                if 'rsi' in signals:
                    rsi = signals['rsi']
                    if rsi > 80:
                        confidence += 12
                        confidence_factors.append(f"Extreme overbought RSI {rsi:.1f} (+12)")
                    elif rsi > 70:
                        confidence += 8
                        confidence_factors.append(f"Overbought RSI {rsi:.1f} (+8)")
                    elif rsi > 65:
                        confidence += 5
                        confidence_factors.append(f"High RSI {rsi:.1f} (+5)")
                
                # Boost for Fisher RSI extreme
                if 'fisher_rsi' in signals:
                    fisher = signals['fisher_rsi']
                    if fisher > 0.7:
                        confidence += 7
                        confidence_factors.append(f"Extreme Fisher {fisher:.2f} (+7)")
                    elif fisher > 0.5:
                        confidence += 4
                        confidence_factors.append(f"High Fisher {fisher:.2f} (+4)")
                
                # MACD bearish confirmation
                if 'macd' in signals and 'macdsignal' in signals:
                    macd_diff = signals['macd'] - signals['macdsignal']
                    if macd_diff < 0:
                        confidence += 5
                        confidence_factors.append("MACD bearish (+5)")
                
                confidence = min(confidence, 80)
            
            # HOLD SIGNAL CONFIDENCE (30-70%)
            else:
                confidence = 45  # Start at 45 for neutral
                confidence_factors.append("No clear signal (base 45)")
                
                # Check for near-entry conditions (potential upcoming buy)
                if 'near_entry' in signals and signals['near_entry']:
                    confidence += 12
                    confidence_factors.append("Near BUY signal (+12)")
                    
                # Check for near-exit conditions (potential upcoming sell)
                if 'near_exit' in signals and signals['near_exit']:
                    confidence -= 8
                    confidence_factors.append("Near SELL signal (-8)")
                
                # Adjust based on RSI position
                if 'rsi' in signals:
                    rsi = signals['rsi']
                    if 45 < rsi < 55:
                        confidence += 8
                        confidence_factors.append(f"Perfect neutral RSI {rsi:.1f} (+8)")
                    elif 40 < rsi < 60:
                        confidence += 5
                        confidence_factors.append(f"Good neutral RSI {rsi:.1f} (+5)")
                    elif 35 < rsi < 40:
                        confidence += 6
                        confidence_factors.append(f"Slight oversold RSI {rsi:.1f} (+6)")
                    elif 30 < rsi < 35:
                        confidence += 8
                        confidence_factors.append(f"Oversold building RSI {rsi:.1f} (+8)")
                    elif rsi < 30:
                        confidence += 10
                        confidence_factors.append(f"Very oversold RSI {rsi:.1f} (+10)")
                    elif 60 < rsi < 65:
                        confidence += 3
                        confidence_factors.append(f"Slightly high RSI {rsi:.1f} (+3)")
                    elif 65 < rsi < 70:
                        confidence -= 3
                        confidence_factors.append(f"Getting hot RSI {rsi:.1f} (-3)")
                    elif rsi > 70:
                        confidence -= 8
                        confidence_factors.append(f"Overbought RSI {rsi:.1f} (-8)")
                
                # Adjust based on trend alignment
                if 'ema_aligned' in signals:
                    if signals['ema_aligned']:
                        confidence += 7
                        confidence_factors.append("Bullish EMA trend (+7)")
                    else:
                        confidence -= 4
                        confidence_factors.append("No EMA trend (-4)")
                
                # Adjust based on volume
                if 'volume_ratio' in signals:
                    vol_ratio = signals['volume_ratio']
                    if 0.9 < vol_ratio < 1.2:
                        confidence += 4
                        confidence_factors.append(f"Stable volume {vol_ratio:.1f}x (+4)")
                    elif 0.8 < vol_ratio < 1.5:
                        confidence += 2
                        confidence_factors.append(f"Normal volume {vol_ratio:.1f}x (+2)")
                    elif vol_ratio > 2.5:
                        confidence -= 6
                        confidence_factors.append(f"Spike warning {vol_ratio:.1f}x (-6)")
                    elif vol_ratio > 1.8:
                        confidence -= 3
                        confidence_factors.append(f"Rising volume {vol_ratio:.1f}x (-3)")
                    elif vol_ratio < 0.5:
                        confidence -= 5
                        confidence_factors.append(f"Very low volume {vol_ratio:.1f}x (-5)")
                
                # Adjust based on MACD trend
                if 'macd' in signals and 'macdsignal' in signals:
                    macd_diff = signals['macd'] - signals['macdsignal']
                    if macd_diff > 0.002:  # Strong bullish
                        confidence += 6
                        confidence_factors.append(f"Strong MACD bullish (+6)")
                    elif macd_diff > 0:
                        confidence += 3
                        confidence_factors.append(f"MACD bullish (+3)")
                    elif macd_diff < -0.002:  # Strong bearish
                        confidence -= 6
                        confidence_factors.append(f"Strong MACD bearish (-6)")
                    elif macd_diff < 0:
                        confidence -= 3
                        confidence_factors.append(f"MACD bearish (-3)")
                    else:
                        confidence += 2
                        confidence_factors.append("MACD neutral (+2)")
                
                # Adjust based on ADX (trend strength)
                if 'adx' in signals:
                    adx = signals['adx']
                    if 25 < adx < 40:
                        confidence += 6
                        confidence_factors.append(f"Good trend ADX {adx:.1f} (+6)")
                    elif 20 < adx < 25:
                        confidence += 3
                        confidence_factors.append(f"Building trend ADX {adx:.1f} (+3)")
                    elif adx < 15:
                        confidence -= 4
                        confidence_factors.append(f"Weak trend ADX {adx:.1f} (-4)")
                    elif adx > 45:
                        confidence -= 2
                        confidence_factors.append(f"Overextended ADX {adx:.1f} (-2)")
                
                # Clamp between 30-70 for HOLD
                confidence = max(30, min(70, confidence))
            
            # Store confidence factors in signals for UI display
            signals['confidence_factors'] = confidence_factors
            
            return int(confidence)
        
        except Exception as e:
            logger.error("Error calculating dynamic confidence: {}", str(e), exc_info=True)
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
                    logger.warning("Skipping {}: {result.get('error')}", str(symbol))
            except Exception as e:
                logger.error("Error analyzing {symbol}: {}", str(e), exc_info=True)
                continue
        
        # Sort by entry signals first, then confidence
        results.sort(key=lambda x: (not x['entry_signal'], -x['confidence_score']))
        
        logger.info(f"Bulk analysis complete: {len(results)} successful results")
        return results

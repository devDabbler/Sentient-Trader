"""
SentientStrategy - Basic Freqtrade strategy for Sentient Trader
Wraps existing technical indicators from analyzers/ for use with Freqtrade
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.strategy.parameters import IntParameter, DecimalParameter, BooleanParameter
import talib.abstract as ta
from technical import qtpylib


class SentientStrategy(IStrategy):
    """
    Sentient Trader Strategy for Freqtrade
    
    Uses EMA crossovers, RSI, MACD, and volume for entry/exit signals.
    Designed for 5m timeframe on crypto pairs via Kraken.
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.10,    # 10% profit target
        "30": 0.05,   # 5% after 30 minutes
        "60": 0.03,   # 3% after 1 hour
        "120": 0.01   # 1% after 2 hours
    }
    
    # Stoploss
    stoploss = -0.05  # 5% stoploss
    
    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '5m'
    
    # Run on every new candle
    process_only_new_candles = True
    
    # Use custom stoploss
    use_custom_stoploss = False
    
    # Number of candles for startup
    startup_candle_count: int = 100
    
    # Order types
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }
    
    # Hyperparameters for optimization
    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    buy_rsi_high = IntParameter(50, 70, default=60, space='buy')
    sell_rsi = IntParameter(60, 85, default=70, space='sell')
    
    ema_fast = IntParameter(8, 20, default=12, space='buy')
    ema_slow = IntParameter(20, 50, default=26, space='buy')
    ema_trend = IntParameter(50, 200, default=100, space='buy')
    
    volume_factor = DecimalParameter(1.0, 3.0, default=1.5, space='buy')
    
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations.
        Used for correlation analysis with BTC/ETH.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        
        # Add BTC and ETH as informative for correlation
        for pair in ['BTC/USD', 'ETH/USD']:
            if pair not in pairs:
                informative_pairs.append((pair, self.timeframe))
                informative_pairs.append((pair, '1h'))  # Higher timeframe
        
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate all indicators for the strategy.
        
        Uses technical indicators similar to those in analyzers/technical.py
        """
        # EMA indicators
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=self.ema_trend.value)
        
        # EMA 20/50/100 for trend alignment
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # ATR for volatility
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Volume indicators
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # MFI - Money Flow Index
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Heikin Ashi for trend confirmation
        dataframe['ha_close'] = (dataframe['open'] + dataframe['high'] + 
                                  dataframe['low'] + dataframe['close']) / 4
        dataframe['ha_open'] = (dataframe['open'].shift(1) + dataframe['close'].shift(1)) / 2
        dataframe.loc[dataframe['ha_open'].isna(), 'ha_open'] = dataframe['open']
        dataframe['ha_high'] = dataframe[['high', 'ha_open', 'ha_close']].max(axis=1)
        dataframe['ha_low'] = dataframe[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Heikin Ashi candle type
        dataframe['ha_bullish'] = dataframe['ha_close'] > dataframe['ha_open']
        dataframe['ha_bearish'] = dataframe['ha_close'] < dataframe['ha_open']
        
        # Trend alignment (EMA 20 > 50 > 100 = bullish)
        dataframe['trend_bullish'] = (
            (dataframe['ema_20'] > dataframe['ema_50']) & 
            (dataframe['ema_50'] > dataframe['ema_100'])
        )
        dataframe['trend_bearish'] = (
            (dataframe['ema_20'] < dataframe['ema_50']) & 
            (dataframe['ema_50'] < dataframe['ema_100'])
        )
        
        # Price momentum
        dataframe['momentum'] = dataframe['close'].pct_change(periods=10) * 100
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate entry signals.
        
        Entry conditions:
        1. EMA fast crosses above EMA slow (bullish crossover)
        2. Price above trend EMA
        3. RSI not overbought (< 60)
        4. Volume above average
        5. Trend aligned (EMA 20 > 50 > 100)
        6. Heikin Ashi bullish
        """
        dataframe.loc[
            (
                # EMA crossover
                (qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])) &
                
                # Price above trend EMA
                (dataframe['close'] > dataframe['ema_trend']) &
                
                # RSI conditions
                (dataframe['rsi'] > self.buy_rsi.value) &
                (dataframe['rsi'] < self.buy_rsi_high.value) &
                
                # Volume confirmation
                (dataframe['volume_ratio'] > self.volume_factor.value) &
                
                # Trend alignment
                (dataframe['trend_bullish']) &
                
                # MACD positive momentum
                (dataframe['macdhist'] > 0) &
                
                # Heikin Ashi bullish
                (dataframe['ha_bullish']) &
                
                # ADX showing trend
                (dataframe['adx'] > 20) &
                
                # Volume check
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate exit signals.
        
        Exit conditions:
        1. RSI overbought
        2. EMA fast crosses below EMA slow
        3. MACD histogram turns negative
        4. Heikin Ashi bearish
        """
        dataframe.loc[
            (
                # RSI overbought
                (dataframe['rsi'] > self.sell_rsi.value) |
                
                # EMA bearish crossover
                (qtpylib.crossed_below(dataframe['ema_fast'], dataframe['ema_slow'])) |
                
                # MACD turns negative with momentum loss
                (
                    (dataframe['macdhist'] < 0) & 
                    (dataframe['macdhist'].shift(1) > 0)
                ) |
                
                # Price breaks below trend
                (
                    (dataframe['close'] < dataframe['ema_trend']) &
                    (dataframe['ha_bearish'])
                )
            ) &
            (dataframe['volume'] > 0),
            'exit_long'] = 1
        
        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
        Customize stake amount based on volatility and confidence.
        
        Uses ATR-based position sizing similar to Sentient Trader risk management.
        """
        # Get the dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            
            # Calculate volatility factor (lower stake for high volatility)
            atr_pct = (last_candle['atr'] / current_rate) * 100
            
            # Adjust stake based on volatility
            if atr_pct > 3:  # High volatility
                stake_multiplier = 0.5
            elif atr_pct > 2:  # Medium volatility
                stake_multiplier = 0.75
            else:  # Low volatility
                stake_multiplier = 1.0
            
            # Adjust based on trend strength
            if last_candle['adx'] > 30:  # Strong trend
                stake_multiplier *= 1.2
            elif last_candle['adx'] < 20:  # Weak trend
                stake_multiplier *= 0.8
            
            adjusted_stake = proposed_stake * stake_multiplier
            
            # Ensure within limits
            if min_stake:
                adjusted_stake = max(adjusted_stake, min_stake)
            adjusted_stake = min(adjusted_stake, max_stake)
            
            return adjusted_stake
        
        return proposed_stake

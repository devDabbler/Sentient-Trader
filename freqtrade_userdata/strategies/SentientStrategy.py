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
    
    # Minimal ROI - slightly higher targets for better R:R
    minimal_roi = {
        "0": 0.05,     # 5% immediate (rare but take it)
        "30": 0.03,    # 3% after 30 min
        "60": 0.025,   # 2.5% after 1 hour
        "120": 0.02,   # 2% after 2 hours
        "240": 0.015,  # 1.5% after 4 hours
        "360": 0.01    # 1% after 6 hours (take small profit)
    }
    
    # Stoploss - give trades room to work
    stoploss = -0.025  # 2.5% stoploss (custom_stoploss tightens over time)
    
    # Trailing stoploss - DISABLED (was causing losses)
    # The custom_stoploss handles profit protection better
    trailing_stop = False
    
    # Timeframe
    timeframe = '5m'
    
    # Run on every new candle
    process_only_new_candles = True
    
    # Enable custom stoploss for tighter control
    use_custom_stoploss = True
    
    # Number of candles for startup
    startup_candle_count: int = 100
    
    # Order types - use exchange stoploss for tighter risk control
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60  # Check every 60 seconds
    }
    
    # Hyperparameters for optimization - BALANCED for quality trades
    buy_rsi = IntParameter(25, 45, default=35, space='buy')  # Moderate - not too oversold
    buy_rsi_high = IntParameter(55, 70, default=65, space='buy')  # Cap before overbought
    sell_rsi = IntParameter(65, 80, default=72, space='sell')  # Exit at overbought
    
    ema_fast = IntParameter(8, 15, default=10, space='buy')  # Balanced fast EMA
    ema_slow = IntParameter(18, 30, default=21, space='buy')  # Balanced slow EMA
    ema_trend = IntParameter(40, 100, default=55, space='buy')  # Medium trend EMA
    
    volume_factor = DecimalParameter(1.0, 2.0, default=1.2, space='buy')  # Require some volume confirmation
    
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
        Simple entry: EMA crossover in uptrend with confirmation.
        Only ONE entry type to avoid overtrading.
        """
        # High-quality entry: EMA cross + ADX trend filter + momentum
        dataframe.loc[
            (
                # EMA golden cross (fresh crossover only)
                (qtpylib.crossed_above(dataframe['ema_fast'], dataframe['ema_slow'])) &
                
                # Price above longer-term trend
                (dataframe['close'] > dataframe['ema_trend']) &
                
                # ADX shows trending market (avoid chop)
                (dataframe['adx'] > 20) &
                
                # RSI in sweet spot (not oversold, not overbought)
                (dataframe['rsi'] > 40) &
                (dataframe['rsi'] < 65) &
                
                # MACD histogram positive AND rising (strong momentum)
                (dataframe['macdhist'] > 0) &
                (dataframe['macdhist'] > dataframe['macdhist'].shift(1)) &
                
                # Volume spike (interest confirmation)
                (dataframe['volume_ratio'] > 1.2) &
                
                # Volume exists
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        No signal exits - let ROI and stoploss handle all exits.
        Signal exits were cutting winners too early.
        """
        # Intentionally empty - rely on ROI/stoploss/trailing
        dataframe['exit_long'] = 0
        return dataframe
    
    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float,
                        after_fill: bool, **kwargs) -> float:
        """
        Simple stoploss: time-based tightening only.
        Let ROI handle profit-taking, stoploss handles risk.
        """
        # Time-based loss cutting (don't hold losers forever)
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
        
        if trade_duration > 240:  # 4+ hours
            return -0.012  # Cut at 1.2%
        elif trade_duration > 180:  # 3+ hours
            return -0.015  # Cut at 1.5%
        elif trade_duration > 120:  # 2+ hours
            return -0.018  # Tighten to 1.8%
        else:
            return -0.025  # Initial 2.5% - give room to breathe
    
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

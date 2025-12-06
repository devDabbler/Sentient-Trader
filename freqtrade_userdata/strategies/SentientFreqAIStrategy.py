"""
SentientFreqAIStrategy - ML-enhanced Freqtrade strategy using FreqAI
Uses LightGBM for ML predictions combined with traditional indicators
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional
import logging

from freqtrade.strategy import IStrategy
from freqtrade.strategy.parameters import IntParameter, DecimalParameter
import talib.abstract as ta
from technical import qtpylib

# FreqAI imports
from freqtrade.freqai.prediction_models.LightGBMRegressor import LightGBMRegressor

logger = logging.getLogger(__name__)


class SentientFreqAIStrategy(IStrategy):
    """
    FreqAI-enhanced Strategy for Sentient Trader
    
    Uses machine learning (LightGBM) to predict price movements
    combined with traditional technical indicators for confirmation.
    
    The ML model predicts future price change, which is used to
    generate high-confidence trading signals.
    """
    
    # Strategy interface version
    INTERFACE_VERSION = 3
    
    # Minimal ROI - more aggressive with ML
    minimal_roi = {
        "0": 0.15,    # 15% profit target
        "60": 0.08,   # 8% after 1 hour
        "180": 0.04,  # 4% after 3 hours
        "360": 0.02   # 2% after 6 hours
    }
    
    # Stoploss
    stoploss = -0.06  # 6% stoploss
    
    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True
    
    # Timeframe
    timeframe = '5m'
    
    # FreqAI configuration
    can_short = False  # Spot trading only
    process_only_new_candles = True
    
    # Number of candles for startup
    startup_candle_count: int = 150
    
    # Hyperparameters
    entry_threshold = DecimalParameter(0.01, 0.05, default=0.02, space='buy')
    exit_threshold = DecimalParameter(-0.03, -0.01, default=-0.015, space='sell')
    
    # Technical confirmation thresholds
    rsi_buy_threshold = IntParameter(25, 45, default=35, space='buy')
    rsi_sell_threshold = IntParameter(65, 85, default=75, space='sell')
    
    # Volume filter
    volume_multiplier = DecimalParameter(1.0, 2.5, default=1.3, space='buy')
    
    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Expand features for all timeframes.
        Called for each informative timeframe.
        
        Creates lagged features and rolling statistics.
        """
        # Basic price features
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        dataframe["%-log_return"] = np.log(dataframe["close"] / dataframe["close"].shift(1))
        
        # Raw OHLCV features
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]
        
        # Price position relative to range
        dataframe["%-close_position"] = (
            (dataframe["close"] - dataframe["low"]) / 
            (dataframe["high"] - dataframe["low"] + 1e-10)
        )
        
        # Volatility features
        dataframe["%-volatility"] = dataframe["close"].rolling(period).std()
        dataframe["%-atr"] = ta.ATR(dataframe, timeperiod=period)
        
        # Momentum features
        dataframe["%-rsi"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-momentum"] = ta.MOM(dataframe, timeperiod=period)
        dataframe["%-roc"] = ta.ROC(dataframe, timeperiod=period)
        
        # MACD features
        macd = ta.MACD(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macdsignal"] = macd["macdsignal"]
        dataframe["%-macdhist"] = macd["macdhist"]
        
        # Bollinger Band features
        bollinger = qtpylib.bollinger_bands(dataframe["close"], window=period, stds=2)
        dataframe["%-bb_width"] = (bollinger["upper"] - bollinger["lower"]) / bollinger["mid"]
        dataframe["%-bb_position"] = (dataframe["close"] - bollinger["lower"]) / (bollinger["upper"] - bollinger["lower"] + 1e-10)
        
        # Volume features
        dataframe["%-volume_sma"] = ta.SMA(dataframe["volume"], timeperiod=period)
        dataframe["%-volume_ratio"] = dataframe["volume"] / (dataframe["%-volume_sma"] + 1e-10)
        
        # Trend features
        for ema_period in [10, 20, 50]:
            dataframe[f"%-ema_{ema_period}"] = ta.EMA(dataframe, timeperiod=ema_period)
            dataframe[f"%-close_ema_{ema_period}_dist"] = (
                (dataframe["close"] - dataframe[f"%-ema_{ema_period}"]) / 
                dataframe["close"] * 100
            )
        
        # ADX for trend strength
        dataframe["%-adx"] = ta.ADX(dataframe, timeperiod=period)
        
        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=period)
        dataframe["%-slowk"] = stoch["slowk"]
        dataframe["%-slowd"] = stoch["slowd"]
        
        # MFI for volume-weighted momentum
        dataframe["%-mfi"] = ta.MFI(dataframe, timeperiod=period)
        
        return dataframe
    
    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Basic feature engineering called once per pair.
        Creates features that don't need period parameter.
        """
        # Candle patterns
        dataframe["%-doji"] = ta.CDLDOJI(dataframe)
        dataframe["%-hammer"] = ta.CDLHAMMER(dataframe)
        dataframe["%-engulfing"] = ta.CDLENGULFING(dataframe)
        dataframe["%-morning_star"] = ta.CDLMORNINGSTAR(dataframe)
        dataframe["%-evening_star"] = ta.CDLEVENINGSTAR(dataframe)
        
        # Heikin Ashi
        dataframe["%-ha_close"] = (
            dataframe["open"] + dataframe["high"] + 
            dataframe["low"] + dataframe["close"]
        ) / 4
        
        dataframe["%-ha_open"] = (
            dataframe["open"].shift(1) + dataframe["close"].shift(1)
        ) / 2
        
        dataframe["%-ha_bullish"] = (dataframe["%-ha_close"] > dataframe["%-ha_open"]).astype(int)
        
        # Higher timeframe trend (using shifted data as proxy)
        dataframe["%-trend_5"] = (dataframe["close"] > dataframe["close"].shift(5)).astype(int)
        dataframe["%-trend_10"] = (dataframe["close"] > dataframe["close"].shift(10)).astype(int)
        dataframe["%-trend_20"] = (dataframe["close"] > dataframe["close"].shift(20)).astype(int)
        
        return dataframe
    
    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Set the target for FreqAI training.
        
        Target: Future price change over next N candles (label_period_candles)
        """
        # Target: percentage change over next 24 candles (2 hours at 5m)
        dataframe["&-target"] = (
            dataframe["close"].shift(-self.freqai_info["feature_parameters"]["label_period_candles"]) / 
            dataframe["close"] - 1
        )
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populate indicators including FreqAI predictions.
        """
        # Start FreqAI
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # Add traditional indicators for confirmation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        
        # Volume
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # Trend alignment
        dataframe['trend_aligned'] = (
            (dataframe['ema_20'] > dataframe['ema_50']) &
            (dataframe['ema_50'] > dataframe['ema_100'])
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals based on FreqAI predictions + technical confirmation.
        
        Entry when:
        1. FreqAI predicts positive return above threshold
        2. Model confidence (do_predict) is good
        3. Technical indicators confirm (RSI, trend, volume)
        """
        dataframe.loc[
            (
                # FreqAI prediction is bullish
                (dataframe["&-target_mean"] > self.entry_threshold.value) &
                
                # Model confidence is good
                (dataframe["do_predict"] == 1) &
                
                # RSI not overbought
                (dataframe["rsi"] < 65) &
                (dataframe["rsi"] > self.rsi_buy_threshold.value) &
                
                # Trend alignment
                (dataframe["trend_aligned"]) &
                
                # Volume confirmation
                (dataframe["volume_ratio"] > self.volume_multiplier.value) &
                
                # MACD positive
                (dataframe["macdhist"] > 0) &
                
                # ADX showing trend
                (dataframe["adx"] > 20) &
                
                # Volume check
                (dataframe["volume"] > 0)
            ),
            'enter_long'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals based on FreqAI predictions + technical confirmation.
        
        Exit when:
        1. FreqAI predicts negative return below threshold
        2. Or technical indicators show reversal
        """
        dataframe.loc[
            (
                # FreqAI predicts decline
                (dataframe["&-target_mean"] < self.exit_threshold.value) |
                
                # RSI overbought
                (dataframe["rsi"] > self.rsi_sell_threshold.value) |
                
                # Trend breakdown
                (
                    (dataframe["ema_20"] < dataframe["ema_50"]) &
                    (dataframe["macdhist"] < 0)
                )
            ) &
            (dataframe["volume"] > 0),
            'exit_long'] = 1
        
        return dataframe
    
    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: Optional[str],
        side: str, **kwargs
    ) -> bool:
        """
        Additional entry confirmation using FreqAI confidence.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            
            # Check model confidence
            if "do_predict" in last_candle and last_candle["do_predict"] != 1:
                logger.info(f"Rejecting entry for {pair}: Low model confidence")
                return False
            
            # Check prediction strength
            if "&-target_mean" in last_candle:
                prediction = last_candle["&-target_mean"]
                if prediction < self.entry_threshold.value * 0.5:
                    logger.info(f"Rejecting entry for {pair}: Weak prediction ({prediction:.4f})")
                    return False
        
        return True
    
    def custom_exit(
        self, pair: str, trade, current_time: datetime, current_rate: float,
        current_profit: float, **kwargs
    ) -> Optional[str]:
        """
        Custom exit logic based on FreqAI predictions.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            
            # Exit if FreqAI strongly predicts decline
            if "&-target_mean" in last_candle:
                prediction = last_candle["&-target_mean"]
                
                if prediction < -0.03 and current_profit > 0:
                    return "freqai_bearish_prediction"
                
                # Take profit if prediction reverses while in profit
                if current_profit > 0.02 and prediction < 0:
                    return "freqai_take_profit"
        
        return None
    
    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: Optional[float], max_stake: float,
        leverage: float, entry_tag: Optional[str], side: str, **kwargs
    ) -> float:
        """
        Adjust stake based on FreqAI confidence and prediction strength.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            stake_multiplier = 1.0
            
            # Adjust based on prediction confidence
            if "&-target_mean" in last_candle:
                prediction = abs(last_candle["&-target_mean"])
                
                if prediction > 0.04:  # Strong prediction
                    stake_multiplier = 1.3
                elif prediction > 0.03:  # Moderate prediction
                    stake_multiplier = 1.0
                else:  # Weak prediction
                    stake_multiplier = 0.7
            
            # Reduce stake if model is uncertain
            if "do_predict" in last_candle:
                if last_candle["do_predict"] != 1:
                    stake_multiplier *= 0.5
            
            # ADX adjustment
            if "adx" in last_candle:
                if last_candle["adx"] > 30:  # Strong trend
                    stake_multiplier *= 1.2
                elif last_candle["adx"] < 20:  # Weak trend
                    stake_multiplier *= 0.7
            
            adjusted_stake = proposed_stake * stake_multiplier
            
            # Ensure within limits
            if min_stake:
                adjusted_stake = max(adjusted_stake, min_stake)
            adjusted_stake = min(adjusted_stake, max_stake)
            
            return adjusted_stake
        
        return proposed_stake

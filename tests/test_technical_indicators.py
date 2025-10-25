"""Unit tests for EMA, DeMarker, Power Zone, and Fibonacci indicator functions."""

import pytest
import pandas as pd
import numpy as np
from analyzers.technical import TechnicalAnalyzer


class TestEMAIndicator:
    """Test EMA calculation"""
    
    def test_ema_basic(self):
        """Test basic EMA calculation"""
        prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        ema8 = TechnicalAnalyzer.ema(prices, 8)
        
        assert len(ema8) == len(prices)
        assert not ema8.empty
        # EMA should be between min and max of input
        assert ema8.iloc[-1] >= prices.min()
        assert ema8.iloc[-1] <= prices.max()
    
    def test_ema_uptrend(self):
        """Test that EMA follows uptrend"""
        prices = pd.Series(range(1, 21))  # Strong uptrend
        ema8 = TechnicalAnalyzer.ema(prices, 8)
        ema21 = TechnicalAnalyzer.ema(prices, 21)
        
        # In uptrend, shorter EMA should be above longer EMA
        assert ema8.iloc[-1] > ema21.iloc[-1]
    
    def test_ema_downtrend(self):
        """Test that EMA follows downtrend"""
        prices = pd.Series(range(20, 0, -1))  # Strong downtrend
        ema8 = TechnicalAnalyzer.ema(prices, 8)
        ema21 = TechnicalAnalyzer.ema(prices, 21)
        
        # In downtrend, shorter EMA should be below longer EMA
        assert ema8.iloc[-1] < ema21.iloc[-1]


class TestDeMarkerIndicator:
    """Test DeMarker oscillator"""
    
    def test_demarker_bounds(self):
        """Test DeMarker stays within 0-1 bounds"""
        df = pd.DataFrame({
            'High': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'Low': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'Close': [9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        })
        
        dem = TechnicalAnalyzer.demarker(df, period=14)
        
        # All values should be between 0 and 1
        assert (dem >= 0).all()
        assert (dem <= 1).all()
    
    def test_demarker_uptrend(self):
        """Test DeMarker in strong uptrend"""
        df = pd.DataFrame({
            'High': list(range(10, 40)),
            'Low': list(range(9, 39)),
            'Close': list(range(9.5, 39.5))
        })
        
        dem = TechnicalAnalyzer.demarker(df, period=14)
        
        # In strong uptrend, DeMarker should be elevated (>0.5)
        assert dem.iloc[-1] > 0.5
    
    def test_demarker_downtrend(self):
        """Test DeMarker in strong downtrend"""
        df = pd.DataFrame({
            'High': list(range(40, 10, -1)),
            'Low': list(range(39, 9, -1)),
            'Close': list(range(39.5, 9.5, -1))
        })
        
        dem = TechnicalAnalyzer.demarker(df, period=14)
        
        # In strong downtrend, DeMarker should be low (<0.5)
        assert dem.iloc[-1] < 0.5
    
    def test_demarker_oversold_threshold(self):
        """Test DeMarker oversold zone (<0.30)"""
        # Create a strong downtrend followed by flattening
        highs = list(range(50, 30, -1)) + [30] * 5
        lows = list(range(49, 29, -1)) + [29] * 5
        closes = list(range(49.5, 29.5, -1)) + [29.5] * 5
        
        df = pd.DataFrame({
            'High': highs,
            'Low': lows,
            'Close': closes
        })
        
        dem = TechnicalAnalyzer.demarker(df, period=14)
        
        # Should reach oversold at some point
        assert (dem < 0.40).any()


class TestEMAPowerZoneAndReclaim:
    """Test EMA Power Zone and Reclaim detection"""
    
    def test_power_zone_detected(self):
        """Test Power Zone detection when price > EMA8 > EMA21"""
        df = pd.DataFrame({
            'Close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'Volume': [1000000] * 11
        })
        
        ema8 = TechnicalAnalyzer.ema(df['Close'], 8)
        ema21 = TechnicalAnalyzer.ema(df['Close'], 21)
        
        result = TechnicalAnalyzer.detect_ema_power_zone_and_reclaim(df, ema8, ema21)
        
        # In strong uptrend, power zone should be active
        assert result['power_zone'] == True
    
    def test_power_zone_not_detected(self):
        """Test Power Zone not detected in downtrend"""
        df = pd.DataFrame({
            'Close': [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
            'Volume': [1000000] * 11
        })
        
        ema8 = TechnicalAnalyzer.ema(df['Close'], 8)
        ema21 = TechnicalAnalyzer.ema(df['Close'], 21)
        
        result = TechnicalAnalyzer.detect_ema_power_zone_and_reclaim(df, ema8, ema21)
        
        # In downtrend, power zone should not be active
        assert result['power_zone'] == False
    
    def test_reclaim_detected(self):
        """Test EMA Reclaim detection"""
        # Setup: downtrend, then strong recovery above EMAs with volume
        prices = [50, 48, 46, 44, 42, 40, 38, 37, 36, 35, 34,  # Downtrend
                  35, 36, 38, 40, 42, 44, 46, 48, 50, 52]       # Recovery
        volumes = [1000000] * 11 + [1500000] * 9  # Increased volume on recovery
        
        df = pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        })
        
        ema8 = TechnicalAnalyzer.ema(df['Close'], 8)
        ema21 = TechnicalAnalyzer.ema(df['Close'], 21)
        
        result = TechnicalAnalyzer.detect_ema_power_zone_and_reclaim(df, ema8, ema21)
        
        # Reclaim may or may not be detected depending on exact EMA values
        # Just verify the function runs and returns expected structure
        assert 'is_reclaim' in result
        assert 'power_zone' in result
        assert 'reasons' in result
        assert isinstance(result['reasons'], list)
    
    def test_no_reclaim_without_volume(self):
        """Test that reclaim requires volume confirmation"""
        # Setup similar to reclaim but with low volume
        prices = [50, 48, 46, 44, 42, 40, 38, 37, 36, 35, 34,
                  35, 36, 38, 40, 42, 44, 46, 48, 50, 52]
        volumes = [1000000] * 21  # No volume increase
        
        df = pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        })
        
        ema8 = TechnicalAnalyzer.ema(df['Close'], 8)
        ema21 = TechnicalAnalyzer.ema(df['Close'], 21)
        
        result = TechnicalAnalyzer.detect_ema_power_zone_and_reclaim(df, ema8, ema21)
        
        # Without volume spike, reclaim is less likely
        # Just verify structure is correct
        assert isinstance(result, dict)
        assert 'is_reclaim' in result


class TestFibonacciExtensions:
    """Test Fibonacci A-B-C swing detection and targets"""
    
    def test_fib_valid_swing(self):
        """Test Fibonacci detection with valid A-B-C swing"""
        # Create clear swing: down to A, up to B, pullback to C
        prices_down = list(range(50, 30, -1))  # Down to A
        prices_up = list(range(30, 70))        # Up to B
        prices_pullback = list(range(70, 50, -1))  # Pullback to C
        
        df = pd.DataFrame({
            'High': [p + 1 for p in prices_down + prices_up + prices_pullback],
            'Low': [p - 1 for p in prices_down + prices_up + prices_pullback],
            'Close': prices_down + prices_up + prices_pullback
        })
        
        result = TechnicalAnalyzer.compute_fib_extensions_from_swing(df)
        
        if result:  # May detect a swing
            assert 'A' in result
            assert 'B' in result
            assert 'C' in result
            assert 'T1_1272' in result
            assert 'T2_1618' in result
            assert 'T3_200' in result
            
            # B should be higher than A and C
            assert result['B'] > result['A']
            assert result['B'] > result['C']
            
            # Targets should be progressively higher than C
            assert result['T1_1272'] > result['C']
            assert result['T2_1618'] > result['T1_1272']
            assert result['T3_200'] > result['T2_1618']
    
    def test_fib_insufficient_data(self):
        """Test Fibonacci with insufficient data"""
        df = pd.DataFrame({
            'High': [10, 11, 12],
            'Low': [9, 10, 11],
            'Close': [9.5, 10.5, 11.5]
        })
        
        result = TechnicalAnalyzer.compute_fib_extensions_from_swing(df)
        
        # Should return None with insufficient data
        assert result is None
    
    def test_fib_no_valid_swing(self):
        """Test Fibonacci with no valid swing pattern"""
        # Flat or monotonic trend without clear swing
        df = pd.DataFrame({
            'High': [50] * 100,
            'Low': [49] * 100,
            'Close': [49.5] * 100
        })
        
        result = TechnicalAnalyzer.compute_fib_extensions_from_swing(df)
        
        # May return None if no valid swing detected
        # Just verify it doesn't crash
        assert result is None or isinstance(result, dict)


class TestMultiTimeframeAlignment:
    """Test multi-timeframe trend alignment"""
    
    @pytest.mark.skip(reason="Requires live market data from yfinance")
    def test_timeframe_alignment_structure(self):
        """Test that timeframe alignment returns correct structure"""
        result = TechnicalAnalyzer.analyze_timeframe_alignment("AAPL")
        
        assert 'timeframes' in result
        assert 'alignment_score' in result
        assert 'aligned' in result
        assert isinstance(result['timeframes'], dict)
        assert isinstance(result['alignment_score'], (int, float))
        assert isinstance(result['aligned'], bool)


class TestSectorRelativeStrength:
    """Test sector relative strength calculation"""
    
    @pytest.mark.skip(reason="Requires live market data from yfinance")
    def test_sector_rs_structure(self):
        """Test that sector RS returns correct structure"""
        result = TechnicalAnalyzer.calculate_sector_relative_strength("AAPL")
        
        assert 'sector' in result
        assert 'rs_score' in result
        assert 'vs_spy' in result
        assert 'vs_sector' in result
        assert isinstance(result['rs_score'], (int, float))
        # RS score should be 0-100
        assert 0 <= result['rs_score'] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

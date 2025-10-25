"""Technical analysis calculations."""

import logging
from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Calculate technical indicators"""
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average using pandas ewm."""
        try:
            return pd.to_numeric(series, errors='coerce').ewm(span=period, adjust=False).mean()
        except Exception:
            return series

    @staticmethod
    def demarker(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        DeMarker oscillator in [0,1].
        DeMM_t = max(High_t - High_{t-1}, 0), DeMm_t = max(Low_{t-1} - Low_t, 0)
        DeM = SMA(DeMM, n) / (SMA(DeMM, n) + SMA(DeMm, n))
        """
        try:
            high = pd.to_numeric(df['High'], errors='coerce')
            low = pd.to_numeric(df['Low'], errors='coerce')
            up = (high - high.shift(1)).clip(lower=0.0)
            dn = (low.shift(1) - low).clip(lower=0.0)
            up_sum = up.rolling(window=period, min_periods=period).sum()
            dn_sum = dn.rolling(window=period, min_periods=period).sum()
            denom = (up_sum + dn_sum).replace(0, np.nan)
            dem = (up_sum / denom).clip(0, 1).fillna(0.5)
            return dem
        except Exception:
            return pd.Series([0.5] * len(df), index=df.index)

    @staticmethod
    def detect_ema_power_zone_and_reclaim(
        df: pd.DataFrame,
        ema8: pd.Series,
        ema21: pd.Series,
        vol_lookback: int = 20
    ) -> dict:
        """
        Detects:
        - power_zone: price > both EMAs and 8>21
        - is_reclaim: prior close below at least one EMA, now above both with rising EMAs, volume > avg, and follow-through
        """
        try:
            close = pd.to_numeric(df['Close'], errors='coerce')
            vol = df['Volume'] if 'Volume' in df.columns else None

            power_zone = bool(close.iloc[-1] > ema8.iloc[-1] > ema21.iloc[-1])

            is_reclaim = False
            reasons = []
            if len(close) >= 3 and ema8.notna().any() and ema21.notna().any():
                prior_below = (close.iloc[-2] < ema8.iloc[-2]) or (close.iloc[-2] < ema21.iloc[-2])
                now_above = (close.iloc[-1] > ema8.iloc[-1]) and (close.iloc[-1] > ema21.iloc[-1])
                # Compute EMA slopes using last two values to avoid typing issues
                try:
                    ema8_last = float(ema8.iloc[-1])
                    ema8_prev = float(ema8.iloc[-2])
                    ema8_up = (ema8_last - ema8_prev) > 0.0
                except Exception:
                    ema8_up = False
                try:
                    ema21_last = float(ema21.iloc[-1])
                    ema21_prev = float(ema21.iloc[-2])
                    ema21_up = (ema21_last - ema21_prev) > 0.0
                except Exception:
                    ema21_up = False

                vol_ok = True
                if vol is not None:
                    avg_vol = pd.to_numeric(vol, errors='coerce').rolling(vol_lookback).mean().iloc[-1]
                    vol_ok = (vol.iloc[-1] > 1.1 * avg_vol) if pd.notna(avg_vol) else True

                follow_through = close.iloc[-1] > close.iloc[-2]

                is_reclaim = bool(prior_below and now_above and ema8_up and ema21_up and vol_ok and follow_through)
                if is_reclaim:
                    reasons.append("Close above 8 & 21 with rising EMAs and follow-through")
                    if vol is not None:
                        reasons.append("Volume > 20D average")

            return {
                "power_zone": power_zone,
                "is_reclaim": is_reclaim,
                "reasons": reasons,
            }
        except Exception:
            return {"power_zone": False, "is_reclaim": False, "reasons": []}

    @staticmethod
    def compute_fib_extensions_from_swing(
        df: pd.DataFrame,
        lookback: int = 180,
        pullback_window: int = 30
    ) -> dict | None:
        """
        Simple A-B-C swing detection for uptrends:
        - B: most recent swing high in last ~60 bars
        - A: lowest low preceding B within lookback
        - C: lowest low after B within pullback_window
        Returns dict with A,B,C and T1/T2/T3 targets.
        """
        try:
            if len(df) < 50:
                return None
            highs = pd.to_numeric(df['High'], errors='coerce').tail(lookback)
            lows = pd.to_numeric(df['Low'], errors='coerce').tail(lookback)
            if highs.empty or lows.empty:
                return None

            recent_highs = highs.tail(min(60, len(highs)))
            b_idx = recent_highs.idxmax()
            B = float(highs.loc[b_idx])

            pre = lows.loc[:b_idx]
            if pre.empty:
                return None
            A = float(pre.min())

            post = lows.loc[b_idx:].head(pullback_window)
            if post.empty:
                return None
            C = float(post.min())

            if not (B > A and B > C and C >= A):
                return None

            swing = B - A
            t1 = C + 1.272 * swing
            t2 = C + 1.618 * swing
            t3a = C + 2.000 * swing
            t3b = C + 2.618 * swing

            return {
                "A": A,
                "B": B,
                "C": C,
                "T1_1272": float(t1),
                "T2_1618": float(t2),
                "T3_200": float(t3a),
                "T3_2618": float(t3b),
            }
        except Exception:
            return None

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            # Ensure numeric dtype
            prices = pd.to_numeric(prices, errors='coerce').dropna()
            # Calculate price changes and force numeric dtype for safe comparisons
            delta = pd.to_numeric(prices.diff(), errors='coerce').fillna(0.0)
            gain = (delta.where(delta > 0.0, 0.0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0.0, 0.0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return round(rsi.iloc[-1], 2)
        except:
            return 50.0
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[str, float]:
        """Calculate MACD indicator"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            if current_macd > current_signal:
                return "BULLISH", round(current_macd - current_signal, 2)
            else:
                return "BEARISH", round(current_macd - current_signal, 2)
        except:
            return "NEUTRAL", 0.0
    
    @staticmethod
    def calculate_support_resistance(prices: pd.Series) -> Tuple[float, float]:
        """Calculate support and resistance levels"""
        try:
            recent_prices = prices.tail(20)
            support = round(recent_prices.min(), 2)
            resistance = round(recent_prices.max(), 2)
            return support, resistance
        except:
            current = prices.iloc[-1]
            return round(current * 0.95, 2), round(current * 1.05, 2)
    
    @staticmethod
    def analyze_timeframe_alignment(
        ticker: str,
        timeframes: list[tuple[str, str]] = None
    ) -> dict:
        """
        Multi-timeframe trend confirmation.
        Returns alignment score and trend by timeframe.
        Default timeframes: 1wk (weekly), 1d (daily), 1h (4-hour-like via hourly).
        """
        if timeframes is None:
            timeframes = [("1wk", "Weekly"), ("1d", "Daily"), ("1h", "4-Hour")]
        
        try:
            stock = yf.Ticker(ticker)
            results = {}
            for interval, label in timeframes:
                try:
                    if interval == "1h":
                        # Get hourly data for ~4-hour equivalent check
                        hist = stock.history(period="5d", interval="1h")
                    elif interval == "1wk":
                        hist = stock.history(period="6mo", interval="1wk")
                    else:
                        hist = stock.history(period="3mo", interval=interval)
                    
                    if hist.empty or len(hist) < 10:
                        results[label] = {"trend": "UNKNOWN", "strength": 0}
                        continue
                    
                    # Compute EMA8/21 for this timeframe
                    ema8 = TechnicalAnalyzer.ema(hist['Close'], 8)
                    ema21 = TechnicalAnalyzer.ema(hist['Close'], 21)
                    
                    if ema8.empty or ema21.empty:
                        results[label] = {"trend": "UNKNOWN", "strength": 0}
                        continue
                    
                    close_val = float(hist['Close'].iloc[-1])
                    ema8_val = float(ema8.iloc[-1])
                    ema21_val = float(ema21.iloc[-1])
                    
                    # Determine trend
                    if close_val > ema8_val > ema21_val:
                        trend = "UPTREND"
                        strength = min(100, ((close_val - ema21_val) / ema21_val) * 100 * 10)
                    elif close_val < ema8_val < ema21_val:
                        trend = "DOWNTREND"
                        strength = min(100, ((ema21_val - close_val) / ema21_val) * 100 * 10)
                    else:
                        trend = "SIDEWAYS"
                        strength = 50
                    
                    results[label] = {"trend": trend, "strength": round(strength, 1)}
                except Exception as e:
                    logger.debug(f"Timeframe {label} fetch error: {e}")
                    results[label] = {"trend": "UNKNOWN", "strength": 0}
            
            # Compute alignment score (0-100)
            trends = [r["trend"] for r in results.values() if r["trend"] != "UNKNOWN"]
            if not trends:
                alignment_score = 0
            else:
                # Count how many agree with the most common trend
                from collections import Counter
                trend_counts = Counter(trends)
                most_common, count = trend_counts.most_common(1)[0]
                alignment_score = (count / len(trends)) * 100
            
            return {
                "timeframes": results,
                "alignment_score": round(alignment_score, 1),
                "aligned": alignment_score >= 66.7  # 2 out of 3 or better
            }
        except Exception as e:
            logger.error(f"Multi-timeframe analysis error: {e}")
            return {
                "timeframes": {},
                "alignment_score": 0,
                "aligned": False
            }

    @staticmethod
    def calculate_sector_relative_strength(ticker: str) -> dict:
        """
        Calculate relative strength vs sector and market (SPY).
        Returns sector info and RS score.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get sector
            sector = info.get('sector', 'Unknown')
            
            # Get stock history (3 months)
            stock_hist = stock.history(period="3mo")
            if stock_hist.empty:
                return {"sector": sector, "rs_score": 50, "vs_spy": 0, "vs_sector": 0}
            
            # Stock performance
            stock_return = ((stock_hist['Close'].iloc[-1] / stock_hist['Close'].iloc[0]) - 1) * 100
            
            # SPY comparison
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="3mo")
            spy_return = 0
            if not spy_hist.empty:
                spy_return = ((spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1) * 100
            
            vs_spy = stock_return - spy_return
            
            # Sector ETF mapping (simplified)
            sector_etf_map = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financial Services": "XLF",
                "Consumer Cyclical": "XLY",
                "Consumer Defensive": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB",
                "Industrials": "XLI",
                "Communication Services": "XLC",
            }
            
            sector_etf = sector_etf_map.get(sector, "SPY")
            sector_return = 0
            try:
                sector_ticker = yf.Ticker(sector_etf)
                sector_hist = sector_ticker.history(period="3mo")
                if not sector_hist.empty:
                    sector_return = ((sector_hist['Close'].iloc[-1] / sector_hist['Close'].iloc[0]) - 1) * 100
            except:
                pass
            
            vs_sector = stock_return - sector_return
            
            # RS Score: 0-100 scale
            # >0 outperforming, <0 underperforming
            # Normalize: +20% = 100, -20% = 0, 0% = 50
            rs_score = max(0, min(100, 50 + (vs_spy * 2.5)))
            
            return {
                "sector": sector,
                "sector_etf": sector_etf,
                "rs_score": round(rs_score, 1),
                "vs_spy": round(vs_spy, 2),
                "vs_sector": round(vs_sector, 2),
                "stock_return_3mo": round(stock_return, 2),
            }
        except Exception as e:
            logger.error(f"Sector RS calculation error: {e}")
            return {
                "sector": "Unknown",
                "sector_etf": "SPY",
                "rs_score": 50,
                "vs_spy": 0,
                "vs_sector": 0,
                "stock_return_3mo": 0,
            }

    @staticmethod
    def calculate_iv_metrics(ticker: str) -> Tuple[float, float]:
        """Calculate IV Rank and IV Percentile (simulated)"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get options data if available
            try:
                options_dates = stock.options
                if options_dates:
                    opt_chain = stock.option_chain(options_dates[0])
                    
                    # Calculate average implied volatility
                    calls_iv = opt_chain.calls['impliedVolatility'].mean()
                    puts_iv = opt_chain.puts['impliedVolatility'].mean()
                    current_iv = (calls_iv + puts_iv) / 2 * 100
                    
                    # Simulate IV rank (in production, use historical IV data)
                    # IV rank = where current IV sits in the range of past year's IV
                    hist = stock.history(period="1y")
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                    
                    # Estimate IV rank (simplified)
                    iv_rank = min(100, max(0, (current_iv / volatility) * 50))
                    iv_percentile = iv_rank  # Simplified
                    
                    return round(iv_rank, 1), round(iv_percentile, 1)
            except:
                pass
            
            # Fallback: estimate from historical volatility
            hist = stock.history(period="1y")
            if not hist.empty:
                volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
                iv_rank = min(100, max(0, volatility))
                return round(iv_rank, 1), round(iv_rank, 1)
            
        except Exception as e:
            logger.error(f"Error calculating IV metrics: {e}")
        
        return 50.0, 50.0

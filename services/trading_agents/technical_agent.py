"""
Technical analysis agent that evaluates price action and indicator strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class TechnicalAgentResult:
    """Structured technical analysis output for the coordinator."""

    symbol: str
    trend: str
    trend_strength: float
    momentum_state: str
    rsi_state: str
    volume_trend: str
    support: Optional[float]
    resistance: Optional[float]
    signals: List[str]
    bias: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trend": self.trend,
            "trend_strength": self.trend_strength,
            "momentum_state": self.momentum_state,
            "rsi_state": self.rsi_state,
            "volume_trend": self.volume_trend,
            "support": self.support,
            "resistance": self.resistance,
            "signals": self.signals,
            "bias": self.bias,
            "confidence": self.confidence,
        }


class TechnicalAgent:
    """Pure technical-analysis agent (no external API calls)."""

    def analyze(
        self,
        symbol: str,
        current_price: float,
        technical_data: Dict[str, Any],
        *,
        pnl_pct: float,
        hold_time_minutes: float,
    ) -> TechnicalAgentResult:
        """
        Evaluate technical posture and return a structured summary.

        Args:
            symbol: Trading symbol (e.g., "BTC/USD").
            current_price: Latest price.
            technical_data: Indicator payload computed upstream.
            pnl_pct: Current profit/loss percentage.
            hold_time_minutes: Duration the position has been open.
        """
        if not technical_data:
            logger.debug(f"TechnicalAgent received no technical data for {symbol}")
            return TechnicalAgentResult(
                symbol=symbol,
                trend="UNKNOWN",
                trend_strength=50.0,
                momentum_state="UNKNOWN",
                rsi_state="UNKNOWN",
                volume_trend="UNKNOWN",
                support=None,
                resistance=None,
                signals=["Insufficient technical data"],
                bias="NEUTRAL",
                confidence=40.0,
            )

        trend = technical_data.get("trend", "NEUTRAL")
        ema_20 = technical_data.get("ema_20")
        ema_50 = technical_data.get("ema_50") or ema_20
        macd_hist = technical_data.get("macd_histogram", 0.0)
        rsi = technical_data.get("rsi")
        volume_change_pct = technical_data.get("volume_change_pct", 0.0)

        # Trend strength: scale the EMA spread into 0-100 range.
        trend_strength = self._calc_trend_strength(ema_20, ema_50, trend)

        # MACD histogram -> momentum label.
        momentum_state = self._determine_momentum(macd_hist)

        # RSI classification.
        rsi_state = self._interpret_rsi(rsi)

        # Volume characterization.
        volume_trend = self._interpret_volume(volume_change_pct)

        signals: List[str] = []
        if trend_strength > 65:
            signals.append("Strong trend alignment")
        if momentum_state.startswith("STRONG"):
            signals.append("MACD momentum confirmation")
        if rsi_state == "NEUTRAL" and trend.startswith("BULL"):
            signals.append("Room before RSI overbought")
        if pnl_pct >= 5:
            signals.append("Consider locking profits (P&L > 5%)")
        if hold_time_minutes > 240 and rsi_state == "OVERBOUGHT":
            signals.append("Extended hold with overbought RSI")

        bias = self._determine_bias(trend, trend_strength, momentum_state)

        confidence = self._calculate_confidence(trend_strength, momentum_state, rsi_state, volume_trend)

        return TechnicalAgentResult(
            symbol=symbol,
            trend=trend,
            trend_strength=trend_strength,
            momentum_state=momentum_state,
            rsi_state=rsi_state,
            volume_trend=volume_trend,
            support=technical_data.get("support"),
            resistance=technical_data.get("resistance"),
            signals=signals,
            bias=bias,
            confidence=confidence,
        )

    def _calc_trend_strength(self, ema_20: Optional[float], ema_50: Optional[float], trend: str) -> float:
        if not ema_20 or not ema_50:
            return 50.0
        spread = abs(ema_20 - ema_50)
        baseline = (ema_20 + ema_50) / 2
        if baseline == 0:
            return 50.0
        pct = min(max((spread / baseline) * 400, 0), 100)  # amplified to emphasize larger spreads
        return pct if trend != "NEUTRAL" else pct / 2

    def _determine_momentum(self, macd_hist: float) -> str:
        if macd_hist is None:
            return "UNKNOWN"
        if macd_hist >= 0.5:
            return "STRONG_BULLISH"
        if 0.1 <= macd_hist < 0.5:
            return "BULLISH"
        if -0.1 < macd_hist < 0.1:
            return "NEUTRAL"
        if -0.5 < macd_hist <= -0.1:
            return "BEARISH"
        return "STRONG_BEARISH"

    def _interpret_rsi(self, rsi: Optional[float]) -> str:
        if rsi is None:
            return "UNKNOWN"
        if rsi >= 70:
            return "OVERBOUGHT"
        if rsi <= 30:
            return "OVERSOLD"
        return "NEUTRAL"

    def _interpret_volume(self, volume_change_pct: float) -> str:
        if volume_change_pct is None:
            return "UNKNOWN"
        if volume_change_pct > 50:
            return "SURGING"
        if volume_change_pct > 10:
            return "INCREASING"
        if volume_change_pct < -30:
            return "COLLAPSING"
        if volume_change_pct < -10:
            return "DECREASING"
        return "NORMAL"

    def _determine_bias(self, trend: str, trend_strength: float, momentum_state: str) -> str:
        if trend.startswith("BULL") and trend_strength >= 60 and "BULL" in momentum_state:
            return "BULLISH"
        if trend.startswith("BEAR") and trend_strength >= 60 and "BEAR" in momentum_state:
            return "BEARISH"
        return "NEUTRAL"

    def _calculate_confidence(
        self,
        trend_strength: float,
        momentum_state: str,
        rsi_state: str,
        volume_trend: str,
    ) -> float:
        confidence = trend_strength * 0.4

        if momentum_state.startswith("STRONG"):
            confidence += 25
        elif momentum_state in ("BULLISH", "BEARISH"):
            confidence += 15
        else:
            confidence += 5

        if rsi_state == "NEUTRAL":
            confidence += 10
        elif rsi_state in ("OVERBOUGHT", "OVERSOLD"):
            confidence -= 10

        if volume_trend in ("SURGING", "INCREASING"):
            confidence += 10
        elif volume_trend in ("COLLAPSING", "DECREASING"):
            confidence -= 10

        return max(0.0, min(100.0, confidence))


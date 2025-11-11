"""
Risk management agent that evaluates position exposure and protective actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from services.ai_crypto_position_manager import MonitoredCryptoPosition


@dataclass
class RiskAgentResult:
    """Structured risk analysis returned to the coordinator."""

    risk_level: float
    position_health: str
    recommended_action: str
    stop_suggestion: Optional[float]
    target_suggestion: Optional[float]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "position_health": self.position_health,
            "recommended_action": self.recommended_action,
            "stop_suggestion": self.stop_suggestion,
            "target_suggestion": self.target_suggestion,
            "notes": self.notes,
        }


class RiskAgent:
    """Evaluates risk exposure using position context and agent data."""

    def analyze(
        self,
        position: "MonitoredCryptoPosition",
        current_price: float,
        pnl_pct: float,
        *,
        technical_view: Dict[str, Any],
        sentiment_view: Dict[str, Any],
    ) -> RiskAgentResult:
        """
        Assess current risk posture and suggest protective adjustments.

        Args:
            position: Current monitored position.
            current_price: Latest market price.
            pnl_pct: Profit/loss percentage.
            technical_view: Output from the technical agent.
            sentiment_view: Output from the sentiment agent.
        """
        risk_level = self._baseline_risk(position, pnl_pct)
        risk_level = self._adjust_for_sentiment(risk_level, sentiment_view)
        risk_level = self._adjust_for_technical(risk_level, technical_view)
        risk_level = max(0.0, min(100.0, risk_level))

        position_health = self._health_bucket(risk_level)
        recommended_action, stop_suggestion, target_suggestion = self._recommend_action(
            position, current_price, pnl_pct, risk_level, technical_view
        )

        notes = self._build_notes(
            pnl_pct,
            risk_level,
            position,
            technical_view,
            sentiment_view,
        )

        return RiskAgentResult(
            risk_level=risk_level,
            position_health=position_health,
            recommended_action=recommended_action,
            stop_suggestion=stop_suggestion,
            target_suggestion=target_suggestion,
            notes=notes,
        )

    def _baseline_risk(self, position: MonitoredCryptoPosition, pnl_pct: float) -> float:
        risk = 50.0

        if position.side == "BUY":
            max_adverse = abs(min(0.0, position.max_adverse_pct))
        else:
            max_adverse = abs(max(0.0, position.max_adverse_pct))

        if max_adverse > 5:
            risk += 10

        if pnl_pct < -3:
            risk += 15
        elif pnl_pct > 5:
            risk -= 10

        if not position.moved_to_breakeven and pnl_pct > position.breakeven_trigger_pct:
            risk -= 5

        return risk

    def _adjust_for_sentiment(self, risk_level: float, sentiment_view: Dict[str, Any]) -> float:
        score = sentiment_view.get("sentiment_score", 50.0)
        trend = sentiment_view.get("trend", "STABLE")

        if score >= 70 and trend == "IMPROVING":
            return risk_level - 10
        if score <= 40 and trend == "DETERIORATING":
            return risk_level + 15
        if score <= 30:
            return risk_level + 10
        if score >= 80:
            return risk_level - 12
        return risk_level

    def _adjust_for_technical(self, risk_level: float, technical_view: Dict[str, Any]) -> float:
        trend = technical_view.get("trend", "NEUTRAL")
        trend_strength = technical_view.get("trend_strength", 50.0)
        rsi_state = technical_view.get("rsi_state", "NEUTRAL")

        if trend.startswith("BULL") and trend_strength >= 70:
            risk_level -= 8
        if trend.startswith("BEAR") and trend_strength >= 70:
            risk_level += 12

        if rsi_state == "OVERBOUGHT":
            risk_level += 5
        elif rsi_state == "OVERSOLD":
            risk_level -= 5

        return risk_level

    def _health_bucket(self, risk_level: float) -> str:
        if risk_level >= 75:
            return "HIGH_RISK"
        if risk_level >= 55:
            return "ELEVATED"
        if risk_level >= 40:
            return "MANAGEABLE"
        return "LOW_RISK"

    def _recommend_action(
        self,
        position: MonitoredCryptoPosition,
        current_price: float,
        pnl_pct: float,
        risk_level: float,
        technical_view: Dict[str, Any],
    ) -> tuple[str, Optional[float], Optional[float]]:
        stop_suggestion: Optional[float] = None
        target_suggestion: Optional[float] = None
        action = "MAINTAIN"

        if risk_level >= 75:
            action = "EXIT_OR_REDUCE"
            stop_suggestion = current_price * (0.99 if position.side == "BUY" else 1.01)
        elif risk_level >= 55:
            action = "TIGHTEN_PROTECTION"
            stop_suggestion = max(position.stop_loss, current_price * 0.985) if position.side == "BUY" else min(
                position.stop_loss, current_price * 1.015
            )
        else:
            trend = technical_view.get("trend", "NEUTRAL")
            if trend.startswith("BULL") and pnl_pct > 3:
                action = "CONSIDER_EXTEND_TARGET"
                resistance = technical_view.get("resistance")
                if resistance:
                    target_suggestion = resistance * 1.01

        return action, stop_suggestion, target_suggestion

    def _build_notes(
        self,
        pnl_pct: float,
        risk_level: float,
        position: MonitoredCryptoPosition,
        technical_view: Dict[str, Any],
        sentiment_view: Dict[str, Any],
    ) -> str:
        notes = [
            f"P&L: {pnl_pct:+.2f}%",
            f"Risk Score: {risk_level:.1f}/100",
        ]

        notes.append(f"Technical bias: {technical_view.get('bias', 'NEUTRAL')}")
        notes.append(f"Sentiment bias: {sentiment_view.get('bias', 'NEUTRAL')}")

        if position.partial_exit_taken:
            notes.append("Partial exit already taken.")
        if position.moved_to_breakeven:
            notes.append("Stop moved to breakeven.")

        return " ".join(notes)


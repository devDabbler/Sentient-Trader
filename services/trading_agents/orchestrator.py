"""
Coordinator that merges agent insights and delegates the final decision to an LLM.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from loguru import logger

from services.crypto_news_analyzer import CryptoNewsSentiment
from services.trading_agents.risk_agent import RiskAgent, RiskAgentResult
from services.trading_agents.sentiment_agent import SentimentAgent, SentimentAgentResult
from services.trading_agents.technical_agent import TechnicalAgent, TechnicalAgentResult

if TYPE_CHECKING:  # pragma: no cover
    from services.ai_crypto_position_manager import AITradeDecision, MonitoredCryptoPosition


@dataclass
class CoordinatorDecisionContext:
    """Inputs required to produce the final coordinated decision."""

    position: "MonitoredCryptoPosition"
    current_price: float
    pnl_pct: float
    hold_time_minutes: float
    technical_data: Dict[str, Any]
    recent_news: Optional[list]
    sentiment_score: Optional[float]
    news_sentiment: Optional[CryptoNewsSentiment]


class TradingAgentOrchestrator:
    """
    Coordinates sentiment, technical, and risk agents to reach a final AI decision.
    """

    def __init__(
        self,
        llm_analyzer: Any,
        sentiment_agent: Optional[SentimentAgent] = None,
        technical_agent: Optional[TechnicalAgent] = None,
        risk_agent: Optional[RiskAgent] = None,
        *,
        sentiment_weight: int = 30,
        technical_weight: int = 40,
        risk_weight: int = 30,
    ) -> None:
        self.llm_analyzer = llm_analyzer
        self.sentiment_agent = sentiment_agent or SentimentAgent(llm_analyzer=llm_analyzer)
        self.technical_agent = technical_agent or TechnicalAgent()
        self.risk_agent = risk_agent or RiskAgent()
        self.agent_weights = {
            "sentiment": sentiment_weight,
            "technical": technical_weight,
            "risk": risk_weight,
        }

    async def make_decision(self, context: CoordinatorDecisionContext) -> Optional[AITradeDecision]:
        """
        Run all agents and synthesize the final AITradeDecision.
        """
        from services.ai_crypto_position_manager import AITradeDecision  # Local import to avoid circular dependency

        if not self.llm_analyzer or not hasattr(self.llm_analyzer, "analyze_with_llm"):
            logger.warning("TradingAgentOrchestrator requires an llm_analyzer with analyze_with_llm")
            return None

        sentiment_task = asyncio.create_task(
            self.sentiment_agent.analyze(
                context.position.pair.split("/")[0],
                news_sentiment=context.news_sentiment,
                hours=2,
            )
        )

        technical_result = self.technical_agent.analyze(
            symbol=context.position.pair,
            current_price=context.current_price,
            technical_data=context.technical_data,
            pnl_pct=context.pnl_pct,
            hold_time_minutes=context.hold_time_minutes,
        )

        sentiment_result = await sentiment_task

        risk_result = self.risk_agent.analyze(
            position=context.position,
            current_price=context.current_price,
            pnl_pct=context.pnl_pct,
            technical_view=technical_result.to_dict(),
            sentiment_view=sentiment_result.to_dict(),
        )

        coordinator_prompt = self._build_coordinator_prompt(
            context=context,
            sentiment=sentiment_result,
            technical=technical_result,
            risk=risk_result,
        )

        response = self.llm_analyzer.analyze_with_llm(coordinator_prompt)
        decision_payload = self._parse_json_response(response)

        if not decision_payload:
            logger.error("Coordinator failed to parse LLM decision payload")
            return None

        metadata = {
            "agent_weights": self.agent_weights,
            "agent_analyses": {
                "sentiment": sentiment_result.to_dict(),
                "technical": technical_result.to_dict(),
                "risk": risk_result.to_dict(),
            },
            "coordinator_prompt": coordinator_prompt,
        }

        return AITradeDecision(
            action=decision_payload.get("action", "HOLD"),
            confidence=float(decision_payload.get("confidence", 0)),
            reasoning=decision_payload.get("reasoning", ""),
            urgency=decision_payload.get("urgency", "LOW"),
            new_stop=decision_payload.get("new_stop"),
            new_target=decision_payload.get("new_target"),
            partial_pct=decision_payload.get("partial_pct"),
            technical_score=float(decision_payload.get("technical_score", 0)),
            trend_score=float(decision_payload.get("trend_score", 0)),
            risk_score=float(decision_payload.get("risk_score", 0)),
            metadata=metadata,
        )

    def _build_coordinator_prompt(
        self,
        *,
        context: CoordinatorDecisionContext,
        sentiment: SentimentAgentResult,
        technical: TechnicalAgentResult,
        risk: RiskAgentResult,
    ) -> str:
        """Compose the orchestration prompt for the coordinator LLM."""
        recent_news_block = ""

        if context.recent_news:
            lines = []
            for news in context.recent_news[:5]:
                emoji = "🟢" if news.get("sentiment") == "BULLISH" else "🔴" if news.get("sentiment") == "BEARISH" else "⚪"
                lines.append(
                    f"{emoji} {news.get('timestamp', 'recent')} — {news.get('title', '')} "
                    f"(Sentiment: {news.get('sentiment')}, Confidence: {news.get('confidence', 0.0):.0%}, "
                    f"Impact: {news.get('impact', 'UNKNOWN')})"
                )
            recent_news_block = "\n".join(lines)

        sentiment_json = json.dumps(sentiment.to_dict(), indent=2)
        technical_json = json.dumps(technical.to_dict(), indent=2)
        risk_json = json.dumps(risk.to_dict(), indent=2)

        rsi_value = context.technical_data.get("rsi")
        rsi_display = f"{rsi_value:.2f}" if rsi_value is not None else "N/A"

        prompt = f"""
You are the multi-agent coordinator for an AI crypto trading assistant.
Three specialized agents have produced analyses for the position below. You MUST respect
the agent weights when synthesizing the final decision.

**Position Context**
- Pair: {context.position.pair}
- Side: {context.position.side}
- Entry: ${context.position.entry_price:,.2f}
- Current Price: ${context.current_price:,.2f}
- P&L: {context.pnl_pct:+.2f}%
- Hold Time: {context.hold_time_minutes:.1f} minutes
- Stop Loss: ${context.position.stop_loss:,.2f}
- Take Profit: ${context.position.take_profit:,.2f}
- Strategy: {context.position.strategy}

**Latest Technical Snapshot**
- RSI: {rsi_display}
- MACD Histogram: {context.technical_data.get('macd_histogram', 0.0):.4f}
- EMA20: ${context.technical_data.get('ema_20', 0.0):,.2f}
- EMA50: ${context.technical_data.get('ema_50', 0.0):,.2f}
- Support: ${context.technical_data.get('support', 0.0):,.2f}
- Resistance: ${context.technical_data.get('resistance', 0.0):,.2f}

**Breaking News (Last 2 Hours)**
{recent_news_block or 'No significant news detected in last 2 hours.'}

**Agent Weighting**
- Sentiment Agent: {self.agent_weights['sentiment']}%
- Technical Agent: {self.agent_weights['technical']}%
- Risk Agent: {self.agent_weights['risk']}%

**Sentiment Agent Analysis**
{sentiment_json}

**Technical Agent Analysis**
{technical_json}

**Risk Agent Analysis**
{risk_json}

Decision Requirements:
1. Weigh each agent according to the weighting table.
2. Resolve disagreements explicitly and record them in `conflicts_resolved`.
3. Recommend one of: HOLD, TIGHTEN_STOP, EXTEND_TARGET, TAKE_PARTIAL, CLOSE_NOW, MOVE_TO_BREAKEVEN.
4. Provide concise reasoning (<= 3 sentences) referencing agent inputs.
5. Recommend updated stop/target only when necessary.

Respond ONLY with valid JSON. Format:
{{
  "action": "HOLD|TIGHTEN_STOP|EXTEND_TARGET|TAKE_PARTIAL|CLOSE_NOW|MOVE_TO_BREAKEVEN",
  "confidence": 0-100,
  "reasoning": "Concise explanation citing agent inputs",
  "urgency": "LOW|MEDIUM|HIGH",
  "new_stop": price_or_null,
  "new_target": price_or_null,
  "partial_pct": percentage_or_null,
  "technical_score": 0-100,
  "trend_score": 0-100,
  "risk_score": 0-100,
  "conflicts_resolved": ["List conflicts you resolved, if any"]
}}
"""

        return prompt

    def _parse_json_response(self, response: Any) -> Optional[Dict[str, Any]]:
        if not response:
            return None
        if isinstance(response, dict):
            return response

        text = str(response)
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        if start_idx == -1 or end_idx <= start_idx:
            return None
        try:
            return json.loads(text[start_idx:end_idx])
        except json.JSONDecodeError as exc:
            logger.error(f"Failed to decode coordinator JSON payload: {exc}")
            return None


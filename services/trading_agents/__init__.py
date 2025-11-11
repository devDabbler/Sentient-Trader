"""
Specialized trading agents used by the multi-agent orchestration layer.

Each agent focuses on a single dimension of the decision process
so that the coordinator LLM can weigh independent viewpoints.
"""

from .sentiment_agent import SentimentAgent, SentimentAgentResult
from .technical_agent import TechnicalAgent, TechnicalAgentResult
from .risk_agent import RiskAgent, RiskAgentResult
from .orchestrator import TradingAgentOrchestrator, CoordinatorDecisionContext

__all__ = [
    "SentimentAgent",
    "SentimentAgentResult",
    "TechnicalAgent",
    "TechnicalAgentResult",
    "RiskAgent",
    "RiskAgentResult",
    "TradingAgentOrchestrator",
    "CoordinatorDecisionContext",
]


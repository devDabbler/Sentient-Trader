"""
Data models for LLM Request Manager
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class LLMPriority(Enum):
    """Priority levels for LLM requests"""
    CRITICAL = 1  # Trade execution validation
    HIGH = 2      # Position monitoring
    MEDIUM = 3    # Opportunity scanning
    LOW = 4       # Informational queries


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENROUTER = "openrouter"
    CLAUDE = "claude"
    OPENAI = "openai"
    GROQ = "groq"
    LOCAL = "local"


@dataclass
class LLMRequest:
    """Represents an LLM API request"""
    prompt: str
    priority: LLMPriority
    service_name: str
    cache_key: Optional[str] = None
    ttl_seconds: int = 300
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Represents an LLM API response"""
    content: str
    request_id: str
    provider: LLMProvider
    model: str
    tokens_used: int
    cost_usd: float
    from_cache: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CachedResponse:
    """Cached LLM response with TTL"""
    response: str
    timestamp: float
    cost_usd: float
    tokens_used: int
    provider: LLMProvider
    model: str


@dataclass
class RateLimitConfig:
    """Rate limiting configuration per provider"""
    max_requests_per_minute: int
    max_concurrent_requests: int
    backoff_seconds: float = 1.0
    max_retries: int = 3


@dataclass
class UsageStats:
    """Usage statistics for a service"""
    service_name: str
    total_requests: int = 0
    cached_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    requests_by_priority: Dict[str, int] = field(default_factory=dict)
    requests_by_provider: Dict[str, int] = field(default_factory=dict)
    last_request_time: Optional[datetime] = None
    errors: int = 0


@dataclass
class LLMManagerConfig:
    """Configuration for LLM Request Manager"""
    primary_provider: LLMProvider = LLMProvider.OPENROUTER
    fallback_providers: List[LLMProvider] = field(default_factory=list)
    enable_caching: bool = True
    default_cache_ttl: int = 300
    default_model: str = "openai/gpt-4o-mini"
    default_max_tokens: int = 2000
    rate_limits: Dict[str, RateLimitConfig] = field(default_factory=dict)
    cost_tracking_enabled: bool = True
    max_queue_size: int = 1000
    log_requests: bool = True


# Cost per 1M tokens (approximate values, update based on actual pricing)
MODEL_COSTS = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.35, "output": 0.40},
    "google/gemini-pro-1.5": {"input": 1.25, "output": 5.00},
    # Groq models (very fast inference, competitive pricing)
    "groq/llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "groq/llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "groq/mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},  # Groq model name without prefix
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a model based on token usage"""
    if model not in MODEL_COSTS:
        # Default fallback cost
        return (input_tokens + output_tokens) / 1_000_000 * 1.0
    
    costs = MODEL_COSTS[model]
    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    return input_cost + output_cost

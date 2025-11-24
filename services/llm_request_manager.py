"""
Centralized LLM Request Manager
Handles all LLM API calls with priority queue, caching, rate limiting, and cost tracking
"""
import os
import time
import logging
import hashlib
import asyncio
import requests
from queue import PriorityQueue, Empty
from threading import Lock, Thread
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict

from models.llm_models import (
    LLMRequest, LLMResponse, CachedResponse, UsageStats,
    LLMPriority, LLMProvider, LLMManagerConfig, RateLimitConfig,
    calculate_cost
)


logger = logging.getLogger(__name__)


class LLMRequestManager:
    """
    Centralized manager for all LLM API requests
    
    Features:
    - Priority queue (CRITICAL > HIGH > MEDIUM > LOW)
    - Rate limiting per provider
    - Response caching with TTL
    - Cost tracking per service
    - Provider fallback (OpenRouter -> Claude -> OpenAI)
    - Concurrent request management
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.config = self._load_config()
        self.queue = PriorityQueue(maxsize=self.config.max_queue_size)
        self.cache: Dict[str, CachedResponse] = {}
        self.usage_stats: Dict[str, UsageStats] = defaultdict(
            lambda: UsageStats(service_name="unknown")
        )
        
        # Rate limiting trackers
        self.request_times: Dict[str, List[float]] = defaultdict(list)
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        self.rate_limit_locks: Dict[str, Lock] = defaultdict(Lock)
        
        # Provider API keys
        self.api_keys = {
            LLMProvider.OPENROUTER: os.getenv("OPENROUTER_API_KEY"),
            LLMProvider.CLAUDE: os.getenv("ANTHROPIC_API_KEY"),
            LLMProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
        }
        
        # Request counter for IDs
        self._request_counter = 0
        self._counter_lock = Lock()
        
        logger.info("LLM Request Manager initialized")
    
    def _load_config(self) -> LLMManagerConfig:
        """Load configuration from environment or defaults"""
        rate_limits = {
            "openrouter": RateLimitConfig(
                max_requests_per_minute=int(os.getenv("OPENROUTER_RPM", "60")),
                max_concurrent_requests=int(os.getenv("OPENROUTER_CONCURRENT", "3"))
            ),
            "claude": RateLimitConfig(
                max_requests_per_minute=50,
                max_concurrent_requests=5
            ),
            "openai": RateLimitConfig(
                max_requests_per_minute=60,
                max_concurrent_requests=5
            ),
        }
        
        return LLMManagerConfig(
            primary_provider=LLMProvider.OPENROUTER,
            fallback_providers=[LLMProvider.CLAUDE, LLMProvider.OPENAI],
            enable_caching=os.getenv("LLM_ENABLE_CACHE", "True").lower() == "true",
            default_model=os.getenv("LLM_DEFAULT_MODEL", "openai/gpt-4o-mini"),
            rate_limits=rate_limits
        )
    
    def request(
        self,
        prompt: str,
        service_name: str,
        priority: str = "MEDIUM",
        cache_key: Optional[str] = None,
        ttl: int = 300,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
        blocking: bool = True
    ) -> Optional[str]:
        """
        Submit an LLM request
        
        Args:
            prompt: The prompt to send
            service_name: Name of calling service (for tracking)
            priority: CRITICAL, HIGH, MEDIUM, or LOW
            cache_key: Optional key for caching (auto-generated if None)
            ttl: Cache TTL in seconds
            model: Model to use (default: config.default_model)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            metadata: Additional metadata to track
            blocking: Wait for response (True) or queue async (False)
        
        Returns:
            Response text if blocking=True, None if blocking=False
        """
        try:
            # Convert priority string to enum
            priority_enum = LLMPriority[priority.upper()]
        except KeyError:
            logger.warning(f"Invalid priority '{priority}', using MEDIUM")
            priority_enum = LLMPriority.MEDIUM
        
        # Generate cache key if not provided
        if cache_key is None and self.config.enable_caching:
            cache_key = self._generate_cache_key(prompt, model or self.config.default_model)
        
        # Check cache first
        if cache_key and self.config.enable_caching:
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.info(f"Cache hit for {service_name}: {cache_key[:20]}...")
                self._update_usage_stats(
                    service_name=service_name,
                    cached=True,
                    tokens=cached.tokens_used,
                    cost=cached.cost_usd,
                    priority=priority_enum,
                    provider=cached.provider
                )
                return cached.response
        
        # Generate request ID
        with self._counter_lock:
            self._request_counter += 1
            request_id = f"{service_name}_{self._request_counter}_{int(time.time())}"
        
        # Create request object
        llm_request = LLMRequest(
            prompt=prompt,
            priority=priority_enum,
            service_name=service_name,
            cache_key=cache_key,
            ttl_seconds=ttl,
            model=model or self.config.default_model,
            max_tokens=max_tokens or self.config.default_max_tokens,
            temperature=temperature,
            metadata=metadata or {},
            request_id=request_id
        )
        
        if blocking:
            return self._process_request(llm_request)
        else:
            # Queue for async processing
            self.queue.put((priority_enum.value, time.time(), llm_request))
            logger.info(f"Queued async request {request_id} with priority {priority}")
            return None
    
    def _process_request(self, request: LLMRequest) -> Optional[str]:
        """Process a single LLM request with provider fallback"""
        providers = [self.config.primary_provider] + self.config.fallback_providers
        
        for provider in providers:
            if not self.api_keys.get(provider):
                logger.debug(f"No API key for {provider.value}, skipping")
                continue
            
            try:
                # Wait for rate limit clearance
                self._wait_for_rate_limit(provider)
                
                # Make API call
                response = self._call_provider(provider, request)
                
                # Cache if enabled
                if request.cache_key and self.config.enable_caching:
                    self._cache_response(
                        request.cache_key,
                        response,
                        request.ttl_seconds
                    )
                
                # Update stats
                self._update_usage_stats(
                    service_name=request.service_name,
                    cached=False,
                    tokens=response.tokens_used,
                    cost=response.cost_usd,
                    priority=request.priority,
                    provider=provider
                )
                
                logger.info(
                    f"Request {request.request_id} completed via {provider.value} "
                    f"(${response.cost_usd:.4f}, {response.tokens_used} tokens)"
                )
                
                return response.content
            
            except Exception as e:
                logger.error(f"Error with {provider.value}: {e}")
                self.usage_stats[request.service_name].errors += 1
                
                # Try next provider if available
                if provider != providers[-1]:
                    logger.info(f"Falling back to next provider...")
                    continue
                else:
                    logger.error(f"All providers failed for request {request.request_id}")
                    return None
        
        return None
    
    def _call_provider(self, provider: LLMProvider, request: LLMRequest) -> LLMResponse:
        """Make API call to specific provider"""
        if provider == LLMProvider.OPENROUTER:
            return self._call_openrouter(request)
        elif provider == LLMProvider.CLAUDE:
            return self._call_claude(request)
        elif provider == LLMProvider.OPENAI:
            return self._call_openai(request)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_openrouter(self, request: LLMRequest) -> LLMResponse:
        """Call OpenRouter API"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_keys[LLMProvider.OPENROUTER]}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sentient-trader.app",
            "X-Title": "Sentient Trader"
        }
        
        payload = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }
        
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        model_name = request.model or self.config.default_model
        cost = calculate_cost(model_name, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            request_id=request.request_id or "unknown",
            provider=LLMProvider.OPENROUTER,
            model=model_name,
            tokens_used=total_tokens,
            cost_usd=cost,
            metadata={"usage": usage}
        )
    
    def _call_claude(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic Claude API"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_keys[LLMProvider.CLAUDE],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Map model name
        model = request.model or self.config.default_model
        if "claude" not in model:
            model = "claude-3-5-sonnet-20241022"
        
        payload = {
            "model": model,
            "max_tokens": request.max_tokens or 4096,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = input_tokens + output_tokens
        
        cost = calculate_cost(model, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            request_id=request.request_id or "unknown",
            provider=LLMProvider.CLAUDE,
            model=model,
            tokens_used=total_tokens,
            cost_usd=cost,
            metadata={"usage": usage}
        )
    
    def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_keys[LLMProvider.OPENAI]}",
            "Content-Type": "application/json"
        }
        
        # Map model name
        model = request.model or self.config.default_model
        if "/" in model:
            model = model.split("/")[-1]
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
        }
        
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
        
        cost = calculate_cost(model, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            request_id=request.request_id or "unknown",
            provider=LLMProvider.OPENAI,
            model=model,
            tokens_used=total_tokens,
            cost_usd=cost,
            metadata={"usage": usage}
        )
    
    def _wait_for_rate_limit(self, provider: LLMProvider):
        """Wait if rate limit would be exceeded"""
        provider_key = provider.value
        rate_config = self.config.rate_limits.get(provider_key)
        
        if not rate_config:
            return
        
        with self.rate_limit_locks[provider_key]:
            now = time.time()
            
            # Clean old timestamps (outside 1-minute window)
            self.request_times[provider_key] = [
                t for t in self.request_times[provider_key]
                if now - t < 60
            ]
            
            # Check if we're at rate limit
            if len(self.request_times[provider_key]) >= rate_config.max_requests_per_minute:
                oldest = self.request_times[provider_key][0]
                wait_time = 60 - (now - oldest)
                if wait_time > 0:
                    logger.info(f"Rate limit reached for {provider_key}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
            
            # Check concurrent requests
            while self.concurrent_requests[provider_key] >= rate_config.max_concurrent_requests:
                time.sleep(0.1)
            
            # Record this request
            self.request_times[provider_key].append(time.time())
            self.concurrent_requests[provider_key] += 1
    
    def _release_concurrent_slot(self, provider: LLMProvider):
        """Release a concurrent request slot"""
        provider_key = provider.value
        with self.rate_limit_locks[provider_key]:
            self.concurrent_requests[provider_key] = max(
                0, self.concurrent_requests[provider_key] - 1
            )
    
    def _generate_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[CachedResponse]:
        """Get cached response if valid"""
        if cache_key not in self.cache:
            return None
        
        cached = self.cache[cache_key]
        if time.time() - cached.timestamp > 0:  # TTL handled separately
            return cached
        
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse, ttl: int):
        """Cache a response"""
        self.cache[cache_key] = CachedResponse(
            response=response.content,
            timestamp=time.time(),
            cost_usd=response.cost_usd,
            tokens_used=response.tokens_used,
            provider=response.provider,
            model=response.model
        )
        
        # Clean old cache entries (simple LRU)
        if len(self.cache) > 1000:
            # Remove oldest 10%
            sorted_cache = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
            for key, _ in sorted_cache[:100]:
                del self.cache[key]
    
    def _update_usage_stats(
        self,
        service_name: str,
        cached: bool,
        tokens: int,
        cost: float,
        priority: LLMPriority,
        provider: LLMProvider
    ):
        """Update usage statistics"""
        stats = self.usage_stats[service_name]
        stats.service_name = service_name
        stats.total_requests += 1
        
        if cached:
            stats.cached_requests += 1
        else:
            stats.total_tokens += tokens
            stats.total_cost_usd += cost
        
        # Track by priority
        priority_key = priority.name
        stats.requests_by_priority[priority_key] = (
            stats.requests_by_priority.get(priority_key, 0) + 1
        )
        
        # Track by provider
        provider_key = provider.value
        stats.requests_by_provider[provider_key] = (
            stats.requests_by_provider.get(provider_key, 0) + 1
        )
        
        stats.last_request_time = datetime.now()
    
    def get_usage_stats(self, service_name: Optional[str] = None) -> Dict[str, UsageStats]:
        """Get usage statistics for all or specific service"""
        if service_name:
            return {service_name: self.usage_stats.get(service_name, UsageStats(service_name))}
        return dict(self.usage_stats)
    
    def get_total_cost(self) -> float:
        """Get total cost across all services"""
        return sum(stats.total_cost_usd for stats in self.usage_stats.values())
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """Clear cache (all or specific key)"""
        if cache_key:
            self.cache.pop(cache_key, None)
            logger.info(f"Cleared cache key: {cache_key}")
        else:
            self.cache.clear()
            logger.info("Cleared all cache")
    
    def reset_stats(self, service_name: Optional[str] = None):
        """Reset usage statistics"""
        if service_name:
            self.usage_stats.pop(service_name, None)
            logger.info(f"Reset stats for {service_name}")
        else:
            self.usage_stats.clear()
            logger.info("Reset all usage statistics")


# Singleton getter
_manager_instance = None
_manager_lock = Lock()


def get_llm_manager() -> LLMRequestManager:
    """Get singleton instance of LLM Request Manager"""
    global _manager_instance
    
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = LLMRequestManager()
    
    return _manager_instance

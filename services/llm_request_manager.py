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

# Task complexity classification for hybrid routing
COMPLEX_TASK_KEYWORDS = [
    'analyze multiple', 'multi-symbol', 'portfolio', 'comprehensive',
    'detailed analysis', 'risk assessment', 'pattern recognition',
    'json format', 'structured output', 'trading signal',
    'entry point', 'exit strategy', 'position sizing',
    'correlat', 'backtest', 'scenario', 'compare'
]

SIMPLE_TASK_KEYWORDS = [
    'sentiment', 'bullish or bearish', 'yes or no', 'confirm',
    'quick check', 'simple', 'brief', 'summarize in one',
    'thumbs up', 'rating', 'score 1-10'
]


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
            LLMProvider.GROQ: os.getenv("GROQ_API_KEY"),
            LLMProvider.LOCAL: "local",  # Always available if Ollama running
        }
        
        # Initialize local Ollama client for hybrid routing
        self.ollama_client = None
        self._init_local_llm()
        
        # Request counter for IDs
        self._request_counter = 0
        self._counter_lock = Lock()
        
        # Hybrid mode configuration
        self.hybrid_mode = os.getenv("LLM_HYBRID_MODE", "true").lower() == "true"
        
        logger.info(f"LLM Request Manager initialized (hybrid_mode={self.hybrid_mode})")
    
    def _init_local_llm(self):
        """Initialize local Ollama client if available"""
        try:
            from .ollama_client import create_ollama_client
            self.ollama_client = create_ollama_client()
            if self.ollama_client._check_ollama_health():
                logger.info("🤖 Local Ollama LLM ready for hybrid routing")
            else:
                self.ollama_client = None
                logger.warning("⚠️ Local Ollama not healthy, hybrid routing will use cloud only")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize local Ollama: {e}")
            self.ollama_client = None
    
    def _classify_task_complexity(self, prompt: str) -> str:
        """
        Classify task complexity for hybrid routing.
        
        Returns:
            'complex' - Route to Groq (better reasoning)
            'simple' - Route to local Ollama (faster, free)
        """
        prompt_lower = prompt.lower()
        
        # Check for complex task indicators
        complex_score = sum(1 for kw in COMPLEX_TASK_KEYWORDS if kw in prompt_lower)
        simple_score = sum(1 for kw in SIMPLE_TASK_KEYWORDS if kw in prompt_lower)
        
        # Length-based heuristic: longer prompts usually need more reasoning
        if len(prompt) > 2000:
            complex_score += 2
        elif len(prompt) < 300:
            simple_score += 1
        
        # JSON output requests typically need better models
        if 'json' in prompt_lower or '{' in prompt:
            complex_score += 1
        
        if complex_score > simple_score:
            return 'complex'
        return 'simple'
    
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
            "groq": RateLimitConfig(
                max_requests_per_minute=30,  # Groq free tier limit
                max_concurrent_requests=5
            ),
            "local": RateLimitConfig(
                max_requests_per_minute=1000,  # Local has no external rate limit
                max_concurrent_requests=2  # But limit concurrent for GPU memory
            ),
        }
        
        # Check if hybrid mode should skip OpenRouter
        use_hybrid = os.getenv("LLM_HYBRID_MODE", "true").lower() == "true"
        
        if use_hybrid:
            # Hybrid mode: LOCAL primary, Groq for complex tasks
            primary = LLMProvider.LOCAL
            fallbacks = [LLMProvider.GROQ, LLMProvider.OPENAI, LLMProvider.CLAUDE]
        else:
            # Standard mode: OpenRouter primary
            primary = LLMProvider.OPENROUTER
            fallbacks = [LLMProvider.GROQ, LLMProvider.CLAUDE, LLMProvider.OPENAI]
        
        return LLMManagerConfig(
            primary_provider=primary,
            fallback_providers=fallbacks,
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
    
    def hybrid_request(
        self,
        prompt: str,
        service_name: str,
        force_provider: Optional[str] = None,
        cache_key: Optional[str] = None,
        ttl: int = 300,
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Intelligent hybrid request that routes based on task complexity.
        
        - Simple tasks → Local Ollama (fast, free)
        - Complex tasks → Groq (better reasoning)
        
        Args:
            prompt: The prompt to send
            service_name: Name of calling service
            force_provider: Force 'local' or 'groq' (bypasses auto-routing)
            cache_key: Optional cache key
            ttl: Cache TTL in seconds
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            metadata: Additional metadata
        
        Returns:
            Response text or None if failed
        """
        # Determine provider based on task complexity
        if force_provider:
            if force_provider.lower() == 'local':
                providers = [LLMProvider.LOCAL, LLMProvider.GROQ]
            elif force_provider.lower() == 'groq':
                providers = [LLMProvider.GROQ, LLMProvider.LOCAL]
            else:
                providers = [LLMProvider.LOCAL, LLMProvider.GROQ]
        else:
            complexity = self._classify_task_complexity(prompt)
            if complexity == 'complex':
                # Complex: Try Groq first (better reasoning), local fallback
                providers = [LLMProvider.GROQ, LLMProvider.LOCAL]
                logger.info(f"🧠 Hybrid routing: COMPLEX task → Groq (with local fallback)")
            else:
                # Simple: Try local first (faster, free), Groq fallback
                providers = [LLMProvider.LOCAL, LLMProvider.GROQ]
                logger.info(f"⚡ Hybrid routing: SIMPLE task → Local Ollama (with Groq fallback)")
        
        # Check cache first
        if cache_key and self.config.enable_caching:
            cached = self._get_cached_response(cache_key)
            if cached:
                logger.info(f"Cache hit for {service_name}")
                return cached.response
        
        # Generate cache key if not provided
        if cache_key is None and self.config.enable_caching:
            cache_key = self._generate_cache_key(prompt, "hybrid")
        
        # Generate request ID
        with self._counter_lock:
            self._request_counter += 1
            request_id = f"{service_name}_hybrid_{self._request_counter}_{int(time.time())}"
        
        # Create request object
        llm_request = LLMRequest(
            prompt=prompt,
            priority=LLMPriority.MEDIUM,
            service_name=service_name,
            cache_key=cache_key,
            ttl_seconds=ttl,
            model="hybrid",  # Will be set per-provider
            max_tokens=max_tokens or 2000,
            temperature=temperature,
            metadata=metadata or {},
            request_id=request_id
        )
        
        # Try providers in order
        for provider in providers:
            # Check if provider is available
            if provider == LLMProvider.LOCAL and not self.ollama_client:
                logger.debug("Local Ollama not available, skipping")
                continue
            if provider == LLMProvider.GROQ and not self.api_keys.get(LLMProvider.GROQ):
                logger.debug("No Groq API key, skipping")
                continue
            
            try:
                self._wait_for_rate_limit(provider)
                response = self._call_provider(provider, llm_request)
                
                # Cache if enabled
                if cache_key and self.config.enable_caching:
                    self._cache_response(cache_key, response, ttl)
                
                # Update stats
                self._update_usage_stats(
                    service_name=service_name,
                    cached=False,
                    tokens=response.tokens_used,
                    cost=response.cost_usd,
                    priority=LLMPriority.MEDIUM,
                    provider=provider
                )
                
                logger.info(
                    f"✅ Hybrid request completed via {provider.value} "
                    f"(${response.cost_usd:.4f}, {response.tokens_used} tokens)"
                )
                return response.content
                
            except Exception as e:
                logger.warning(f"Hybrid routing: {provider.value} failed: {e}")
                continue
        
        logger.error(f"All hybrid providers failed for {service_name}")
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
        if provider == LLMProvider.LOCAL:
            return self._call_local(request)
        elif provider == LLMProvider.OPENROUTER:
            return self._call_openrouter(request)
        elif provider == LLMProvider.CLAUDE:
            return self._call_claude(request)
        elif provider == LLMProvider.OPENAI:
            return self._call_openai(request)
        elif provider == LLMProvider.GROQ:
            return self._call_groq(request)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _call_local(self, request: LLMRequest) -> LLMResponse:
        """
        Call local Ollama LLM - free, private, GPU-accelerated
        
        Uses the existing OllamaClient for consistency.
        """
        if not self.ollama_client:
            raise RuntimeError("Local Ollama client not available")
        
        start_time = time.time()
        
        # Use Ollama client's generate method
        content = self.ollama_client.generate_trading_signal(
            prompt=request.prompt,
            analysis_type=request.metadata.get('analysis_type', 'trading_signals'),
            temperature=request.temperature
        )
        
        if not content:
            raise RuntimeError("Local Ollama returned empty response")
        
        generation_time = time.time() - start_time
        
        # Estimate tokens (rough approximation: ~4 chars per token)
        prompt_tokens = len(request.prompt) // 4
        output_tokens = len(content) // 4
        total_tokens = prompt_tokens + output_tokens
        
        return LLMResponse(
            content=content,
            request_id=request.request_id or "unknown",
            provider=LLMProvider.LOCAL,
            model=self.ollama_client.config.model,
            tokens_used=total_tokens,
            cost_usd=0.0,  # Local is free!
            metadata={
                "generation_time": generation_time,
                "local_model": self.ollama_client.config.model
            }
        )
    
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
    
    def _call_groq(self, request: LLMRequest) -> LLMResponse:
        """
        Call Groq API - Ultra-fast inference with Llama models
        
        Groq provides extremely fast inference speeds using custom LPU hardware.
        Supported models: llama-3.1-8b-instant, llama-3.1-70b-versatile, mixtral-8x7b-32768
        """
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_keys[LLMProvider.GROQ]}",
            "Content-Type": "application/json"
        }
        
        # Map model name - strip groq/ prefix if present
        model = request.model or "llama-3.1-8b-instant"
        if model.startswith("groq/"):
            model = model.replace("groq/", "")
        
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
        
        # Use model with groq/ prefix for cost calculation
        cost_model = f"groq/{model}" if not model.startswith("groq/") else model
        cost = calculate_cost(cost_model, input_tokens, output_tokens)
        
        return LLMResponse(
            content=content,
            request_id=request.request_id or "unknown",
            provider=LLMProvider.GROQ,
            model=model,
            tokens_used=total_tokens,
            cost_usd=cost,
            metadata={"usage": usage, "groq_inference": True}
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
    
    def record_external_usage(
        self,
        service_name: str,
        provider: str = "local_ollama",
        tokens: int = 0,
        cost: float = 0.0,
        cached: bool = False,
        priority: str = "MEDIUM"
    ):
        """
        Record usage from external LLM calls (e.g., HybridLLMAnalyzer direct calls)
        This allows tracking all LLM usage in one place for the usage dashboard.
        
        Args:
            service_name: Name of the service making the request
            provider: Provider name (e.g., 'local_ollama', 'openrouter', 'groq')
            tokens: Number of tokens used
            cost: Cost in USD
            cached: Whether this was a cached response
            priority: Priority level string
        """
        # Map string priority to enum
        priority_map = {
            'CRITICAL': LLMPriority.CRITICAL,
            'HIGH': LLMPriority.HIGH,
            'MEDIUM': LLMPriority.MEDIUM,
            'LOW': LLMPriority.LOW
        }
        priority_enum = priority_map.get(priority.upper(), LLMPriority.MEDIUM)
        
        # Map string provider to enum (use OPENROUTER as fallback for tracking)
        provider_map = {
            'local_ollama': LLMProvider.OPENROUTER,  # Track as separate category
            'openrouter': LLMProvider.OPENROUTER,
            'groq': LLMProvider.GROQ,
            'claude': LLMProvider.CLAUDE,
            'openai': LLMProvider.OPENAI
        }
        provider_enum = provider_map.get(provider.lower(), LLMProvider.OPENROUTER)
        
        # For local ollama, track with a custom provider string
        stats = self.usage_stats[service_name]
        stats.service_name = service_name
        stats.total_requests += 1
        
        if cached:
            stats.cached_requests += 1
        else:
            stats.total_tokens += tokens
            stats.total_cost_usd += cost
        
        # Track by priority
        stats.requests_by_priority[priority] = (
            stats.requests_by_priority.get(priority, 0) + 1
        )
        
        # Track by provider (use actual string for better differentiation)
        stats.requests_by_provider[provider] = (
            stats.requests_by_provider.get(provider, 0) + 1
        )
        
        stats.last_request_time = datetime.now()
        logger.debug(f"Recorded external usage: {service_name} via {provider} ({tokens} tokens, ${cost:.4f})")
    
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

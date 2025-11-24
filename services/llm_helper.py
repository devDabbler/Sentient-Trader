"""
LLM Helper - Easy integration adapter for existing services
Provides simple wrapper functions to migrate from direct API calls to LLM Request Manager

IMPORTANT: This module uses LAZY IMPORTS to prevent hangs when running as Windows Service/Task Scheduler.
DO NOT add module-level imports of llm_request_manager - it must be imported inside functions only.
"""
import logging
from typing import Optional, Dict, Any

# DO NOT IMPORT llm_request_manager at module level - causes Task Scheduler hangs
# from services.llm_request_manager import get_llm_manager  # MOVED TO FUNCTION LEVEL


logger = logging.getLogger(__name__)


class LLMHelper:
    """
    Helper class for easy LLM integration
    
    Usage:
        # Old code:
        response = openrouter_client.chat(prompt)
        
        # New code:
        from services.llm_helper import llm_request
        response = llm_request(prompt, service_name="my_service")
    """
    
    def __init__(self, service_name: str, default_priority: str = "MEDIUM"):
        self.service_name = service_name
        self.default_priority = default_priority
        self._manager = None  # Lazy initialization
    
    @property
    def manager(self):
        """Lazy-load the LLM manager only when actually needed"""
        if self._manager is None:
            from services.llm_request_manager import get_llm_manager
            self._manager = get_llm_manager()
        return self._manager
    
    def request(
        self,
        prompt: str,
        priority: Optional[str] = None,
        cache_key: Optional[str] = None,
        ttl: int = 300,
        model: Optional[str] = None,
        temperature: float = 0.7,
        blocking: bool = True
    ) -> Optional[str]:
        """
        Make an LLM request using the centralized manager
        
        Args:
            prompt: The prompt to send
            priority: CRITICAL, HIGH, MEDIUM, or LOW (default: service default)
            cache_key: Optional cache key
            ttl: Cache TTL in seconds
            model: Model to use
            temperature: Sampling temperature
            blocking: Wait for response
        
        Returns:
            Response text
        """
        return self.manager.request(
            prompt=prompt,
            service_name=self.service_name,
            priority=priority or self.default_priority,
            cache_key=cache_key,
            ttl=ttl,
            model=model,
            temperature=temperature,
            blocking=blocking
        )
    
    def critical_request(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a CRITICAL priority request (trade execution, etc.)"""
        kwargs['priority'] = "CRITICAL"
        return self.request(prompt, **kwargs)
    
    def high_request(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a HIGH priority request (position monitoring, etc.)"""
        kwargs['priority'] = "HIGH"
        return self.request(prompt, **kwargs)
    
    def medium_request(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a MEDIUM priority request (opportunity scanning, etc.)"""
        kwargs['priority'] = "MEDIUM"
        return self.request(prompt, **kwargs)
    
    def low_request(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a LOW priority request (informational, etc.)"""
        kwargs['priority'] = "LOW"
        return self.request(prompt, **kwargs)
    
    def cached_request(
        self,
        prompt: str,
        cache_key: str,
        ttl: int = 300,
        **kwargs
    ) -> Optional[str]:
        """Make a request with explicit caching"""
        return self.request(prompt, cache_key=cache_key, ttl=ttl, **kwargs)


# Convenience functions for quick integration

def llm_request(
    prompt: str,
    service_name: str,
    priority: str = "MEDIUM",
    cache_key: Optional[str] = None,
    ttl: int = 300,
    model: Optional[str] = None,
    temperature: float = 0.7
) -> Optional[str]:
    """
    Quick LLM request function
    
    Example:
        from services.llm_helper import llm_request
        
        response = llm_request(
            "Analyze this stock: AAPL",
            service_name="stock_analyzer",
            priority="HIGH"
        )
    """
    from services.llm_request_manager import get_llm_manager
    manager = get_llm_manager()
    return manager.request(
        prompt=prompt,
        service_name=service_name,
        priority=priority,
        cache_key=cache_key,
        ttl=ttl,
        model=model,
        temperature=temperature
    )


def llm_cached_request(
    prompt: str,
    service_name: str,
    cache_key: str,
    ttl: int = 300,
    priority: str = "MEDIUM"
) -> Optional[str]:
    """
    Quick cached LLM request
    
    Example:
        from services.llm_helper import llm_cached_request
        
        # Auto-cached with key
        response = llm_cached_request(
            f"Analyze {symbol}",
            service_name="analyzer",
            cache_key=f"analysis_{symbol}",
            ttl=300
        )
    """
    from services.llm_request_manager import get_llm_manager
    manager = get_llm_manager()
    return manager.request(
        prompt=prompt,
        service_name=service_name,
        priority=priority,
        cache_key=cache_key,
        ttl=ttl
    )


def get_llm_helper(service_name: str, default_priority: str = "MEDIUM") -> LLMHelper:
    """
    Get an LLM helper instance for a service
    
    Example:
        from services.llm_helper import get_llm_helper
        
        # In your service __init__:
        self.llm = get_llm_helper("dex_hunter", default_priority="HIGH")
        
        # Later in your code:
        response = self.llm.request("Analyze this token...")
        response = self.llm.critical_request("Validate this trade...")
        response = self.llm.cached_request("Get sentiment...", cache_key="sentiment_xyz")
    """
    return LLMHelper(service_name, default_priority)


# Migration helpers

def migrate_openrouter_call(
    service_name: str,
    prompt: str,
    priority: str = "MEDIUM",
    cache: bool = True,
    cache_ttl: int = 300
) -> Optional[str]:
    """
    Drop-in replacement for OpenRouter calls
    
    Before:
        response = openrouter_client.chat(prompt)
    
    After:
        from services.llm_helper import migrate_openrouter_call
        response = migrate_openrouter_call("my_service", prompt)
    """
    cache_key = None
    if cache:
        # Auto-generate cache key from prompt
        import hashlib
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
    
    return llm_request(
        prompt=prompt,
        service_name=service_name,
        priority=priority,
        cache_key=cache_key,
        ttl=cache_ttl
    )


class LLMServiceMixin:
    """
    Mixin class for services to easily add LLM capabilities
    
    Usage:
        class MyService(LLMServiceMixin):
            def __init__(self):
                super().__init__()
                self._init_llm("my_service", default_priority="HIGH")
            
            def analyze(self, data):
                response = self.llm_request("Analyze this...")
                return response
    """
    
    def _init_llm(self, service_name: str, default_priority: str = "MEDIUM"):
        """Initialize LLM helper for this service"""
        self.llm_helper = get_llm_helper(service_name, default_priority)
    
    def llm_request(self, prompt: str, **kwargs) -> Optional[str]:
        """Make an LLM request"""
        if not hasattr(self, 'llm_helper'):
            raise RuntimeError("LLM helper not initialized. Call _init_llm() first.")
        return self.llm_helper.request(prompt, **kwargs)
    
    def llm_critical(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a CRITICAL priority request"""
        if not hasattr(self, 'llm_helper'):
            raise RuntimeError("LLM helper not initialized. Call _init_llm() first.")
        return self.llm_helper.critical_request(prompt, **kwargs)
    
    def llm_high(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a HIGH priority request"""
        if not hasattr(self, 'llm_helper'):
            raise RuntimeError("LLM helper not initialized. Call _init_llm() first.")
        return self.llm_helper.high_request(prompt, **kwargs)
    
    def llm_medium(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a MEDIUM priority request"""
        if not hasattr(self, 'llm_helper'):
            raise RuntimeError("LLM helper not initialized. Call _init_llm() first.")
        return self.llm_helper.medium_request(prompt, **kwargs)
    
    def llm_low(self, prompt: str, **kwargs) -> Optional[str]:
        """Make a LOW priority request"""
        if not hasattr(self, 'llm_helper'):
            raise RuntimeError("LLM helper not initialized. Call _init_llm() first.")
        return self.llm_helper.low_request(prompt, **kwargs)
    
    def llm_cached(self, prompt: str, cache_key: str, ttl: int = 300, **kwargs) -> Optional[str]:
        """Make a cached request"""
        if not hasattr(self, 'llm_helper'):
            raise RuntimeError("LLM helper not initialized. Call _init_llm() first.")
        return self.llm_helper.cached_request(prompt, cache_key, ttl, **kwargs)

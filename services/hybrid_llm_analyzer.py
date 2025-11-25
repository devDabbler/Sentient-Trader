"""
Hybrid LLM Analyzer for Sentient Trader
Supports both local Ollama models (GPU-accelerated) and cloud OpenRouter models
Automatically switches between local and cloud based on availability and performance

NOTE: For new code, use the centralized LLM Request Manager instead:
    from services.llm_helper import get_llm_helper
    llm = get_llm_helper("my_service", default_priority="MEDIUM")
    response = llm.request("Your prompt here")

This class remains for backward compatibility with existing code.
"""

import os
import json
import time
from loguru import logger
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import requests
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Ollama client for local GPU-accelerated LLM support
try:
    from .ollama_client import create_ollama_client, OllamaClient, OllamaConfig
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not available - will use only OpenRouter")

# Import LLM request manager for usage tracking
try:
    from .llm_request_manager import get_llm_manager
    LLM_MANAGER_AVAILABLE = True
except ImportError:
    LLM_MANAGER_AVAILABLE = False


@dataclass
class LLMResponse:
    """Standardized LLM response across different providers"""
    content: str
    model_used: str
    provider: str
    generation_time: float
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None


@dataclass
class LLMConfig:
    """Configuration for hybrid LLM usage"""
    prefer_local: bool = os.getenv('PREFER_LOCAL_LLM', 'true').lower() == 'true'  # Prefer local Ollama over cloud
    local_model: str = "qwen2.5:7b"
    # Use env var or default to a reliable free model
    cloud_model: str = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-small-3.1-24b-instruct:free')
    fallback_enabled: bool = True
    max_local_timeout: int = 120  # Increased for longer prompts
    max_cloud_timeout: int = 30


class HybridLLMAnalyzer:
    """
    Intelligent LLM analyzer that automatically chooses between:
    1. Local Ollama models (GPU-accelerated, private, free)
    2. Cloud OpenRouter models (various providers, fast, costs money)
    
    Optimized for trading analysis with automatic fallback and performance monitoring.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize hybrid LLM analyzer"""
        self.config = config or LLMConfig()
        self.ollama_client: Optional[OllamaClient] = None
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')  # Groq fallback
        
        # Performance tracking
        self.local_performance_history = []
        self.cloud_performance_history = []
        self.groq_performance_history = []
        
        # Initialize local Ollama client if available
        if OLLAMA_AVAILABLE and self.config.prefer_local:
            try:
                self.ollama_client = create_ollama_client(self.config.local_model)
                if self.ollama_client._check_ollama_health():
                    logger.success(f"🤖 Local Ollama ready: {self.config.local_model}")
                else:
                    logger.warning("⚠️ Local Ollama not healthy, will use cloud fallback")
                    self.ollama_client = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize Ollama: {e}")
                self.ollama_client = None
        
        # Validate cloud backup
        if not self.ollama_client or self.config.fallback_enabled:
            if not self.openrouter_api_key:
                logger.warning("⚠️ No OpenRouter API key found - cloud fallback unavailable")
            else:
                logger.info(f"☁️ Cloud backup ready: {self.config.cloud_model}")
    
    def analyze_with_llm(
        self,
        prompt: str,
        analysis_type: str = 'trading_signals',
        force_provider: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Main analysis method with intelligent provider selection
        
        Args:
            prompt: Analysis prompt
            analysis_type: Type of analysis (trading_signals, market_analysis, etc.)
            force_provider: Force specific provider ('local' or 'cloud')
            temperature: Override default temperature
            
        Returns:
            LLM response string or None if all methods fail
        """
        start_time = time.time()
        
        # Determine best provider
        if force_provider == 'local':
            providers_to_try = ['local']
        elif force_provider == 'cloud':
            providers_to_try = ['cloud']
        else:
            providers_to_try = self._get_optimal_provider_order()
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                if provider == 'local' and self.ollama_client:
                    logger.info(f"🤖 Using local Ollama: {self.config.local_model}")
                    response = self.ollama_client.generate_trading_signal(
                        prompt=prompt,
                        analysis_type=analysis_type,
                        temperature=temperature
                    )
                    
                    if response:
                        generation_time = time.time() - start_time
                        self._record_performance('local', generation_time, success=True)
                        # Track in LLM manager for usage dashboard
                        if LLM_MANAGER_AVAILABLE:
                            try:
                                get_llm_manager().record_external_usage(
                                    service_name=f"hybrid_{analysis_type}",
                                    provider="local_ollama",
                                    tokens=len(response) // 4,  # Rough estimate
                                    cost=0.0  # Local is free
                                )
                            except Exception:
                                pass
                        return response
                    else:
                        logger.warning(f"⚠️ Local Ollama returned no response, trying fallback...")
                        self._record_performance('local', time.time() - start_time, success=False)
                        
                elif provider == 'cloud' and self.openrouter_api_key:
                    logger.info(f"☁️ Using cloud OpenRouter: {self.config.cloud_model}")
                    response = self._call_openrouter(prompt, temperature)
                    
                    if response:
                        generation_time = time.time() - start_time
                        self._record_performance('cloud', generation_time, success=True)
                        # Track in LLM manager for usage dashboard
                        if LLM_MANAGER_AVAILABLE:
                            try:
                                tokens_est = len(response) // 4
                                # Rough cost estimate for free tier models
                                cost_est = 0.0 if 'free' in self.config.cloud_model else tokens_est * 0.00001
                                get_llm_manager().record_external_usage(
                                    service_name=f"hybrid_{analysis_type}",
                                    provider="openrouter",
                                    tokens=tokens_est,
                                    cost=cost_est
                                )
                            except Exception:
                                pass
                        return response
                    else:
                        self._record_performance('cloud', time.time() - start_time, success=False)
                        
                elif provider == 'groq' and self.groq_api_key:
                    logger.info(f"⚡ Using Groq (fast fallback): llama-3.1-8b-instant")
                    response = self._call_groq(prompt, temperature)
                    
                    if response:
                        generation_time = time.time() - start_time
                        self._record_performance('groq', generation_time, success=True)
                        # Track in LLM manager for usage dashboard
                        if LLM_MANAGER_AVAILABLE:
                            try:
                                tokens_est = len(response) // 4
                                get_llm_manager().record_external_usage(
                                    service_name=f"hybrid_{analysis_type}",
                                    provider="groq",
                                    tokens=tokens_est,
                                    cost=0.0  # Groq free tier
                                )
                            except Exception:
                                pass
                        return response
                    else:
                        self._record_performance('groq', time.time() - start_time, success=False)
                        
            except Exception as e:
                last_error = e
                logger.error(f"❌ {provider} provider failed: {e}")
                self._record_performance(provider, time.time() - start_time, success=False)
                continue
        
        # All providers failed
        total_time = time.time() - start_time
        pass  # logger.error(f"❌ All LLM providers failed after {}s {total_time:.1f}")
        if last_error:
            logger.error(f"Last error: {last_error}")
        
        return None
    
    def _get_optimal_provider_order(self) -> List[str]:
        """Determine optimal order to try providers based on config and performance"""
        providers = []
        
        # Check recent performance
        local_avg_time = self._get_average_performance('local')
        cloud_avg_time = self._get_average_performance('cloud')
        
        if self.config.prefer_local and self.ollama_client:
            # Prefer local unless it's significantly slower
            if local_avg_time is None or cloud_avg_time is None or local_avg_time <= cloud_avg_time * 2:
                providers.extend(['local', 'cloud'])
            else:
                providers.extend(['cloud', 'local'])
        else:
            # Prefer cloud for speed
            if self.openrouter_api_key:
                providers.append('cloud')
            if self.ollama_client:
                providers.append('local')
        
        # Always add Groq as final fallback if available
        if self.groq_api_key and 'groq' not in providers:
            providers.append('groq')
        
        return providers
    
    def _call_openrouter(self, prompt: str, temperature: Optional[float] = None) -> Optional[str]:
        """Call OpenRouter API (cloud fallback)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/sentient-trader",
                "X-Title": "Sentient Trader"
            }
            
            payload = {
                "model": self.config.cloud_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert trading analyst. Provide precise, actionable analysis in valid JSON format when requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature or 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.max_cloud_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            return None
    
    def _call_groq(self, prompt: str, temperature: Optional[float] = None) -> Optional[str]:
        """Call Groq API - ultra-fast inference fallback"""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-8b-instant",  # Fast default
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert trading analyst. Provide precise, actionable analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature or 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['choices'][0]['message']['content']
            else:
                logger.error(f"Groq API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Groq: {e}")
            return None
    
    def _record_performance(self, provider: str, generation_time: float, success: bool):
        """Record performance metrics for provider optimization"""
        performance_data = {
            'timestamp': time.time(),
            'generation_time': generation_time,
            'success': success
        }
        
        if provider == 'local':
            self.local_performance_history.append(performance_data)
            # Keep only last 20 entries
            if len(self.local_performance_history) > 20:
                self.local_performance_history.pop(0)
        elif provider == 'cloud':
            self.cloud_performance_history.append(performance_data)
            if len(self.cloud_performance_history) > 20:
                self.cloud_performance_history.pop(0)
        elif provider == 'groq':
            self.groq_performance_history.append(performance_data)
            if len(self.groq_performance_history) > 20:
                self.groq_performance_history.pop(0)
    
    def _get_average_performance(self, provider: str) -> Optional[float]:
        """Get average successful response time for provider"""
        if provider == 'local':
            history = self.local_performance_history
        elif provider == 'cloud':
            history = self.cloud_performance_history
        elif provider == 'groq':
            history = self.groq_performance_history
        else:
            return None
        
        successful_times = [
            entry['generation_time'] for entry in history 
            if entry['success']
        ]
        
        if successful_times:
            return sum(successful_times) / len(successful_times)
        return None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for both providers"""
        stats = {
            'local': {
                'available': self.ollama_client is not None,
                'model': self.config.local_model,
                'total_requests': len(self.local_performance_history),
                'successful_requests': len([e for e in self.local_performance_history if e['success']]),
                'avg_response_time': self._get_average_performance('local')
            },
            'cloud': {
                'available': self.openrouter_api_key is not None,
                'model': self.config.cloud_model,
                'total_requests': len(self.cloud_performance_history),
                'successful_requests': len([e for e in self.cloud_performance_history if e['success']]),
                'avg_response_time': self._get_average_performance('cloud')
            }
        }
        return stats
    
    def switch_local_model(self, model_name: str) -> bool:
        """Switch to a different local Ollama model"""
        if not self.ollama_client:
            logger.error("No Ollama client available")
            return False
        
        if self.ollama_client.switch_model(model_name):
            self.config.local_model = model_name
            logger.success(f"🔄 Switched local model to: {model_name}")
            return True
        return False
    
    def switch_cloud_model(self, model_name: str) -> bool:
        """Switch to a different cloud model"""
        self.config.cloud_model = model_name
        logger.success(f"🔄 Switched cloud model to: {model_name}")
        return True
    
    def list_available_models(self) -> Dict:
        """List available models for both providers"""
        models = {
            'local': [],
            'cloud': [
                "google/gemini-2.0-flash-exp:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemini-2.5-flash",
                "openai/gpt-4o",
                "anthropic/claude-3-haiku"
            ]
        }
        
        if self.ollama_client:
            models['local'] = self.ollama_client.list_available_models()
        
        return models
    
    async def analyze_batch_async(
        self, 
        prompts: List[str], 
        analysis_type: str = 'trading_signals',
        max_concurrent: int = 3
    ) -> List[Optional[str]]:
        """Async batch processing for multiple analyses"""
        if self.ollama_client:
            # Use Ollama's built-in batch processing
            prompt_tuples = [(prompt, analysis_type) for prompt in prompts]
            return self.ollama_client.batch_analyze(prompt_tuples, max_concurrent)
        else:
            # Fallback to sequential cloud processing
            results = []
            for prompt in prompts:
                result = self.analyze_with_llm(prompt, analysis_type)
                results.append(result)
            return results


# Factory functions for easy integration with existing Sentient Trader code
def create_hybrid_llm_analyzer(
    prefer_local: bool = True,
    local_model: str = "qwen2.5:7b",
    cloud_model: Optional[str] = None  # Use env var default if not specified
) -> HybridLLMAnalyzer:
    """Create optimized hybrid LLM analyzer for trading"""
    # Use env var if cloud_model not explicitly provided
    if cloud_model is None:
        cloud_model = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-small-3.1-24b-instruct:free')
    
    config = LLMConfig(
        prefer_local=prefer_local,
        local_model=local_model,
        cloud_model=cloud_model,
        fallback_enabled=True
    )
    return HybridLLMAnalyzer(config)


def get_best_trading_analyzer() -> HybridLLMAnalyzer:
    """Get the best available LLM analyzer for trading with smart defaults"""
    # Check what's available and configure accordingly
    local_available = OLLAMA_AVAILABLE
    cloud_available = bool(os.getenv('OPENROUTER_API_KEY'))
    
    # Check env var for preference (defaults to True if not set)
    prefer_local_env = os.getenv('PREFER_LOCAL_LLM', 'true').lower() == 'true'
    
    if local_available and cloud_available:
        # Both available - use env var preference
        if prefer_local_env:
            logger.info("🚀 Both local and cloud LLM available - preferring local (PREFER_LOCAL_LLM=true)")
        else:
            logger.info("☁️ Both local and cloud LLM available - preferring cloud (PREFER_LOCAL_LLM=false)")
        return create_hybrid_llm_analyzer(prefer_local=prefer_local_env)
    elif local_available:
        # Only local available
        logger.info("🤖 Only local LLM available")
        return create_hybrid_llm_analyzer(prefer_local=True)
    elif cloud_available:
        # Only cloud available
        logger.info("☁️ Only cloud LLM available")
        return create_hybrid_llm_analyzer(prefer_local=False)
    else:
        # Nothing available
        logger.error("❌ No LLM providers available!")
        raise RuntimeError("No LLM providers available. Please set up Ollama or OpenRouter API key.")


if __name__ == "__main__":
    # Test the hybrid analyzer
    analyzer = get_best_trading_analyzer()
    
    test_prompt = """
    Analyze AAPL with the following data:
    Price: $150.25 (+2.1%)
    RSI: 65, MACD: Bullish crossover
    News: Strong iPhone sales
    
    Provide trading signal in JSON format.
    """
    
    print("Testing hybrid LLM analyzer...")
    result = analyzer.analyze_with_llm(test_prompt, 'trading_signals')
    
    if result:
        print("✅ Test successful!")
        print(f"Response: {result[:200]}...")
        
        # Show performance stats
        stats = analyzer.get_performance_stats()
        print("\nPerformance Stats:")
        for provider, data in stats.items():
            print(f"{provider}: {data}")
    else:
        print("❌ Test failed!")

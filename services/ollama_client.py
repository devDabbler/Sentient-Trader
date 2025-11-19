"""
Ollama Local LLM Client for Sentient Trader
High-performance local LLM integration using GPU-accelerated Ollama models
"""

from loguru import logger
import requests
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class OllamaConfig:
    """Configuration for Ollama client"""
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5-coder:32b"  # Default to best trading model
    timeout: int = 120  # Increased for longer prompts
    temperature: float = 0.3  # Lower for more consistent trading analysis
    max_tokens: int = 2000
    stream: bool = False
    
    
class OllamaClient:
    """
    High-performance Ollama client optimized for trading analysis.
    Provides both sync and async methods for maximum performance.
    """
    
    def __init__(self, config: Optional[OllamaConfig] = None):
        """Initialize Ollama client with configuration"""
        self.config = config or OllamaConfig()
        
        # Override model from environment if set
        env_model = os.getenv('OLLAMA_MODEL', os.getenv('AI_TRADING_MODEL'))
        if env_model and 'ollama' in env_model.lower():
            # Extract model name from ollama format (e.g., "ollama/qwen2.5-coder:32b" -> "qwen2.5-coder:32b")
            self.config.model = env_model.split('/', 1)[-1] if '/' in env_model else env_model
        
        # Trading-specific system prompts for different use cases
        self.system_prompts = {
            'trading_signals': """You are an expert day trader and quantitative analyst. Analyze market data and provide precise trading signals with specific entry/exit points. Focus on risk management and probability-based decisions. Always respond in valid JSON format.""",
            
            'market_analysis': """You are a professional market analyst specializing in real-time market conditions. Provide comprehensive analysis considering technical indicators, market sentiment, and risk factors. Be concise and actionable.""",
            
            'risk_assessment': """You are a risk management expert for trading operations. Evaluate position sizing, stop-loss levels, and portfolio exposure. Prioritize capital preservation and consistent returns over aggressive growth.""",
            
            'crypto_analysis': """You are a cryptocurrency trading specialist. Analyze crypto markets considering 24/7 trading, high volatility, social sentiment, and on-chain metrics. Adapt traditional TA for crypto market dynamics."""
        }
        
        logger.info(f"🤖 Ollama Client initialized with model: {self.config.model}")
        logger.info(f"🌐 Base URL: {self.config.base_url}")
        self._check_ollama_health()
    
    def _check_ollama_health(self) -> bool:
        """Check if Ollama service is running and model is available"""
        try:
            # Check service health
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.config.model in model_names:
                    logger.success(f"✅ Ollama ready with model: {self.config.model}")
                    return True
                else:
                    logger.warning(f"⚠️ Model {self.config.model} not found. Available: {model_names}")
                    if model_names:
                        # Use first available model as fallback
                        self.config.model = model_names[0]
                        logger.info(f"🔄 Using fallback model: {self.config.model}")
                        return True
                    else:
                        logger.error("❌ No Ollama models available")
                        return False
            else:
                logger.error(f"❌ Ollama service not responding: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"❌ Cannot connect to Ollama: {e}")
            logger.info("💡 Make sure Ollama is running: ollama serve")
            return False
    
    def generate_trading_signal(
        self,
        prompt: str,
        analysis_type: str = 'trading_signals',
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Generate trading signal using local Ollama model
        
        Args:
            prompt: Trading analysis prompt
            analysis_type: Type of analysis (trading_signals, market_analysis, etc.)
            temperature: Override default temperature
            
        Returns:
            LLM response string or None if error
        """
        try:
            system_prompt = self.system_prompts.get(analysis_type, self.system_prompts['trading_signals'])
            
            payload = {
                "model": self.config.model,
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": self.config.stream,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            logger.info(f"🧠 Generating {analysis_type} with {self.config.model}...")
            logger.debug(f"API URL: {self.config.base_url}/api/generate")
            logger.debug(f"Stream mode: {self.config.stream}")
            start_time = time.time()
            
            response = requests.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            logger.debug(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                generation_time = time.time() - start_time
                
                if generated_text:
                    logger.success(f"✅ Generated {len(generated_text)} chars in {generation_time:.1f}s")
                    return generated_text
                else:
                    logger.warning(f"⚠️ Ollama returned empty response in {generation_time:.1f}s")
                    logger.debug(f"Full result: {result}")
                    return None
            else:
                logger.error(f"❌ Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ Ollama request timeout after {self.config.timeout}s")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"❌ Cannot connect to Ollama at {self.config.base_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error generating trading signal: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    async def generate_trading_signal_async(
        self,
        prompt: str,
        analysis_type: str = 'trading_signals',
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Async version for non-blocking trading signal generation
        """
        try:
            system_prompt = self.system_prompts.get(analysis_type, self.system_prompts['trading_signals'])
            
            payload = {
                "model": self.config.model,
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": self.config.stream,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": self.config.max_tokens,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            logger.info(f"🧠 [Async] Generating {analysis_type} with {self.config.model}...")
            start_time = time.time()
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
                async with session.post(f"{self.config.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        generated_text = result.get('response', '')
                        generation_time = time.time() - start_time
                        
                        logger.success(f"✅ [Async] Generated {len(generated_text)} chars in {generation_time:.1f}s")
                        return generated_text
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ [Async] Ollama API error: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ [Async] Error generating trading signal: {e}")
            return None
    
    def batch_analyze(
        self,
        prompts: List[Tuple[str, str]],  # [(prompt, analysis_type), ...]
        max_concurrent: int = 3
    ) -> List[Optional[str]]:
        """
        Batch process multiple trading analyses
        
        Args:
            prompts: List of (prompt, analysis_type) tuples
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of responses in same order as prompts
        """
        async def batch_process():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single(prompt_data):
                prompt, analysis_type = prompt_data
                async with semaphore:
                    return await self.generate_trading_signal_async(prompt, analysis_type)
            
            tasks = [process_single(prompt_data) for prompt_data in prompts]
            return await asyncio.gather(*tasks)
        
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(batch_process())
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(batch_process())
            finally:
                loop.close()
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        try:
            response = requests.post(
                f"{self.config.base_url}/api/show",
                json={"name": self.config.model},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get model info: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def list_available_models(self) -> List[str]:
        """List all available Ollama models"""
        try:
            response = requests.get(f"{self.config.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            else:
                return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        available_models = self.list_available_models()
        if model_name in available_models:
            self.config.model = model_name
            logger.info(f"🔄 Switched to model: {model_name}")
            return True
        else:
            logger.error(f"❌ Model {model_name} not available. Available: {available_models}")
            return False


def create_ollama_client(model: Optional[str] = None) -> OllamaClient:
    """
    Factory function to create optimized Ollama client
    
    Args:
        model: Override default model
        
    Returns:
        Configured OllamaClient instance
    """
    config = OllamaConfig()
    if model:
        config.model = model
    
    return OllamaClient(config)


# Sentient Trader integration functions
def create_trading_signal_generator(model: str = "qwen2.5-coder:32b") -> OllamaClient:
    """Create Ollama client specifically optimized for trading signal generation"""
    config = OllamaConfig(
        model=model,
        temperature=0.2,  # Lower temperature for more consistent signals
        max_tokens=1500   # Enough for detailed trading signals
    )
    return OllamaClient(config)


def create_market_analyzer(model: str = "qwen2.5-coder:32b") -> OllamaClient:
    """Create Ollama client for comprehensive market analysis"""
    config = OllamaConfig(
        model=model,
        temperature=0.3,  # Slightly higher for more nuanced analysis
        max_tokens=2000   # More tokens for detailed analysis
    )
    return OllamaClient(config)


if __name__ == "__main__":
    # Test the Ollama client
    client = create_ollama_client()
    
    # Test trading signal generation
    test_prompt = """
    Analyze AAPL stock with the following data:
    Price: $150.25 (+2.1%)
    Volume: 85M (1.5x average)
    RSI: 65
    MACD: Bullish crossover
    News: New iPhone sales beating expectations
    
    Provide a trading signal in JSON format with entry, target, stop-loss, and reasoning.
    """
    
    result = client.generate_trading_signal(test_prompt)
    if result:
        print("✅ Ollama client test successful!")
        print(f"Response: {result[:200]}...")
    else:
        print("❌ Ollama client test failed!")

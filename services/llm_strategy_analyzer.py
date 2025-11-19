"""
LLM Strategy Analyzer for Option Alpha Bot Critique
Analyzes bot configurations and provides intelligent recommendations
"""

import os
import json
import time
from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests

# Import Ollama client for local GPU-accelerated LLM support
try:
    from .ollama_client import create_ollama_client, OllamaClient
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not available - will use only OpenRouter")


# Enhanced logging for OpenRouter integration (Loguru uses enable/disable, not setLevel)
logger.enable("llm_strategy_analyzer")

@dataclass
class StrategyAnalysis:
    """Results of strategy analysis"""
    bot_name: str
    risk_score: float  # 0-1 scale
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    conflicts: List[str]
    overall_rating: str
    confidence: float

class LLMStrategyAnalyzer:
    """Analyzes Option Alpha bot strategies using LLM"""
    
    # Class-level rate limiting: track last request time and minimum delay between requests
    _last_request_time = 0
    _min_request_delay = 2.0  # Minimum seconds between requests (for free tier models)
    _rate_limit_backoff_until = 0  # Timestamp until which we should back off
    _rate_limited_models = set()  # Track which models are currently rate-limited
    _model_blacklist_until = {}  # Track when rate-limited models can be retried
    
    # Fallback free models organized by provider (to avoid provider-level rate limits)
    # When one provider is rate-limited, we try a different provider's model
    # Verified working free models as of 2025-01-07
    FALLBACK_FREE_MODELS = {
        "google": [
            "google/gemini-2.0-flash-exp:free",
            "google/gemini-1.5-flash:free"
        ],
        "meta": [
            "meta-llama/llama-3.3-70b-instruct:free"
        ],
        "qwen": [
            "qwen/qwen3-235b-a22b:free"
        ],
        "tngtech": [
            "tngtech/deepseek-r1t-chimera:free"
        ]
    }
    
    @staticmethod
    def _get_provider_from_model(model: str) -> Optional[str]:
        """Extract provider name from model string"""
        if not model:
            return None
        parts = model.split("/")
        if len(parts) > 0:
            return parts[0].lower()
        return None
    
    @staticmethod
    def _get_fallback_models(current_model: str) -> List[str]:
        """
        Get list of fallback models from different providers.
        This helps when rate limits are provider-specific (upstream), not API key-specific.
        """
        current_provider = LLMStrategyAnalyzer._get_provider_from_model(current_model)
        fallbacks = []
        
        # Add models from other providers first (most likely to work)
        for provider, models in LLMStrategyAnalyzer.FALLBACK_FREE_MODELS.items():
            if provider != current_provider:
                fallbacks.extend(models)
        
        # Then add models from same provider but different models (in case it's model-specific)
        if current_provider and current_provider in LLMStrategyAnalyzer.FALLBACK_FREE_MODELS:
            for model in LLMStrategyAnalyzer.FALLBACK_FREE_MODELS[current_provider]:
                if model != current_model:
                    fallbacks.append(model)
        
        return fallbacks
    
    def __init__(self, provider: str = "openrouter", model: Optional[str] = None, api_key: Optional[str] = None):
        # Auto-detect model if not provided
        if not model:
            model = os.getenv("AI_ANALYZER_MODEL", "google/gemini-2.0-flash-exp:free")
        
        # Auto-detect provider from model name if using default provider
        if provider == "openrouter" and model and (model.lower().startswith("ollama/") or model.lower().startswith("ollama:")):
            provider = "ollama"
            logger.info(f"🔄 Auto-detected Ollama model from name: {model}, switching provider to 'ollama'")
        
        self.provider = provider
        self.model = model
        self.api_key = api_key

        if provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.model:
                self.model = os.getenv("AI_ANALYZER_MODEL", "google/gemini-2.0-flash-exp:free")
            
            # Adjust rate limiting based on model type (free models need more throttling)
            if self.model and ":free" in self.model.lower():
                LLMStrategyAnalyzer._min_request_delay = 3.0  # More delay for free models
        elif provider == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.model:
                self.model = "gpt-4-turbo"
        elif provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ValueError("Ollama provider requested but Ollama client not available")
            self.base_url = "http://localhost:11434"
            # Strip 'ollama/' prefix if present
            if self.model and self.model.startswith("ollama/"):
                self.model = self.model.replace("ollama/", "")
            self.ollama_client = create_ollama_client(model=self.model)
            logger.info(f"Using local Ollama with model: {self.model}")
            # No API key check needed for Ollama
            return 
        # Add other providers as needed
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        if not self.api_key:
            raise ValueError(f"API key for {provider} not found. Please set the appropriate environment variable.")
        
        logger.info(f"LLM Strategy Analyzer initialized with {provider} using model: {self.model}")
        
    
    def analyze_bot_strategy(self, bot_config: Dict) -> StrategyAnalysis:
        """Analyze a bot strategy configuration"""
        try:
            logger.info(f"Starting strategy analysis for {bot_config.get('name', 'Unknown Bot')}")
            logger.debug(f"Using model: {self.model}")
            
            # Prepare analysis prompt
            prompt = self._create_analysis_prompt(bot_config)
            logger.debug(f"Created analysis prompt (length: {len(prompt)} chars)")
            
            # Get LLM response
            logger.info(f"Calling LLM API via {self.provider}...")
            response = self._call_llm_api(prompt)
            logger.debug(f"Received LLM response (length: {len(response)} chars)")
            
            # Parse response
            analysis = self._parse_analysis_response(response, bot_config)
            logger.info(f"Strategy analysis completed for {bot_config.get('name', 'Unknown Bot')}")
            logger.info(f"Analysis result: {analysis.overall_rating}, Risk: {analysis.risk_score:.2f}, Confidence: {analysis.confidence:.2f}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing strategy: {e}", exc_info=True)
            return self._create_error_analysis(bot_config, str(e))
    
    def _create_analysis_prompt(self, bot_config: Dict) -> str:
        """Create detailed prompt for strategy analysis"""
        
        # Extract key information from bot config
        bot_name = bot_config.get('name', 'Unknown Bot')
        description = bot_config.get('description', 'No description')
        automations = bot_config.get('automations', [])
        inputs = bot_config.get('inputs', {})
        safeguards = bot_config.get('safeguards', {})
        
        prompt = f"""
You are an expert options trading strategist and risk manager. Analyze the following Option Alpha bot configuration and provide a comprehensive critique.

BOT CONFIGURATION:
Name: {bot_name}
Description: {description}

AUTOMATIONS:
{json.dumps(automations, indent=2)}

BOT INPUTS:
{json.dumps(inputs, indent=2)}

SAFEGUARDS:
{json.dumps(safeguards, indent=2)}

ANALYSIS REQUIREMENTS:
Please provide a detailed analysis covering:

1. RISK ASSESSMENT (0-1 scale):
   - Overall risk level
   - Position sizing adequacy
   - Diversification concerns
   - Market condition sensitivity

2. STRATEGY STRENGTHS:
   - What the bot does well
   - Effective risk management features
   - Smart automation choices

3. POTENTIAL WEAKNESSES:
   - Risk management gaps
   - Overly aggressive parameters
   - Missing safeguards
   - Potential conflicts

4. SPECIFIC RECOMMENDATIONS:
   - Concrete improvements
   - Parameter adjustments
   - Additional safeguards
   - Better automation logic

5. CONFLICTS & ISSUES:
   - Conflicting parameters
   - Logic inconsistencies
   - Overlapping rules

6. OVERALL RATING:
   - Excellent, Good, Fair, Poor
   - Confidence level (0-1)

RESPOND IN VALID JSON FORMAT:
{{
    "risk_score": 0.75,
    "strengths": ["List of strengths"],
    "weaknesses": ["List of weaknesses"], 
    "recommendations": ["List of recommendations"],
    "conflicts": ["List of conflicts"],
    "overall_rating": "Good",
    "confidence": 0.85
}}

Focus on practical, actionable insights for options trading. Consider the specific strategy (BPS/BCS with hedging) and current market conditions.
"""
        
        return prompt
    
    def analyze_with_llm(self, prompt: str) -> str:
        """
        General-purpose LLM analysis method for AI trade reviewer and other services.
        Wrapper around _call_llm_api for compatibility with external callers.
        
        Args:
            prompt: The analysis prompt/query
            
        Returns:
            str: LLM response text
        """
        return self._call_llm_api(prompt)
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call the selected LLM API. Returns string or empty on error."""
        if self.provider == "ollama":
            try:
                logger.info(f"🤖 Calling local Ollama with model: {self.model}")
                response = self.ollama_client.generate_trading_signal(prompt, analysis_type="market_analysis")
                if response:
                    logger.info(f"✅ Ollama call successful - received {len(response)} characters")
                    return response
                else:
                    logger.error("❌ Ollama returned empty response")
                    return ""
            except Exception as e:
                logger.error(f"❌ Ollama error: {e}", exc_info=True)
                return ""

        try:
            logger.info(f"🤖 Calling {self.provider} API with model: {self.model}")
            logger.debug(f"Base URL: {self.base_url}")
            logger.debug(f"API Key present: {bool(self.api_key)}, Length: {len(self.api_key) if self.api_key else 0}")
            
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": "https://github.com/sentient-trader",
                    "X-Title": "Sentient Trader"
                }
            )
            
            logger.debug(f"Created OpenAI client for {self.provider}")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert options trading strategist and risk manager."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            content = getattr(response.choices[0].message, 'content', None)
            logger.info(f"✅ {self.provider} API call successful - received {len(content) if content else 0} characters")
            logger.debug(f"{self.provider} API response: {str(content)[:200]}...")
            return content or ""
            
        except Exception as e:
            logger.error(f"❌ {self.provider} API error: {e}", exc_info=True)
            logger.error(f"API Key configured: {bool(self.api_key)}")
            logger.error(f"Model: {self.model}")
            logger.error(f"Base URL: {self.base_url}")
            return ""
    
    def _call_openrouter(self, prompt: str, max_retries: int = 3, try_fallbacks: bool = True) -> Optional[str]:
        """
        Call OpenRouter API directly - used by ai_confidence_scanner
        This is a simplified wrapper around _call_llm_api for direct calls
        
        Args:
            prompt: The prompt to send to the API
            max_retries: Maximum number of retry attempts for rate-limited requests
            try_fallbacks: If True, try fallback models from different providers when rate-limited
            
        Returns:
            API response content or None on failure
        """
        models_to_try = [self.model]
        
        # If fallbacks are enabled and we're using a free model, prepare fallback list
        if try_fallbacks and self.model and ":free" in self.model.lower():
            fallback_models = self._get_fallback_models(self.model)
            models_to_try.extend(fallback_models)
            logger.debug(f"🔄 Prepared {len(fallback_models)} fallback models from different providers")
            logger.debug(f"📋 Fallback models: {fallback_models}")
            
            # Validate models - check for old/invalid model names
            invalid_models = [m for m in fallback_models if "llama-3.1-8b" in m or "llama-3.2-3b" in m]
            if invalid_models:
                logger.warning(f"⚠️ Found outdated model names in fallback list: {invalid_models}. Please restart the application to load updated models.")
        
        # Try each model in sequence
        for model_idx, model_to_use in enumerate(models_to_try):
            # Check if this model is currently blacklisted
            current_time = time.time()
            if model_to_use in LLMStrategyAnalyzer._model_blacklist_until:
                blacklist_until = LLMStrategyAnalyzer._model_blacklist_until[model_to_use]
                if current_time < blacklist_until:
                    wait_time = blacklist_until - current_time
                    logger.debug(f"⏸️ Model {model_to_use} is blacklisted for {wait_time:.1f}s more, skipping...")
                    continue
            
            # Rate limiting: wait if we're in a backoff period
            if current_time < LLMStrategyAnalyzer._rate_limit_backoff_until:
                wait_time = LLMStrategyAnalyzer._rate_limit_backoff_until - current_time
                logger.warning(f"⏳ Rate limit backoff active, waiting {wait_time:.1f}s before request...")
                time.sleep(wait_time)
                current_time = time.time()
            
            # Throttling: ensure minimum delay between requests
            time_since_last = current_time - LLMStrategyAnalyzer._last_request_time
            if time_since_last < LLMStrategyAnalyzer._min_request_delay:
                wait_time = LLMStrategyAnalyzer._min_request_delay - time_since_last
                logger.debug(f"⏱️ Rate limiting: waiting {wait_time:.1f}s before next request...")
                time.sleep(wait_time)
            
            # Retry loop with exponential backoff for this model
            initial_delay = 5.0  # Start with 5 seconds
            backoff_factor = 2.0
            
            for attempt in range(max_retries + 1):
                try:
                    # Update last request time
                    LLMStrategyAnalyzer._last_request_time = time.time()
                    
                    model_label = f"{model_to_use} (fallback {model_idx})" if model_idx > 0 else model_to_use
                    logger.info(f"🤖 Direct OpenRouter call with model: {model_label} (attempt {attempt + 1}/{max_retries + 1})")
                    logger.debug(f"API Key present: {bool(self.api_key)}")
                    
                    # Use requests for direct API call
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/sentient-trader",
                            "X-Title": "Sentient Trader"
                        },
                        json={
                            "model": model_to_use,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "temperature": 0.3,
                            "max_tokens": 1500
                        },
                        timeout=30
                    )
                    
                    logger.info(f"OpenRouter response status: {response.status_code}")
                    
                    # Handle successful response
                    if response.status_code == 200:
                        data = response.json()
                        content = data['choices'][0]['message']['content']
                        model_used = model_to_use if model_idx == 0 else f"{model_to_use} (fallback)"
                        logger.info(f"✅ OpenRouter API call successful with {model_used} - received {len(content)} characters")
                        logger.debug(f"OpenRouter response: {str(content)[:200]}...")
                        # Reset backoff on success
                        LLMStrategyAnalyzer._rate_limit_backoff_until = 0
                        # Remove from blacklist if it was there
                        if model_to_use in LLMStrategyAnalyzer._model_blacklist_until:
                            del LLMStrategyAnalyzer._model_blacklist_until[model_to_use]
                        return content
                    
                    # Handle rate limiting (429)
                    elif response.status_code == 429:
                        error_msg = ""
                        is_provider_limited = False
                        try:
                            error_data = response.json()
                            error_msg = error_data.get('error', {}).get('message', response.text[:200])
                            # Check if error mentions specific provider/model
                            if "upstream" in error_msg.lower() or "provider" in error_msg.lower():
                                is_provider_limited = True
                        except:
                            error_msg = response.text[:200]
                        
                        # If it's a provider-specific rate limit, blacklist this model and try next fallback
                        if is_provider_limited and model_idx < len(models_to_try) - 1:
                            # Blacklist this model for 10 minutes
                            LLMStrategyAnalyzer._model_blacklist_until[model_to_use] = time.time() + 600
                            provider = self._get_provider_from_model(model_to_use)
                            logger.warning(
                                f"⚠️ Provider-specific rate limit detected for {model_to_use} ({provider}). "
                                f"Trying fallback model from different provider..."
                            )
                            break  # Break out of retry loop, try next model
                        
                        # Try to parse Retry-After header (if present)
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                            except ValueError:
                                wait_time = initial_delay * (backoff_factor ** attempt)
                        else:
                            # Calculate exponential backoff
                            wait_time = initial_delay * (backoff_factor ** attempt)
                        
                        # Cap maximum wait time at 5 minutes
                        wait_time = min(wait_time, 300)
                        
                        # Set global backoff to prevent other requests
                        LLMStrategyAnalyzer._rate_limit_backoff_until = time.time() + wait_time
                        
                        # If this is the last model, retry with backoff
                        if model_idx == len(models_to_try) - 1 and attempt < max_retries:
                            logger.warning(
                                f"⚠️ OpenRouter rate limit (429) - {error_msg}. "
                                f"Retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries + 1})..."
                            )
                            time.sleep(wait_time)
                            continue
                        elif model_idx < len(models_to_try) - 1:
                            # Try next fallback model
                            logger.warning(
                                f"⚠️ Rate limit on {model_to_use}, trying next fallback model..."
                            )
                            break  # Break to try next model
                        else:
                            # All models exhausted
                            logger.error(
                                f"❌ OpenRouter rate limit (429) after trying {len(models_to_try)} models. "
                                f"Error: {error_msg}"
                            )
                            logger.info(
                                "💡 Tips to resolve rate limits:\n"
                                "   1. Add your own provider API keys to OpenRouter: https://openrouter.ai/settings/integrations\n"
                                "   2. Switch to a paid model instead of free tier\n"
                                "   3. Reduce request frequency or batch requests\n"
                                "   4. Wait a few minutes and try again\n"
                                "   5. Consider using multiple OpenRouter API keys with different models"
                            )
                            return None
                    
                    # Handle 404 (model not found) - skip to next fallback
                    elif response.status_code == 404:
                        error_msg = ""
                        try:
                            error_data = response.json()
                            error_msg = error_data.get('error', {}).get('message', response.text[:200])
                        except Exception as e:
                            error_msg = response.text[:200] if hasattr(response, 'text') else str(e)
                        
                        logger.warning(f"⚠️ Model not found (404): {model_to_use} - {error_msg}")
                        
                        # Blacklist this invalid model permanently (until restart)
                        if model_to_use not in LLMStrategyAnalyzer._model_blacklist_until:
                            LLMStrategyAnalyzer._model_blacklist_until[model_to_use] = time.time() + 86400  # 24 hours
                        
                        # If we have more fallbacks, try next model
                        if model_idx < len(models_to_try) - 1:
                            logger.info(f"🔄 Skipping invalid model {model_to_use}, trying next fallback model...")
                            break  # Exit retry loop, move to next model
                        else:
                            logger.error(f"❌ All fallback models exhausted (404 on last model: {model_to_use})")
                            return None
                    
                    # Handle other errors
                    else:
                        error_msg = response.text[:500]
                        logger.error(f"❌ OpenRouter API error: {response.status_code} - {error_msg}")
                        # If it's a server error (5xx) and we have fallbacks, try next model
                        if model_idx < len(models_to_try) - 1 and response.status_code >= 500:
                            logger.warning(f"⚠️ Server error on {model_to_use}, trying fallback...")
                            break
                        # Don't retry on non-rate-limit client errors (4xx except 404/429)
                        return None
                
                except requests.exceptions.Timeout:
                    if attempt < max_retries and model_idx == len(models_to_try) - 1:
                        wait_time = initial_delay * (backoff_factor ** attempt)
                        logger.warning(f"⏱️ OpenRouter timeout, retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    elif model_idx < len(models_to_try) - 1:
                        logger.warning(f"⏱️ Timeout on {model_to_use}, trying fallback model...")
                        break
                    else:
                        logger.error("❌ OpenRouter API timeout after all retries and fallbacks")
                        return None
                
                except Exception as e:
                    logger.error(f"❌ Error calling OpenRouter: {e}", exc_info=True)
                    logger.error(f"API Key configured: {bool(self.api_key)}")
                    # If we have fallbacks, try next model
                    if model_idx < len(models_to_try) - 1:
                        logger.warning(f"⚠️ Error on {model_to_use}, trying fallback model...")
                        break
                    # Don't retry on unexpected errors
                    return None
        
        # All models exhausted
        logger.error(f"❌ All {len(models_to_try)} models exhausted, request failed")
        return None
    
    def _parse_analysis_response(self, response: str, bot_config: Dict) -> StrategyAnalysis:
        """Parse LLM response into StrategyAnalysis object"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            analysis_data = json.loads(json_str)
            
            return StrategyAnalysis(
                bot_name=bot_config.get('name', 'Unknown Bot'),
                risk_score=float(analysis_data.get('risk_score', 0.5)),
                strengths=analysis_data.get('strengths', []),
                weaknesses=analysis_data.get('weaknesses', []),
                recommendations=analysis_data.get('recommendations', []),
                conflicts=analysis_data.get('conflicts', []),
                overall_rating=analysis_data.get('overall_rating', 'Fair'),
                confidence=float(analysis_data.get('confidence', 0.5))
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_error_analysis(bot_config, f"Failed to parse response: {e}")
    
    def _create_error_analysis(self, bot_config: Dict, error: str) -> StrategyAnalysis:
        """Create error analysis when LLM fails"""
        return StrategyAnalysis(
            bot_name=bot_config.get('name', 'Unknown Bot'),
            risk_score=0.5,
            strengths=[],
            weaknesses=[f"Analysis failed: {error}"],
            recommendations=["Fix LLM integration", "Check API credentials"],
            conflicts=[],
            overall_rating="Unknown",
            confidence=0.0
        )

# Utility functions for bot configuration extraction
def extract_bot_config_from_screenshot() -> Dict:
    """Extract bot configuration from the provided screenshot description"""
    
    # Based on the screenshot description, create a structured config
    config = {
        "name": "[↑↓] BPS/BCS XSP/SPX",
        "description": "XSP/SPX - BPS or BCS with hedging",
        "version": "21",
        "date": "Aug 8, 2025",
        "automations": {
            "scanners": [
                "Manage VIX",
                "Evaluate Market >100EMA/<100EMA with 1 input - SPX",
                "Open a new BPS/BCS position-V6 with 9 inputs - SPX"
            ],
            "monitors": [
                "Manage Positions-V6 with 9 inputs"
            ],
            "triggers": [
                "Initialize Zero Position Bot Tags",
                "Market open Mon-Fri"
            ]
        },
        "inputs": {
            "new_position": True,
            "high_frequency_trades": True,
            "max_position_size": "6% of net liquid",
            "hedge": True,
            "symbol": "SPX",
            "max_regular_positions": 10,
            "early_warning_close": True,
            "position_size_early_warning": "4% of net liquid",
            "hedge_expiration": "3 to 7 days"
        },
        "safeguards": {
            "allocation": "$5,000",
            "daily_positions": 10,
            "position_limit": 10,
            "day_trading": "Allowed"
        },
        "scan_speeds": {
            "automations": "Every 15m",
            "exit_options": "Every 1m"
        },
        "activity_alerts": [
            "Open position",
            "Close position", 
            "Automation warning",
            "Automation error"
        ]
    }
    
    return config

def create_strategy_comparison(analyses: List[StrategyAnalysis]) -> Dict:
    """Compare multiple strategy analyses"""
    if not analyses:
        return {}
    
    comparison = {
        "total_strategies": len(analyses),
        "average_risk_score": sum(a.risk_score for a in analyses) / len(analyses),
        "average_confidence": sum(a.confidence for a in analyses) / len(analyses),
        "rating_distribution": {},
        "common_strengths": [],
        "common_weaknesses": [],
        "top_recommendations": []
    }
    
    # Count ratings
    for analysis in analyses:
        rating = analysis.overall_rating
        comparison["rating_distribution"][rating] = comparison["rating_distribution"].get(rating, 0) + 1
    
    # Find common themes
    all_strengths = [strength for analysis in analyses for strength in analysis.strengths]
    all_weaknesses = [weakness for analysis in analyses for weakness in analysis.weaknesses]
    all_recommendations = [rec for analysis in analyses for rec in analysis.recommendations]
    
    # Simple frequency analysis (could be enhanced with NLP)
    from collections import Counter
    comparison["common_strengths"] = [item for item, count in Counter(all_strengths).most_common(3)]
    comparison["common_weaknesses"] = [item for item, count in Counter(all_weaknesses).most_common(3)]
    comparison["top_recommendations"] = [item for item, count in Counter(all_recommendations).most_common(5)]
    
    return comparison

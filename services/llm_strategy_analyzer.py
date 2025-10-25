"""
LLM Strategy Analyzer for Option Alpha Bot Critique
Analyzes bot configurations and provides intelligent recommendations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

# Enhanced logging for OpenRouter integration
logger.setLevel(logging.DEBUG)

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
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.model = model or self._get_default_model()
        # Allow passing API key at runtime (from UI) or fall back to env var
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._get_api_key()
        self.base_url = self._get_base_url()
        
    def _get_default_model(self) -> str:
        """Get default model based on provider"""
        defaults = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-5-sonnet-20241022",
            "google": "gemini-pro",
            "openrouter": "meta-llama/llama-3.3-70b-instruct"
        }
        return defaults.get(self.provider, "gpt-4-turbo-preview")
    
    def _get_api_key(self) -> str:
        """Get API key from environment"""
        key_mapping = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "google": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY"
        }
        
        env_var = key_mapping.get(self.provider)
        if not env_var:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key not found. Set {env_var} environment variable.")
        
        return api_key
    
    def _get_base_url(self) -> Optional[str]:
        """Get base URL for API calls"""
        if self.provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        return None
    
    def analyze_bot_strategy(self, bot_config: Dict) -> StrategyAnalysis:
        """Analyze a bot strategy configuration"""
        try:
            logger.info(f"Starting strategy analysis for {bot_config.get('name', 'Unknown Bot')}")
            logger.debug(f"Using provider: {self.provider}, model: {self.model}")
            
            # Prepare analysis prompt
            prompt = self._create_analysis_prompt(bot_config)
            logger.debug(f"Created analysis prompt (length: {len(prompt)} chars)")
            
            # Get LLM response
            logger.info("Calling LLM API...")
            response = self._call_llm(prompt)
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
    
    def _call_llm(self, prompt: str) -> str:
        """Call the appropriate LLM API. Returns a string (may be empty on error)."""
        try:
            if self.provider == "openai":
                return self._call_openai(prompt)
            elif self.provider == "anthropic":
                return self._call_anthropic(prompt)
            elif self.provider == "google":
                return self._call_google(prompt)
            elif self.provider == "openrouter":
                return self._call_openrouter(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return ""
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API. Returns response text or empty string on error."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
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
            return content or ""
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ""
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API. Returns response text or empty string on error."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Try to extract text safely
            try:
                content = getattr(response, 'content', None)
                if content and len(content) > 0:
                    block = content[0]
                    text = getattr(block, 'text', None) or getattr(block, 'content', None)
                    return str(text) if text is not None else str(block)
                return str(response) or ""
            except Exception:
                return str(response) or ""
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return ""
    
    def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API. Returns string or empty on error."""
        try:
            import google.generativeai as genai
            # Some google SDK versions use configure or different entrypoints; attempt best-effort
            # Different versions of google.generativeai expose different APIs; attempt best-effort call
            try:
                model_cls = getattr(genai, 'GenerativeModel', None)
                if model_cls is not None:
                    m = model_cls(self.model)
                    response = m.generate_content(
                        prompt,
                        generation_config=(getattr(genai, 'types', {}).get('GenerationConfig', lambda **k: None)(
                            temperature=0.3,
                            max_output_tokens=2000
                        ))
                    )
                    return getattr(response, 'text', str(response) or "")
            except Exception:
                # Give up gracefully on SDK mismatch
                return ""

            return ""
        except Exception as e:
            logger.error(f"Google API error: {e}")
            return ""
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API (via OpenAI-compatible client). Returns string or empty on error."""
        try:
            logger.debug(f"Calling OpenRouter API with model: {self.model}")
            logger.debug(f"Base URL: {self.base_url}")
            logger.debug(f"API Key: {self.api_key[:10]}..." if self.api_key else "No API key")
            
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            logger.debug("Created OpenAI client for OpenRouter")
            
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
            logger.debug(f"OpenRouter API response received: {str(content)[:200]}...")
            return content or ""
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}", exc_info=True)
            return ""
    
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

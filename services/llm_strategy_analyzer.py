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
    
    def __init__(self, provider: str = "openrouter", model: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key

        if provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            if not self.model:
                self.model = os.getenv("AI_ANALYZER_MODEL", "google/gemini-2.0-flash-exp:free")
        elif provider == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.model:
                self.model = "gpt-4-turbo"
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
    
    
    
    
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call the selected LLM API. Returns string or empty on error."""
        try:
            logger.debug(f"Calling {self.provider} API with model: {self.model}")
            logger.debug(f"Base URL: {self.base_url}")
            
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
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
            logger.debug(f"{self.provider} API response received: {str(content)[:200]}...")
            return content or ""
            
        except Exception as e:
            logger.error(f"{self.provider} API error: {e}", exc_info=True)
            return ""
    
    def _call_openrouter(self, prompt: str) -> Optional[str]:
        """
        Call OpenRouter API directly - used by ai_confidence_scanner
        This is a simplified wrapper around _call_llm_api for direct calls
        """
        try:
            logger.debug(f"Direct OpenRouter call with model: {self.model}")
            
            # Use requests for direct API call
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
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
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                logger.debug(f"OpenRouter response received: {str(content)[:200]}...")
                return content
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}", exc_info=True)
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

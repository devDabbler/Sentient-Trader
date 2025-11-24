"""
AI-powered validator for advanced option strategies.
Uses LLM to analyze if a strategy is viable given current market conditions.
"""

import os
import json
from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from models.reddit_strategies import CustomStrategy, MarketCondition
from models.analysis import StockAnalysis
from services.llm_helper import get_llm_helper



@dataclass
class StrategyValidation:
    """Results of strategy validation"""
    strategy_name: str
    is_viable: bool
    viability_score: float  # 0-1 scale
    market_alignment: str  # "Excellent", "Good", "Fair", "Poor"
    strengths: List[str]
    concerns: List[str]
    recommendations: List[str]
    missing_conditions: List[str]
    red_flags_detected: List[str]
    confidence: float
    reasoning: str


class StrategyValidator:
    """
    Validates advanced option strategies using AI analysis of current market conditions.
    """
    
    def __init__(self, provider: str = "openrouter", model: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize strategy validator with LLM Request Manager"""
        if provider != "openrouter" or model or api_key:
            logger.warning("⚠️ provider, model, and api_key parameters are deprecated")
        
        # Initialize LLM Request Manager helper (LOW priority for informational analysis)
        try:
            self.llm_helper = get_llm_helper("reddit_strategy_validator", default_priority="LOW")
            logger.success("🚀 Strategy Validator using LLM Request Manager")
        except Exception as e:
            logger.error(f"Failed to initialize LLM helper: {e}")
            raise
    
    
    def validate_strategy(
        self, 
        strategy: CustomStrategy, 
        ticker: str,
        analysis: Optional[StockAnalysis] = None,
        market_context: Optional[Dict] = None
    ) -> StrategyValidation:
        """
        Validate if a Reddit strategy is viable for the given ticker and market conditions.
        
        Args:
            strategy: The CustomStrategy to validate
            ticker: The ticker symbol to apply the strategy to
            analysis: Optional StockAnalysis with technical/fundamental data
            market_context: Optional dict with additional market context (VIX, news, etc.)
        
        Returns:
            StrategyValidation with AI assessment
        """
        try:
            logger.info("Validating strategy '{}' for {ticker}", str(strategy.name))
            
            # Create validation prompt
            prompt = self._create_validation_prompt(strategy, ticker, analysis, market_context)
            logger.debug(f"Created validation prompt (length: {len(prompt)} chars)")
            
            # Get LLM response using centralized manager
            logger.info("Calling LLM API for strategy validation...")
            # Use LOW priority with caching (10 min TTL for strategy validation)
            cache_key = f"strategy_val_{strategy.name}_{ticker}"
            response = self.llm_helper.low_request(
                prompt,
                cache_key=cache_key,
                ttl=600,  # 10 minutes cache
                temperature=0.4
            )
            
            if not response:
                raise Exception("No response from LLM")
            
            logger.debug(f"Received LLM response (length: {len(response)} chars)")
            
            # Parse response
            validation = self._parse_validation_response(response, strategy)
            logger.info(f"Validation completed: {validation.market_alignment}, Viable: {validation.is_viable}, Score: {validation.viability_score:.2f}")
            
            return validation
            
        except Exception as e:
            logger.error("Error validating strategy: {}", str(e), exc_info=True)
            return self._create_error_validation(strategy, str(e))
    
    
    def _create_validation_prompt(
        self, 
        strategy: CustomStrategy, 
        ticker: str,
        analysis: Optional[StockAnalysis],
        market_context: Optional[Dict]
    ) -> str:
        """Create detailed prompt for strategy validation"""
        
        # Build market conditions section
        market_conditions_text = "No market analysis provided."
        if analysis:
            market_conditions_text = f"""
CURRENT MARKET CONDITIONS FOR {ticker}:
- Price: ${analysis.price:.2f} (Change: {analysis.change_pct:+.2f}%)
- Trend: {analysis.trend}
- RSI: {analysis.rsi:.1f}
- MACD Signal: {analysis.macd_signal}
- IV Rank: {analysis.iv_rank:.1f}%
- IV Percentile: {analysis.iv_percentile:.1f}%
- Support: ${analysis.support:.2f}
- Resistance: ${analysis.resistance:.2f}
- Volume: {analysis.volume:,} (Avg: {analysis.avg_volume:,})
- Sentiment Score: {analysis.sentiment_score:.2f}
- Earnings Days Away: {analysis.earnings_days_away if analysis.earnings_days_away else 'N/A'}
"""
        
        if market_context:
            market_conditions_text += f"\nADDITIONAL MARKET CONTEXT:\n{json.dumps(market_context, indent=2)}\n"
        
        # Build strategy details
        parameters_text = "\n".join([
            f"  - {p.name}: {p.value} ({p.description})"
            for p in strategy.parameters
        ])
        
        setup_rules_text = "\n".join([
            f"  {r.priority}. {r.condition}\n     Action: {r.action}\n     Notes: {r.notes or 'N/A'}"
            for r in strategy.setup_rules
        ])
        
        risk_mgmt_text = "\n".join([
            f"  - {r.rule_type.upper()}: {r.value}\n    {r.description} (Mandatory: {r.mandatory})"
            for r in strategy.risk_management
        ])
        
        validation_checklist_text = "\n".join([f"  - {item}" for item in strategy.validation_checklist])
        red_flags_text = "\n".join([f"  - {flag}" for flag in strategy.red_flags])
        warnings_text = "\n".join([f"  {warning}" for warning in strategy.warnings])
        
        prompt = f"""
You are an expert options trading strategist and risk manager. You are evaluating whether an advanced option strategy is viable for a specific ticker given current market conditions.

STRATEGY TO VALIDATE:
===================
Name: {strategy.name}
Source: {strategy.source}
Type: {strategy.strategy_type.value}

DESCRIPTION:
{strategy.description}

PHILOSOPHY:
{strategy.philosophy}

KEY PERFORMANCE METRICS (from source):
{json.dumps(strategy.key_metrics, indent=2)}

SUITABLE PRODUCTS:
{', '.join(strategy.suitable_products)}

REQUIRED CONDITIONS:
{', '.join([c.value for c in strategy.required_conditions])}

UNSUITABLE CONDITIONS:
{', '.join([c.value for c in strategy.unsuitable_conditions])}

STRATEGY PARAMETERS:
{parameters_text}

SETUP RULES:
{setup_rules_text}

RISK MANAGEMENT:
{risk_mgmt_text}

EXPERIENCE LEVEL: {strategy.experience_level}
RISK LEVEL: {strategy.risk_level}
CAPITAL REQUIREMENT: {strategy.capital_requirement}
TYPICAL WIN RATE: {strategy.typical_win_rate or 'N/A'}

VALIDATION CHECKLIST:
{validation_checklist_text}

RED FLAGS TO WATCH FOR:
{red_flags_text}

WARNINGS:
{warnings_text}

ADDITIONAL NOTES:
{strategy.notes or 'None'}

{market_conditions_text}

VALIDATION TASK:
===============
Please analyze whether this strategy is viable for {ticker} given the current market conditions.

Consider:
1. Does {ticker} match the suitable products for this strategy?
2. Are the required market conditions present?
3. Are any unsuitable conditions present?
4. Does the current technical/fundamental analysis support this strategy?
5. Are there any red flags from the strategy's red flag list?
6. What items from the validation checklist are satisfied vs missing?
7. What are the specific risks for {ticker} with this strategy?
8. What modifications or adjustments would improve viability?

RESPOND IN VALID JSON FORMAT:
{{
    "is_viable": true/false,
    "viability_score": 0.75,
    "market_alignment": "Excellent/Good/Fair/Poor",
    "strengths": ["List of strengths for this ticker/market condition"],
    "concerns": ["List of concerns or risks"],
    "recommendations": ["Specific recommendations for implementation"],
    "missing_conditions": ["Required conditions that are NOT currently met"],
    "red_flags_detected": ["Any red flags from the strategy list that apply"],
    "confidence": 0.85,
    "reasoning": "Detailed explanation of the viability assessment"
}}

Be thorough, practical, and honest. If the strategy is not suitable, say so clearly. If modifications are needed, specify them.
"""
        
        return prompt
    
<<<<<<< HEAD
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call the selected LLM API"""
        try:
            logger.debug(f"Calling {self.provider} API with model: {self.model}")
            
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert options trading strategist and risk manager specializing in validating trading strategies."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2500
            )
            
            content = getattr(response.choices[0].message, 'content', None)
            logger.debug(f"{self.provider} API response received: {str(content)[:200]}...")
            return content or ""
            
        except Exception as e:
            logger.error("{self.provider} API error: {}", str(e), exc_info=True)
            return ""
    
    
=======
>>>>>>> 9653b474 (WIP: saving changes before rebase)
    def _parse_validation_response(self, response: str, strategy: CustomStrategy) -> StrategyValidation:
        """Parse LLM response into StrategyValidation object"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            validation_data = json.loads(json_str)
            
            return StrategyValidation(
                strategy_name=strategy.name,
                is_viable=validation_data.get('is_viable', False),
                viability_score=float(validation_data.get('viability_score', 0.5)),
                market_alignment=validation_data.get('market_alignment', 'Fair'),
                strengths=validation_data.get('strengths', []),
                concerns=validation_data.get('concerns', []),
                recommendations=validation_data.get('recommendations', []),
                missing_conditions=validation_data.get('missing_conditions', []),
                red_flags_detected=validation_data.get('red_flags_detected', []),
                confidence=float(validation_data.get('confidence', 0.5)),
                reasoning=validation_data.get('reasoning', 'No reasoning provided')
            )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return self._create_error_validation(strategy, f"Failed to parse response: {e}")
    
    
    def _create_error_validation(self, strategy: CustomStrategy, error: str) -> StrategyValidation:
        """Create error validation when LLM fails"""
        return StrategyValidation(
            strategy_name=strategy.name,
            is_viable=False,
            viability_score=0.0,
            market_alignment="Unknown",
            strengths=[],
            concerns=[f"Validation failed: {error}"],
            recommendations=["Fix LLM integration", "Check API credentials", "Retry validation"],
            missing_conditions=["Unable to determine"],
            red_flags_detected=[],
            confidence=0.0,
            reasoning=f"Validation error: {error}"
        )
    
    
    def batch_validate_strategies(
        self,
        strategies: List[CustomStrategy],
        ticker: str,
        analysis: Optional[StockAnalysis] = None,
        market_context: Optional[Dict] = None
    ) -> List[StrategyValidation]:
        """
        Validate multiple strategies at once and return ranked results.
        """
        validations = []
        
        for strategy in strategies:
            validation = self.validate_strategy(strategy, ticker, analysis, market_context)
            validations.append(validation)
        
        # Sort by viability score (highest first)
        validations.sort(key=lambda v: v.viability_score, reverse=True)
        
        return validations
    
    
    def compare_strategies(
        self,
        validations: List[StrategyValidation]
    ) -> Dict:
        """
        Compare multiple strategy validations and provide summary insights.
        """
        if not validations:
            return {}
        
        viable_strategies = [v for v in validations if v.is_viable]
        
        comparison = {
            "total_strategies": len(validations),
            "viable_count": len(viable_strategies),
            "average_viability_score": sum(v.viability_score for v in validations) / len(validations),
            "average_confidence": sum(v.confidence for v in validations) / len(validations),
            "best_strategy": validations[0].strategy_name if validations else None,
            "best_score": validations[0].viability_score if validations else 0.0,
            "alignment_distribution": {},
            "common_concerns": [],
            "top_recommendations": []
        }
        
        # Count alignment distribution
        for validation in validations:
            alignment = validation.market_alignment
            comparison["alignment_distribution"][alignment] = comparison["alignment_distribution"].get(alignment, 0) + 1
        
        # Find common themes
        all_concerns = [concern for v in validations for concern in v.concerns]
        all_recommendations = [rec for v in validations for rec in v.recommendations]
        
        from collections import Counter
        comparison["common_concerns"] = [item for item, count in Counter(all_concerns).most_common(5)]
        comparison["top_recommendations"] = [item for item, count in Counter(all_recommendations).most_common(5)]
        
        return comparison


def format_validation_report(validation: StrategyValidation) -> str:
    """
    Format a validation result into a readable report.
    """
    report = f"""
{'='*80}
STRATEGY VALIDATION REPORT
{'='*80}

Strategy: {validation.strategy_name}
Viable: {'✅ YES' if validation.is_viable else '❌ NO'}
Viability Score: {validation.viability_score:.1%}
Market Alignment: {validation.market_alignment}
Confidence: {validation.confidence:.1%}

REASONING:
{validation.reasoning}

STRENGTHS:
"""
    
    for strength in validation.strengths:
        report += f"  ✅ {strength}\n"
    
    if not validation.strengths:
        report += "  (None identified)\n"
    
    report += "\nCONCERNS:\n"
    for concern in validation.concerns:
        report += f"  ⚠️ {concern}\n"
    
    if not validation.concerns:
        report += "  (None identified)\n"
    
    if validation.missing_conditions:
        report += "\nMISSING CONDITIONS:\n"
        for condition in validation.missing_conditions:
            report += f"  ❌ {condition}\n"
    
    if validation.red_flags_detected:
        report += "\n🚩 RED FLAGS DETECTED:\n"
        for flag in validation.red_flags_detected:
            report += f"  🚩 {flag}\n"
    
    report += "\nRECOMMENDATIONS:\n"
    for rec in validation.recommendations:
        report += f"  💡 {rec}\n"
    
    if not validation.recommendations:
        report += "  (None provided)\n"
    
    report += f"\n{'='*80}\n"
    
    return report

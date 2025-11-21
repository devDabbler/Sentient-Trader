"""
AI-Enhanced Crypto Scanner

Uses LLM to provide intelligent confidence analysis for crypto trading opportunities.
Adds AI reasoning on top of quantitative scoring with crypto-specific factors.

Similar to AIConfidenceScanner but adapted for cryptocurrency markets:
- 24/7 market analysis (no trading hours)
- Social sentiment and narrative importance
- On-chain metrics consideration
- Higher volatility tolerance
- Instant settlement implications
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from loguru import logger
from .crypto_scanner import CryptoOpportunityScanner, CryptoOpportunity
from clients.kraken_client import KrakenClient
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()



@dataclass
class AICryptoOpportunity(CryptoOpportunity):
    """Extended crypto opportunity with AI confidence analysis"""
    ai_confidence: str = "MEDIUM"  # AI-assessed confidence
    ai_reasoning: str = ""  # AI explanation
    ai_risks: str = ""  # AI-identified risks
    ai_rating: float = 0.0  # 0-10 rating from AI
    social_narrative: str = ""  # Current social media narrative
    market_cycle_phase: str = "UNKNOWN"  # Where we are in the cycle


class AICryptoScanner:
    """
    Scanner that adds AI-powered confidence analysis to crypto opportunities.
    Can work with or without LLM API keys.
    """
    
    def __init__(self, kraken_client: KrakenClient, config=None, use_llm: Optional[bool] = None):
        """
        Initialize AI crypto scanner
        
        Args:
            kraken_client: KrakenClient instance
            config: Trading configuration
            use_llm: Whether to use LLM for analysis. 
                    If None, auto-detect based on API keys.
        """
        self.base_scanner = CryptoOpportunityScanner(kraken_client, config)
        self.client = kraken_client
        self.config = config
        
        # Auto-detect if we should use LLM
        if use_llm is None:
            self.use_llm = self._check_llm_available()
        else:
            self.use_llm = use_llm
        
        if self.use_llm:
            try:
                from .llm_strategy_analyzer import LLMStrategyAnalyzer
                api_key = os.getenv('OPENROUTER_API_KEY')
                model = os.getenv('AI_CRYPTO_MODEL', 'google/gemini-2.0-flash-exp:free')
                
                if not api_key:
                    from utils.config_loader import get_api_key as get_key_helper
                    api_key = get_key_helper('OPENROUTER_API_KEY', 'openrouter')
                
                if not api_key:
                    logger.error("❌ OPENROUTER_API_KEY not found - AI crypto analysis disabled")
                    self.use_llm = False
                    self.llm_analyzer = None
                else:
                    self.llm_analyzer = LLMStrategyAnalyzer(provider="openrouter", model=model, api_key=api_key)
                    logger.info(f"✅ AI Crypto Scanner initialized with OpenRouter")
                    logger.info(f"   Model: {model}")
            except Exception as e:
                logger.error("❌ LLM initialization failed: {}", str(e), exc_info=True)
                self.use_llm = False
    
    def _check_llm_available(self) -> bool:
        """Check if OpenRouter LLM API key is available"""
        has_key = bool(os.getenv('OPENROUTER_API_KEY'))
        if not has_key:
            logger.warning("⚠️ OpenRouter API key not found in environment")
        return has_key
    
    def scan_with_ai_confidence(
        self,
        strategy: str = 'ALL',
        top_n: int = 10,
        min_score: float = 60.0,
        min_ai_confidence: Optional[str] = None
    ) -> List[AICryptoOpportunity]:
        """
        Scan for crypto opportunities with AI confidence analysis
        
        Args:
            strategy: 'SCALP', 'MOMENTUM', 'SWING', or 'ALL'
            top_n: Number of top opportunities to return
            min_score: Minimum quantitative score threshold
            min_ai_confidence: Filter by AI confidence ('HIGH', 'MEDIUM', 'LOW')
            
        Returns:
            List of AICryptoOpportunity objects with AI analysis
        """
        logger.info(f"🤖 AI Crypto Scanner: Finding {top_n} opportunities with AI confidence")
        
        # Get base opportunities from quantitative scanner
        base_opportunities = self.base_scanner.scan_opportunities(
            strategy=strategy,
            top_n=top_n * 2,  # Get more to filter by AI confidence
            min_score=min_score * 0.8  # Lower threshold, AI will refine
        )
        
        if not base_opportunities:
            logger.warning("No base opportunities found")
            return []
        
        logger.info(f"Found {len(base_opportunities)} base opportunities, adding AI analysis...")
        
        # Add AI confidence analysis to each opportunity
        ai_opportunities = []
        for opp in base_opportunities:
            ai_opp = self._add_ai_confidence(opp)
            
            # Filter by AI confidence if specified
            if min_ai_confidence and ai_opp.ai_confidence != min_ai_confidence:
                if min_ai_confidence == 'HIGH' and ai_opp.ai_confidence not in ['HIGH']:
                    continue
                elif min_ai_confidence == 'MEDIUM' and ai_opp.ai_confidence not in ['HIGH', 'MEDIUM']:
                    continue
            
            ai_opportunities.append(ai_opp)
        
        # Re-sort by combined score (quant + AI rating)
        ai_opportunities.sort(
            key=lambda x: (x.score * 0.6) + (x.ai_rating * 10 * 0.4),
            reverse=True
        )
        
        logger.info(f"✅ Returning top {min(top_n, len(ai_opportunities} AI-analyzed crypto opportunities"))
        
        return ai_opportunities[:top_n]
    
    def _add_ai_confidence(self, opportunity: CryptoOpportunity) -> AICryptoOpportunity:
        """
        Add AI confidence analysis to a crypto opportunity
        
        Args:
            opportunity: Base CryptoOpportunity object
            
        Returns:
            AICryptoOpportunity with AI analysis
        """
        if not self.use_llm:
            # Fallback to rule-based confidence
            return self._rule_based_confidence(opportunity)
        
        try:
            # Create prompt for LLM
            prompt = self._create_crypto_analysis_prompt(opportunity)
            
            # Get LLM response
            response = self._query_llm(prompt, symbol=opportunity.symbol)
            
            # Parse response
            ai_data = self._parse_llm_response(response, opportunity)
            
            # Create AI-enhanced opportunity
            ai_opp = AICryptoOpportunity(
                **asdict(opportunity),
                ai_confidence=ai_data.get('confidence', 'MEDIUM'),
                ai_reasoning=ai_data.get('reasoning', ''),
                ai_risks=ai_data.get('risks', ''),
                ai_rating=ai_data.get('rating', 5.0),
                social_narrative=ai_data.get('social_narrative', ''),
                market_cycle_phase=ai_data.get('market_cycle', 'UNKNOWN')
            )
            
            logger.info(f"🤖 AI Analysis for {opportunity.symbol}: {ai_opp.ai_confidence} confidence, {ai_opp.ai_rating}/10 rating")
            
            return ai_opp
            
        except Exception as e:
            logger.error(f"Error adding AI confidence to {opportunity.symbol}: {e}")
            return self._rule_based_confidence(opportunity)
    
    def _create_crypto_analysis_prompt(self, opp: CryptoOpportunity) -> str:
        """Create analysis prompt for crypto opportunity"""
        
        prompt = f"""Analyze this cryptocurrency trading opportunity and provide your confidence assessment.

**Asset:** {opp.symbol} ({opp.base_asset})
**Current Price:** ${opp.current_price:,.2f}
**24h Change:** {opp.change_pct_24h:+.2f}%
**24h Volume:** ${opp.volume_24h:,.0f}
**Volume Ratio:** {opp.volume_ratio:.2f}x average
**Volatility (24h):** {opp.volatility_24h:.2f}%

**Quantitative Score:** {opp.score:.1f}/100
**Strategy:** {opp.strategy}
**Risk Level:** {opp.risk_level}
**Reason:** {opp.reason}

**Context:**
- Crypto markets trade 24/7 with no weekend breaks
- Higher volatility is normal in crypto (3-5x stocks)
- Social sentiment and narratives drive crypto prices significantly
- This is for {opp.strategy} trading strategy

**Your Task:**
Provide a JSON response with:
1. **confidence**: "HIGH", "MEDIUM", or "LOW" - your overall confidence in this trade
2. **rating**: Number 0-10 - how strong is this opportunity?
3. **reasoning**: 2-3 sentences explaining WHY this is a good/bad trade right now
4. **risks**: Main risks to watch for (2-3 bullet points)
5. **social_narrative**: Current market narrative for this crypto (if known)
6. **market_cycle**: Where we are in the cycle: "ACCUMULATION", "MARKUP", "DISTRIBUTION", "MARKDOWN", or "UNKNOWN"

Consider:
- Is the volume surge sustainable or a pump?
- Does the volatility present opportunity or danger?
- What's the current sentiment around this crypto?
- Is the strategy (scalp/momentum/swing) appropriate for current conditions?
- Any recent news or events affecting this asset?

Respond ONLY with valid JSON, no extra text:
```json
{{
  "confidence": "HIGH|MEDIUM|LOW",
  "rating": 0-10,
  "reasoning": "...",
  "risks": "...",
  "social_narrative": "...",
  "market_cycle": "..."
}}
```"""
        return prompt
    
    def _query_llm(self, prompt: str, symbol: str) -> str:
        """Query LLM with the analysis prompt"""
        try:
            if not self.llm_analyzer:
                return ""
              # Try hybrid analyzer first
            if hasattr(self.llm_analyzer, 'analyze_with_llm'):
                response = self.llm_analyzer.analyze_with_llm(prompt)  # type: ignore
            else:
                # Fallback to original method
                response = self.llm_analyzer._call_openrouter(prompt)
            
            return response if response else ""
            
        except Exception as e:
            logger.error(f"LLM query failed for {symbol}: {e}")
            return ""
    
    def _parse_llm_response(self, response: str, opp: CryptoOpportunity) -> Dict:
        """Parse LLM response and extract AI analysis data"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            # Validate and normalize data
            return {
                'confidence': data.get('confidence', 'MEDIUM').upper(),
                'rating': float(data.get('rating', 5.0)),
                'reasoning': data.get('reasoning', 'Analysis unavailable'),
                'risks': data.get('risks', 'Standard crypto market risks apply'),
                'social_narrative': data.get('social_narrative', 'Unknown'),
                'market_cycle': data.get('market_cycle', 'UNKNOWN').upper()
            }
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response for {opp.symbol}: {e}")
            logger.debug(f"Response was: {response}")
            
            # Return fallback data
            return {
                'confidence': 'MEDIUM',
                'rating': 5.0,
                'reasoning': 'AI analysis temporarily unavailable',
                'risks': 'Standard market risks apply',
                'social_narrative': 'Unknown',
                'market_cycle': 'UNKNOWN'
            }
    
    def _rule_based_confidence(self, opportunity: CryptoOpportunity) -> AICryptoOpportunity:
        """
        Fallback rule-based confidence when LLM is not available
        
        Args:
            opportunity: Base opportunity
            
        Returns:
            AICryptoOpportunity with rule-based analysis
        """
        # Determine confidence based on quantitative score
        if opportunity.score >= 80:
            ai_confidence = "HIGH"
            ai_rating = 8.0
        elif opportunity.score >= 65:
            ai_confidence = "MEDIUM"
            ai_rating = 6.0
        else:
            ai_confidence = "LOW"
            ai_rating = 4.0
        
        # Adjust for risk level
        risk_adjustments = {
            'LOW': 1.0,
            'MEDIUM': 0.5,
            'HIGH': 0.0,
            'EXTREME': -1.0
        }
        ai_rating += risk_adjustments.get(opportunity.risk_level, 0.0)
        ai_rating = max(0.0, min(10.0, ai_rating))  # Clamp to 0-10
        
        # Generate reasoning
        reasoning = f"Score: {opportunity.score:.1f}/100. {opportunity.reason}"
        
        # Generate risks based on risk level and volatility
        if opportunity.risk_level == 'EXTREME':
            risks = "⚠️ EXTREME volatility - tight stops essential. High risk of sudden reversal."
        elif opportunity.risk_level == 'HIGH':
            risks = "High volatility requires careful position sizing. Monitor closely for reversals."
        elif opportunity.volatility_24h > 10:
            risks = "Elevated volatility present. Use appropriate stop losses."
        else:
            risks = "Standard crypto market risks. Use proper risk management."
        
        # Determine market cycle based on momentum
        if opportunity.change_pct_24h > 10:
            market_cycle = "MARKUP"
        elif opportunity.change_pct_24h < -10:
            market_cycle = "MARKDOWN"
        else:
            market_cycle = "UNKNOWN"
        
        return AICryptoOpportunity(
            **asdict(opportunity),
            ai_confidence=ai_confidence,
            ai_reasoning=reasoning,
            ai_risks=risks,
            ai_rating=ai_rating,
            social_narrative="Analysis based on technical indicators",
            market_cycle_phase=market_cycle
        )
    
    def get_buzzing_cryptos(self, top_n: int = 10) -> List[AICryptoOpportunity]:
        """
        Find buzzing/trending cryptocurrencies
        Focus on volume surges and social momentum
        
        Args:
            top_n: Number of results to return
            
        Returns:
            List of buzzing crypto opportunities
        """
        logger.info(f"🔥 Scanning for {top_n} buzzing cryptocurrencies...")
        
        opportunities = self.scan_with_ai_confidence(
            strategy='MOMENTUM',
            top_n=top_n,
            min_score=50.0  # Lower threshold for buzzing assets
        )
        
        # Filter and re-sort by volume ratio (buzzing = high volume)
        buzzing = [opp for opp in opportunities if opp.volume_ratio >= 1.5]
        buzzing.sort(key=lambda x: x.volume_ratio, reverse=True)
        
        return buzzing[:top_n]
    
    def get_hottest_cryptos(self, top_n: int = 10) -> List[AICryptoOpportunity]:
        """
        Find the hottest cryptocurrencies with strongest momentum
        Focus on price action and momentum
        
        Args:
            top_n: Number of results to return
            
        Returns:
            List of hottest crypto opportunities
        """
        logger.info(f"🌶️ Scanning for {top_n} hottest cryptocurrencies...")
        
        opportunities = self.scan_with_ai_confidence(
            strategy='MOMENTUM',
            top_n=top_n * 2,
            min_score=60.0
        )
        
        # Filter for strong momentum and re-sort
        hottest = [opp for opp in opportunities if abs(opp.change_pct_24h) >= 3.0]
        hottest.sort(key=lambda x: abs(x.change_pct_24h), reverse=True)
        
        return hottest[:top_n]


# Convenience function for quick access
def create_ai_crypto_scanner(kraken_client: KrakenClient, config=None) -> AICryptoScanner:
    """Create and return an AI crypto scanner instance"""
    return AICryptoScanner(kraken_client, config, use_llm=True)

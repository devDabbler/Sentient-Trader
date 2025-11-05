"""
AI-Enhanced Confidence Scanner

Uses LLM to provide intelligent confidence analysis for top trades.
Adds AI reasoning on top of quantitative scoring.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
from .top_trades_scanner import TopTradesScanner, TopTrade
from dotenv import load_dotenv

# Load environment variables
load_dotenv()



@dataclass
class AIConfidenceTrade(TopTrade):
    """Extended trade with AI confidence analysis"""
    ai_confidence: str = "MEDIUM"  # AI-assessed confidence
    ai_reasoning: str = ""  # AI explanation
    ai_risks: str = ""  # AI-identified risks
    ai_rating: float = 0.0  # 0-10 rating from AI


class AIConfidenceScanner:
    """
    Scanner that adds AI-powered confidence analysis to top trades.
    Can work with or without LLM API keys.
    """
    
    def __init__(self, use_llm: bool = None):
        """
        Initialize AI scanner
        
        Args:
            use_llm: Whether to use LLM for analysis. 
                    If None, auto-detect based on API keys.
        """
        self.scanner = TopTradesScanner()
        
        # Auto-detect if we should use LLM
        if use_llm is None:
            self.use_llm = self._check_llm_available()
        else:
            self.use_llm = use_llm
        
        if self.use_llm:
            try:
                from .llm_strategy_analyzer import LLMStrategyAnalyzer
                api_key = os.getenv('OPENROUTER_API_KEY')
                model = os.getenv('AI_CONFIDENCE_MODEL', 'google/gemini-2.0-flash-exp:free')
                
                if not api_key:
                    logger.error("❌ OPENROUTER_API_KEY not found - AI analysis disabled")
                    self.use_llm = False
                    self.llm_analyzer = None
                else:
                    self.llm_analyzer = LLMStrategyAnalyzer(provider="openrouter", model=model)
                    logger.info(f"✅ AI Confidence Scanner initialized with OpenRouter")
                    logger.info(f"   Model: {model}")
                    logger.info(f"   API Key: {'*' * (len(api_key) - 8) + api_key[-8:]}")
            except Exception as e:
                logger.error(f"❌ LLM initialization failed: {e}", exc_info=True)
                self.use_llm = False
    
    def _check_llm_available(self) -> bool:
        """Check if OpenRouter LLM API key is available"""
        has_key = bool(os.getenv('OPENROUTER_API_KEY'))
        if not has_key:
            logger.warning("⚠️ OpenRouter API key not found in environment")
        return has_key
    
    def _generate_ai_confidence(self, trade: TopTrade, trade_type: str) -> Dict:
        """
        Generate AI confidence analysis for a trade
        
        Args:
            trade: TopTrade object
            trade_type: 'options' or 'penny_stock'
        
        Returns:
            Dict with ai_confidence, ai_reasoning, ai_risks, ai_rating
        """
        if not self.use_llm:
            # Fallback to rule-based confidence
            return self._rule_based_confidence(trade, trade_type)
        
        try:
            # Create prompt for LLM
            prompt = self._create_analysis_prompt(trade, trade_type)
            
            # Get LLM response
            response = self._query_llm(prompt, ticker=trade.ticker)
            
            # Parse response
            return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"Error generating AI confidence: {e}")
            return self._rule_based_confidence(trade, trade_type)
    
    def _rule_based_confidence(self, trade: TopTrade, trade_type: str) -> Dict:
        """
        Fallback rule-based confidence when LLM not available
        """
        # Calculate AI confidence based on score and other factors
        score = trade.score
        
        if score >= 85:
            ai_conf = "VERY HIGH"
            # Cap at 10.0: 9.0 + min((score-85)/15, 1.0)
            ai_rating = min(10.0, 9.0 + (score - 85) / 15)
        elif score >= 75:
            ai_conf = "HIGH"
            ai_rating = min(10.0, 7.5 + (score - 75) / 10)
        elif score >= 60:
            ai_conf = "MEDIUM-HIGH"
            ai_rating = min(10.0, 6.0 + (score - 60) / 15)
        elif score >= 45:
            ai_conf = "MEDIUM"
            ai_rating = min(10.0, 4.5 + (score - 45) / 15)
        else:
            ai_conf = "LOW"
            ai_rating = max(1.0, min(10.0, score / 45 * 4.5))
        
        # Generate reasoning
        reasons = []
        
        if trade_type == 'options':
            if trade.volume_ratio > 2.5:
                reasons.append(f"Exceptional volume ({trade.volume_ratio:.1f}x average) indicates strong interest")
            if abs(trade.change_pct) > 5:
                reasons.append(f"Significant price movement ({trade.change_pct:+.1f}%) creates opportunities")
            if trade.confidence in ['HIGH', 'VERY HIGH']:
                reasons.append("Statistical indicators show high probability setup")
        else:  # penny_stock
            if trade.score >= 75:
                reasons.append("Strong composite score across momentum, valuation, and catalysts")
            if trade.volume_ratio > 2.0:
                reasons.append(f"Volume spike ({trade.volume_ratio:.1f}x) suggests increasing attention")
            if trade.confidence == 'VERY HIGH':
                reasons.append("Multiple bullish indicators aligned")
        
        ai_reasoning = " | ".join(reasons) if reasons else "Standard quantitative analysis supports this opportunity."
        
        # Identify risks
        risks = []
        
        if trade.risk_level == 'H':
            risks.append("High risk classification requires careful position sizing")
        if trade_type == 'options' and abs(trade.change_pct) < 1:
            risks.append("Low price movement may limit options profitability")
        if trade_type == 'penny_stock' and trade.score < 50:
            risks.append("Below-average scores suggest elevated uncertainty")
        
        ai_risks = " | ".join(risks) if risks else "Standard market risks apply; use appropriate position sizing."
        
        return {
            'ai_confidence': ai_conf,
            'ai_reasoning': ai_reasoning,
            'ai_risks': ai_risks,
            'ai_rating': round(ai_rating, 1)
        }
    
    def _create_analysis_prompt(self, trade: TopTrade, trade_type: str) -> str:
        """Create prompt for LLM analysis"""
        return f"""
You are an expert trading analyst. Analyze this {trade_type} opportunity:

Ticker: {trade.ticker}
Score: {trade.score}/100
Price: ${trade.price}
Change: {trade.change_pct:+.2f}%
Volume Ratio: {trade.volume_ratio:.2f}x
Quantitative Confidence: {trade.confidence}
Risk Level: {trade.risk_level}
Reason: {trade.reason}

Provide:
1. AI Confidence Level (VERY HIGH/HIGH/MEDIUM-HIGH/MEDIUM/LOW)
2. AI Reasoning (why this is or isn't a good trade)
3. AI-Identified Risks (what could go wrong)
4. AI Rating (0-10 scale)

Be concise but insightful. Focus on actionable analysis.
"""
    
    def _query_llm(self, prompt: str, ticker: str = None) -> str:
        """Query LLM for analysis"""
        try:
            # Use the LLM analyzer's _call_llm method directly
            system_prompt = """You are a professional trading analyst. Provide concise, actionable insights in this exact format:

1. AI Confidence: [VERY HIGH/HIGH/MEDIUM-HIGH/MEDIUM/LOW]
2. AI Rating: [0-10]
3. AI Reasoning: [Why this trade - be specific]
4. AI Risks: [What could go wrong - be specific]

Be concise but insightful. Focus on actionable analysis."""
            
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Call LLM directly
            ticker_str = f" on {ticker}" if ticker else ""
            logger.info(f"🤖 Querying LLM for AI confidence analysis{ticker_str}...")
            response = self.llm_analyzer._call_openrouter(full_prompt)
            
            if not response:
                logger.error(f"❌ Empty LLM response{ticker_str}")
                raise Exception("Empty LLM response")
            
            logger.info(f"✅ Received LLM response{ticker_str} ({len(response)} characters)")
            return response
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured data"""
        import re
        
        result = {
            'ai_confidence': 'MEDIUM',
            'ai_reasoning': 'Standard analysis supports this opportunity.',
            'ai_risks': 'Standard market risks apply.',
            'ai_rating': 5.0
        }
        
        try:
            # Parse line by line
            lines = response.strip().split('\n')
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                
                line_lower = line_stripped.lower()
                
                # Parse AI Confidence
                if 'ai confidence' in line_lower or (line_stripped.startswith('1.') and 'confidence' in line_lower):
                    if 'very high' in line_lower:
                        result['ai_confidence'] = 'VERY HIGH'
                    elif 'medium-high' in line_lower or 'medium high' in line_lower:
                        result['ai_confidence'] = 'MEDIUM-HIGH'
                    elif 'high' in line_lower:
                        result['ai_confidence'] = 'HIGH'
                    elif 'low' in line_lower:
                        result['ai_confidence'] = 'LOW'
                    elif 'medium' in line_lower:
                        result['ai_confidence'] = 'MEDIUM'
                
                # Parse AI Rating
                elif 'ai rating' in line_lower or (line_stripped.startswith('2.') and 'rating' in line_lower):
                    # Extract text after colon to avoid list number prefix
                    if ':' in line_stripped:
                        rating_text = line_stripped.split(':', 1)[1].strip()
                        numbers = re.findall(r'\d+\.?\d*', rating_text)
                    else:
                        numbers = re.findall(r'\d+\.?\d*', line_stripped)
                    
                    if numbers:
                        rating = float(numbers[0])
                        # Ensure 0-10 range
                        result['ai_rating'] = max(0.0, min(10.0, rating))
                
                # Parse AI Reasoning
                elif 'ai reasoning' in line_lower or (line_stripped.startswith('3.') and 'reasoning' in line_lower):
                    # Get the text after the colon
                    if ':' in line_stripped:
                        reasoning = line_stripped.split(':', 1)[1].strip()
                        # Also grab next lines if they don't start with a number
                        for j in range(i+1, min(i+5, len(lines))):
                            next_line = lines[j].strip()
                            if next_line and not re.match(r'^\d+\.', next_line):
                                reasoning += ' ' + next_line
                            else:
                                break
                        result['ai_reasoning'] = reasoning
                
                # Parse AI Risks
                elif 'ai risk' in line_lower or (line_stripped.startswith('4.') and 'risk' in line_lower):
                    # Get the text after the colon
                    if ':' in line_stripped:
                        risks = line_stripped.split(':', 1)[1].strip()
                        # Also grab next lines if they don't start with a number
                        for j in range(i+1, min(i+5, len(lines))):
                            next_line = lines[j].strip()
                            if next_line and not re.match(r'^\d+\.', next_line):
                                risks += ' ' + next_line
                            else:
                                break
                        result['ai_risks'] = risks
            
            # Ensure we have valid content
            if not result['ai_reasoning'] or len(result['ai_reasoning']) < 10:
                result['ai_reasoning'] = 'Trade shows favorable characteristics based on quantitative analysis.'
            
            if not result['ai_risks'] or len(result['ai_risks']) < 10:
                result['ai_risks'] = 'Monitor standard market risks and use appropriate position sizing.'
            
            # Log successful parse
            logger.debug(f"Parsed: confidence={result['ai_confidence']}, rating={result['ai_rating']}")
                
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response was: {response}")
        
        return result
    
    def scan_top_options_with_ai(self, top_n: int = 20, 
                                  min_ai_rating: float = 5.0,
                                  min_score: float = 50.0,
                                  min_price: float = None,
                                  max_price: float = None) -> List[AIConfidenceTrade]:
        """
        Scan for top options with AI confidence analysis
        
        Args:
            top_n: Number of trades to scan (scans more, returns filtered)
            min_ai_rating: Minimum AI rating to include (0-10), default 5.0
            min_score: Minimum quantitative score (0-100), default 50.0
            min_price: Minimum stock price filter (optional)
            max_price: Maximum stock price filter (optional)
        
        Returns:
            List of AIConfidenceTrade objects with AI analysis (quality filtered)
        """
        logger.info(f"Scanning top {top_n} options with AI confidence (min rating: {min_ai_rating}, min score: {min_score})...")
        
        # Get base trades - scan more to ensure we have enough after filtering
        scan_count = min(top_n * 3, 50)
        trades = self.scanner.scan_top_options_trades(top_n=scan_count)
        logger.info(f"Base scanner returned {len(trades)} trades before AI filtering")
        
        # EARLY PRICE FILTERING (before expensive LLM calls)
        if min_price or max_price:
            price_filtered = []
            price_rejected = 0
            for trade in trades:
                if min_price and trade.price < min_price:
                    price_rejected += 1
                    continue
                if max_price and trade.price > max_price:
                    price_rejected += 1
                    logger.debug(f"💰 Price Filter: Skipping {trade.ticker} (${trade.price:.2f} > ${max_price:.2f})")
                    continue
                price_filtered.append(trade)
            
            if price_rejected > 0:
                logger.info(f"💰 EARLY Price Filter: Removed {price_rejected} trades outside ${min_price or 0:.2f}-${max_price or 999999:.2f} range (BEFORE AI analysis)")
            trades = price_filtered
        
        # Add AI analysis to each
        ai_trades = []
        skipped_low_score = 0
        skipped_low_ai = 0
        
        for trade in trades:
            # Skip low-quality trades early
            if trade.score < min_score:
                skipped_low_score += 1
                logger.debug(f"Skipping {trade.ticker}: score {trade.score:.1f} < min {min_score}")
                continue
                
            ai_analysis = self._generate_ai_confidence(trade, 'options')
            
            # Only include if meets minimum rating AND score
            if ai_analysis['ai_rating'] >= min_ai_rating and trade.score >= min_score:
                logger.info(f"✓ {trade.ticker}: score={trade.score:.1f}, AI rating={ai_analysis['ai_rating']:.1f}, confidence={ai_analysis['ai_confidence']}")
                ai_trade = AIConfidenceTrade(
                    ticker=trade.ticker,
                    score=trade.score,
                    price=trade.price,
                    change_pct=trade.change_pct,
                    volume=trade.volume,
                    volume_ratio=trade.volume_ratio,
                    reason=trade.reason,
                    trade_type='options',
                    confidence=trade.confidence,
                    risk_level=trade.risk_level,
                    ai_confidence=ai_analysis['ai_confidence'],
                    ai_reasoning=ai_analysis['ai_reasoning'],
                    ai_risks=ai_analysis['ai_risks'],
                    ai_rating=ai_analysis['ai_rating']
                )
                ai_trades.append(ai_trade)
            else:
                skipped_low_ai += 1
                logger.info(f"✗ {trade.ticker}: AI rating {ai_analysis['ai_rating']:.1f} < min {min_ai_rating} (score was {trade.score:.1f})")
            
            # Stop if we have enough quality trades
            if len(ai_trades) >= top_n:
                break
        
        # Sort by AI rating (primary) and score (secondary)
        ai_trades.sort(key=lambda x: (x.ai_rating, x.score), reverse=True)
        
        # Return only top_n after filtering
        filtered_trades = ai_trades[:top_n]
        logger.info(f"Found {len(filtered_trades)} quality AI-analyzed options trades (from {len(trades)} scanned)")
        logger.info(f"Filtering summary: {skipped_low_score} skipped for low score, {skipped_low_ai} skipped for low AI rating")
        return filtered_trades
    
    def scan_top_penny_stocks_with_ai(self, top_n: int = 20,
                                      min_ai_rating: float = 5.0,
                                      min_score: float = 55.0) -> List[AIConfidenceTrade]:
        """
        Scan for top penny stocks with AI confidence analysis
        
        Args:
            top_n: Number of stocks to scan (scans more, returns filtered)
            min_ai_rating: Minimum AI rating to include (0-10), default 5.0
            min_score: Minimum composite score (0-100), default 55.0
        
        Returns:
            List of AIConfidenceTrade objects with AI analysis (quality filtered)
        """
        logger.info(f"Scanning top {top_n} penny stocks with AI confidence (min rating: {min_ai_rating}, min score: {min_score})...")
        
        # Get base trades - scan more to ensure we have enough after filtering
        scan_count = min(top_n * 2, 30)
        trades = self.scanner.scan_top_penny_stocks(top_n=scan_count)
        
        # Add AI analysis to each
        ai_trades = []
        for trade in trades:
            # Skip low-quality trades early
            if trade.score < min_score:
                continue
                
            ai_analysis = self._generate_ai_confidence(trade, 'penny_stock')
            
            # Only include if meets minimum rating AND score
            if ai_analysis['ai_rating'] >= min_ai_rating and trade.score >= min_score:
                ai_trade = AIConfidenceTrade(
                    ticker=trade.ticker,
                    score=trade.score,
                    price=trade.price,
                    change_pct=trade.change_pct,
                    volume=trade.volume,
                    volume_ratio=trade.volume_ratio,
                    reason=trade.reason,
                    trade_type='penny_stock',
                    confidence=trade.confidence,
                    risk_level=trade.risk_level,
                    ai_confidence=ai_analysis['ai_confidence'],
                    ai_reasoning=ai_analysis['ai_reasoning'],
                    ai_risks=ai_analysis['ai_risks'],
                    ai_rating=ai_analysis['ai_rating']
                )
                ai_trades.append(ai_trade)
            
            # Stop if we have enough quality trades
            if len(ai_trades) >= top_n:
                break
        
        # Sort by AI rating (primary) and score (secondary)
        ai_trades.sort(key=lambda x: (x.ai_rating, x.score), reverse=True)
        
        # Return only top_n after filtering
        filtered_trades = ai_trades[:top_n]
        logger.info(f"Found {len(filtered_trades)} quality AI-analyzed penny stocks (from {len(trades)} scanned)")
        return filtered_trades
    
    def get_ai_insights(self, trades: List[AIConfidenceTrade]) -> Dict:
        """Generate AI insights summary"""
        if not trades:
            return {
                'total': 0,
                'avg_ai_rating': 0,
                'very_high_confidence': 0,
                'top_pick': None,
                'avg_quant_score': 0
            }
        
        return {
            'total': len(trades),
            'avg_ai_rating': round(sum(t.ai_rating for t in trades) / len(trades), 1),
            'very_high_confidence': len([t for t in trades if t.ai_confidence == 'VERY HIGH']),
            'top_pick': trades[0].ticker if trades else None,
            'avg_quant_score': round(sum(t.score for t in trades) / len(trades), 1),
            'llm_enabled': self.use_llm
        }

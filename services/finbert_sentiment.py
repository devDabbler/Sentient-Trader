"""
FinBERT Sentiment Analysis Service
Provides specialized financial sentiment analysis using FinBERT model
More accurate than general-purpose sentiment for financial news/social media

FinBERT is trained on financial texts and understands:
- Financial jargon ("bullish", "bear market", "rally", "correction")
- Context-specific sentiment (e.g., "stock plunged" is clearly negative)
- Nuanced market language
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger
import json

# Try to import transformers (optional dependency)
try:
    from transformers import BertTokenizer, BertForSequenceClassification, pipeline
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning(
        "⚠️ FinBERT dependencies not available. "
        "Install with: pip install transformers torch"
    )


@dataclass
class FinBERTSentiment:
    """FinBERT sentiment analysis result"""
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float  # 0-1 scale
    scores: Dict[str, float]  # {'positive': 0.85, 'negative': 0.10, 'neutral': 0.05}
    text_analyzed: str


class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT
    
    Model: yiyanghkust/finbert-tone
    - Trained on financial news and social media
    - 3 classes: positive, negative, neutral
    - High accuracy on financial texts
    """
    
    def __init__(self, use_gpu: bool = False, fallback_to_llm: bool = True):
        """
        Initialize FinBERT analyzer
        
        Args:
            use_gpu: Use GPU acceleration if available
            fallback_to_llm: If FinBERT unavailable, use LLM-based sentiment
        """
        self.use_gpu = use_gpu and torch.cuda.is_available() if FINBERT_AVAILABLE else False
        self.fallback_to_llm = fallback_to_llm
        self.finbert_available = FINBERT_AVAILABLE
        
        if FINBERT_AVAILABLE:
            try:
                logger.info("🧠 Loading FinBERT model (yiyanghkust/finbert-tone)...")
                
                # Use pipeline for simpler interface
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="yiyanghkust/finbert-tone",
                    tokenizer="yiyanghkust/finbert-tone",
                    device=0 if self.use_gpu else -1,
                    top_k=None  # Return all scores
                )
                
                logger.info(f"✅ FinBERT model loaded successfully (GPU: {self.use_gpu})")
                
            except Exception as e:
                logger.error(f"❌ Failed to load FinBERT model: {e}")
                self.finbert_available = False
                
                if fallback_to_llm:
                    logger.info("📡 Will use LLM-based sentiment analysis as fallback")
                    self._init_llm_fallback()
        else:
            if fallback_to_llm:
                logger.info("📡 Using LLM-based sentiment analysis (FinBERT not installed)")
                self._init_llm_fallback()
    
    def _init_llm_fallback(self):
        """Initialize LLM-based sentiment fallback using LLM Request Manager"""
        try:
            from services.llm_helper import get_llm_helper
            
            self.llm_helper = get_llm_helper("finbert_sentiment", default_priority="LOW")
            logger.info("✅ LLM sentiment fallback initialized with LLM Request Manager")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize LLM fallback: {e}")
            self.llm_helper = None
    
    def analyze_sentiment(
        self,
        text: str,
        return_all_scores: bool = True
    ) -> FinBERTSentiment:
        """
        Analyze financial sentiment of text
        
        Args:
            text: Text to analyze (news headline, tweet, article excerpt)
            return_all_scores: If True, return scores for all sentiments
        
        Returns:
            FinBERTSentiment with sentiment, confidence, and scores
        """
        if not text or not text.strip():
            return self._neutral_sentiment(text)
        
        # Try FinBERT first
        if self.finbert_available:
            try:
                return self._analyze_with_finbert(text, return_all_scores)
            except Exception as e:
                logger.error(f"FinBERT analysis failed: {e}")
                # Fall through to LLM fallback
        
        # Fallback to LLM-based sentiment
        if self.fallback_to_llm and hasattr(self, 'llm_helper') and self.llm_helper:
            try:
                return self._analyze_with_llm(text)
            except Exception as e:
                logger.error(f"LLM sentiment analysis failed: {e}")
        
        # Last resort: neutral sentiment
        return self._neutral_sentiment(text)
    
    def _analyze_with_finbert(
        self,
        text: str,
        return_all_scores: bool = True
    ) -> FinBERTSentiment:
        """Analyze using FinBERT model"""
        
        # Truncate text to model's max length (512 tokens)
        # Roughly 4 chars per token, so ~2000 chars max
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated text to {max_chars} chars for FinBERT")
        
        # Run sentiment analysis
        results = self.sentiment_pipeline(text)
        
        # Results format: [[{'label': 'positive', 'score': 0.85}, ...]]
        if not results or not results[0]:
            return self._neutral_sentiment(text)
        
        # Get all scores
        scores_dict = {}
        for result in results[0]:
            label = result['label'].lower()
            score = result['score']
            scores_dict[label] = score
        
        # Find dominant sentiment
        dominant_sentiment = max(scores_dict, key=scores_dict.get)
        confidence = scores_dict[dominant_sentiment]
        
        return FinBERTSentiment(
            sentiment=dominant_sentiment,
            confidence=confidence,
            scores=scores_dict,
            text_analyzed=text[:100] + "..." if len(text) > 100 else text
        )
    
    def _analyze_with_llm(self, text: str) -> FinBERTSentiment:
        """Fallback: Analyze using LLM Request Manager"""
        
        prompt = f"""Analyze the financial sentiment of this text.

Text: \"\"\"{text}\"\"\"

Respond in JSON format:
{{
    "label": "positive", "negative", or "neutral",
    "score": confidence score 0.0-1.0,
    "positive": positive score 0.0-1.0,
    "negative": negative score 0.0-1.0,
    "neutral": neutral score 0.0-1.0
}}

Be precise and consider financial context.
"""
        
        try:
            # Use LOW priority with caching (1 min TTL for sentiment analysis)
            import hashlib
            cache_key = f"sentiment_{hashlib.md5(text.encode()).hexdigest()[:16]}"
            response = self.llm_helper.low_request(
                prompt,
                cache_key=cache_key,
                ttl=60,  # 1 minute cache
                temperature=0.2  # Low temperature for consistent sentiment
            )
            
            if not response:
                return self._neutral_sentiment(text)
            
            # Parse JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return self._neutral_sentiment(text)
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            sentiment = data.get('sentiment', 'neutral').lower()
            confidence = float(data.get('confidence', 0.5))
            
            # Create scores dict
            scores = {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0
            }
            scores[sentiment] = confidence
            
            # Distribute remaining probability
            remaining = 1.0 - confidence
            other_sentiments = [s for s in scores.keys() if s != sentiment]
            for s in other_sentiments:
                scores[s] = remaining / len(other_sentiments)
            
            return FinBERTSentiment(
                sentiment=sentiment,
                confidence=confidence,
                scores=scores,
                text_analyzed=text[:100] + "..." if len(text) > 100 else text
            )
            
        except Exception as e:
            logger.error(f"Error in LLM sentiment analysis: {e}")
            return self._neutral_sentiment(text)
    
    def _neutral_sentiment(self, text: str) -> FinBERTSentiment:
        """Return neutral sentiment as fallback"""
        return FinBERTSentiment(
            sentiment='neutral',
            confidence=0.5,
            scores={'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
            text_analyzed=text[:100] + "..." if len(text) > 100 else text
        )
    
    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> List[FinBERTSentiment]:
        """
        Analyze multiple texts efficiently in batches
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
        
        Returns:
            List of FinBERTSentiment results
        """
        results = []
        
        if not self.finbert_available:
            # Process one by one with LLM fallback
            for text in texts:
                results.append(self.analyze_sentiment(text))
            return results
        
        # Process in batches for efficiency
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # FinBERT pipeline handles batches automatically
                batch_results = self.sentiment_pipeline(batch)
                
                for text, result in zip(batch, batch_results):
                    # Parse result
                    scores_dict = {}
                    for item in result:
                        label = item['label'].lower()
                        score = item['score']
                        scores_dict[label] = score
                    
                    dominant_sentiment = max(scores_dict, key=scores_dict.get)
                    confidence = scores_dict[dominant_sentiment]
                    
                    results.append(FinBERTSentiment(
                        sentiment=dominant_sentiment,
                        confidence=confidence,
                        scores=scores_dict,
                        text_analyzed=text[:100] + "..." if len(text) > 100 else text
                    ))
                    
            except Exception as e:
                logger.error(f"Batch analysis failed: {e}")
                # Fall back to individual analysis
                for text in batch:
                    results.append(self.analyze_sentiment(text))
        
        return results
    
    def get_aggregated_sentiment(
        self,
        sentiments: List[FinBERTSentiment],
        weighted: bool = True
    ) -> Dict:
        """
        Aggregate multiple sentiment analyses
        
        Args:
            sentiments: List of FinBERTSentiment results
            weighted: If True, weight by confidence scores
        
        Returns:
            Dict with aggregated sentiment analysis
        """
        if not sentiments:
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 50.0,  # 0-100 scale
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_analyzed': 0
            }
        
        positive_count = sum(1 for s in sentiments if s.sentiment == 'positive')
        negative_count = sum(1 for s in sentiments if s.sentiment == 'negative')
        neutral_count = sum(1 for s in sentiments if s.sentiment == 'neutral')
        
        if weighted:
            # Weight by confidence
            positive_weight = sum(s.confidence for s in sentiments if s.sentiment == 'positive')
            negative_weight = sum(s.confidence for s in sentiments if s.sentiment == 'negative')
            neutral_weight = sum(s.confidence for s in sentiments if s.sentiment == 'neutral')
            
            total_weight = positive_weight + negative_weight + neutral_weight
            
            if total_weight > 0:
                # Calculate sentiment score (0-100: 0=very bearish, 50=neutral, 100=very bullish)
                sentiment_score = ((positive_weight - negative_weight) / total_weight) * 50 + 50
            else:
                sentiment_score = 50.0
            
            avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)
        else:
            # Simple counting
            total = len(sentiments)
            sentiment_score = ((positive_count - negative_count) / total) * 50 + 50
            avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)
        
        # Determine overall sentiment
        if sentiment_score > 60:
            overall_sentiment = 'bullish'
        elif sentiment_score < 40:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': sentiment_score,
            'confidence': avg_confidence,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_analyzed': len(sentiments),
            'weighted': weighted
        }


# Singleton instance for easy access
_finbert_analyzer = None

def get_finbert_analyzer() -> FinBERTSentimentAnalyzer:
    """Get or create singleton FinBERT analyzer instance"""
    global _finbert_analyzer
    
    if _finbert_analyzer is None:
        _finbert_analyzer = FinBERTSentimentAnalyzer(
            use_gpu=False,  # Set to True if you have GPU
            fallback_to_llm=True
        )
    
    return _finbert_analyzer


# Example usage
if __name__ == "__main__":
    # Test the analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    test_texts = [
        "Bitcoin ETF sees $850M inflow, largest since May",
        "Crypto exchange reports security breach, $50M stolen",
        "Federal Reserve maintains interest rates at current levels",
        "Stock market rallies on strong earnings reports",
        "Company announces layoffs due to economic uncertainty"
    ]
    
    print("=" * 80)
    print("FinBERT Sentiment Analysis Test")
    print("=" * 80)
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result.sentiment.upper()} (Confidence: {result.confidence:.2%})")
        print(f"Scores: {result.scores}")
    
    print("\n" + "=" * 80)
    print("Aggregated Sentiment")
    print("=" * 80)
    
    results = [analyzer.analyze_sentiment(text) for text in test_texts]
    aggregated = analyzer.get_aggregated_sentiment(results, weighted=True)
    print(f"Overall: {aggregated['overall_sentiment'].upper()}")
    print(f"Sentiment Score: {aggregated['sentiment_score']:.1f}/100")
    print(f"Average Confidence: {aggregated['confidence']:.2%}")
    print(f"Distribution: +{aggregated['positive_count']} / ~{aggregated['neutral_count']} / -{aggregated['negative_count']}")


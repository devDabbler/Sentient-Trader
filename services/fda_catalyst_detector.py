"""
FDA and Healthcare Catalyst Detection Service

Detects FDA approvals, clinical trials, and healthcare news that can drive
massive price movements in pharma/biotech penny stocks.
"""

from loguru import logger
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re



# Healthcare/Pharma sector keywords and classifications
HEALTHCARE_SECTORS = {
    'pharma': ['pharmaceutical', 'pharma', 'drug', 'medicine', 'therapeutics'],
    'biotech': ['biotech', 'biopharmaceutical', 'genomics', 'gene therapy', 'biologics'],
    'medical_device': ['medical device', 'surgical', 'implant', 'diagnostic'],
    'healthcare': ['healthcare', 'health services', 'hospital', 'clinic']
}

# FDA-related keywords (prioritized by impact)
FDA_CATALYSTS = {
    # Highest impact (30-50+ points)
    'approval': {
        'keywords': ['fda approval', 'approved by fda', 'nda approval', 'bla approval', 
                    'breakthrough therapy', 'priority review', 'accelerated approval',
                    'fda clearance', '510(k) clearance', 'pma approval'],
        'score': 50,
        'confidence': 'VERY HIGH'
    },
    
    # High impact (25-40 points)
    'positive_trial': {
        'keywords': ['phase 3 success', 'phase 3 positive', 'met primary endpoint',
                    'met all endpoints', 'statistically significant', 'trial success',
                    'positive data', 'positive results', 'successful trial'],
        'score': 40,
        'confidence': 'HIGH'
    },
    
    # Medium-high impact (20-30 points)
    'trial_advancement': {
        'keywords': ['phase 3 trial', 'phase 3 study', 'pivotal trial', 
                    'registration trial', 'initiated phase 3', 'dosing first patient',
                    'fast track designation', 'orphan drug', 'rare disease'],
        'score': 30,
        'confidence': 'HIGH'
    },
    
    # Medium impact (15-25 points)
    'phase_2': {
        'keywords': ['phase 2 positive', 'phase 2 success', 'phase 2 results',
                    'phase 2 data', 'initiated phase 2', 'interim data'],
        'score': 25,
        'confidence': 'MEDIUM-HIGH'
    },
    
    # Lower-medium impact (10-20 points)
    'early_stage': {
        'keywords': ['phase 1', 'preclinical', 'ind application', 'investigational',
                    'early stage', 'research collaboration', 'drug candidate'],
        'score': 15,
        'confidence': 'MEDIUM'
    },
    
    # Regulatory milestones (15-30 points)
    'regulatory': {
        'keywords': ['fda meeting', 'pdufa date', 'prescription drug user fee',
                    'advisory committee', 'regulatory submission', 'rolling review',
                    'fda feedback', 'type c meeting'],
        'score': 20,
        'confidence': 'MEDIUM-HIGH'
    },
    
    # Partnerships (20-35 points)
    'partnership': {
        'keywords': ['licensing deal', 'collaboration agreement', 'strategic partnership',
                    'co-development', 'option agreement', 'acquisition', 'merger'],
        'score': 30,
        'confidence': 'HIGH'
    },
    
    # Negative catalysts (penalties)
    'negative': {
        'keywords': ['fda rejection', 'crl issued', 'complete response letter',
                    'trial failed', 'missed endpoint', 'halted trial', 'safety concern',
                    'fda warning', 'adverse event', 'trial terminated'],
        'score': -40,
        'confidence': 'HIGH'
    }
}

# Disease area focus (some have higher commercial potential)
HIGH_VALUE_INDICATIONS = [
    'oncology', 'cancer', 'tumor', 'carcinoma',
    'alzheimer', 'parkinson', 'neurodegenerative',
    'diabetes', 'cardiovascular', 'obesity',
    'rare disease', 'orphan drug',
    'covid', 'pandemic', 'infectious disease'
]


@dataclass
class FDACatalyst:
    """Container for FDA catalyst information"""
    ticker: str
    catalyst_type: str
    description: str
    score_boost: float
    confidence: str
    keywords_found: List[str]
    news_date: Optional[datetime] = None
    is_healthcare: bool = False
    sector_type: str = ""


class FDACatalystDetector:
    """Detects FDA and healthcare catalysts for stocks"""
    
    def __init__(self):
        from loguru import logger
        self.logger = logger
        self.fda_keywords = [keyword for catalyst_info in FDA_CATALYSTS.values() for keyword in catalyst_info['keywords']]
    
    def is_healthcare_stock(self, ticker: str, info: Dict = None) -> Tuple[bool, str]:
        """
        Determine if a stock is in healthcare/pharma sector
        
        Args:
            ticker: Stock ticker
            info: Optional yfinance info dict to avoid extra API calls
            
        Returns:
            Tuple of (is_healthcare: bool, sector_type: str)
        """
        try:
            if info is None:
                stock = yf.Ticker(ticker)
                info = stock.info
            
            # Check sector and industry
            sector = info.get('sector', '').lower()
            industry = info.get('industry', '').lower()
            business_summary = info.get('longBusinessSummary', '').lower()
            
            # Combine all text for comprehensive check
            combined_text = f"{sector} {industry} {business_summary}"
            
            # Check against each healthcare category
            for category, keywords in HEALTHCARE_SECTORS.items():
                for keyword in keywords:
                    if keyword in combined_text:
                        self.logger.info(f"✅ {ticker} identified as {category} stock")
                        return True, category
            
            return False, ""
            
        except Exception as e:
            self.logger.debug(f"Error checking healthcare status for {ticker}: {e}")
            return False, ""
    
    def detect_fda_catalysts(self, ticker: str, timeframe_days: int = 30) -> Optional[FDACatalyst]:
        """
        Detect FDA and healthcare catalysts from recent news
        
        Args:
            ticker: Stock ticker
            timeframe_days: How many days back to check news
            
        Returns:
            FDACatalyst object if found, None otherwise
        """
        try:
            stock = yf.Ticker(ticker)
            
            # First check if it's a healthcare stock
            is_healthcare, sector_type = self.is_healthcare_stock(ticker, stock.info)
            
            # Get recent news
            news = stock.news if hasattr(stock, 'news') else []
            
            if not news:
                # Fallback: check with info summary
                return self._check_from_info(ticker, stock.info, is_healthcare, sector_type)
            
            # Analyze news for catalysts
            all_keywords_found = []
            best_catalyst = None
            max_score = 0
            
            cutoff_date = datetime.now() - timedelta(days=timeframe_days)
            
            for article in news[:10]:  # Check most recent 10 articles
                try:
                    # Get article details
                    title = article.get('title', '').lower()
                    summary = article.get('summary', '').lower()
                    published = article.get('providerPublishTime', 0)
                    
                    # Check if recent enough
                    if published:
                        article_date = datetime.fromtimestamp(published)
                        if article_date < cutoff_date:
                            continue
                    
                    combined_text = f"{title} {summary}"
                    
                    # Check for FDA catalysts
                    for catalyst_type, catalyst_info in FDA_CATALYSTS.items():
                        keywords_found = []
                        
                        for keyword in catalyst_info['keywords']:
                            if keyword in combined_text:
                                keywords_found.append(keyword)
                        
                        if keywords_found:
                            score = catalyst_info['score']
                            
                            # Boost score for high-value indications
                            for indication in HIGH_VALUE_INDICATIONS:
                                if indication in combined_text:
                                    score = int(score * 1.2)
                                    keywords_found.append(f"high-value: {indication}")
                                    break
                            
                            # Boost score if healthcare stock
                            if is_healthcare:
                                score = int(score * 1.15)
                            
                            # Track best catalyst
                            if abs(score) > abs(max_score):
                                max_score = score
                                best_catalyst = FDACatalyst(
                                    ticker=ticker,
                                    catalyst_type=catalyst_type,
                                    description=title,
                                    score_boost=score,
                                    confidence=catalyst_info['confidence'],
                                    keywords_found=keywords_found,
                                    news_date=article_date if published else None,
                                    is_healthcare=is_healthcare,
                                    sector_type=sector_type
                                )
                                all_keywords_found.extend(keywords_found)
                
                except Exception as e:
                    self.logger.debug(f"Error parsing news article for {ticker}: {e}")
                    continue
            
            if best_catalyst:
                self.logger.info(f"🎯 FDA CATALYST DETECTED for {ticker}:")
                self.logger.info(f"   Type: {best_catalyst.catalyst_type}")
                self.logger.info(f"   Score Boost: +{best_catalyst.score_boost}")
                self.logger.info(f"   Confidence: {best_catalyst.confidence}")
                pass  # self.logger.info(f"   Keywords: {', '.join(best_catalyst.keywords_found[:5]}"))
            
            return best_catalyst
            
        except Exception as e:
            self.logger.error(f"Error detecting FDA catalysts for {ticker}: {e}")
            return None
    
    def _check_from_info(self, ticker: str, info: Dict, 
                        is_healthcare: bool, sector_type: str) -> Optional[FDACatalyst]:
        """Fallback: check for catalysts from stock info"""
        try:
            business_summary = info.get('longBusinessSummary', '').lower()
            
            if not business_summary:
                return None
            
            # Quick scan for major keywords
            for catalyst_type, catalyst_info in FDA_CATALYSTS.items():
                for keyword in catalyst_info['keywords'][:3]:  # Check top 3 keywords
                    if keyword in business_summary:
                        return FDACatalyst(
                            ticker=ticker,
                            catalyst_type=catalyst_type,
                            description=f"Company focus: {keyword}",
                            score_boost=int(catalyst_info['score'] * 0.5),  # Half score for general info
                            confidence='MEDIUM',
                            keywords_found=[keyword],
                            is_healthcare=is_healthcare,
                            sector_type=sector_type
                        )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error checking info for {ticker}: {e}")
            return None
    
    def enhance_catalyst_score(self, base_catalyst_score: float, 
                              ticker: str, existing_catalyst_text: str = "") -> Tuple[float, str]:
        """
        Enhance catalyst score with FDA detection
        
        Args:
            base_catalyst_score: Existing catalyst score
            ticker: Stock ticker
            existing_catalyst_text: Existing catalyst description
            
        Returns:
            Tuple of (enhanced_score, enhanced_catalyst_text)
        """
        try:
            # Detect FDA catalysts
            fda_catalyst = self.detect_fda_catalysts(ticker)
            
            if fda_catalyst:
                # Enhance score
                enhanced_score = min(100, max(0, base_catalyst_score + fda_catalyst.score_boost))
                
                # Enhance text
                catalyst_emoji = self._get_catalyst_emoji(fda_catalyst.catalyst_type)
                enhanced_text = f"{catalyst_emoji} FDA: {fda_catalyst.catalyst_type.replace('_', ' ').title()}"
                
                if existing_catalyst_text:
                    enhanced_text = f"{existing_catalyst_text} | {enhanced_text}"
                
                self.logger.info(f"Enhanced {ticker} catalyst score: {base_catalyst_score:.0f} → {enhanced_score:.0f}")
                
                return enhanced_score, enhanced_text
            
            return base_catalyst_score, existing_catalyst_text
            
        except Exception as e:
            self.logger.error(f"Error enhancing catalyst score for {ticker}: {e}")
            return base_catalyst_score, existing_catalyst_text
    
    @staticmethod
    def _get_catalyst_emoji(catalyst_type: str) -> str:
        """Get emoji for catalyst type"""
        emoji_map = {
            'approval': '💊',
            'positive_trial': '✅',
            'trial_advancement': '🧪',
            'phase_2': '🔬',
            'early_stage': '🧬',
            'regulatory': '📋',
            'partnership': '🤝',
            'negative': '❌'
        }
        return emoji_map.get(catalyst_type, '📰')
    
    def get_catalyst_summary(self, ticker: str) -> Dict:
        """
        Get comprehensive catalyst summary for a stock
        
        Returns:
            Dict with catalyst information
        """
        try:
            is_healthcare, sector_type = self.is_healthcare_stock(ticker)
            catalyst = self.detect_fda_catalysts(ticker)
            
            if catalyst:
                return {
                    'has_catalyst': True,
                    'catalyst_type': catalyst.catalyst_type,
                    'catalyst_description': catalyst.description,
                    'score_boost': catalyst.score_boost,
                    'confidence': catalyst.confidence,
                    'is_healthcare': is_healthcare,
                    'sector_type': sector_type,
                    'keywords': ', '.join(catalyst.keywords_found[:5])
                }
            
            return {
                'has_catalyst': False,
                'is_healthcare': is_healthcare,
                'sector_type': sector_type
            }
            
        except Exception as e:
            self.logger.error(f"Error getting catalyst summary for {ticker}: {e}")
            return {'has_catalyst': False, 'error': str(e)}


# Helper function for easy integration
def get_fda_boost_for_ticker(ticker: str) -> Tuple[float, str]:
    """
    Quick helper to get FDA score boost for a ticker
    
    Args:
        ticker: Stock ticker
        
    Returns:
        Tuple of (score_boost, catalyst_description)
    """
    detector = FDACatalystDetector()
    catalyst = detector.detect_fda_catalysts(ticker)
    
    if catalyst:
        return catalyst.score_boost, catalyst.description
    
    return 0.0, ""


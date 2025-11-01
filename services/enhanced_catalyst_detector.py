"""
Enhanced Catalyst Detector

Extends SEC filing detection with:
- 8-K parsing for specific material events
- Paid promotion detection
- Press release scanning
- Financing event detection
"""

import logging
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests

logger = logging.getLogger(__name__)


class EnhancedCatalystDetector:
    """Enhanced catalyst detection for penny stocks"""
    
    # Paid promotion keywords (common in pump schemes)
    PAID_PROMOTION_KEYWORDS = [
        'sponsored', 'paid advertisement', 'paid promotion', 'compensation',
        'we have been compensated', 'this is a paid', 'promotional consideration',
        'financial compensation', 'paid to promote', 'third party paid'
    ]
    
    # Financing keywords (often dilutive for penny stocks)
    FINANCING_KEYWORDS = [
        'registered direct offering', 'public offering', 'private placement',
        'direct offering', 'shelf offering', 'at-the-market offering',
        'warrant exercise', 'convertible note', 'dilution'
    ]
    
    # Material event keywords in 8-K
    MATERIAL_EVENT_TYPES = {
        'Item 1.01': 'Entry into Material Agreement',
        'Item 1.02': 'Termination of Material Agreement',
        'Item 1.03': 'Bankruptcy',
        'Item 2.01': 'Completion of Acquisition',
        'Item 2.03': 'Creation of Direct Financial Obligation',
        'Item 3.02': 'Unregistered Sales of Equity Securities',
        'Item 4.01': 'Changes in Accountant',
        'Item 5.02': 'Departure/Election of Directors or Officers',
        'Item 7.01': 'Regulation FD Disclosure',
        'Item 8.01': 'Other Events',
    }
    
    @staticmethod
    def detect_paid_promotion(text: str) -> Dict[str, any]:
        """
        Detect if content is a paid promotion.
        
        Args:
            text: News article or press release text
            
        Returns:
            Dict with detection results
        """
        text_lower = text.lower()
        detected_keywords = []
        
        for keyword in EnhancedCatalystDetector.PAID_PROMOTION_KEYWORDS:
            if keyword in text_lower:
                detected_keywords.append(keyword)
        
        is_paid = len(detected_keywords) > 0
        risk_level = "CRITICAL" if len(detected_keywords) >= 2 else "HIGH" if is_paid else "LOW"
        
        return {
            'is_paid_promotion': is_paid,
            'risk_level': risk_level,
            'detected_keywords': detected_keywords,
            'warning': '⚠️ PAID PROMOTION DETECTED - High pump risk' if is_paid else None
        }
    
    @staticmethod
    def detect_financing_event(text: str, filing_type: str = None) -> Dict[str, any]:
        """
        Detect financing events (often dilutive).
        
        Args:
            text: Filing or news text
            filing_type: SEC filing type if applicable
            
        Returns:
            Dict with financing detection
        """
        text_lower = text.lower()
        detected_keywords = []
        
        for keyword in EnhancedCatalystDetector.FINANCING_KEYWORDS:
            if keyword in text_lower:
                detected_keywords.append(keyword)
        
        is_financing = len(detected_keywords) > 0
        
        # Check if it's an S-1, S-3, or 424B filing (offerings)
        is_offering_filing = filing_type in ['S-1', 'S-3', 'S-1/A', 'S-3/A', '424B3', '424B5']
        
        if is_offering_filing:
            is_financing = True
            detected_keywords.append(f'SEC Filing: {filing_type}')
        
        # Determine dilution risk
        if 'warrant' in text_lower or 'convertible' in text_lower:
            dilution_risk = "HIGH"
        elif 'registered direct' in text_lower:
            dilution_risk = "MEDIUM"
        elif is_offering_filing:
            dilution_risk = "MEDIUM"
        else:
            dilution_risk = "LOW" if not is_financing else "MEDIUM"
        
        return {
            'is_financing_event': is_financing,
            'dilution_risk': dilution_risk,
            'detected_keywords': detected_keywords,
            'warning': f'⚠️ FINANCING EVENT - {dilution_risk} dilution risk' if is_financing else None
        }
    
    @staticmethod
    def parse_8k_items(filing_text: str) -> List[Dict[str, str]]:
        """
        Parse 8-K filing to extract reported items.
        
        Args:
            filing_text: Raw 8-K filing text
            
        Returns:
            List of item dicts with type and description
        """
        items = []
        
        for item_code, description in EnhancedCatalystDetector.MATERIAL_EVENT_TYPES.items():
            # Look for item references in text
            pattern = rf'{item_code}\b'
            if re.search(pattern, filing_text, re.IGNORECASE):
                items.append({
                    'item_code': item_code,
                    'description': description,
                    'is_critical': item_code in ['Item 1.03', 'Item 2.01', 'Item 3.02', 'Item 5.02']
                })
        
        return items
    
    @staticmethod
    def fetch_8k_content(filing_url: str) -> Optional[str]:
        """
        Fetch 8-K filing content from SEC EDGAR.
        
        Args:
            filing_url: URL to SEC filing
            
        Returns:
            Filing text content or None
        """
        try:
            headers = {'User-Agent': 'Sentient Trader/1.0'}
            response = requests.get(filing_url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching 8-K content: {e}")
            return None
    
    @staticmethod
    def analyze_catalyst_quality(catalyst_text: str, source: str = 'news') -> Dict[str, any]:
        """
        Analyze catalyst quality and authenticity.
        
        Args:
            catalyst_text: Catalyst description or news text
            source: 'news', 'sec_filing', 'press_release', 'social'
            
        Returns:
            Dict with quality assessment
        """
        # Check for paid promotion
        paid_promo = EnhancedCatalystDetector.detect_paid_promotion(catalyst_text)
        
        # Check for financing
        financing = EnhancedCatalystDetector.detect_financing_event(catalyst_text)
        
        # Determine credibility
        credibility_score = 100
        warnings = []
        
        if paid_promo['is_paid_promotion']:
            credibility_score -= 50
            warnings.append(paid_promo['warning'])
        
        if financing['is_financing_event'] and financing['dilution_risk'] in ['HIGH', 'CRITICAL']:
            credibility_score -= 30
            warnings.append(financing['warning'])
        
        # Source credibility
        if source == 'sec_filing':
            source_weight = 1.0  # Most credible
        elif source == 'press_release':
            source_weight = 0.8
        elif source == 'news':
            source_weight = 0.9
        else:  # social
            source_weight = 0.5  # Least credible
        
        final_score = max(0, credibility_score * source_weight)
        
        # Classification
        if final_score >= 80:
            quality = "HIGH"
            emoji = "✅"
        elif final_score >= 60:
            quality = "MEDIUM"
            emoji = "⚠️"
        else:
            quality = "LOW"
            emoji = "❌"
        
        return {
            'catalyst_quality': quality,
            'credibility_score': round(final_score, 1),
            'source': source,
            'source_weight': source_weight,
            'emoji': emoji,
            'warnings': warnings,
            'paid_promotion_detected': paid_promo['is_paid_promotion'],
            'financing_event_detected': financing['is_financing_event'],
            'recommendation': f"{emoji} {quality} QUALITY CATALYST (Score: {final_score:.0f}/100)"
        }
    
    @staticmethod
    def scan_news_for_catalysts(news_items: List[Dict], ticker: str) -> Dict[str, any]:
        """
        Scan news items for catalysts and detect promotions.
        
        Args:
            news_items: List of news item dicts with 'title', 'summary', 'url'
            ticker: Stock ticker
            
        Returns:
            Dict with catalyst analysis
        """
        catalysts = []
        paid_promotions = 0
        financing_events = 0
        
        for news in news_items:
            title = news.get('title', '')
            summary = news.get('summary', '')
            combined_text = f"{title} {summary}"
            
            # Analyze quality
            analysis = EnhancedCatalystDetector.analyze_catalyst_quality(
                combined_text, source='news'
            )
            
            if analysis['paid_promotion_detected']:
                paid_promotions += 1
            
            if analysis['financing_event_detected']:
                financing_events += 1
            
            catalysts.append({
                **news,
                **analysis
            })
        
        # Overall assessment
        if paid_promotions >= 2:
            overall_warning = "🚨 MULTIPLE PAID PROMOTIONS DETECTED - Potential pump scheme"
            risk_level = "CRITICAL"
        elif paid_promotions == 1:
            overall_warning = "⚠️ Paid promotion detected - Exercise caution"
            risk_level = "HIGH"
        elif financing_events >= 2:
            overall_warning = "⚠️ Multiple financing events - Dilution risk"
            risk_level = "MEDIUM"
        else:
            overall_warning = None
            risk_level = "LOW"
        
        return {
            'ticker': ticker,
            'catalyst_count': len(catalysts),
            'paid_promotions': paid_promotions,
            'financing_events': financing_events,
            'risk_level': risk_level,
            'overall_warning': overall_warning,
            'catalysts': catalysts
        }


class StockLiquidityChecker:
    """Check stock liquidity for execution risk"""
    
    @staticmethod
    def check_stock_liquidity(ticker: str, current_price: float, volume: int, 
                             avg_volume: int, bid: float = None, ask: float = None) -> Dict[str, any]:
        """
        Check stock liquidity and execution risk.
        
        Args:
            ticker: Stock ticker
            current_price: Current price
            volume: Today's volume
            avg_volume: Average daily volume
            bid: Current bid price (optional)
            ask: Current ask price (optional)
            
        Returns:
            Dict with liquidity assessment
        """
        warnings = []
        risk_factors = []
        
        # Volume ratio check
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0
        
        if avg_volume < 100_000:
            risk_factors.append("Very low average volume (<100K)")
            volume_risk = "CRITICAL"
        elif avg_volume < 500_000:
            risk_factors.append("Low average volume (<500K)")
            volume_risk = "HIGH"
        elif avg_volume < 1_000_000:
            risk_factors.append("Below-average volume (<1M)")
            volume_risk = "MEDIUM"
        else:
            volume_risk = "LOW"
        
        # Bid-ask spread check (if available)
        spread_risk = "UNKNOWN"
        if bid and ask and bid > 0:
            spread = ask - bid
            spread_pct = (spread / current_price) * 100
            
            if spread_pct > 5.0:
                risk_factors.append(f"Wide bid-ask spread ({spread_pct:.1f}%)")
                spread_risk = "HIGH"
                warnings.append(f"⚠️ Wide spread: ${spread:.2f} ({spread_pct:.1f}%)")
            elif spread_pct > 2.0:
                spread_risk = "MEDIUM"
            else:
                spread_risk = "LOW"
        
        # Dollar volume check (for position sizing)
        dollar_volume = current_price * avg_volume
        
        if dollar_volume < 100_000:  # <$100K daily
            risk_factors.append(f"Very low dollar volume (${dollar_volume/1000:.0f}K/day)")
            dollar_risk = "CRITICAL"
        elif dollar_volume < 500_000:  # <$500K daily
            risk_factors.append(f"Low dollar volume (${dollar_volume/1000:.0f}K/day)")
            dollar_risk = "HIGH"
        elif dollar_volume < 1_000_000:  # <$1M daily
            dollar_risk = "MEDIUM"
        else:
            dollar_risk = "LOW"
        
        # Overall liquidity score
        risk_levels = [volume_risk, spread_risk, dollar_risk]
        risk_counts = {
            'CRITICAL': sum(1 for r in risk_levels if r == 'CRITICAL'),
            'HIGH': sum(1 for r in risk_levels if r == 'HIGH'),
            'MEDIUM': sum(1 for r in risk_levels if r == 'MEDIUM'),
        }
        
        if risk_counts['CRITICAL'] > 0:
            overall_risk = "CRITICAL"
            recommendation = "❌ AVOID - Illiquid stock, high execution risk"
        elif risk_counts['HIGH'] >= 2:
            overall_risk = "HIGH"
            recommendation = "⚠️ HIGH RISK - Use limit orders, small size only"
        elif risk_counts['HIGH'] >= 1 or risk_counts['MEDIUM'] >= 2:
            overall_risk = "MEDIUM"
            recommendation = "⚠️ CAUTION - Use limit orders, avoid market orders"
        else:
            overall_risk = "LOW"
            recommendation = "✅ ADEQUATE LIQUIDITY"
        
        # Position size recommendation (% of daily volume)
        if overall_risk in ["CRITICAL", "HIGH"]:
            max_position_pct = 5.0  # Max 5% of daily volume
        elif overall_risk == "MEDIUM":
            max_position_pct = 10.0  # Max 10% of daily volume
        else:
            max_position_pct = 20.0  # Max 20% of daily volume
        
        max_shares = int(avg_volume * (max_position_pct / 100))
        
        return {
            'ticker': ticker,
            'overall_risk': overall_risk,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'warnings': warnings,
            'volume_risk': volume_risk,
            'spread_risk': spread_risk,
            'dollar_risk': dollar_risk,
            'avg_daily_volume': avg_volume,
            'dollar_volume': round(dollar_volume, 2),
            'volume_ratio': round(volume_ratio, 2),
            'max_position_shares': max_shares,
            'max_position_pct_of_volume': max_position_pct
        }


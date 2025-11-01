"""
SEC Filing Detector

Monitors SEC EDGAR filings for material events (8-K), insider trading,
and quarterly/annual reports.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from models.alerts import TradingAlert, AlertType, AlertPriority
from .base_detector import BaseEventDetector

logger = logging.getLogger(__name__)


class SECDetector(BaseEventDetector):
    """Detects SEC filings for watchlist tickers"""
    
    # Filing type priorities
    FILING_PRIORITIES = {
        '8-K': AlertPriority.CRITICAL,      # Material events
        '8-K/A': AlertPriority.CRITICAL,    # Amended material events
        '4': AlertPriority.HIGH,             # Insider trading
        '3': AlertPriority.MEDIUM,           # Initial insider ownership
        '5': AlertPriority.MEDIUM,           # Annual insider trading
        '10-Q': AlertPriority.MEDIUM,        # Quarterly report
        '10-K': AlertPriority.MEDIUM,        # Annual report
        '10-Q/A': AlertPriority.LOW,         # Amended quarterly
        '10-K/A': AlertPriority.LOW,         # Amended annual
        'S-1': AlertPriority.HIGH,           # IPO registration
        'S-3': AlertPriority.MEDIUM,         # Securities registration
        'DEF 14A': AlertPriority.LOW,        # Proxy statement
    }
    
    def __init__(self, alert_system, ticker_manager=None, my_tickers_only: bool = True):
        """
        Initialize SEC detector
        
        Args:
            alert_system: AlertSystem instance
            ticker_manager: TickerManager instance
            my_tickers_only: Filter to watchlist only
        """
        super().__init__(alert_system, ticker_manager, my_tickers_only)
        self.sec_api_key = os.getenv('SEC_API_KEY')  # Optional, for rate limits
        self.user_agent = "Sentient Trader/1.0 (trading@example.com)"
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            CIK string or None
        """
        try:
            # SEC ticker to CIK mapping
            url = "https://www.sec.gov/files/company_tickers.json"
            headers = {'User-Agent': self.user_agent}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            companies = response.json()
            
            # Search for ticker
            for company in companies.values():
                if company.get('ticker', '').upper() == ticker.upper():
                    cik = str(company.get('cik_str', '')).zfill(10)
                    return cik
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting CIK for {ticker}: {e}")
            return None
    
    def get_recent_filings(self, ticker: str, cik: str, hours_back: int = 24) -> List[Dict]:
        """
        Get recent SEC filings for a company
        
        Args:
            ticker: Ticker symbol
            cik: Company CIK
            hours_back: How many hours back to check
            
        Returns:
            List of filing dicts
        """
        try:
            # SEC EDGAR API
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            headers = {'User-Agent': self.user_agent}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            recent_filings = data.get('filings', {}).get('recent', {})
            
            if not recent_filings:
                return []
            
            # Parse filings
            filings = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            filing_dates = recent_filings.get('filingDate', [])
            form_types = recent_filings.get('form', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_documents = recent_filings.get('primaryDocument', [])
            
            for i in range(min(len(filing_dates), 20)):  # Check last 20 filings
                try:
                    filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d')
                    
                    if filing_date >= cutoff_time:
                        form_type = form_types[i]
                        accession = accession_numbers[i]
                        primary_doc = primary_documents[i] if i < len(primary_documents) else ''
                        
                        # Build filing URL
                        accession_clean = accession.replace('-', '')
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{primary_doc}"
                        
                        filings.append({
                            'ticker': ticker,
                            'form_type': form_type,
                            'filing_date': filing_date,
                            'accession_number': accession,
                            'url': filing_url
                        })
                except Exception as e:
                    logger.debug(f"Error parsing filing {i}: {e}")
                    continue
            
            return filings
            
        except Exception as e:
            logger.debug(f"Error getting SEC filings for {ticker}: {e}")
            return []
    
    def get_filing_description(self, form_type: str) -> str:
        """
        Get human-readable description of filing type
        
        Args:
            form_type: SEC form type
            
        Returns:
            Description string
        """
        descriptions = {
            '8-K': 'Material Event Report',
            '8-K/A': 'Amended Material Event',
            '4': 'Insider Trading Statement',
            '3': 'Initial Insider Ownership',
            '5': 'Annual Insider Trading',
            '10-Q': 'Quarterly Report',
            '10-K': 'Annual Report',
            '10-Q/A': 'Amended Quarterly Report',
            '10-K/A': 'Amended Annual Report',
            'S-1': 'IPO Registration',
            'S-3': 'Securities Registration',
            'DEF 14A': 'Proxy Statement'
        }
        return descriptions.get(form_type, form_type)
    
    def create_sec_alert(self, filing: Dict) -> Optional[TradingAlert]:
        """
        Create a SEC filing alert
        
        Args:
            filing: Filing dict
            
        Returns:
            TradingAlert or None
        """
        ticker = filing['ticker']
        form_type = filing['form_type']
        
        # Get priority for this filing type
        priority = self.FILING_PRIORITIES.get(form_type, AlertPriority.LOW)
        
        # Skip low priority filings unless they're very recent
        if priority == AlertPriority.LOW:
            hours_old = (datetime.now() - filing['filing_date']).total_seconds() / 3600
            if hours_old > 6:
                return None
        
        # Build message
        description = self.get_filing_description(form_type)
        
        emoji_map = {
            AlertPriority.CRITICAL: '🚨',
            AlertPriority.HIGH: '📋',
            AlertPriority.MEDIUM: '📄',
            AlertPriority.LOW: '📝'
        }
        emoji = emoji_map.get(priority, '📄')
        
        time_ago = self._format_time_ago(filing['filing_date'])
        message = f"{emoji} SEC Filing: {form_type} - {description} ({time_ago})"
        
        # Create alert
        alert = TradingAlert(
            ticker=ticker,
            alert_type=AlertType.SEC_FILING,
            priority=priority,
            message=message,
            confidence_score=0.0,
            details={
                'form_type': form_type,
                'description': description,
                'filing_date': filing['filing_date'].isoformat(),
                'accession_number': filing['accession_number'],
                'url': filing['url']
            }
        )
        
        return alert
    
    def _format_time_ago(self, dt: datetime) -> str:
        """Format datetime as 'X hours ago' or 'X minutes ago'"""
        delta = datetime.now() - dt
        
        if delta.days > 0:
            return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    
    def detect(self) -> List[TradingAlert]:
        """
        Detect SEC filings
        
        Returns:
            List of SEC filing alerts
        """
        alerts = []
        
        # Get watchlist tickers
        my_tickers = self._get_my_tickers()
        
        if not my_tickers:
            logger.warning("No tickers in watchlist for SEC detection")
            return alerts
        
        logger.info(f"Checking SEC filings for {len(my_tickers)} tickers...")
        
        for ticker in my_tickers:
            try:
                # Get CIK
                cik = self.get_company_cik(ticker)
                
                if not cik:
                    logger.debug(f"Could not find CIK for {ticker}")
                    continue
                
                # Get recent filings
                filings = self.get_recent_filings(ticker, cik, hours_back=24)
                
                for filing in filings:
                    alert = self.create_sec_alert(filing)
                    
                    if alert:
                        alerts.append(alert)
                        self.trigger_alert(alert)
                        logger.info(f"SEC alert: {ticker} - {filing['form_type']}")
            
            except Exception as e:
                logger.error(f"Error processing SEC filings for {ticker}: {e}")
                continue
        
        return alerts

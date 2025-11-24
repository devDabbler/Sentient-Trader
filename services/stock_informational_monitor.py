"""
Stock Informational Monitor Service
Monitors stocks without executing trades - alerts only

Integrates with LLM Request Manager for cost-efficient scanning
Uses LOW priority LLM requests with aggressive caching
"""

import time
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import os

from services.llm_helper import get_llm_helper, LLMServiceMixin
from services.alert_system import get_alert_system
from services.ai_confidence_scanner import AIConfidenceScanner

try:
    import config_stock_informational as cfg
    SCAN_INTERVAL_MINUTES = cfg.SCAN_INTERVAL_MINUTES
    CACHE_TTL_SECONDS = cfg.CACHE_TTL_SECONDS
    WATCHLIST = cfg.WATCHLIST
    MIN_ENSEMBLE_SCORE = cfg.MIN_ENSEMBLE_SCORE
    ENABLE_ALERTS = cfg.ENABLE_ALERTS
    LOG_OPPORTUNITIES = cfg.LOG_OPPORTUNITIES
    OPPORTUNITIES_LOG_PATH = cfg.OPPORTUNITIES_LOG_PATH
except ImportError:
    # Fallback to default settings
    SCAN_INTERVAL_MINUTES = 30
    CACHE_TTL_SECONDS = 900
    WATCHLIST = []
    MIN_ENSEMBLE_SCORE = 60
    ENABLE_ALERTS = True
    LOG_OPPORTUNITIES = True
    OPPORTUNITIES_LOG_PATH = "logs/stock_opportunities.json"


logger = logging.getLogger(__name__)


@dataclass
class StockOpportunity:
    """Detected stock opportunity"""
    symbol: str
    opportunity_type: str  # ENTRY, BREAKOUT, SETUP, WATCH, etc.
    ensemble_score: int  # 0-100
    confidence: float  # 0-1.0
    price: float
    reasoning: str
    technical_summary: str
    timeframe_alignment: str  # TRIPLE_THREAT, DUAL_ALIGN, SINGLE, NONE
    alert_priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


class StockInformationalMonitor(LLMServiceMixin):
    """
    Monitors stocks without trading - alerts only
    
    Features:
    - Cost-efficient LLM usage (LOW priority, aggressive caching)
    - Multi-factor validation (Technical + ML + LLM)
    - Event monitoring (earnings, SEC, news)
    - Priority-based Discord alerts
    - No trade execution
    """
    
    def __init__(
        self,
        watchlist: Optional[List[str]] = None,
        scan_interval_minutes: Optional[int] = None,
        min_score: Optional[int] = None
    ):
        """
        Initialize Stock Informational Monitor
        
        Args:
            watchlist: List of tickers to monitor (or use config)
            scan_interval_minutes: Scan frequency (or use config)
            min_score: Minimum ensemble score for alerts (or use config)
        """
        super().__init__()
        
        # Initialize LLM helper with LOW priority
        self._init_llm("stock_informational_monitor", default_priority="LOW")
        
        # Configuration
        self.watchlist = watchlist or WATCHLIST
        self.scan_interval = scan_interval_minutes or SCAN_INTERVAL_MINUTES
        self.min_score = min_score or MIN_ENSEMBLE_SCORE
        
        # Services
        self.alert_system = get_alert_system()
        self.confidence_scanner = AIConfidenceScanner()
        
        # State
        self.opportunities: List[StockOpportunity] = []
        self.last_scan_time: Optional[datetime] = None
        self.is_running = False
        
        logger.info(
            f"Stock Informational Monitor initialized "
            f"({len(self.watchlist)} tickers, {self.scan_interval}min interval)"
        )
    
    def scan_ticker(self, symbol: str) -> Optional[StockOpportunity]:
        """
        Scan a single ticker for opportunities
        
        Args:
            symbol: Ticker symbol to scan
            
        Returns:
            StockOpportunity if found, None otherwise
        """
        try:
            logger.info(f"Scanning {symbol}...")
            
            # Get confidence analysis (already migrated to LLM manager)
            analysis = self.confidence_scanner.analyze_ticker(symbol)
            
            if not analysis:
                logger.debug(f"No analysis for {symbol}")
                return None
            
            ensemble_score = analysis.get('ensemble_score', 0)
            
            # Filter by minimum score
            if ensemble_score < self.min_score:
                logger.debug(f"{symbol} score {ensemble_score} below threshold {self.min_score}")
                return None
            
            # Use LLM helper for qualitative reasoning
            prompt = f"""
            Analyze this stock opportunity for {symbol}:
            
            Ensemble Score: {ensemble_score}/100
            Technical Setup: {analysis.get('technical_setup', 'N/A')}
            ML Confidence: {analysis.get('ml_confidence', 0):.2%}
            Sentiment Score: {analysis.get('sentiment_score', 50)}/100
            
            Provide a concise 2-3 sentence reasoning for why this is (or isn't) a good opportunity.
            Focus on what makes it compelling right now.
            """
            
            # Use cached LOW priority request (15min TTL from config)
            reasoning = self.llm_cached(
                prompt=prompt,
                cache_key=f"stock_reasoning_{symbol}_{int(time.time() / CACHE_TTL_SECONDS)}",
                ttl=CACHE_TTL_SECONDS
            )
            
            if not reasoning:
                reasoning = "LLM analysis unavailable - using technical signals only"
            
            # Determine timeframe alignment
            timeframe_alignment = self._determine_alignment(analysis)
            
            # Determine alert priority
            alert_priority = self._determine_priority(
                ensemble_score,
                timeframe_alignment,
                analysis
            )
            
            # Create opportunity
            opportunity = StockOpportunity(
                symbol=symbol,
                opportunity_type=analysis.get('setup_type', 'SETUP'),
                ensemble_score=ensemble_score,
                confidence=analysis.get('confidence', 0.5),
                price=analysis.get('current_price', 0.0),
                reasoning=reasoning.strip(),
                technical_summary=analysis.get('technical_setup', ''),
                timeframe_alignment=timeframe_alignment,
                alert_priority=alert_priority,
                timestamp=datetime.now().isoformat(),
                metadata={
                    'ml_confidence': analysis.get('ml_confidence', 0),
                    'sentiment_score': analysis.get('sentiment_score', 50),
                    'volume_surge': analysis.get('volume_surge', False),
                    'volatility': analysis.get('volatility', 0),
                }
            )
            
            logger.info(
                f"Found opportunity: {symbol} "
                f"(Score: {ensemble_score}, Priority: {alert_priority})"
            )
            
            return opportunity
        
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None
    
    def _determine_alignment(self, analysis: Dict) -> str:
        """Determine timeframe alignment level"""
        aligned_count = analysis.get('timeframes_aligned', 0)
        
        if aligned_count >= 3:
            return "TRIPLE_THREAT"
        elif aligned_count == 2:
            return "DUAL_ALIGN"
        elif aligned_count == 1:
            return "SINGLE"
        else:
            return "NONE"
    
    def _determine_priority(
        self,
        score: int,
        alignment: str,
        analysis: Dict
    ) -> str:
        """Determine alert priority based on multiple factors"""
        
        # CRITICAL: Triple threat or very high score
        if alignment == "TRIPLE_THREAT" or score >= 85:
            return "CRITICAL"
        
        # HIGH: Dual alignment or high score
        if alignment == "DUAL_ALIGN" or score >= 75:
            return "HIGH"
        
        # MEDIUM: Single alignment or moderate score
        if alignment == "SINGLE" or score >= 65:
            return "MEDIUM"
        
        # LOW: Below threshold but still valid
        return "LOW"
    
    def scan_all_tickers(self) -> List[StockOpportunity]:
        """
        Scan all tickers in watchlist
        
        Returns:
            List of detected opportunities
        """
        opportunities = []
        
        logger.info(f"Starting scan of {len(self.watchlist)} tickers...")
        
        for symbol in self.watchlist:
            try:
                opportunity = self.scan_ticker(symbol)
                if opportunity:
                    opportunities.append(opportunity)
                    
                    # Send alert
                    if ENABLE_ALERTS:
                        self._send_alert(opportunity)
                    
                    # Log opportunity
                    if LOG_OPPORTUNITIES:
                        self._log_opportunity(opportunity)
                
                # Small delay to respect rate limits
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        logger.info(f"Scan complete: Found {len(opportunities)} opportunities")
        
        self.opportunities = opportunities
        self.last_scan_time = datetime.now()
        
        return opportunities
    
    def _send_alert(self, opportunity: StockOpportunity):
        """Send Discord alert for opportunity"""
        try:
            alert_title = f"📊 {opportunity.symbol} - {opportunity.opportunity_type}"
            
            alert_message = f"""
**Score:** {opportunity.ensemble_score}/100
**Confidence:** {opportunity.confidence:.1%}
**Price:** ${opportunity.price:.2f}
**Alignment:** {opportunity.timeframe_alignment}

**Reasoning:**
{opportunity.reasoning}

**Technical:**
{opportunity.technical_summary}
            """.strip()
            
            self.alert_system.send_alert(
                title=alert_title,
                message=alert_message,
                priority=opportunity.alert_priority,
                metadata={
                    'symbol': opportunity.symbol,
                    'score': opportunity.ensemble_score,
                    'type': 'STOCK_OPPORTUNITY'
                }
            )
        
        except Exception as e:
            logger.error(f"Error sending alert for {opportunity.symbol}: {e}")
    
    def _log_opportunity(self, opportunity: StockOpportunity):
        """Log opportunity to file"""
        try:
            log_path = OPPORTUNITIES_LOG_PATH
            
            # Create logs directory if needed
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            # Load existing logs
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Append new opportunity
            logs.append(opportunity.to_dict())
            
            # Keep last 1000 entries
            logs = logs[-1000:]
            
            # Save
            with open(log_path, 'w') as f:
                json.dump(logs, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error logging opportunity: {e}")
    
    def run_continuous(self):
        """Run continuous monitoring loop"""
        self.is_running = True
        logger.info("Starting continuous monitoring...")
        
        try:
            while self.is_running:
                # Scan all tickers
                opportunities = self.scan_all_tickers()
                
                # Log summary
                logger.info(
                    f"Scan cycle complete: "
                    f"{len(opportunities)} opportunities found, "
                    f"next scan in {self.scan_interval} minutes"
                )
                
                # Wait for next interval
                time.sleep(self.scan_interval * 60)
        
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.is_running = False
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            self.is_running = False
    
    def stop(self):
        """Stop continuous monitoring"""
        self.is_running = False
        logger.info("Stopping monitoring...")
    
    def get_recent_opportunities(self, limit: int = 50) -> List[StockOpportunity]:
        """Get recent opportunities"""
        return self.opportunities[-limit:]
    
    def get_opportunities_by_priority(self, priority: str) -> List[StockOpportunity]:
        """Get opportunities by alert priority"""
        return [
            opp for opp in self.opportunities
            if opp.alert_priority == priority.upper()
        ]


# Singleton instance
_monitor_instance = None


def get_stock_informational_monitor(
    watchlist: Optional[List[str]] = None,
    **kwargs
) -> StockInformationalMonitor:
    """
    Get singleton instance of Stock Informational Monitor
    
    Args:
        watchlist: Optional custom watchlist
        **kwargs: Additional configuration
        
    Returns:
        StockInformationalMonitor instance
    """
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = StockInformationalMonitor(
            watchlist=watchlist,
            **kwargs
        )
    
    return _monitor_instance


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create monitor
    monitor = get_stock_informational_monitor()
    
    # Run continuous monitoring
    try:
        monitor.run_continuous()
    except KeyboardInterrupt:
        monitor.stop()
        logger.info("Monitor stopped")

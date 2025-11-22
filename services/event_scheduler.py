"""
Event Scheduler Service

Manages scheduled execution of event detectors with configurable intervals.
Runs detectors at optimal times to catch market-moving events.
"""

from loguru import logger
import threading
import time
from datetime import datetime, time as dt_time
from typing import Dict, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from services.alert_system import AlertSystem
from services.ticker_manager import TickerManager
from services.event_detectors.earnings_detector import EarningsDetector
from services.event_detectors.news_detector import NewsDetector
from services.event_detectors.sec_detector import SECDetector
from services.event_detectors.economic_detector import EconomicDetector



class EventScheduler:
    """Schedules and manages event detection jobs"""
    
    def __init__(self, alert_system: AlertSystem, ticker_manager: TickerManager, 
                 position_tracker=None, tradier_client=None, ibkr_client=None):
        """
        Initialize event scheduler
        
        Args:
            alert_system: AlertSystem instance for notifications
            ticker_manager: TickerManager instance for watchlist
            position_tracker: Optional position tracker for earnings alerts
            tradier_client: Optional Tradier client for position checking
            ibkr_client: Optional IBKR client for position checking
        """
        self.alert_system = alert_system
        self.ticker_manager = ticker_manager
        self.position_tracker = position_tracker
        self.tradier_client = tradier_client
        self.ibkr_client = ibkr_client
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        
        # Initialize detectors
        self.earnings_detector = EarningsDetector(
            alert_system=alert_system,
            ticker_manager=ticker_manager,
            my_tickers_only=True,
            position_tracker=position_tracker,
            tradier_client=tradier_client,
            ibkr_client=ibkr_client
        )
        
        self.news_detector = NewsDetector(
            alert_system=alert_system,
            ticker_manager=ticker_manager,
            my_tickers_only=True
        )
        
        self.sec_detector = SECDetector(
            alert_system=alert_system,
            ticker_manager=ticker_manager,
            my_tickers_only=True
        )
        
        self.economic_detector = EconomicDetector(
            alert_system=alert_system,
            ticker_manager=ticker_manager,
            my_tickers_only=True
        )
        
        # Job statistics
        self.job_stats: Dict[str, Dict] = {
            'earnings': {'runs': 0, 'alerts': 0, 'last_run': None, 'errors': 0},
            'news': {'runs': 0, 'alerts': 0, 'last_run': None, 'errors': 0},
            'sec': {'runs': 0, 'alerts': 0, 'last_run': None, 'errors': 0},
            'economic': {'runs': 0, 'alerts': 0, 'last_run': None, 'errors': 0}
        }
    
    def _run_detector(self, detector_name: str, detector):
        """
        Run a detector and update statistics
        
        Args:
            detector_name: Name of detector for logging
            detector: Detector instance
        """
        try:
            logger.info(f"Running {detector_name} detector...")
            alerts = detector.run_detection()
            
            # Update stats
            self.job_stats[detector_name]['runs'] += 1
            self.job_stats[detector_name]['alerts'] += len(alerts)
            self.job_stats[detector_name]['last_run'] = datetime.now()
            
            pass  # logger.info(f"{detector_name} detector completed: {len(alerts))} alerts generated")
            
        except Exception as e:
            logger.error("Error running {detector_name} detector: {}", str(e), exc_info=True)
            self.job_stats[detector_name]['errors'] += 1
    
    def setup_schedules(self):
        """Setup all scheduled jobs with optimal timing"""
        
        # EARNINGS: Daily at 4:00 PM ET (after market close)
        # Check once per day for upcoming earnings
        self.scheduler.add_job(
            func=lambda: self._run_detector('earnings', self.earnings_detector),
            trigger=CronTrigger(hour=16, minute=0),  # 4:00 PM
            id='earnings_daily',
            name='Earnings Detection (Daily)',
            replace_existing=True
        )
        logger.info("Scheduled: Earnings detection daily at 4:00 PM ET")
        
        # NEWS: Every 30 minutes during market hours (9:30 AM - 4:00 PM ET)
        # More frequent during trading to catch breaking news
        self.scheduler.add_job(
            func=lambda: self._run_detector('news', self.news_detector),
            trigger=IntervalTrigger(minutes=30),
            id='news_frequent',
            name='News Detection (Every 30 min)',
            replace_existing=True
        )
        logger.info("Scheduled: News detection every 30 minutes")
        
        # SEC FILINGS: Every hour during market hours + after-hours sweep
        # SEC filings can come anytime, but most common during business hours
        self.scheduler.add_job(
            func=lambda: self._run_detector('sec', self.sec_detector),
            trigger=IntervalTrigger(hours=1),
            id='sec_hourly',
            name='SEC Filing Detection (Hourly)',
            replace_existing=True
        )
        logger.info("Scheduled: SEC filing detection every hour")
        
        # ECONOMIC CALENDAR: Daily at 8:00 AM ET (before market open)
        # Check economic calendar once per day in the morning
        self.scheduler.add_job(
            func=lambda: self._run_detector('economic', self.economic_detector),
            trigger=CronTrigger(hour=8, minute=0),  # 8:00 AM
            id='economic_daily',
            name='Economic Calendar (Daily)',
            replace_existing=True
        )
        logger.info("Scheduled: Economic calendar check daily at 8:00 AM ET")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        try:
            self.setup_schedules()
            self.scheduler.start()
            self.is_running = True
            logger.info("Event scheduler started successfully")
            
            # Run initial detection for all detectors
            logger.info("Running initial detection sweep...")
            self._run_detector('earnings', self.earnings_detector)
            self._run_detector('news', self.news_detector)
            self._run_detector('sec', self.sec_detector)
            self._run_detector('economic', self.economic_detector)
            
        except Exception as e:
            logger.error("Failed to start scheduler: {}", str(e), exc_info=True)
            self.is_running = False
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        try:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("Event scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    def get_status(self) -> Dict:
        """
        Get scheduler status and statistics
        
        Returns:
            Dict with status information
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            })
        
        return {
            'is_running': self.is_running,
            'jobs': jobs,
            'statistics': self.job_stats
        }
    
    def run_detector_now(self, detector_name: str) -> bool:
        """
        Manually trigger a detector to run immediately
        
        Args:
            detector_name: Name of detector ('earnings', 'news', 'sec', 'economic')
            
        Returns:
            True if successful
        """
        detector_map = {
            'earnings': self.earnings_detector,
            'news': self.news_detector,
            'sec': self.sec_detector,
            'economic': self.economic_detector
        }
        
        detector = detector_map.get(detector_name)
        if not detector:
            logger.error(f"Unknown detector: {detector_name}")
            return False
        
        try:
            self._run_detector(detector_name, detector)
            return True
        except Exception as e:
            logger.error(f"Error running {detector_name} detector: {e}")
            return False


# Global scheduler instance
_global_scheduler: Optional[EventScheduler] = None


def get_event_scheduler(alert_system: AlertSystem = None, 
                       ticker_manager: TickerManager = None,
                       position_tracker=None,
                       tradier_client=None,
                       ibkr_client=None) -> EventScheduler:
    """
    Get or create global event scheduler instance
    
    Args:
        alert_system: AlertSystem instance (required for first call)
        ticker_manager: TickerManager instance (required for first call)
        position_tracker: Optional position tracker
        tradier_client: Optional Tradier client for position checking
        ibkr_client: Optional IBKR client for position checking
        
    Returns:
        EventScheduler instance
    """
    global _global_scheduler
    
    if _global_scheduler is None:
        if alert_system is None or ticker_manager is None:
            raise ValueError("alert_system and ticker_manager required for first initialization")
        
        _global_scheduler = EventScheduler(
            alert_system=alert_system,
            ticker_manager=ticker_manager,
            position_tracker=position_tracker,
            tradier_client=tradier_client,
            ibkr_client=ibkr_client
        )
    
    return _global_scheduler

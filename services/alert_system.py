"""Real-time alert system for high-confidence trading setups."""

from loguru import logger
import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Callable, Set
from dataclasses import field
from src.integrations.discord_webhook import send_discord_alert
from models.alerts import AlertType, AlertPriority, TradingAlert



class AlertSystem:
    """Manages trading alerts and notifications"""
    
    def __init__(self, alert_log_path: str = "logs/trading_alerts.json"):
        """Initialize alert system"""
        self.alert_log_path = alert_log_path
        self.alerts: List[TradingAlert] = []
        self.alert_callbacks: List[Callable[[TradingAlert], None]] = [send_discord_alert]
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        log_dir = os.path.dirname(self.alert_log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    def add_callback(self, callback: Callable[[TradingAlert], None]):
        """Add a callback to be triggered when alerts are created"""
        self.alert_callbacks.append(callback)
    
    def send_alert(self, title: str, message: str, priority: str = "MEDIUM", metadata: Optional[Dict] = None):
        """
        Simplified alert sending method (convenience wrapper)
        Creates a TradingAlert and triggers it
        """
        try:
            # Map string priority to AlertPriority enum
            priority_map = {
                "CRITICAL": AlertPriority.CRITICAL,
                "HIGH": AlertPriority.HIGH,
                "MEDIUM": AlertPriority.MEDIUM,
                "LOW": AlertPriority.LOW
            }
            alert_priority = priority_map.get(priority.upper(), AlertPriority.MEDIUM)
            
            # Extract ticker/symbol from metadata
            ticker = metadata.get('symbol', 'UNKNOWN') if metadata else 'UNKNOWN'
            
            # Check if this is a DEX Hunter / pump chaser alert
            is_dex_alert = title in ["LAUNCH_DETECTED", "DEX_PUMP", "TOKEN_LAUNCH"] or \
                           "LAUNCH" in title.upper() or \
                           "PUMP" in title.upper() or \
                           (metadata and metadata.get('source') in ['dex_hunter', 'dex_launch', 'pump_chaser'])
            
            # Create TradingAlert (uses 'ticker' and 'details' not 'symbol' and 'metadata')
            alert = TradingAlert(
                ticker=ticker,
                alert_type=AlertType.HIGH_CONFIDENCE,  # Generic opportunity type
                priority=alert_priority,
                message=f"{title}\n\n{message}",
                timestamp=datetime.now(),
                confidence_score=metadata.get('score', 0.0) if metadata else 0.0,
                details={**(metadata or {}), '_is_dex_alert': is_dex_alert}
            )
            
            # Trigger it
            self.trigger_alert(alert)
            
        except Exception as e:
            logger.error(f"Error in send_alert: {e}")
            raise
    
    def trigger_alert(self, alert: TradingAlert):
        """Trigger an alert and execute callbacks"""
        self.alerts.append(alert)
        
        # Log to file
        self._log_alert(alert)
        
        # Execute callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        logger.info(f"Alert triggered: {alert}")
    
    def _log_alert(self, alert: TradingAlert):
        """Log alert to file"""
        try:
            alerts_data = []
            if os.path.exists(self.alert_log_path):
                try:
                    with open(self.alert_log_path, 'r') as f:
                        content = f.read().strip()
                        if content:  # Only parse if file has content
                            alerts_data = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    # File exists but is empty or corrupted - start fresh
                    alerts_data = []
            
            alerts_data.append(alert.to_dict())
            
            # Keep only last 1000 alerts
            if len(alerts_data) > 1000:
                alerts_data = alerts_data[-1000:]
            
            with open(self.alert_log_path, 'w') as f:
                json.dump(alerts_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
    
    def get_recent_alerts(self, count: int = 50, priority: Optional[AlertPriority] = None) -> List[TradingAlert]:
        """Get recent alerts, optionally filtered by priority"""
        alerts = self.alerts[-count:] if len(self.alerts) > count else self.alerts
        
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def clear_alerts(self):
        """Clear all in-memory alerts (does not affect log file)"""
        self.alerts = []


class SetupDetector:
    """Detects high-confidence trading setups and generates alerts"""
    
    def __init__(self, alert_system: AlertSystem, my_tickers_only: bool = False, ticker_manager=None):
        """Initialize setup detector
        
        Args:
            alert_system: Alert system to send notifications
            my_tickers_only: If True, only generate alerts for tickers in 'My Tickers'
            ticker_manager: TickerManager instance for filtering
        """
        self.alert_system = alert_system
        self.my_tickers_only = my_tickers_only
        self.ticker_manager = ticker_manager
        self._my_tickers_cache: Optional[Set[str]] = None
        self._cache_time: Optional[datetime] = None
    
    def _get_my_tickers(self) -> Set[str]:
        """Get 'My Tickers' with caching (1 minute cache)"""
        now = datetime.now()
        
        # Refresh cache if older than 1 minute
        if self._my_tickers_cache is None or self._cache_time is None or \
           (now - self._cache_time).total_seconds() > 60:
            if self.ticker_manager:
                try:
                    tickers_list = self.ticker_manager.get_all_tickers(limit=1000)
                    self._my_tickers_cache = set(t['ticker'].upper() for t in tickers_list)
                    self._cache_time = now
                    logger.info(f"My Tickers cache refreshed: {len(self._my_tickers_cache)} tickers")
                except Exception as e:
                    logger.error(f"Error loading My Tickers: {e}")
                    self._my_tickers_cache = set()
            else:
                self._my_tickers_cache = set()
        
        return self._my_tickers_cache or set()
    
    def _should_alert(self, ticker: str) -> bool:
        """Check if alert should be generated for this ticker"""
        if not self.my_tickers_only:
            return True
        
        my_tickers = self._get_my_tickers()
        return ticker.upper() in my_tickers
    
    def analyze_for_alerts(self, analysis) -> List[TradingAlert]:
        """
        Analyze stock analysis for alert-worthy setups.
        Returns list of triggered alerts.
        """
        alerts = []
        
        # Check if we should generate alerts for this ticker
        if not self._should_alert(analysis.ticker):
            logger.debug(f"Skipping alerts for {analysis.ticker} (not in My Tickers)")
            return alerts
        
        # 1. EMA Reclaim Alert (CRITICAL)
        if analysis.ema_reclaim:
            alert = TradingAlert(
                ticker=analysis.ticker,
                alert_type=AlertType.EMA_RECLAIM,
                priority=AlertPriority.CRITICAL,
                message=f"🔥 EMA RECLAIM CONFIRMED - High probability bullish setup",
                confidence_score=analysis.confidence_score,
                details={
                    "ema8": analysis.ema8,
                    "ema21": analysis.ema21,
                    "price": analysis.price,
                    "demarker": analysis.demarker
                }
            )
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
        
        # 2. Triple Threat: Reclaim + Aligned + Strong RS (CRITICAL)
        if (analysis.ema_reclaim and 
            analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned') and
            analysis.sector_rs and analysis.sector_rs.get('rs_score', 0) > 60):
            
            alert = TradingAlert(
                ticker=analysis.ticker,
                alert_type=AlertType.HIGH_CONFIDENCE,
                priority=AlertPriority.CRITICAL,
                message=f"🚀 TRIPLE THREAT SETUP - Reclaim + Aligned + Sector Leader (Confidence: {analysis.confidence_score:.0f})",
                confidence_score=analysis.confidence_score,
                details={
                    "ema_reclaim": True,
                    "timeframe_alignment": analysis.timeframe_alignment.get('alignment_score'),
                    "sector_rs": analysis.sector_rs.get('rs_score'),
                    "fib_targets": analysis.fib_targets
                }
            )
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
        
        # 3. Timeframe Alignment (HIGH)
        elif analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned'):
            alert = TradingAlert(
                ticker=analysis.ticker,
                alert_type=AlertType.TIMEFRAME_ALIGNED,
                priority=AlertPriority.HIGH,
                message=f"✅ Multi-Timeframe Aligned ({analysis.timeframe_alignment.get('alignment_score', 0):.0f}%)",
                confidence_score=analysis.confidence_score,
                details={
                    "timeframes": analysis.timeframe_alignment.get('timeframes', {}),
                    "alignment_score": analysis.timeframe_alignment.get('alignment_score')
                }
            )
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
        
        # 4. Sector Leader (HIGH)
        if analysis.sector_rs and analysis.sector_rs.get('rs_score', 0) > 70:
            alert = TradingAlert(
                ticker=analysis.ticker,
                alert_type=AlertType.SECTOR_LEADER,
                priority=AlertPriority.HIGH,
                message=f"💪 Strong Sector Leader (RS: {analysis.sector_rs.get('rs_score', 0):.1f})",
                confidence_score=analysis.confidence_score,
                details={
                    "sector": analysis.sector_rs.get('sector'),
                    "rs_score": analysis.sector_rs.get('rs_score'),
                    "vs_spy": analysis.sector_rs.get('vs_spy'),
                    "vs_sector": analysis.sector_rs.get('vs_sector')
                }
            )
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
        
        # 5. Fibonacci Setup with DeMarker Entry (HIGH)
        if (analysis.fib_targets and 
            analysis.demarker is not None and analysis.demarker <= 0.30 and
            "UPTREND" in analysis.trend):
            
            alert = TradingAlert(
                ticker=analysis.ticker,
                alert_type=AlertType.FIBONACCI_SETUP,
                priority=AlertPriority.HIGH,
                message=f"🎯 Fibonacci Pullback Entry - DeMarker oversold in uptrend",
                confidence_score=analysis.confidence_score,
                details={
                    "demarker": analysis.demarker,
                    "fib_targets": analysis.fib_targets,
                    "trend": analysis.trend,
                    "price": analysis.price
                }
            )
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
        
        # 6. DeMarker Entry Signal (MEDIUM)
        elif analysis.demarker is not None:
            if analysis.demarker <= 0.30 and "UPTREND" in analysis.trend:
                alert = TradingAlert(
                    ticker=analysis.ticker,
                    alert_type=AlertType.DEMARKER_ENTRY,
                    priority=AlertPriority.MEDIUM,
                    message=f"📉 DeMarker Oversold Entry ({analysis.demarker:.2f}) in uptrend",
                    confidence_score=analysis.confidence_score,
                    details={"demarker": analysis.demarker, "trend": analysis.trend}
                )
                alerts.append(alert)
                self.alert_system.trigger_alert(alert)
            elif analysis.demarker >= 0.70 and "DOWNTREND" in analysis.trend:
                alert = TradingAlert(
                    ticker=analysis.ticker,
                    alert_type=AlertType.DEMARKER_ENTRY,
                    priority=AlertPriority.MEDIUM,
                    message=f"📈 DeMarker Overbought Entry ({analysis.demarker:.2f}) in downtrend",
                    confidence_score=analysis.confidence_score,
                    details={"demarker": analysis.demarker, "trend": analysis.trend}
                )
                alerts.append(alert)
                self.alert_system.trigger_alert(alert)
        
        # 7. High Confidence Setup (general) (HIGH)
        if analysis.confidence_score >= 85 and not any(a.alert_type == AlertType.HIGH_CONFIDENCE for a in alerts):
            alert = TradingAlert(
                ticker=analysis.ticker,
                alert_type=AlertType.HIGH_CONFIDENCE,
                priority=AlertPriority.HIGH,
                message=f"⭐ High Confidence Setup (Score: {analysis.confidence_score:.0f})",
                confidence_score=analysis.confidence_score,
                details={
                    "rsi": analysis.rsi,
                    "trend": analysis.trend,
                    "iv_rank": analysis.iv_rank
                }
            )
            alerts.append(alert)
            self.alert_system.trigger_alert(alert)
        
        return alerts


# Convenience functions for common callback types

def console_callback(alert: TradingAlert):
    """Print alert to console"""
    print(f"\n{'='*60}")
    print(f"🔔 TRADING ALERT: {alert.ticker}")
    print(f"{'='*60}")
    print(f"Priority: {alert.priority.value}")
    print(f"Type: {alert.alert_type.value}")
    print(f"Message: {alert.message}")
    print(f"Confidence: {alert.confidence_score:.1f}")
    print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    if alert.details:
        print(f"Details: {json.dumps(alert.details, indent=2)}")
    print(f"{'='*60}\n")


def email_callback_stub(alert: TradingAlert):
    """Stub for email notification (implement with your email service)"""
    # TODO: Implement actual email sending
    logger.info(f"Would send email alert for {alert.ticker}: {alert.message}")


def webhook_callback_stub(alert: TradingAlert):
    """Stub for webhook notification (implement with your webhook service)"""
    # TODO: Implement actual webhook POST
    logger.info(f"Would send webhook for {alert.ticker}: {alert.message}")


# Global alert system instance
_global_alert_system = None

def get_alert_system() -> AlertSystem:
    """Get or create global alert system instance"""
    global _global_alert_system
    if _global_alert_system is None:
        _global_alert_system = AlertSystem()
    return _global_alert_system

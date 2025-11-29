"""
Service Orchestrator - Unified Background Service Management

Coordinates all background services to prevent:
- Resource conflicts (API rate limits, CPU overload)
- Duplicate processing
- Missed opportunities due to service conflicts

Features:
- Priority-based scheduling (critical services run first)
- Staggered execution (services don't overlap)
- Health monitoring with auto-recovery
- Discord alert queue integration
- One-click workflow modes
"""

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger
import queue


class ServicePriority(Enum):
    """Service execution priority"""
    CRITICAL = 1      # AI Position Manager (manages active trades)
    HIGH = 2          # Breakout Monitor (time-sensitive signals)
    MEDIUM = 3        # DEX Launch Monitor
    LOW = 4           # Stock Monitor, ORB Scanner
    BACKGROUND = 5    # Discord bot, data sync


class ServiceState(Enum):
    """Service lifecycle state"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COOLDOWN = "cooldown"  # Waiting between scans


@dataclass
class ServiceHealth:
    """Health metrics for a service"""
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    success_count: int = 0
    avg_run_time_seconds: float = 0.0
    alerts_generated: int = 0
    
    def to_dict(self) -> dict:
        return {
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "avg_run_time_seconds": round(self.avg_run_time_seconds, 2),
            "alerts_generated": self.alerts_generated
        }


@dataclass
class ManagedService:
    """A service managed by the orchestrator"""
    name: str
    display_name: str
    priority: ServicePriority
    interval_seconds: int
    category: str  # crypto, stocks, infrastructure
    
    # Runtime state
    state: ServiceState = ServiceState.STOPPED
    health: ServiceHealth = field(default_factory=ServiceHealth)
    
    # Execution control
    last_execution_start: Optional[datetime] = None
    next_scheduled_run: Optional[datetime] = None
    execution_lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)  # Can't run simultaneously
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "priority": self.priority.name,
            "interval_seconds": self.interval_seconds,
            "category": self.category,
            "state": self.state.value,
            "health": self.health.to_dict(),
            "next_scheduled_run": self.next_scheduled_run.isoformat() if self.next_scheduled_run else None,
            "depends_on": self.depends_on,
            "conflicts_with": self.conflicts_with
        }


@dataclass
class AlertQueueItem:
    """An alert waiting for review/action"""
    id: str
    source: str  # discord, scanner, manual
    symbol: str
    asset_type: str  # crypto, stock
    alert_type: str  # ENTRY, BREAKOUT, EXIT, etc.
    price: Optional[float]
    target: Optional[float]
    stop_loss: Optional[float]
    reasoning: str
    confidence: str  # HIGH, MEDIUM, LOW
    timestamp: datetime
    status: str = "pending"  # pending, approved, rejected, expired
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "symbol": self.symbol,
            "asset_type": self.asset_type,
            "alert_type": self.alert_type,
            "price": self.price,
            "target": self.target,
            "stop_loss": self.stop_loss,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }


class WorkflowMode(Enum):
    """Pre-configured workflow modes"""
    STOPPED = "stopped"           # All services stopped
    SAFE = "safe"                 # Only AI Position Manager (monitor existing)
    DISCOVERY = "discovery"       # Scanners ON, no auto-trade
    ACTIVE = "active"             # Full automation with approvals
    AGGRESSIVE = "aggressive"     # Fast intervals, lower thresholds


# Service definitions with conflict rules
SERVICE_DEFINITIONS = {
    "sentient-crypto-ai-trader": {
        "display_name": "AI Crypto Position Manager",
        "priority": ServicePriority.CRITICAL,
        "default_interval": 120,
        "category": "crypto",
        "conflicts_with": [],  # Can always run
        "depends_on": []
    },
    "sentient-crypto-breakout": {
        "display_name": "Crypto Breakout Scanner",
        "priority": ServicePriority.HIGH,
        "default_interval": 180,
        "category": "crypto",
        "conflicts_with": ["sentient-dex-launch"],  # Don't scan simultaneously
        "depends_on": []
    },
    "sentient-dex-launch": {
        "display_name": "DEX Launch Monitor",
        "priority": ServicePriority.MEDIUM,
        "default_interval": 30,
        "category": "crypto",
        "conflicts_with": ["sentient-crypto-breakout"],
        "depends_on": []
    },
    "sentient-stock-monitor": {
        "display_name": "Stock Monitor",
        "priority": ServicePriority.LOW,
        "default_interval": 300,
        "category": "stocks",
        "conflicts_with": ["sentient-orb-fvg"],
        "depends_on": []
    },
    "sentient-orb-fvg": {
        "display_name": "ORB+FVG Scanner",
        "priority": ServicePriority.LOW,
        "default_interval": 60,
        "category": "stocks",
        "conflicts_with": ["sentient-stock-monitor"],
        "depends_on": []
    },
    "sentient-discord-approval": {
        "display_name": "Discord Approval Bot",
        "priority": ServicePriority.BACKGROUND,
        "default_interval": 0,  # Always running, no interval
        "category": "infrastructure",
        "conflicts_with": [],
        "depends_on": []
    }
}


class ServiceOrchestrator:
    """
    Central coordinator for all background services.
    Prevents conflicts, manages priorities, and provides unified control.
    """
    
    def __init__(
        self,
        state_file: str = "data/orchestrator_state.json",
        alert_queue_file: str = "data/alert_queue.json"
    ):
        self.state_file = Path(__file__).parent.parent / state_file
        self.alert_queue_file = Path(__file__).parent.parent / alert_queue_file
        
        # Managed services
        self.services: Dict[str, ManagedService] = {}
        
        # Alert queue
        self.alert_queue: List[AlertQueueItem] = []
        self.alert_queue_lock = threading.Lock()
        
        # Orchestration state
        self.current_mode = WorkflowMode.STOPPED
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.execution_lock = threading.Lock()
        
        # Callbacks
        self.on_alert_callback: Optional[Callable[[AlertQueueItem], None]] = None
        self.on_service_state_change: Optional[Callable[[str, ServiceState], None]] = None
        
        # Initialize services
        self._init_services()
        self._load_state()
        
        logger.info("=" * 70)
        logger.info("🎯 SERVICE ORCHESTRATOR INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"   Services: {len(self.services)}")
        logger.info(f"   Pending Alerts: {len(self.alert_queue)}")
        logger.info(f"   Mode: {self.current_mode.value}")
        logger.info("=" * 70)
    
    def _init_services(self):
        """Initialize managed service instances"""
        for name, config in SERVICE_DEFINITIONS.items():
            self.services[name] = ManagedService(
                name=name,
                display_name=config["display_name"],
                priority=config["priority"],
                interval_seconds=config["default_interval"],
                category=config["category"],
                conflicts_with=config.get("conflicts_with", []),
                depends_on=config.get("depends_on", [])
            )
    
    def refresh_state(self):
        """Manually refresh state from disk (useful for cross-process sync)"""
        self._load_state()

    def _load_state(self):
        """Load persisted orchestrator state"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    
                # Restore mode
                mode_str = data.get("current_mode", "stopped")
                self.current_mode = WorkflowMode(mode_str)
                
                # Restore service intervals from config
                intervals_file = self.state_file.parent / "service_intervals.json"
                if intervals_file.exists():
                    with open(intervals_file, 'r') as f:
                        intervals = json.load(f)
                    for name, config in intervals.items():
                        if name in self.services:
                            interval_key = "check_interval_seconds" if "check" in str(config) else "scan_interval_seconds"
                            if interval_key in config:
                                self.services[name].interval_seconds = config[interval_key]
            
            # Load alert queue
            if self.alert_queue_file.exists():
                with open(self.alert_queue_file, 'r') as f:
                    queue_data = json.load(f)
                for item in queue_data:
                    self.alert_queue.append(AlertQueueItem(
                        id=item["id"],
                        source=item["source"],
                        symbol=item["symbol"],
                        asset_type=item["asset_type"],
                        alert_type=item["alert_type"],
                        price=item.get("price"),
                        target=item.get("target"),
                        stop_loss=item.get("stop_loss"),
                        reasoning=item["reasoning"],
                        confidence=item["confidence"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        status=item.get("status", "pending"),
                        expires_at=datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None,
                        metadata=item.get("metadata", {})
                    ))
                    
        except Exception as e:
            logger.warning(f"Could not load orchestrator state: {e}")
    
    def _save_state(self):
        """Persist orchestrator state"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state = {
                "current_mode": self.current_mode.value,
                "services": {name: svc.to_dict() for name, svc in self.services.items()},
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save alert queue separately
            with self.alert_queue_lock:
                queue_data = [item.to_dict() for item in self.alert_queue]
            with open(self.alert_queue_file, 'w') as f:
                json.dump(queue_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save orchestrator state: {e}")
    
    # =========================================================================
    # WORKFLOW MODES
    # =========================================================================
    
    def set_mode(self, mode: WorkflowMode) -> bool:
        """
        Switch to a workflow mode. This configures which services run.
        
        Modes:
        - STOPPED: All services off
        - SAFE: Only AI Position Manager (monitor existing positions)
        - DISCOVERY: Scanners ON, alerts queue for review, no auto-trade
        - ACTIVE: Full automation with Discord approvals
        - AGGRESSIVE: Fast intervals, more sensitive triggers
        """
        logger.info(f"🔄 Switching to mode: {mode.value}")
        
        self.current_mode = mode
        
        # Configure services based on mode
        if mode == WorkflowMode.STOPPED:
            for svc in self.services.values():
                svc.state = ServiceState.STOPPED
                
        elif mode == WorkflowMode.SAFE:
            # Only AI Position Manager
            for name, svc in self.services.items():
                if name == "sentient-crypto-ai-trader":
                    svc.state = ServiceState.RUNNING
                elif name == "sentient-discord-approval":
                    svc.state = ServiceState.RUNNING  # Keep Discord for approvals
                else:
                    svc.state = ServiceState.STOPPED
                    
        elif mode == WorkflowMode.DISCOVERY:
            # Scanners + AI Monitor, alerts queue for review
            for name, svc in self.services.items():
                if svc.category == "crypto" or name == "sentient-discord-approval":
                    svc.state = ServiceState.RUNNING
                else:
                    svc.state = ServiceState.STOPPED
                    
        elif mode == WorkflowMode.ACTIVE:
            # All crypto services active
            for name, svc in self.services.items():
                if svc.category in ["crypto", "infrastructure"]:
                    svc.state = ServiceState.RUNNING
                else:
                    svc.state = ServiceState.STOPPED
                    
        elif mode == WorkflowMode.AGGRESSIVE:
            # All services, faster intervals
            for svc in self.services.values():
                svc.state = ServiceState.RUNNING
                # Reduce intervals by 50% (min 30s)
                svc.interval_seconds = max(30, svc.interval_seconds // 2)
        
        self._save_state()
        logger.info(f"✅ Mode set to: {mode.value}")
        return True
    
    def get_mode(self) -> WorkflowMode:
        """Get current workflow mode"""
        return self.current_mode
    
    # =========================================================================
    # ALERT QUEUE MANAGEMENT
    # =========================================================================
    
    def add_alert(
        self,
        symbol: str,
        alert_type: str,
        source: str = "scanner",
        asset_type: str = "crypto",
        price: Optional[float] = None,
        target: Optional[float] = None,
        stop_loss: Optional[float] = None,
        reasoning: str = "",
        confidence: str = "MEDIUM",
        expires_minutes: int = 60,
        metadata: Optional[Dict] = None
    ) -> AlertQueueItem:
        """
        Add an alert to the queue for review.
        Called by scanners, Discord listener, or manual input.
        """
        alert_id = f"{symbol}_{alert_type}_{int(datetime.now().timestamp())}"
        
        alert = AlertQueueItem(
            id=alert_id,
            source=source,
            symbol=symbol,
            asset_type=asset_type,
            alert_type=alert_type,
            price=price,
            target=target,
            stop_loss=stop_loss,
            reasoning=reasoning,
            confidence=confidence,
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=expires_minutes),
            metadata=metadata or {}
        )
        
        with self.alert_queue_lock:
            # Check for duplicate (same symbol + type within 5 minutes)
            for existing in self.alert_queue:
                if (existing.symbol == symbol and 
                    existing.alert_type == alert_type and
                    existing.status == "pending" and
                    (datetime.now() - existing.timestamp).seconds < 300):
                    logger.debug(f"Duplicate alert suppressed: {symbol} {alert_type}")
                    return existing
            
            self.alert_queue.append(alert)
            
            # Keep queue manageable (last 100 alerts)
            if len(self.alert_queue) > 100:
                # Remove oldest non-pending alerts
                self.alert_queue = [a for a in self.alert_queue if a.status == "pending"][-100:]
        
        logger.info(f"📥 Alert queued: {symbol} {alert_type} ({confidence})")
        
        # Trigger callback if set
        if self.on_alert_callback:
            try:
                self.on_alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        self._save_state()
        return alert
    
    def reload_alerts_from_file(self):
        """Reload alert queue from file (for cross-process sync)"""
        try:
            if self.alert_queue_file.exists():
                with open(self.alert_queue_file, 'r') as f:
                    queue_data = json.load(f)
                
                with self.alert_queue_lock:
                    # Build set of existing IDs to avoid duplicates
                    existing_ids = {a.id for a in self.alert_queue}
                    
                    for item in queue_data:
                        if item["id"] not in existing_ids:
                            self.alert_queue.append(AlertQueueItem(
                                id=item["id"],
                                source=item["source"],
                                symbol=item["symbol"],
                                asset_type=item["asset_type"],
                                alert_type=item["alert_type"],
                                price=item.get("price"),
                                target=item.get("target"),
                                stop_loss=item.get("stop_loss"),
                                reasoning=item["reasoning"],
                                confidence=item["confidence"],
                                timestamp=datetime.fromisoformat(item["timestamp"]),
                                status=item.get("status", "pending"),
                                expires_at=datetime.fromisoformat(item["expires_at"]) if item.get("expires_at") else None,
                                metadata=item.get("metadata", {})
                            ))
        except Exception as e:
            logger.debug(f"Could not reload alerts from file: {e}")
    
    def get_pending_alerts(self, asset_type: Optional[str] = None) -> List[AlertQueueItem]:
        """Get all pending alerts, optionally filtered by asset type"""
        # Reload from file to pick up alerts from other processes
        self.reload_alerts_from_file()
        
        with self.alert_queue_lock:
            pending = [a for a in self.alert_queue if a.status == "pending"]
            
            # Remove expired
            now = datetime.now()
            for alert in pending:
                if alert.expires_at and alert.expires_at < now:
                    alert.status = "expired"
            
            pending = [a for a in pending if a.status == "pending"]
            
            if asset_type:
                pending = [a for a in pending if a.asset_type == asset_type]
            
            return sorted(pending, key=lambda x: x.timestamp, reverse=True)
    
    def approve_alert(self, alert_id: str, add_to_watchlist: bool = True) -> bool:
        """Approve an alert and optionally add to watchlist"""
        with self.alert_queue_lock:
            for alert in self.alert_queue:
                if alert.id == alert_id:
                    alert.status = "approved"
                    
                    if add_to_watchlist:
                        self._add_to_watchlist(alert)
                    
                    self._save_state()
                    logger.info(f"✅ Alert approved: {alert.symbol}")
                    return True
        return False
    
    def reject_alert(self, alert_id: str) -> bool:
        """Reject an alert"""
        with self.alert_queue_lock:
            for alert in self.alert_queue:
                if alert.id == alert_id:
                    alert.status = "rejected"
                    self._save_state()
                    logger.info(f"❌ Alert rejected: {alert.symbol}")
                    return True
        return False
    
    def clear_expired_alerts(self) -> int:
        """Clear all expired alerts, return count cleared"""
        with self.alert_queue_lock:
            now = datetime.now()
            before = len(self.alert_queue)
            self.alert_queue = [
                a for a in self.alert_queue 
                if a.status == "pending" and (not a.expires_at or a.expires_at > now)
            ]
            cleared = before - len(self.alert_queue)
        
        if cleared > 0:
            self._save_state()
            logger.info(f"🧹 Cleared {cleared} expired alerts")
        return cleared
    
    def _add_to_watchlist(self, alert: AlertQueueItem):
        """Add approved alert to appropriate watchlist"""
        try:
            if alert.asset_type == "crypto":
                # Add to crypto watchlist
                watchlist_file = self.state_file.parent / "service_watchlists.json"
                watchlists = {}
                if watchlist_file.exists():
                    with open(watchlist_file, 'r') as f:
                        watchlists = json.load(f)
                
                # Add to breakout scanner watchlist
                service_name = "sentient-crypto-breakout"
                if service_name not in watchlists:
                    watchlists[service_name] = {"tickers": []}
                
                # Format symbol for Kraken (e.g., BTC -> BTC/USD)
                symbol = alert.symbol
                if "/" not in symbol:
                    symbol = f"{symbol}/USD"
                
                if symbol not in watchlists[service_name]["tickers"]:
                    watchlists[service_name]["tickers"].append(symbol)
                    watchlists[service_name]["updated"] = datetime.now().isoformat()
                    
                    with open(watchlist_file, 'w') as f:
                        json.dump(watchlists, f, indent=2)
                    
                    logger.info(f"📋 Added {symbol} to crypto watchlist")
            else:
                # Add to stock watchlist via TickerManager
                try:
                    from services.ticker_manager import TickerManager
                    tm = TickerManager()
                    tm.add_ticker(alert.symbol, notes=f"From {alert.source}: {alert.reasoning[:100]}")
                    logger.info(f"📋 Added {alert.symbol} to stock watchlist")
                except Exception as e:
                    logger.warning(f"Could not add to stock watchlist: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to add to watchlist: {e}")
    
    # =========================================================================
    # SERVICE CONTROL
    # =========================================================================
    
    def get_service_status(self, service_name: str) -> Optional[Dict]:
        """Get status of a specific service"""
        if service_name in self.services:
            return self.services[service_name].to_dict()
        return None
    
    def get_all_services_status(self) -> Dict[str, Dict]:
        """Get status of all services"""
        return {name: svc.to_dict() for name, svc in self.services.items()}
    
    def can_service_run(self, service_name: str) -> Tuple[bool, str]:
        """Check if a service can run (no conflicts)"""
        if service_name not in self.services:
            return False, "Unknown service"
        
        service = self.services[service_name]
        
        # Check conflicts
        for conflict_name in service.conflicts_with:
            if conflict_name in self.services:
                conflict_svc = self.services[conflict_name]
                if conflict_svc.state == ServiceState.RUNNING:
                    # Check if it's actually executing right now
                    if conflict_svc.last_execution_start:
                        elapsed = (datetime.now() - conflict_svc.last_execution_start).seconds
                        if elapsed < 60:  # Assume max 60s execution time
                            return False, f"Conflicts with {conflict_svc.display_name} (running)"
        
        # Check dependencies
        for dep_name in service.depends_on:
            if dep_name in self.services:
                dep_svc = self.services[dep_name]
                if dep_svc.state != ServiceState.RUNNING:
                    return False, f"Dependency {dep_svc.display_name} not running"
        
        return True, "OK"
    
    def record_service_run(
        self,
        service_name: str,
        success: bool,
        run_time_seconds: float,
        alerts_generated: int = 0,
        error_message: Optional[str] = None
    ):
        """Record a service execution result"""
        if service_name not in self.services:
            return
        
        service = self.services[service_name]
        health = service.health
        
        health.last_run = datetime.now()
        
        if success:
            health.last_success = datetime.now()
            health.success_count += 1
            health.alerts_generated += alerts_generated
        else:
            health.error_count += 1
            health.last_error = error_message
        
        # Update average run time
        total_runs = health.success_count + health.error_count
        health.avg_run_time_seconds = (
            (health.avg_run_time_seconds * (total_runs - 1) + run_time_seconds) / total_runs
        )
        
        # Schedule next run
        service.next_scheduled_run = datetime.now() + timedelta(seconds=service.interval_seconds)
        
        self._save_state()
    
    # =========================================================================
    # DASHBOARD DATA
    # =========================================================================
    
    def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data for Control Panel"""
        pending_alerts = self.get_pending_alerts()
        
        # Group alerts by type
        crypto_alerts = [a for a in pending_alerts if a.asset_type == "crypto"]
        stock_alerts = [a for a in pending_alerts if a.asset_type == "stock"]
        
        # Service health summary
        services_running = sum(1 for s in self.services.values() if s.state == ServiceState.RUNNING)
        services_error = sum(1 for s in self.services.values() if s.state == ServiceState.ERROR)
        
        # Recent activity
        recent_alerts = sorted(
            [a for a in self.alert_queue if a.timestamp > datetime.now() - timedelta(hours=1)],
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]
        
        return {
            "mode": self.current_mode.value,
            "services": {
                "total": len(self.services),
                "running": services_running,
                "error": services_error,
                "details": self.get_all_services_status()
            },
            "alerts": {
                "pending_total": len(pending_alerts),
                "crypto_pending": len(crypto_alerts),
                "stock_pending": len(stock_alerts),
                "recent": [a.to_dict() for a in recent_alerts]
            },
            "last_updated": datetime.now().isoformat()
        }


# Global instance
_orchestrator: Optional[ServiceOrchestrator] = None


def get_orchestrator() -> ServiceOrchestrator:
    """Get or create global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ServiceOrchestrator()
    return _orchestrator

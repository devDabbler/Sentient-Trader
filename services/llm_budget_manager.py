"""
LLM Budget Manager
==================

Tracks and enforces LLM API call budgets to prevent accidental overspending.
Designed for services with limited free tiers (e.g., 100 calls/month).

Features:
- Persistent storage (survives restarts)
- Monthly reset
- Daily caps (optional)
- Usage statistics
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class BudgetStats:
    """Budget usage statistics"""
    monthly_limit: int = 100
    monthly_used: int = 0
    daily_limit: int = 5  # Optional daily cap
    daily_used: int = 0
    last_reset_month: int = 0
    last_reset_day: int = 0
    last_updated: str = ""
    
    # History
    total_calls_ever: int = 0
    calls_by_service: Dict[str, int] = field(default_factory=dict)


class LLMBudgetManager:
    """
    Manages LLM API call budget with persistent tracking.
    
    Usage:
        budget = LLMBudgetManager(monthly_limit=100, daily_limit=5)
        
        if budget.can_use_call():
            # Make LLM call
            result = await llm_client.call(...)
            budget.increment_calls(service_name="x_sentiment")
        else:
            # Fall back to rule-based
            result = rule_based_analysis(...)
    """
    
    def __init__(
        self,
        monthly_limit: int = 100,
        daily_limit: int = 5,
        storage_path: Optional[str] = None
    ):
        """
        Initialize budget manager.
        
        Args:
            monthly_limit: Maximum calls per calendar month
            daily_limit: Maximum calls per day (0 = unlimited)
            storage_path: Path to persist budget data
        """
        self.monthly_limit = monthly_limit
        self.daily_limit = daily_limit
        
        # Storage path (defaults to data folder)
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(__file__).parent.parent / "data" / "llm_budget.json"
        
        # Load or create stats
        self.stats = self._load_stats()
        
        # Check for month/day rollover
        self._check_reset()
        
        logger.info(
            f"✅ LLM Budget Manager initialized: "
            f"{self.stats.monthly_used}/{self.monthly_limit} monthly, "
            f"{self.stats.daily_used}/{self.daily_limit} daily"
        )
    
    def can_use_call(self) -> bool:
        """
        Check if a call is allowed within budget.
        
        Returns:
            True if call is allowed, False if budget exhausted
        """
        self._check_reset()
        
        # Check monthly limit
        if self.stats.monthly_used >= self.monthly_limit:
            logger.warning(
                f"⚠️ Monthly LLM budget exhausted: "
                f"{self.stats.monthly_used}/{self.monthly_limit}"
            )
            return False
        
        # Check daily limit (if set)
        if self.daily_limit > 0 and self.stats.daily_used >= self.daily_limit:
            logger.warning(
                f"⚠️ Daily LLM budget exhausted: "
                f"{self.stats.daily_used}/{self.daily_limit}"
            )
            return False
        
        return True
    
    def increment_calls(self, count: int = 1, service_name: str = "unknown"):
        """
        Record LLM call(s) usage.
        
        Args:
            count: Number of calls to record
            service_name: Service that made the call (for tracking)
        """
        self._check_reset()
        
        self.stats.monthly_used += count
        self.stats.daily_used += count
        self.stats.total_calls_ever += count
        
        # Track by service
        if service_name not in self.stats.calls_by_service:
            self.stats.calls_by_service[service_name] = 0
        self.stats.calls_by_service[service_name] += count
        
        self.stats.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Save to disk
        self._save_stats()
        
        logger.info(
            f"📊 LLM call recorded ({service_name}): "
            f"{self.stats.monthly_used}/{self.monthly_limit} monthly, "
            f"{self.stats.daily_used}/{self.daily_limit} daily"
        )
    
    def get_remaining(self) -> Dict[str, int]:
        """Get remaining budget"""
        self._check_reset()
        return {
            "monthly_remaining": max(0, self.monthly_limit - self.stats.monthly_used),
            "daily_remaining": max(0, self.daily_limit - self.stats.daily_used) if self.daily_limit > 0 else -1,
            "monthly_used": self.stats.monthly_used,
            "daily_used": self.stats.daily_used,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get full budget statistics"""
        remaining = self.get_remaining()
        return {
            **remaining,
            "monthly_limit": self.monthly_limit,
            "daily_limit": self.daily_limit,
            "total_calls_ever": self.stats.total_calls_ever,
            "calls_by_service": self.stats.calls_by_service,
            "last_updated": self.stats.last_updated,
        }
    
    def reset_monthly(self):
        """Manually reset monthly counter (for testing)"""
        self.stats.monthly_used = 0
        self.stats.last_reset_month = datetime.now(timezone.utc).month
        self._save_stats()
        logger.info("🔄 Monthly LLM budget reset")
    
    def reset_daily(self):
        """Manually reset daily counter"""
        self.stats.daily_used = 0
        self.stats.last_reset_day = datetime.now(timezone.utc).day
        self._save_stats()
        logger.info("🔄 Daily LLM budget reset")
    
    def _check_reset(self):
        """Check if month/day rolled over and reset counters"""
        now = datetime.now(timezone.utc)
        
        # Monthly reset
        if now.month != self.stats.last_reset_month:
            logger.info(f"🔄 New month detected - resetting monthly LLM budget")
            self.stats.monthly_used = 0
            self.stats.last_reset_month = now.month
            self._save_stats()
        
        # Daily reset
        if now.day != self.stats.last_reset_day:
            logger.debug(f"🔄 New day detected - resetting daily LLM budget")
            self.stats.daily_used = 0
            self.stats.last_reset_day = now.day
            self._save_stats()
    
    def _load_stats(self) -> BudgetStats:
        """Load stats from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    return BudgetStats(**data)
        except Exception as e:
            logger.warning(f"Could not load budget stats: {e}")
        
        # Create new stats
        now = datetime.now(timezone.utc)
        return BudgetStats(
            monthly_limit=self.monthly_limit,
            daily_limit=self.daily_limit,
            last_reset_month=now.month,
            last_reset_day=now.day,
            last_updated=now.isoformat(),
        )
    
    def _save_stats(self):
        """Save stats to disk"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                # Convert dataclass to dict, handling nested dict
                data = asdict(self.stats)
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save budget stats: {e}")


# ============================================================================
# Singleton & Convenience Functions
# ============================================================================

_budget_manager: Optional[LLMBudgetManager] = None


def get_llm_budget_manager(
    monthly_limit: int = 100,
    daily_limit: int = 5
) -> LLMBudgetManager:
    """Get or create singleton budget manager"""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = LLMBudgetManager(
            monthly_limit=monthly_limit,
            daily_limit=daily_limit
        )
    return _budget_manager


def can_use_llm_call() -> bool:
    """Quick check if LLM call is within budget"""
    return get_llm_budget_manager().can_use_call()


def record_llm_call(service_name: str = "unknown"):
    """Quick helper to record an LLM call"""
    get_llm_budget_manager().increment_calls(service_name=service_name)


def get_llm_budget_remaining() -> Dict[str, int]:
    """Quick helper to get remaining budget"""
    return get_llm_budget_manager().get_remaining()


# Export
__all__ = [
    "LLMBudgetManager",
    "BudgetStats",
    "get_llm_budget_manager",
    "can_use_llm_call",
    "record_llm_call",
    "get_llm_budget_remaining",
]

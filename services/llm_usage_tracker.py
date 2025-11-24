"""
LLM Usage Tracker and Cost Monitor
Provides utilities for tracking, analyzing, and displaying LLM usage statistics
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from models.llm_models import UsageStats, LLMPriority, LLMProvider
from services.llm_request_manager import get_llm_manager


logger = logging.getLogger(__name__)


class LLMUsageTracker:
    """Tracks and analyzes LLM usage across services"""
    
    def __init__(self, log_path: str = "logs/llm_usage.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.manager = get_llm_manager()
    
    def get_current_stats(self, service_name: Optional[str] = None) -> Dict[str, UsageStats]:
        """Get current usage statistics"""
        return self.manager.get_usage_stats(service_name)
    
    def get_total_cost(self) -> float:
        """Get total cost across all services"""
        return self.manager.get_total_cost()
    
    def get_service_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Get breakdown of usage by service
        
        Returns:
            {
                "service_name": {
                    "requests": 120,
                    "cached": 45,
                    "tokens": 150000,
                    "cost": 0.45,
                    "cache_hit_rate": 0.375
                }
            }
        """
        stats = self.get_current_stats()
        breakdown = {}
        
        for service_name, service_stats in stats.items():
            cache_hit_rate = (
                service_stats.cached_requests / service_stats.total_requests
                if service_stats.total_requests > 0 else 0.0
            )
            
            breakdown[service_name] = {
                "requests": service_stats.total_requests,
                "cached": service_stats.cached_requests,
                "tokens": service_stats.total_tokens,
                "cost": service_stats.total_cost_usd,
                "cache_hit_rate": cache_hit_rate,
                "errors": service_stats.errors
            }
        
        return breakdown
    
    def get_priority_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of requests by priority
        
        Returns:
            {"CRITICAL": 10, "HIGH": 50, "MEDIUM": 100, "LOW": 200}
        """
        stats = self.get_current_stats()
        priority_counts = {p.name: 0 for p in LLMPriority}
        
        for service_stats in stats.values():
            for priority, count in service_stats.requests_by_priority.items():
                priority_counts[priority] = priority_counts.get(priority, 0) + count
        
        return priority_counts
    
    def get_provider_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of requests by provider
        
        Returns:
            {"openrouter": 300, "claude": 20, "openai": 10}
        """
        stats = self.get_current_stats()
        provider_counts = {p.value: 0 for p in LLMProvider}
        
        for service_stats in stats.values():
            for provider, count in service_stats.requests_by_provider.items():
                provider_counts[provider] = provider_counts.get(provider, 0) + count
        
        return provider_counts
    
    def get_cost_by_service(self) -> List[Tuple[str, float]]:
        """
        Get services ranked by cost
        
        Returns:
            [("service_name", 1.25), ("another_service", 0.50), ...]
        """
        breakdown = self.get_service_breakdown()
        return sorted(
            [(service, data["cost"]) for service, data in breakdown.items()],
            key=lambda x: x[1],
            reverse=True
        )
    
    def get_efficiency_metrics(self) -> Dict[str, float]:
        """
        Calculate efficiency metrics
        
        Returns:
            {
                "overall_cache_hit_rate": 0.45,
                "avg_cost_per_request": 0.0015,
                "avg_tokens_per_request": 500,
                "total_cost_saved_by_cache": 0.25
            }
        """
        stats = self.get_current_stats()
        
        total_requests = sum(s.total_requests for s in stats.values())
        total_cached = sum(s.cached_requests for s in stats.values())
        total_tokens = sum(s.total_tokens for s in stats.values())
        total_cost = sum(s.total_cost_usd for s in stats.values())
        
        if total_requests == 0:
            return {
                "overall_cache_hit_rate": 0.0,
                "avg_cost_per_request": 0.0,
                "avg_tokens_per_request": 0.0,
                "total_cost_saved_by_cache": 0.0
            }
        
        cache_hit_rate = total_cached / total_requests
        avg_cost_per_request = total_cost / (total_requests - total_cached) if total_requests > total_cached else 0.0
        avg_tokens_per_request = total_tokens / (total_requests - total_cached) if total_requests > total_cached else 0.0
        cost_saved = total_cached * avg_cost_per_request
        
        return {
            "overall_cache_hit_rate": cache_hit_rate,
            "avg_cost_per_request": avg_cost_per_request,
            "avg_tokens_per_request": avg_tokens_per_request,
            "total_cost_saved_by_cache": cost_saved
        }
    
    def save_snapshot(self):
        """Save current statistics to JSON log"""
        stats = self.get_current_stats()
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "total_cost": self.get_total_cost(),
            "services": {
                name: {
                    "total_requests": s.total_requests,
                    "cached_requests": s.cached_requests,
                    "total_tokens": s.total_tokens,
                    "total_cost_usd": s.total_cost_usd,
                    "requests_by_priority": s.requests_by_priority,
                    "requests_by_provider": s.requests_by_provider,
                    "errors": s.errors
                }
                for name, s in stats.items()
            }
        }
        
        # Load existing log
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {"snapshots": []}
        
        # Append snapshot
        log_data["snapshots"].append(snapshot)
        
        # Keep only last 1000 snapshots
        if len(log_data["snapshots"]) > 1000:
            log_data["snapshots"] = log_data["snapshots"][-1000:]
        
        # Save
        with open(self.log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Saved LLM usage snapshot to {self.log_path}")
    
    def get_historical_cost(self, days: int = 7) -> List[Tuple[str, float]]:
        """
        Get historical cost data for last N days
        
        Returns:
            [("2025-11-23", 1.25), ("2025-11-22", 1.50), ...]
        """
        if not self.log_path.exists():
            return []
        
        with open(self.log_path, 'r') as f:
            log_data = json.load(f)
        
        # Group by date
        daily_costs = {}
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for snapshot in log_data.get("snapshots", []):
            timestamp = datetime.fromisoformat(snapshot["timestamp"])
            if timestamp >= cutoff_date:
                date_key = timestamp.strftime("%Y-%m-%d")
                cost = snapshot.get("total_cost", 0.0)
                
                if date_key not in daily_costs:
                    daily_costs[date_key] = cost
                else:
                    # Use max cost for the day (cumulative)
                    daily_costs[date_key] = max(daily_costs[date_key], cost)
        
        return sorted(daily_costs.items())
    
    def format_report(self, detailed: bool = False) -> str:
        """
        Format usage report as text
        
        Args:
            detailed: Include detailed breakdown by service
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("LLM USAGE REPORT")
        lines.append("=" * 60)
        
        # Total cost
        total_cost = self.get_total_cost()
        lines.append(f"Total Cost: ${total_cost:.4f}")
        lines.append("")
        
        # Efficiency metrics
        efficiency = self.get_efficiency_metrics()
        lines.append("EFFICIENCY METRICS")
        lines.append("-" * 60)
        lines.append(f"Cache Hit Rate: {efficiency['overall_cache_hit_rate']:.1%}")
        lines.append(f"Avg Cost/Request: ${efficiency['avg_cost_per_request']:.4f}")
        lines.append(f"Avg Tokens/Request: {efficiency['avg_tokens_per_request']:.0f}")
        lines.append(f"Cost Saved by Cache: ${efficiency['total_cost_saved_by_cache']:.4f}")
        lines.append("")
        
        # Priority breakdown
        priority_breakdown = self.get_priority_breakdown()
        lines.append("REQUESTS BY PRIORITY")
        lines.append("-" * 60)
        for priority, count in sorted(priority_breakdown.items(), key=lambda x: LLMPriority[x[0]].value):
            lines.append(f"{priority:10} {count:6d}")
        lines.append("")
        
        # Provider breakdown
        provider_breakdown = self.get_provider_breakdown()
        lines.append("REQUESTS BY PROVIDER")
        lines.append("-" * 60)
        for provider, count in sorted(provider_breakdown.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                lines.append(f"{provider:15} {count:6d}")
        lines.append("")
        
        if detailed:
            # Service breakdown
            lines.append("COST BY SERVICE")
            lines.append("-" * 60)
            for service, cost in self.get_cost_by_service():
                breakdown = self.get_service_breakdown()[service]
                cache_pct = breakdown["cache_hit_rate"] * 100
                lines.append(
                    f"{service:20} ${cost:8.4f} "
                    f"({breakdown['requests']:4d} req, {cache_pct:5.1f}% cached)"
                )
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def alert_on_cost_threshold(self, threshold_usd: float) -> bool:
        """
        Check if total cost exceeds threshold
        
        Args:
            threshold_usd: Alert threshold in USD
        
        Returns:
            True if threshold exceeded
        """
        total_cost = self.get_total_cost()
        
        if total_cost >= threshold_usd:
            logger.warning(
                f"LLM cost threshold exceeded: ${total_cost:.4f} >= ${threshold_usd:.4f}"
            )
            return True
        
        return False
    
    def reset_all_stats(self):
        """Reset all usage statistics"""
        self.manager.reset_stats()
        logger.info("Reset all LLM usage statistics")


# Singleton getter
_tracker_instance = None


def get_llm_usage_tracker() -> LLMUsageTracker:
    """Get singleton instance of LLM Usage Tracker"""
    global _tracker_instance
    
    if _tracker_instance is None:
        _tracker_instance = LLMUsageTracker()
    
    return _tracker_instance

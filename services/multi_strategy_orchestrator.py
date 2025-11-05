"""
Multi-Strategy Orchestrator
Runs multiple trading strategies simultaneously with independent risk management
"""

from loguru import logger
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio



@dataclass
class StrategyAllocation:
    """Capital allocation for each strategy"""
    strategy_name: str
    capital_pct: float  # % of total capital
    max_positions: int
    scan_interval_minutes: int
    config_module: str  # e.g., 'config_background_trader', 'config_swing_trader'


class MultiStrategyOrchestrator:
    """
    Orchestrates multiple trading strategies with independent:
    - Risk management
    - Capital allocation
    - Execution timing
    """
    
    def __init__(self, total_capital: float, allocations: List[StrategyAllocation]):
        """
        Initialize multi-strategy orchestrator
        
        Args:
            total_capital: Total trading capital
            allocations: List of strategy allocations
        """
        self.total_capital = total_capital
        self.allocations = allocations
        self.strategy_runners = {}
        self.strategy_stats = {}
        
        # Validate allocations
        total_pct = sum(a.capital_pct for a in allocations)
        if abs(total_pct - 100.0) > 0.01:
            raise ValueError(f"Allocations must sum to 100%, got {total_pct}%")
        
        logger.info(f"Multi-Strategy Orchestrator initialized with ${total_capital:,.2f}")
        for alloc in allocations:
            capital = total_capital * (alloc.capital_pct / 100)
            logger.info(f"  - {alloc.strategy_name}: ${capital:,.2f} ({alloc.capital_pct}%)")
    
    def get_strategy_capital(self, strategy_name: str) -> float:
        """Get allocated capital for a strategy"""
        alloc = next((a for a in self.allocations if a.strategy_name == strategy_name), None)
        if not alloc:
            return 0
        return self.total_capital * (alloc.capital_pct / 100)
    
    def start_all_strategies(self):
        """Start all strategies with their independent scanners"""
        logger.info("Starting all trading strategies...")
        
        for alloc in self.allocations:
            try:
                self._start_strategy(alloc)
            except Exception as e:
                logger.error(f"Failed to start {alloc.strategy_name}: {e}")
    
    def _start_strategy(self, alloc: StrategyAllocation):
        """Start a single strategy"""
        logger.info(f"Starting {alloc.strategy_name}...")
        
        # Import config module dynamically
        import importlib
        try:
            config = importlib.import_module(alloc.config_module)
        except ImportError:
            logger.error(f"Config module {alloc.config_module} not found")
            return
        
        # Initialize strategy runner
        # (You'll need to implement specific strategy runners)
        self.strategy_stats[alloc.strategy_name] = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'last_scan': None
        }
        
        logger.info(f"✅ {alloc.strategy_name} started")
    
    def get_combined_stats(self) -> Dict:
        """Get combined statistics across all strategies"""
        total_trades = sum(s['trades'] for s in self.strategy_stats.values())
        total_pnl = sum(s['pnl'] for s in self.strategy_stats.values())
        total_wins = sum(s['wins'] for s in self.strategy_stats.values())
        total_losses = sum(s['losses'] for s in self.strategy_stats.values())
        
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'strategies': self.strategy_stats
        }
    
    def rebalance_capital(self):
        """Rebalance capital allocations based on performance"""
        logger.info("Rebalancing capital allocations...")
        
        # Get current performance
        stats = self.get_combined_stats()
        
        # Simple rebalancing: Move capital from losers to winners
        # (This is a placeholder - implement your own logic)
        for strategy_name, strategy_stats in stats['strategies'].items():
            if strategy_stats['pnl'] < 0:
                logger.warning(f"{strategy_name} is negative, consider reducing allocation")
            elif strategy_stats['pnl'] > 0 and strategy_stats['win_rate'] > 60:
                logger.info(f"{strategy_name} performing well, consider increasing allocation")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

def create_default_multi_strategy() -> MultiStrategyOrchestrator:
    """
    Create a default multi-strategy portfolio
    
    Allocation:
    - 40% Scalping (short-term, high frequency)
    - 30% Swing Trading (medium-term, quality setups)
    - 20% Options Premium (income generation)
    - 10% Breakout/Buzzing (high risk/reward)
    """
    
    allocations = [
        StrategyAllocation(
            strategy_name="Scalping",
            capital_pct=40.0,
            max_positions=10,
            scan_interval_minutes=5,
            config_module="config_background_trader"
        ),
        StrategyAllocation(
            strategy_name="Swing Trading",
            capital_pct=30.0,
            max_positions=5,
            scan_interval_minutes=30,
            config_module="config_swing_trader"
        ),
        # Add more strategies as you implement them
        # StrategyAllocation(
        #     strategy_name="Options Premium",
        #     capital_pct=20.0,
        #     max_positions=3,
        #     scan_interval_minutes=1440,  # Daily
        #     config_module="config_options_premium"
        # ),
        # StrategyAllocation(
        #     strategy_name="Breakout Scanner",
        #     capital_pct=10.0,
        #     max_positions=2,
        #     scan_interval_minutes=60,
        #     config_module="config_breakout_scanner"
        # ),
    ]
    
    # Get account balance (you'll need to get this from Tradier)
    # For now, using a placeholder
    total_capital = 10000.0  # Replace with actual account balance
    
    return MultiStrategyOrchestrator(total_capital, allocations)


if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = create_default_multi_strategy()
    stats = orchestrator.get_combined_stats()
    print(f"Combined stats: {stats}")

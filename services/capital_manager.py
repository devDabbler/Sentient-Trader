"""
Capital Manager
Tracks available capital, allocations, and ensures intelligent position sizing
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class PositionAllocation:
    """Tracks capital allocated to a position"""
    ticker: str
    strategy: str
    capital_allocated: float
    timestamp: datetime
    status: str = "OPEN"  # OPEN, CLOSED
    
    # Trade details
    entry_price: Optional[float] = None
    quantity: Optional[int] = None
    current_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class CapitalManager:
    """
    Manages trading capital with intelligent allocation and risk management
    """
    
    def __init__(self, total_capital: float, max_position_pct: float = 5.0,
                 reserve_cash_pct: float = 10.0):
        """
        Initialize capital manager
        
        Args:
            total_capital: Total trading capital available
            max_position_pct: Maximum % of capital per position (default 5%)
            reserve_cash_pct: % of capital to keep in reserve (default 10%)
        """
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.reserve_cash_pct = reserve_cash_pct
        
        # Calculate usable capital
        self.reserve_cash = total_capital * (reserve_cash_pct / 100)
        self.usable_capital = total_capital - self.reserve_cash
        
        # Track allocations
        self.positions: List[PositionAllocation] = []
        
        logger.info(f"💰 Capital Manager Initialized:")
        logger.info(f"   Total Capital: ${total_capital:,.2f}")
        logger.info(f"   Usable Capital: ${self.usable_capital:,.2f}")
        logger.info(f"   Reserve Cash: ${self.reserve_cash:,.2f} ({reserve_cash_pct}%)")
        logger.info(f"   Max Per Position: ${self.get_max_position_size():,.2f} ({max_position_pct}%)")
    
    def get_available_capital(self) -> float:
        """Get currently available capital for new trades"""
        allocated = sum(
            pos.capital_allocated 
            for pos in self.positions 
            if pos.status == "OPEN"
        )
        available = self.usable_capital - allocated
        return max(0, available)
    
    def get_allocated_capital(self) -> float:
        """Get total capital currently allocated"""
        return sum(
            pos.capital_allocated 
            for pos in self.positions 
            if pos.status == "OPEN"
        )
    
    def get_max_position_size(self) -> float:
        """Get maximum capital allowed for a single position"""
        # Use total capital for consistency with strategy selector
        return self.total_capital * (self.max_position_pct / 100)
    
    def get_utilization_pct(self) -> float:
        """Get capital utilization percentage"""
        allocated = self.get_allocated_capital()
        return (allocated / self.usable_capital * 100) if self.usable_capital > 0 else 0
    
    def can_allocate(self, amount: float, ticker: str) -> tuple[bool, str]:
        """
        Check if we can allocate capital for a new position
        
        Returns:
            (can_allocate: bool, reason: str)
        """
        available = self.get_available_capital()
        max_position = self.get_max_position_size()
        
        # Check 1: Sufficient available capital
        if amount > available:
            return False, f"Insufficient capital: ${available:,.2f} available, ${amount:,.2f} requested"
        
        # Check 2: Not exceeding max position size
        if amount > max_position:
            return False, f"Exceeds max position size: ${amount:,.2f} > ${max_position:,.2f}"
        
        # Check 3: Not already in this ticker (optional - can be removed if you want multiple positions in same ticker)
        existing = [p for p in self.positions if p.ticker == ticker and p.status == "OPEN"]
        if existing:
            return False, f"Already have open position in {ticker}"
        
        return True, "OK"
    
    def allocate_capital(self, ticker: str, strategy: str, 
                        position_size_pct: float,
                        entry_price: Optional[float] = None,
                        quantity: Optional[int] = None) -> Optional[PositionAllocation]:
        """
        Allocate capital for a new position
        
        Args:
            ticker: Stock ticker
            strategy: Trading strategy being used
            position_size_pct: Requested position size as % of total capital
            entry_price: Entry price (optional)
            quantity: Share quantity (optional)
            
        Returns:
            PositionAllocation if successful, None if failed
        """
        # Calculate capital to allocate
        requested_capital = self.total_capital * (position_size_pct / 100)
        
        # Check if we can allocate
        can_allocate, reason = self.can_allocate(requested_capital, ticker)
        
        if not can_allocate:
            logger.warning(f"❌ Cannot allocate ${requested_capital:,.2f} for {ticker}: {reason}")
            return None
        
        # Create allocation
        allocation = PositionAllocation(
            ticker=ticker,
            strategy=strategy,
            capital_allocated=requested_capital,
            timestamp=datetime.now(),
            status="OPEN",
            entry_price=entry_price,
            quantity=quantity,
            current_value=requested_capital,  # Initial value = allocated capital
            unrealized_pnl=0.0
        )
        
        self.positions.append(allocation)
        
        available = self.get_available_capital()
        utilization = self.get_utilization_pct()
        
        logger.info(f"✅ Allocated ${requested_capital:,.2f} for {ticker} ({strategy})")
        logger.info(f"   Available Capital: ${available:,.2f} ({100 - utilization:.1f}% free)")
        logger.info(f"   Total Positions: {len([p for p in self.positions if p.status == 'OPEN'])}")
        
        return allocation
    
    def release_capital(self, ticker: str, pnl: float = 0.0):
        """
        Release capital from a closed position
        
        Args:
            ticker: Stock ticker to release
            pnl: Profit/Loss from the trade
        """
        # Find open positions for this ticker
        positions_to_close = [p for p in self.positions if p.ticker == ticker and p.status == "OPEN"]
        
        if not positions_to_close:
            logger.warning(f"No open position found for {ticker}")
            return
        
        for pos in positions_to_close:
            pos.status = "CLOSED"
            pos.unrealized_pnl = pnl
            
            # Update total capital based on P&L
            self.total_capital += pnl
            self.usable_capital = self.total_capital - self.reserve_cash
            
            logger.info(f"💵 Released ${pos.capital_allocated:,.2f} from {ticker}")
            logger.info(f"   P&L: ${pnl:+,.2f}")
            logger.info(f"   New Total Capital: ${self.total_capital:,.2f}")
    
    def update_position(self, ticker: str, current_price: float):
        """
        Update position value based on current price
        
        Args:
            ticker: Stock ticker
            current_price: Current market price
        """
        positions = [p for p in self.positions if p.ticker == ticker and p.status == "OPEN"]
        
        for pos in positions:
            if pos.entry_price and pos.quantity:
                pos.current_value = current_price * pos.quantity
                pos.unrealized_pnl = pos.current_value - (pos.entry_price * pos.quantity)
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        open_positions = [p for p in self.positions if p.status == "OPEN"]
        closed_positions = [p for p in self.positions if p.status == "CLOSED"]
        
        total_allocated = sum(p.capital_allocated for p in open_positions)
        total_unrealized_pnl = sum(p.unrealized_pnl or 0 for p in open_positions)
        total_realized_pnl = sum(p.unrealized_pnl or 0 for p in closed_positions)
        
        return {
            'total_capital': self.total_capital,
            'usable_capital': self.usable_capital,
            'reserve_cash': self.reserve_cash,
            'allocated_capital': total_allocated,
            'available_capital': self.get_available_capital(),
            'utilization_pct': self.get_utilization_pct(),
            'open_positions': len(open_positions),
            'closed_positions': len(closed_positions),
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'positions': {
                'open': [
                    {
                        'ticker': p.ticker,
                        'strategy': p.strategy,
                        'allocated': p.capital_allocated,
                        'current_value': p.current_value,
                        'unrealized_pnl': p.unrealized_pnl,
                        'entry_price': p.entry_price,
                        'quantity': p.quantity
                    }
                    for p in open_positions
                ],
                'closed': [
                    {
                        'ticker': p.ticker,
                        'strategy': p.strategy,
                        'allocated': p.capital_allocated,
                        'pnl': p.unrealized_pnl
                    }
                    for p in closed_positions[-10:]  # Last 10 closed positions
                ]
            }
        }
    
    def get_strategy_allocation(self) -> Dict[str, float]:
        """Get capital allocation breakdown by strategy"""
        strategy_totals = {}
        
        for pos in self.positions:
            if pos.status == "OPEN":
                strategy = pos.strategy
                strategy_totals[strategy] = strategy_totals.get(strategy, 0) + pos.capital_allocated
        
        return strategy_totals
    
    def print_summary(self):
        """Print a nice summary of capital status"""
        summary = self.get_position_summary()
        strategy_breakdown = self.get_strategy_allocation()
        
        print("\n" + "="*80)
        print("💰 CAPITAL MANAGEMENT SUMMARY")
        print("="*80)
        
        print(f"\n📊 Capital Overview:")
        print(f"   Total Capital:      ${summary['total_capital']:>12,.2f}")
        print(f"   Usable Capital:     ${summary['usable_capital']:>12,.2f}")
        print(f"   Reserve Cash:       ${summary['reserve_cash']:>12,.2f}")
        print(f"   Allocated:          ${summary['allocated_capital']:>12,.2f}")
        print(f"   Available:          ${summary['available_capital']:>12,.2f}")
        print(f"   Utilization:        {summary['utilization_pct']:>12.1f}%")
        
        print(f"\n📈 Performance:")
        print(f"   Unrealized P&L:     ${summary['unrealized_pnl']:>12,.2f}")
        print(f"   Realized P&L:       ${summary['realized_pnl']:>12,.2f}")
        print(f"   Total P&L:          ${summary['total_pnl']:>12,.2f}")
        
        print(f"\n🎯 Positions:")
        print(f"   Open:               {summary['open_positions']:>12}")
        print(f"   Closed (Total):     {summary['closed_positions']:>12}")
        
        if strategy_breakdown:
            print(f"\n📋 Strategy Breakdown:")
            for strategy, amount in sorted(strategy_breakdown.items(), key=lambda x: x[1], reverse=True):
                pct = (amount / summary['usable_capital'] * 100) if summary['usable_capital'] > 0 else 0
                print(f"   {strategy:20s} ${amount:>10,.2f} ({pct:>5.1f}%)")
        
        if summary['positions']['open']:
            print(f"\n💼 Open Positions:")
            for pos in summary['positions']['open']:
                pnl_str = f"${pos['unrealized_pnl']:+,.2f}" if pos['unrealized_pnl'] else "N/A"
                print(f"   {pos['ticker']:6s} | {pos['strategy']:15s} | "
                      f"${pos['allocated']:>10,.2f} | P&L: {pnl_str}")
        
        print("="*80 + "\n")


# Global instance (singleton pattern)
_capital_manager = None


def get_capital_manager(total_capital: Optional[float] = None,
                        max_position_pct: float = 5.0,
                        reserve_cash_pct: float = 10.0) -> CapitalManager:
    """
    Get or create global capital manager instance
    
    Args:
        total_capital: Initial capital (only used on first call)
        max_position_pct: Max % per position
        reserve_cash_pct: % to keep in reserve
        
    Returns:
        CapitalManager instance
    """
    global _capital_manager
    
    if _capital_manager is None:
        if total_capital is None:
            raise ValueError("Must provide total_capital for first initialization")
        _capital_manager = CapitalManager(total_capital, max_position_pct, reserve_cash_pct)
    
    return _capital_manager


def reset_capital_manager():
    """Reset the global capital manager (for testing)"""
    global _capital_manager
    _capital_manager = None

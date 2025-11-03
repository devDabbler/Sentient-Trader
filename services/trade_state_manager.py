"""
Trade State Manager - Persistent state tracking for auto-trader
Tracks positions, orders, and trade history across restarts
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"

@dataclass
class TradeRecord:
    """Record of a trade"""
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    entry_price: float
    entry_time: str
    order_id: Optional[str] = None
    bracket_order_ids: Optional[List[str]] = None  # [take_profit_id, stop_loss_id]
    status: str = TradeStatus.OPEN.value
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None  # Why trade was opened/closed
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

class TradeStateManager:
    """
    Manages persistent state for the auto-trader
    Saves to JSON file so state survives restarts
    """
    
    def __init__(self, state_file: str = "data/trade_state.json"):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # State data
        self.open_positions: Dict[str, TradeRecord] = {}
        self.pending_orders: Dict[str, TradeRecord] = {}
        self.closed_trades: List[TradeRecord] = []
        self.trade_history: List[Dict] = []
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # Load existing state
        self._load_state()
        logger.info(f"📊 Trade State Manager initialized: {len(self.open_positions)} open positions")
    
    def _load_state(self):
        """Load state from JSON file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                # Load open positions
                self.open_positions = {
                    symbol: TradeRecord.from_dict(record)
                    for symbol, record in data.get('open_positions', {}).items()
                }
                
                # Load pending orders
                self.pending_orders = {
                    order_id: TradeRecord.from_dict(record)
                    for order_id, record in data.get('pending_orders', {}).items()
                }
                
                # Load closed trades
                self.closed_trades = [
                    TradeRecord.from_dict(record)
                    for record in data.get('closed_trades', [])
                ]
                
                # Load statistics
                self.total_trades = data.get('total_trades', 0)
                self.winning_trades = data.get('winning_trades', 0)
                self.losing_trades = data.get('losing_trades', 0)
                self.total_pnl = data.get('total_pnl', 0.0)
                
                logger.info(f"✅ Loaded state: {len(self.open_positions)} positions, {len(self.closed_trades)} closed trades")
                
        except Exception as e:
            logger.error(f"❌ Error loading state: {e}")
            # Start fresh if can't load
            self.open_positions = {}
            self.pending_orders = {}
            self.closed_trades = []
    
    def _save_state(self):
        """Save state to JSON file"""
        try:
            data = {
                'open_positions': {
                    symbol: record.to_dict()
                    for symbol, record in self.open_positions.items()
                },
                'pending_orders': {
                    order_id: record.to_dict()
                    for order_id, record in self.pending_orders.items()
                },
                'closed_trades': [record.to_dict() for record in self.closed_trades],
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'total_pnl': self.total_pnl,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.debug(f"💾 State saved to {self.state_file}")
            
        except Exception as e:
            logger.error(f"❌ Error saving state: {e}")
    
    def record_trade_opened(self, symbol: str, side: str, quantity: int, entry_price: float, 
                           order_id: Optional[str] = None, bracket_order_ids: Optional[List[str]] = None,
                           reason: str = "Signal"):
        """Record a new trade being opened"""
        trade = TradeRecord(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now().isoformat(),
            order_id=order_id,
            bracket_order_ids=bracket_order_ids,
            status=TradeStatus.OPEN.value,
            reason=reason
        )
        
        self.open_positions[symbol] = trade
        self.total_trades += 1
        self._save_state()
        
        logger.info(f"📝 Recorded trade opened: {side} {quantity} {symbol} @ ${entry_price:.2f}")
    
    def record_trade_closed(self, symbol: str, exit_price: float, reason: str = "Signal"):
        """Record a trade being closed"""
        if symbol not in self.open_positions:
            logger.warning(f"⚠️  Trying to close {symbol} but no open position found")
            return
        
        trade = self.open_positions[symbol]
        trade.status = TradeStatus.CLOSED.value
        trade.exit_price = exit_price
        trade.exit_time = datetime.now().isoformat()
        trade.reason = reason
        
        # Calculate P&L
        if trade.side == 'BUY':
            # Long position: profit when price rises
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            # Short position: profit when price drops
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Update statistics
        self.total_pnl += trade.pnl
        if trade.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Move to closed trades
        self.closed_trades.append(trade)
        del self.open_positions[symbol]
        
        self._save_state()
        
        logger.info(f"📝 Recorded trade closed: {symbol} @ ${exit_price:.2f}, P&L: ${trade.pnl:.2f}")
    
    def has_open_position(self, symbol: str) -> bool:
        """Check if we have an open position for this symbol"""
        return symbol in self.open_positions
    
    def get_open_position(self, symbol: str) -> Optional[TradeRecord]:
        """Get open position for symbol"""
        return self.open_positions.get(symbol)
    
    def get_all_open_positions(self) -> Dict[str, TradeRecord]:
        """Get all open positions"""
        return self.open_positions
    
    def sync_with_broker(self, broker_positions: List[Dict]):
        """
        Sync our state with actual broker positions
        Call this after restart to ensure consistency
        """
        broker_symbols = {pos['symbol'] for pos in broker_positions}
        our_symbols = set(self.open_positions.keys())
        
        # Find positions we think we have but broker doesn't
        orphaned = our_symbols - broker_symbols
        if orphaned:
            logger.warning(f"⚠️  Found orphaned positions (we think we have but broker doesn't): {orphaned}")
            for symbol in orphaned:
                # Mark as closed (probably closed while we were offline)
                trade = self.open_positions[symbol]
                trade.status = TradeStatus.CLOSED.value
                trade.exit_time = datetime.now().isoformat()
                trade.reason = "Closed while offline"
                self.closed_trades.append(trade)
                del self.open_positions[symbol]
        
        # Find positions broker has but we don't know about
        unknown = broker_symbols - our_symbols
        if unknown:
            logger.warning(f"⚠️  Found unknown positions (broker has but we don't know about): {unknown}")
            for symbol in unknown:
                pos = next(p for p in broker_positions if p['symbol'] == symbol)
                # Calculate per-share entry price from cost_basis
                quantity = abs(int(float(pos.get('quantity', 0))))
                cost_basis = float(pos.get('cost_basis', 0))
                entry_price = cost_basis / quantity if quantity > 0 else 0
                
                # Add them to our tracking
                self.record_trade_opened(
                    symbol=symbol,
                    side='BUY' if float(pos.get('quantity', 0)) > 0 else 'SELL',
                    quantity=quantity,
                    entry_price=entry_price,
                    reason="Found on startup"
                )
                logger.info(f"   📊 {symbol}: {quantity} shares @ ${entry_price:.2f} per share (cost basis: ${cost_basis:.2f})")
        
        if orphaned or unknown:
            self._save_state()
            logger.info(f"✅ Synced state with broker: {len(self.open_positions)} positions")
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        avg_win = sum(t.pnl for t in self.closed_trades if t.pnl > 0) / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = sum(t.pnl for t in self.closed_trades if t.pnl < 0) / self.losing_trades if self.losing_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'open_positions': len(self.open_positions)
        }
    
    def clear_old_trades(self, days: int = 30):
        """Clear closed trades older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        
        self.closed_trades = [
            trade for trade in self.closed_trades
            if datetime.fromisoformat(trade.exit_time).timestamp() > cutoff
        ]
        
        self._save_state()
        logger.info(f"🗑️  Cleared trades older than {days} days")


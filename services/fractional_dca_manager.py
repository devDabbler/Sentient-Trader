"""
Fractional DCA (Dollar-Cost Averaging) Manager
Automates recurring fractional share purchases with AI confirmation
"""

from loguru import logger
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass, asdict
import json


@dataclass
class DCASchedule:
    """Represents a DCA schedule for a single stock"""
    ticker: str
    amount_per_interval: float
    frequency: str  # 'daily', 'weekly', 'monthly'
    next_buy_date: datetime
    min_confidence: float = 60.0
    max_price: Optional[float] = None  # Optional price limit
    strategy: str = "AI"  # Default analysis strategy
    active: bool = True
    created_date: Optional[datetime] = None
    total_invested: float = 0.0
    total_shares: float = 0.0
    last_buy_date: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['next_buy_date'] = self.next_buy_date.isoformat() if self.next_buy_date else None
        data['created_date'] = self.created_date.isoformat() if self.created_date else None
        data['last_buy_date'] = self.last_buy_date.isoformat() if self.last_buy_date else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DCASchedule':
        """Create from dictionary"""
        # Convert ISO strings back to datetime
        if data.get('next_buy_date'):
            data['next_buy_date'] = datetime.fromisoformat(data['next_buy_date'])
        if data.get('created_date'):
            data['created_date'] = datetime.fromisoformat(data['created_date'])
        if data.get('last_buy_date') and data['last_buy_date']:
            data['last_buy_date'] = datetime.fromisoformat(data['last_buy_date'])
        return cls(**data)


@dataclass
class DCATransaction:
    """Represents a single DCA purchase"""
    ticker: str
    date: datetime
    amount: float
    price: float
    shares: float
    confidence: float
    strategy: str
    ai_recommendation: str = "BUY"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['date'] = self.date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DCATransaction':
        """Create from dictionary"""
        if data.get('date'):
            data['date'] = datetime.fromisoformat(data['date'])
        return cls(**data)


class FractionalDCAManager:
    """
    Manages automated fractional share DCA strategies
    """
    
    def __init__(self, supabase_client=None):
        self.supabase = supabase_client
        self.schedules: Dict[str, DCASchedule] = {}
        self.transactions: List[DCATransaction] = []
        logger.info("✅ Fractional DCA Manager initialized")
    
    def add_schedule(self, ticker: str, amount: float, frequency: str, 
                    min_confidence: float = 60.0, strategy: str = "AI",
                    max_price: Optional[float] = None) -> DCASchedule:
        """
        Add a new DCA schedule
        
        Args:
            ticker: Stock ticker
            amount: Dollar amount per interval
            frequency: 'daily', 'weekly', 'monthly'
            min_confidence: Minimum AI confidence to execute (0-100)
            strategy: Analysis strategy to use
            max_price: Optional max price limit
        """
        # Calculate next buy date based on frequency
        now = datetime.now()
        if frequency == 'daily':
            next_buy = now + timedelta(days=1)
        elif frequency == 'weekly':
            next_buy = now + timedelta(weeks=1)
        elif frequency == 'monthly':
            next_buy = now + timedelta(days=30)
        else:
            raise ValueError(f"Invalid frequency: {frequency}")
        
        schedule = DCASchedule(
            ticker=ticker,
            amount_per_interval=amount,
            frequency=frequency,
            next_buy_date=next_buy,
            min_confidence=min_confidence,
            max_price=max_price,
            strategy=strategy
        )
        
        self.schedules[ticker] = schedule
        logger.info(f"📅 Added DCA schedule: {ticker} - ${amount} {frequency}")
        
        # Save to database if available
        if self.supabase:
            self._save_schedule_to_db(schedule)
        
        return schedule
    
    def remove_schedule(self, ticker: str) -> bool:
        """Remove a DCA schedule"""
        if ticker in self.schedules:
            del self.schedules[ticker]
            logger.info(f"🗑️ Removed DCA schedule for {ticker}")
            return True
        return False
    
    def pause_schedule(self, ticker: str) -> bool:
        """Pause a DCA schedule"""
        if ticker in self.schedules:
            self.schedules[ticker].active = False
            logger.info(f"⏸️ Paused DCA schedule for {ticker}")
            return True
        return False
    
    def resume_schedule(self, ticker: str) -> bool:
        """Resume a paused DCA schedule"""
        if ticker in self.schedules:
            self.schedules[ticker].active = True
            logger.info(f"▶️ Resumed DCA schedule for {ticker}")
            return True
        return False
    
    def get_due_schedules(self) -> List[DCASchedule]:
        """Get all schedules that are due for execution"""
        now = datetime.now()
        due_schedules = []
        
        for schedule in self.schedules.values():
            if schedule.active and schedule.next_buy_date <= now:
                due_schedules.append(schedule)
        
        return due_schedules
    
    def should_execute_buy(self, ticker: str, current_price: float, 
                          confidence: float, recommendation: str) -> Tuple[bool, str]:
        """
        Determine if a DCA buy should be executed based on AI analysis
        
        Returns:
            (should_buy, reason)
        """
        if ticker not in self.schedules:
            return False, "No schedule found"
        
        schedule = self.schedules[ticker]
        
        # Check if schedule is active
        if not schedule.active:
            return False, "Schedule is paused"
        
        # Check confidence threshold
        if confidence < schedule.min_confidence:
            return False, f"Confidence {confidence:.1f}% below minimum {schedule.min_confidence}%"
        
        # Check max price limit
        if schedule.max_price and current_price > schedule.max_price:
            return False, f"Price ${current_price:.2f} exceeds max ${schedule.max_price:.2f}"
        
        # Check AI recommendation
        if recommendation not in ["BUY", "STRONG_BUY"]:
            return False, f"AI recommends {recommendation}, not BUY"
        
        return True, "All conditions met"
    
    def record_transaction(self, ticker: str, amount: float, price: float, 
                          shares: float, confidence: float, strategy: str,
                          recommendation: str = "BUY") -> DCATransaction:
        """Record a DCA transaction"""
        transaction = DCATransaction(
            ticker=ticker,
            date=datetime.now(),
            amount=amount,
            price=price,
            shares=shares,
            confidence=confidence,
            strategy=strategy,
            ai_recommendation=recommendation
        )
        
        self.transactions.append(transaction)
        
        # Update schedule
        if ticker in self.schedules:
            schedule = self.schedules[ticker]
            schedule.total_invested += amount
            schedule.total_shares += shares
            schedule.last_buy_date = datetime.now()
            
            # Calculate next buy date
            if schedule.frequency == 'daily':
                schedule.next_buy_date = datetime.now() + timedelta(days=1)
            elif schedule.frequency == 'weekly':
                schedule.next_buy_date = datetime.now() + timedelta(weeks=1)
            elif schedule.frequency == 'monthly':
                schedule.next_buy_date = datetime.now() + timedelta(days=30)
        
        logger.info(f"✅ Recorded DCA transaction: {ticker} - {shares:.4f} shares @ ${price:.2f}")
        
        # Save to database if available
        if self.supabase:
            self._save_transaction_to_db(transaction)
        
        return transaction
    
    def get_position_summary(self, ticker: str) -> Optional[Dict]:
        """Get summary of DCA position for a ticker"""
        if ticker not in self.schedules:
            return None
        
        schedule = self.schedules[ticker]
        ticker_transactions = [t for t in self.transactions if t.ticker == ticker]
        
        if not ticker_transactions:
            return {
                'ticker': ticker,
                'total_invested': 0.0,
                'total_shares': 0.0,
                'average_cost': 0.0,
                'transaction_count': 0,
                'last_buy_date': None,
                'next_buy_date': schedule.next_buy_date,
                'frequency': schedule.frequency,
                'active': schedule.active
            }
        
        average_cost = schedule.total_invested / schedule.total_shares if schedule.total_shares > 0 else 0
        
        return {
            'ticker': ticker,
            'total_invested': schedule.total_invested,
            'total_shares': schedule.total_shares,
            'average_cost': average_cost,
            'transaction_count': len(ticker_transactions),
            'last_buy_date': schedule.last_buy_date,
            'next_buy_date': schedule.next_buy_date,
            'frequency': schedule.frequency,
            'amount_per_interval': schedule.amount_per_interval,
            'min_confidence': schedule.min_confidence,
            'strategy': schedule.strategy,
            'active': schedule.active
        }
    
    def get_all_positions(self) -> List[Dict]:
        """Get summary of all DCA positions"""
        positions = []
        for ticker in self.schedules.keys():
            pos = self.get_position_summary(ticker)
            if pos:
                positions.append(pos)
        return positions
    
    def calculate_portfolio_performance(self, current_prices: Dict[str, float]) -> Dict:
        """
        Calculate overall portfolio performance
        
        Args:
            current_prices: Dict of ticker -> current price
        """
        total_invested = 0.0
        current_value = 0.0
        positions = []
        
        for ticker, schedule in self.schedules.items():
            if schedule.total_shares > 0:
                current_price = current_prices.get(ticker, 0)
                position_value = schedule.total_shares * current_price
                position_gain = position_value - schedule.total_invested
                position_gain_pct = (position_gain / schedule.total_invested * 100) if schedule.total_invested > 0 else 0
                
                total_invested += schedule.total_invested
                current_value += position_value
                
                positions.append({
                    'ticker': ticker,
                    'shares': schedule.total_shares,
                    'invested': schedule.total_invested,
                    'current_value': position_value,
                    'gain': position_gain,
                    'gain_pct': position_gain_pct,
                    'average_cost': schedule.total_invested / schedule.total_shares if schedule.total_shares > 0 else 0,
                    'current_price': current_price
                })
        
        total_gain = current_value - total_invested
        total_gain_pct = (total_gain / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'total_invested': total_invested,
            'current_value': current_value,
            'total_gain': total_gain,
            'total_gain_pct': total_gain_pct,
            'positions': positions,
            'position_count': len(positions)
        }
    
    def get_transaction_history(self, ticker: Optional[str] = None, 
                               limit: int = 100) -> List[DCATransaction]:
        """Get transaction history, optionally filtered by ticker"""
        if ticker:
            transactions = [t for t in self.transactions if t.ticker == ticker]
        else:
            transactions = self.transactions
        
        # Sort by date descending
        transactions.sort(key=lambda x: x.date, reverse=True)
        
        return transactions[:limit]
    
    def export_to_csv(self, filepath: str):
        """Export transaction history to CSV"""
        if not self.transactions:
            logger.warning("No transactions to export")
            return
        
        df = pd.DataFrame([t.to_dict() for t in self.transactions])
        df.to_csv(filepath, index=False)
        logger.info(f"📊 Exported {len(self.transactions)} transactions to {filepath}")
    
    def _save_schedule_to_db(self, schedule: DCASchedule):
        """Save schedule to Supabase (if configured)"""
        try:
            if not self.supabase:
                return
            
            data = schedule.to_dict()
            data['user_id'] = 'default'  # Replace with actual user ID if multi-user
            
            # Upsert (insert or update)
            self.supabase.table('dca_schedules').upsert(data).execute()
            logger.debug(f"💾 Saved DCA schedule to database: {schedule.ticker}")
        except Exception as e:
            logger.error(f"Error saving DCA schedule to database: {e}")
    
    def _save_transaction_to_db(self, transaction: DCATransaction):
        """Save transaction to Supabase (if configured)"""
        try:
            if not self.supabase:
                return
            
            data = transaction.to_dict()
            data['user_id'] = 'default'  # Replace with actual user ID if multi-user
            
            self.supabase.table('dca_transactions').insert(data).execute()
            logger.debug(f"💾 Saved DCA transaction to database: {transaction.ticker}")
        except Exception as e:
            logger.error(f"Error saving DCA transaction to database: {e}")
    
    def load_from_db(self):
        """Load schedules and transactions from database"""
        try:
            if not self.supabase:
                logger.warning("No Supabase client - cannot load from database")
                return
            
            # Load schedules
            response = self.supabase.table('dca_schedules').select('*').execute()
            if response.data:
                for row in response.data:
                    schedule = DCASchedule.from_dict(row)
                    self.schedules[schedule.ticker] = schedule
                logger.info(f"📥 Loaded {len(response.data)} DCA schedules from database")
            
            # Load transactions
            response = self.supabase.table('dca_transactions').select('*').execute()
            if response.data:
                self.transactions = [DCATransaction.from_dict(row) for row in response.data]
                logger.info(f"📥 Loaded {len(self.transactions)} DCA transactions from database")
        
        except Exception as e:
            logger.error(f"Error loading from database: {e}")

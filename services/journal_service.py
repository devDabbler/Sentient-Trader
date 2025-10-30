"""Trade journaling and telemetry service"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

from services.agents.messages import JournalEntry, SetupType, TradingMode

logger = logging.getLogger(__name__)


@dataclass
class TradeStats:
    """Aggregate trade statistics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_r_multiple: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hold_time_minutes: float = 0.0
    
    # Per-setup stats
    setup_stats: Dict[str, Dict] = None


class JournalService:
    """
    Persistent trade journal with SQLite backend and CSV export.
    Tracks all trades with settlement dates and running settled cash.
    """
    
    def __init__(self, db_path: str = "data/trade_journal.db"):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                mode TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                shares INTEGER NOT NULL,
                bucket_index INTEGER NOT NULL,
                settlement_date TEXT NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                realized_pnl REAL,
                r_multiple REAL,
                hold_time_minutes INTEGER,
                settled_cash_after REAL,
                exit_reason TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create indices for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entry_time ON trades(entry_time)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_setup_type ON trades(setup_type)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Journal database initialized at {self.db_path}")
    
    def record_trade(self, entry: JournalEntry) -> int:
        """
        Record a trade to the journal.
        Returns the trade ID.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO trades (
                symbol, setup_type, mode, side, entry_time, entry_price,
                stop_price, target_price, shares, bucket_index, settlement_date,
                exit_time, exit_price, realized_pnl, r_multiple, hold_time_minutes,
                settled_cash_after, exit_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.symbol,
            entry.setup_type.value if isinstance(entry.setup_type, SetupType) else entry.setup_type,
            entry.mode.value if isinstance(entry.mode, TradingMode) else entry.mode,
            entry.side,
            entry.entry_time.isoformat(),
            entry.entry_price,
            entry.stop_price,
            entry.target_price,
            entry.shares,
            entry.bucket_index,
            entry.settlement_date.isoformat() if isinstance(entry.settlement_date, (date, datetime)) else entry.settlement_date,
            entry.exit_time.isoformat() if entry.exit_time else None,
            entry.exit_price,
            entry.realized_pnl,
            entry.r_multiple,
            entry.hold_time_minutes,
            entry.settled_cash_after,
            entry.exit_reason,
            datetime.now().isoformat()
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Recorded trade #{trade_id}: {entry.symbol} {entry.side} {entry.shares} @ ${entry.entry_price:.2f}")
        return trade_id
    
    def update_trade_exit(
        self,
        symbol: str,
        entry_time: datetime,
        exit_price: float,
        exit_time: datetime,
        settled_cash_after: float,
        exit_reason: str = "manual"
    ) -> bool:
        """
        Update trade with exit information.
        Returns True if trade was found and updated.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find open trade
        cursor.execute("""
            SELECT id, entry_price, stop_price, target_price, shares, side
            FROM trades
            WHERE symbol = ? AND entry_time = ? AND exit_time IS NULL
        """, (symbol, entry_time.isoformat()))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            logger.warning(f"No open trade found for {symbol} @ {entry_time}")
            return False
        
        trade_id, entry_price, stop_price, target_price, shares, side = row
        
        # Calculate PnL and R multiple
        if side.upper() == "BUY":
            pnl = (exit_price - entry_price) * shares
            risk = entry_price - stop_price
        else:  # SELL/SHORT
            pnl = (entry_price - exit_price) * shares
            risk = stop_price - entry_price
        
        r_multiple = (exit_price - entry_price) / risk if risk > 0 else 0
        hold_time_minutes = int((exit_time - entry_time).total_seconds() / 60)
        
        # Update trade
        cursor.execute("""
            UPDATE trades
            SET exit_time = ?,
                exit_price = ?,
                realized_pnl = ?,
                r_multiple = ?,
                hold_time_minutes = ?,
                settled_cash_after = ?,
                exit_reason = ?
            WHERE id = ?
        """, (
            exit_time.isoformat(),
            exit_price,
            pnl,
            r_multiple,
            hold_time_minutes,
            settled_cash_after,
            exit_reason,
            trade_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated trade #{trade_id}: {symbol} exit @ ${exit_price:.2f}, PnL=${pnl:.2f}, R={r_multiple:.2f}")
        return True
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        setup_type: Optional[SetupType] = None,
        only_closed: bool = False
    ) -> List[JournalEntry]:
        """
        Query trades with filters.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        
        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND entry_time <= ?"
            params.append(end_date.isoformat())
        
        if setup_type:
            query += " AND setup_type = ?"
            params.append(setup_type.value)
        
        if only_closed:
            query += " AND exit_time IS NOT NULL"
        
        query += " ORDER BY entry_time DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to JournalEntry objects
        entries = []
        for row in rows:
            entry = self._row_to_journal_entry(row)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _row_to_journal_entry(self, row: tuple) -> Optional[JournalEntry]:
        """Convert database row to JournalEntry"""
        try:
            return JournalEntry(
                symbol=row[1],
                setup_type=SetupType(row[2]),
                mode=TradingMode(row[3]),
                side=row[4],
                entry_time=datetime.fromisoformat(row[5]),
                entry_price=row[6],
                stop_price=row[7],
                target_price=row[8],
                shares=row[9],
                bucket_index=row[10],
                settlement_date=datetime.fromisoformat(row[11]).date() if row[11] else datetime.now().date(),
                exit_time=datetime.fromisoformat(row[12]) if row[12] else None,
                exit_price=row[13],
                realized_pnl=row[14],
                r_multiple=row[15],
                hold_time_minutes=row[16],
                settled_cash_after=row[17],
                exit_reason=row[18]
            )
        except Exception as e:
            logger.error(f"Error converting row to JournalEntry: {e}")
            return None
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> TradeStats:
        """
        Calculate aggregate statistics for closed trades.
        """
        trades = self.get_trades(
            start_date=start_date,
            end_date=end_date,
            only_closed=True
        )
        
        if not trades:
            return TradeStats(setup_stats={})
        
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.realized_pnl and t.realized_pnl > 0)
        losing_trades = sum(1 for t in trades if t.realized_pnl and t.realized_pnl < 0)
        
        total_pnl = sum(t.realized_pnl for t in trades if t.realized_pnl)
        
        wins = [t.realized_pnl for t in trades if t.realized_pnl and t.realized_pnl > 0]
        losses = [abs(t.realized_pnl) for t in trades if t.realized_pnl and t.realized_pnl < 0]
        
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        r_multiples = [t.r_multiple for t in trades if t.r_multiple]
        avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0
        
        hold_times = [t.hold_time_minutes for t in trades if t.hold_time_minutes]
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
        
        # Per-setup stats
        setup_stats = {}
        for setup in SetupType:
            setup_trades = [t for t in trades if t.setup_type == setup]
            if setup_trades:
                setup_wins = sum(1 for t in setup_trades if t.realized_pnl and t.realized_pnl > 0)
                setup_total = len(setup_trades)
                setup_pnl = sum(t.realized_pnl for t in setup_trades if t.realized_pnl)
                
                setup_stats[setup.value] = {
                    'trades': setup_total,
                    'wins': setup_wins,
                    'win_rate': setup_wins / setup_total if setup_total > 0 else 0,
                    'total_pnl': setup_pnl
                }
        
        return TradeStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_r_multiple=avg_r_multiple,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_hold_time_minutes=avg_hold_time,
            setup_stats=setup_stats
        )
    
    def export_to_csv(self, output_path: str, start_date: Optional[datetime] = None):
        """Export trades to CSV"""
        trades = self.get_trades(start_date=start_date)
        
        if not trades:
            logger.warning("No trades to export")
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'symbol', 'setup_type', 'mode', 'side', 'entry_time', 'entry_price',
                'stop_price', 'target_price', 'shares', 'bucket_index', 'settlement_date',
                'exit_time', 'exit_price', 'realized_pnl', 'r_multiple', 'hold_time_minutes',
                'settled_cash_after', 'exit_reason'
            ])
            writer.writeheader()
            
            for trade in trades:
                row = asdict(trade)
                # Convert enums to strings
                row['setup_type'] = row['setup_type'].value if isinstance(row['setup_type'], SetupType) else row['setup_type']
                row['mode'] = row['mode'].value if isinstance(row['mode'], TradingMode) else row['mode']
                # Convert datetimes to ISO strings
                row['entry_time'] = row['entry_time'].isoformat() if row['entry_time'] else None
                row['exit_time'] = row['exit_time'].isoformat() if row['exit_time'] else None
                row['settlement_date'] = row['settlement_date'].isoformat() if row['settlement_date'] else None
                # Remove event_id
                row.pop('event_id', None)
                writer.writerow(row)
        
        logger.info(f"Exported {len(trades)} trades to {output_path}")
    
    def print_summary(self, days: int = 30):
        """Print trade statistics summary"""
        start_date = datetime.now() - timedelta(days=days)
        stats = self.get_statistics(start_date=start_date)
        
        print(f"\n{'='*60}")
        print(f"TRADE JOURNAL SUMMARY (Last {days} days)")
        print(f"{'='*60}")
        print(f"Total Trades: {stats.total_trades}")
        print(f"Win Rate: {stats.win_rate*100:.1f}% ({stats.winning_trades}W / {stats.losing_trades}L)")
        print(f"Total P&L: ${stats.total_pnl:.2f}")
        print(f"Avg Win: ${stats.avg_win:.2f}")
        print(f"Avg Loss: ${stats.avg_loss:.2f}")
        print(f"Profit Factor: {stats.profit_factor:.2f}")
        print(f"Avg R Multiple: {stats.avg_r_multiple:.2f}R")
        print(f"Avg Hold Time: {stats.avg_hold_time_minutes:.0f} minutes")
        
        if stats.setup_stats:
            print(f"\n{'='*60}")
            print("PER-SETUP PERFORMANCE")
            print(f"{'='*60}")
            for setup_name, setup_stat in stats.setup_stats.items():
                print(f"\n{setup_name}:")
                print(f"  Trades: {setup_stat['trades']}")
                print(f"  Win Rate: {setup_stat['win_rate']*100:.1f}%")
                print(f"  Total P&L: ${setup_stat['total_pnl']:.2f}")
        
        print(f"{'='*60}\n")


# Singleton instance
_journal_service: Optional[JournalService] = None


def get_journal_service() -> JournalService:
    """Get or create singleton journal service"""
    global _journal_service
    if _journal_service is None:
        _journal_service = JournalService()
    return _journal_service


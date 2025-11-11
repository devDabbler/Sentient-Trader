"""
Unified Trade Journal Service
Automatic journaling for all trades across stocks and crypto with AI decision tracking
"""

from loguru import logger
import sqlite3
import csv
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum


class TradeType(Enum):
    """Type of trade"""
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"
    OPTION = "OPTION"


class TradeStatus(Enum):
    """Trade status"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"


@dataclass
class UnifiedTradeEntry:
    """
    Unified trade entry that works for stocks, crypto, and options
    """
    # Core fields
    trade_id: str
    trade_type: str  # STOCK, CRYPTO, OPTION
    symbol: str
    side: str  # BUY, SELL
    entry_time: datetime
    entry_price: float
    quantity: float  # Shares for stocks, volume for crypto
    position_size_usd: float
    
    # Risk management
    stop_loss: float
    take_profit: float
    risk_pct: float
    reward_pct: float
    risk_reward_ratio: float
    
    # Strategy & setup
    strategy: str
    setup_type: Optional[str] = None
    timeframe: Optional[str] = None
    
    # Exit data (filled when closed)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    hold_time_minutes: Optional[int] = None
    r_multiple: Optional[float] = None
    
    # AI tracking
    ai_managed: bool = False
    ai_decisions: List[Dict] = field(default_factory=list)
    ai_adjustments_count: int = 0
    moved_to_breakeven: bool = False
    trailing_stop_activated: bool = False
    partial_exit_taken: bool = False
    
    # Market conditions at entry
    rsi_entry: Optional[float] = None
    macd_entry: Optional[float] = None
    volume_change_entry: Optional[float] = None
    trend_entry: Optional[str] = None
    
    # Market conditions at exit
    rsi_exit: Optional[float] = None
    macd_exit: Optional[float] = None
    volume_change_exit: Optional[float] = None
    trend_exit: Optional[str] = None
    
    # Performance tracking
    max_favorable_pct: float = 0.0
    max_adverse_pct: float = 0.0
    
    # Metadata
    broker: Optional[str] = None  # KRAKEN, TRADIER, IBKR
    order_id: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Status
    status: str = TradeStatus.OPEN.value
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AIDecisionLog:
    """Log entry for AI decision"""
    timestamp: datetime
    action: str  # HOLD, TIGHTEN_STOP, EXTEND_TARGET, TAKE_PARTIAL, CLOSE_NOW
    confidence: float
    reasoning: str
    technical_score: float = 0.0
    trend_score: float = 0.0
    risk_score: float = 0.0
    new_stop: Optional[float] = None
    new_target: Optional[float] = None
    partial_pct: Optional[float] = None


@dataclass
class JournalStats:
    """Aggregate statistics"""
    # Overall stats
    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    # P&L
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Performance metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_r_multiple: float = 0.0
    expectancy: float = 0.0
    
    # Holding time
    avg_hold_time_minutes: float = 0.0
    avg_win_hold_time: float = 0.0
    avg_loss_hold_time: float = 0.0
    
    # AI performance
    ai_managed_trades: int = 0
    ai_win_rate: float = 0.0
    ai_avg_adjustments: float = 0.0
    breakeven_moves: int = 0
    trailing_stops: int = 0
    partial_exits: int = 0
    
    # By type
    stock_trades: int = 0
    crypto_trades: int = 0
    option_trades: int = 0
    
    # By strategy
    strategy_stats: Dict[str, Dict] = field(default_factory=dict)


class UnifiedTradeJournal:
    """
    Unified trade journal for stocks, crypto, and options
    Automatically logs all trades with AI decision tracking
    """
    
    def __init__(self, db_path: str = "data/unified_trade_journal.db"):
        """
        Initialize unified journal
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"Unified Trade Journal initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                trade_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                position_size_usd REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                risk_pct REAL NOT NULL,
                reward_pct REAL NOT NULL,
                risk_reward_ratio REAL NOT NULL,
                strategy TEXT NOT NULL,
                setup_type TEXT,
                timeframe TEXT,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                realized_pnl REAL,
                pnl_pct REAL,
                hold_time_minutes INTEGER,
                r_multiple REAL,
                ai_managed INTEGER DEFAULT 0,
                ai_adjustments_count INTEGER DEFAULT 0,
                moved_to_breakeven INTEGER DEFAULT 0,
                trailing_stop_activated INTEGER DEFAULT 0,
                partial_exit_taken INTEGER DEFAULT 0,
                rsi_entry REAL,
                macd_entry REAL,
                volume_change_entry REAL,
                trend_entry TEXT,
                rsi_exit REAL,
                macd_exit REAL,
                volume_change_exit REAL,
                trend_exit TEXT,
                max_favorable_pct REAL DEFAULT 0.0,
                max_adverse_pct REAL DEFAULT 0.0,
                broker TEXT,
                order_id TEXT,
                notes TEXT,
                tags TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # AI decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT NOT NULL,
                technical_score REAL DEFAULT 0.0,
                trend_score REAL DEFAULT 0.0,
                risk_score REAL DEFAULT 0.0,
                new_stop REAL,
                new_target REAL,
                partial_pct REAL,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)
        
        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_id ON trades(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entry_time ON trades(entry_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_type ON trades(trade_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_trade_id ON ai_decisions(trade_id)")
        
        conn.commit()
        conn.close()
    
    def log_trade_entry(self, entry: UnifiedTradeEntry) -> bool:
        """
        Log trade entry
        
        Args:
            entry: Trade entry object
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, trade_type, symbol, side, entry_time, entry_price,
                    quantity, position_size_usd, stop_loss, take_profit,
                    risk_pct, reward_pct, risk_reward_ratio, strategy, setup_type,
                    timeframe, exit_time, exit_price, exit_reason, realized_pnl,
                    pnl_pct, hold_time_minutes, r_multiple, ai_managed,
                    ai_adjustments_count, moved_to_breakeven, trailing_stop_activated,
                    partial_exit_taken, rsi_entry, macd_entry, volume_change_entry,
                    trend_entry, rsi_exit, macd_exit, volume_change_exit, trend_exit,
                    max_favorable_pct, max_adverse_pct, broker, order_id, notes,
                    tags, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                         ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.trade_id, entry.trade_type, entry.symbol, entry.side,
                entry.entry_time.isoformat(), entry.entry_price, entry.quantity,
                entry.position_size_usd, entry.stop_loss, entry.take_profit,
                entry.risk_pct, entry.reward_pct, entry.risk_reward_ratio,
                entry.strategy, entry.setup_type, entry.timeframe,
                entry.exit_time.isoformat() if entry.exit_time else None,
                entry.exit_price, entry.exit_reason, entry.realized_pnl,
                entry.pnl_pct, entry.hold_time_minutes, entry.r_multiple,
                1 if entry.ai_managed else 0, entry.ai_adjustments_count,
                1 if entry.moved_to_breakeven else 0,
                1 if entry.trailing_stop_activated else 0,
                1 if entry.partial_exit_taken else 0,
                entry.rsi_entry, entry.macd_entry, entry.volume_change_entry,
                entry.trend_entry, entry.rsi_exit, entry.macd_exit,
                entry.volume_change_exit, entry.trend_exit,
                entry.max_favorable_pct, entry.max_adverse_pct,
                entry.broker, entry.order_id, entry.notes,
                json.dumps(entry.tags), entry.status, entry.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📝 Journaled {entry.trade_type} trade: {entry.symbol} {entry.side} @ ${entry.entry_price:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error logging trade entry: {e}", exc_info=True)
            return False
    
    def log_ai_decision(self, trade_id: str, decision: AIDecisionLog) -> bool:
        """
        Log AI decision for a trade
        
        Args:
            trade_id: Trade identifier
            decision: AI decision object
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ai_decisions (
                    trade_id, timestamp, action, confidence, reasoning,
                    technical_score, trend_score, risk_score, new_stop,
                    new_target, partial_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, decision.timestamp.isoformat(), decision.action,
                decision.confidence, decision.reasoning, decision.technical_score,
                decision.trend_score, decision.risk_score, decision.new_stop,
                decision.new_target, decision.partial_pct
            ))
            
            # Update trade AI adjustments count
            cursor.execute("""
                UPDATE trades
                SET ai_adjustments_count = ai_adjustments_count + 1
                WHERE trade_id = ?
            """, (trade_id,))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"🤖 Logged AI decision for {trade_id}: {decision.action} (confidence: {decision.confidence:.0f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error logging AI decision: {e}", exc_info=True)
            return False
    
    def update_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        market_conditions: Optional[Dict] = None
    ) -> bool:
        """
        Update trade with exit information
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_time: Exit time
            exit_reason: Reason for exit
            market_conditions: Optional market conditions at exit
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get trade entry data
            cursor.execute("""
                SELECT entry_time, entry_price, stop_loss, quantity, side, position_size_usd
                FROM trades
                WHERE trade_id = ?
            """, (trade_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"Trade {trade_id} not found")
                conn.close()
                return False
            
            entry_time_str, entry_price, stop_loss, quantity, side, position_size_usd = row
            entry_time = datetime.fromisoformat(entry_time_str)
            
            # Calculate P&L
            if side.upper() == "BUY":
                pnl = (exit_price - entry_price) * quantity
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                risk = entry_price - stop_loss
            else:  # SELL
                pnl = (entry_price - exit_price) * quantity
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                risk = stop_loss - entry_price
            
            # Calculate R multiple
            if risk > 0:
                price_change = exit_price - entry_price if side.upper() == "BUY" else entry_price - exit_price
                r_multiple = price_change / risk
            else:
                r_multiple = 0
            
            # Calculate hold time
            hold_time_minutes = int((exit_time - entry_time).total_seconds() / 60)
            
            # Update trade
            update_sql = """
                UPDATE trades
                SET exit_time = ?,
                    exit_price = ?,
                    exit_reason = ?,
                    realized_pnl = ?,
                    pnl_pct = ?,
                    hold_time_minutes = ?,
                    r_multiple = ?,
                    status = ?
            """
            params = [
                exit_time.isoformat(), exit_price, exit_reason,
                pnl, pnl_pct, hold_time_minutes, r_multiple,
                TradeStatus.CLOSED.value
            ]
            
            # Add market conditions if provided
            if market_conditions:
                update_sql += """,
                    rsi_exit = ?,
                    macd_exit = ?,
                    volume_change_exit = ?,
                    trend_exit = ?
                """
                params.extend([
                    market_conditions.get('rsi'),
                    market_conditions.get('macd'),
                    market_conditions.get('volume_change'),
                    market_conditions.get('trend')
                ])
            
            update_sql += " WHERE trade_id = ?"
            params.append(trade_id)
            
            cursor.execute(update_sql, params)
            conn.commit()
            conn.close()
            
            logger.info(f"📝 Updated trade exit: {trade_id} @ ${exit_price:.4f}, P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%), R: {r_multiple:.2f}R")
            return True
            
        except Exception as e:
            logger.error(f"Error updating trade exit: {e}", exc_info=True)
            return False
    
    def get_trades(
        self,
        trade_type: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strategy: Optional[str] = None,
        ai_managed_only: bool = False,
        limit: int = 100
    ) -> List[UnifiedTradeEntry]:
        """
        Query trades with filters
        
        Args:
            trade_type: Filter by STOCK, CRYPTO, OPTION
            symbol: Filter by symbol
            status: Filter by status (OPEN, CLOSED, PARTIAL)
            start_date: Start date
            end_date: End date
            strategy: Filter by strategy
            ai_managed_only: Only AI-managed trades
            limit: Max results
        
        Returns:
            List of trade entries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if trade_type:
                query += " AND trade_type = ?"
                params.append(trade_type)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND entry_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND entry_time <= ?"
                params.append(end_date.isoformat())
            
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            if ai_managed_only:
                query += " AND ai_managed = 1"
            
            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            trades = []
            for row in rows:
                trade = self._row_to_trade_entry(row)
                if trade:
                    trades.append(trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error querying trades: {e}", exc_info=True)
            return []
    
    def _row_to_trade_entry(self, row: tuple) -> Optional[UnifiedTradeEntry]:
        """Convert database row to trade entry"""
        try:
            # Parse tags from JSON
            tags = json.loads(row[42]) if row[42] else []
            
            return UnifiedTradeEntry(
                trade_id=row[1],
                trade_type=row[2],
                symbol=row[3],
                side=row[4],
                entry_time=datetime.fromisoformat(row[5]),
                entry_price=row[6],
                quantity=row[7],
                position_size_usd=row[8],
                stop_loss=row[9],
                take_profit=row[10],
                risk_pct=row[11],
                reward_pct=row[12],
                risk_reward_ratio=row[13],
                strategy=row[14],
                setup_type=row[15],
                timeframe=row[16],
                exit_time=datetime.fromisoformat(row[17]) if row[17] else None,
                exit_price=row[18],
                exit_reason=row[19],
                realized_pnl=row[20],
                pnl_pct=row[21],
                hold_time_minutes=row[22],
                r_multiple=row[23],
                ai_managed=bool(row[24]),
                ai_adjustments_count=row[25],
                moved_to_breakeven=bool(row[26]),
                trailing_stop_activated=bool(row[27]),
                partial_exit_taken=bool(row[28]),
                rsi_entry=row[29],
                macd_entry=row[30],
                volume_change_entry=row[31],
                trend_entry=row[32],
                rsi_exit=row[33],
                macd_exit=row[34],
                volume_change_exit=row[35],
                trend_exit=row[36],
                max_favorable_pct=row[37],
                max_adverse_pct=row[38],
                broker=row[39],
                order_id=row[40],
                notes=row[41],
                tags=tags,
                status=row[43],
                created_at=datetime.fromisoformat(row[44])
            )
        except Exception as e:
            logger.error(f"Error converting row to trade entry: {e}")
            return None
    
    def get_ai_decisions(self, trade_id: str) -> List[AIDecisionLog]:
        """Get all AI decisions for a trade"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, action, confidence, reasoning,
                       technical_score, trend_score, risk_score,
                       new_stop, new_target, partial_pct
                FROM ai_decisions
                WHERE trade_id = ?
                ORDER BY timestamp ASC
            """, (trade_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            decisions = []
            for row in rows:
                decision = AIDecisionLog(
                    timestamp=datetime.fromisoformat(row[0]),
                    action=row[1],
                    confidence=row[2],
                    reasoning=row[3],
                    technical_score=row[4],
                    trend_score=row[5],
                    risk_score=row[6],
                    new_stop=row[7],
                    new_target=row[8],
                    partial_pct=row[9]
                )
                decisions.append(decision)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error getting AI decisions: {e}", exc_info=True)
            return []
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        trade_type: Optional[str] = None
    ) -> JournalStats:
        """Calculate aggregate statistics"""
        trades = self.get_trades(
            trade_type=trade_type,
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        if not trades:
            return JournalStats()
        
        # Basic counts
        total_trades = len(trades)
        open_trades = sum(1 for t in trades if t.status == TradeStatus.OPEN.value)
        closed_trades = sum(1 for t in trades if t.status == TradeStatus.CLOSED.value)
        
        # Only analyze closed trades for P&L
        closed = [t for t in trades if t.status == TradeStatus.CLOSED.value]
        
        if not closed:
            return JournalStats(
                total_trades=total_trades,
                open_trades=open_trades,
                closed_trades=closed_trades
            )
        
        # Win/Loss counts
        winning_trades = sum(1 for t in closed if t.realized_pnl and t.realized_pnl > 0)
        losing_trades = sum(1 for t in closed if t.realized_pnl and t.realized_pnl < 0)
        breakeven_trades = sum(1 for t in closed if t.realized_pnl and abs(t.realized_pnl) < 1)
        
        # P&L calculations
        total_pnl = sum(t.realized_pnl for t in closed if t.realized_pnl)
        
        wins = [t for t in closed if t.realized_pnl and t.realized_pnl > 0]
        losses = [t for t in closed if t.realized_pnl and t.realized_pnl < 0]
        
        avg_win = sum(t.realized_pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.realized_pnl for t in losses) / len(losses)) if losses else 0
        
        avg_win_pct = sum(t.pnl_pct for t in wins if t.pnl_pct) / len(wins) if wins else 0
        avg_loss_pct = abs(sum(t.pnl_pct for t in losses if t.pnl_pct) / len(losses)) if losses else 0
        
        largest_win = max((t.realized_pnl for t in wins), default=0)
        largest_loss = min((t.realized_pnl for t in losses), default=0)
        
        # Performance metrics
        win_rate = winning_trades / closed_trades if closed_trades > 0 else 0
        
        total_wins = sum(t.realized_pnl for t in wins)
        total_losses = abs(sum(t.realized_pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        r_multiples = [t.r_multiple for t in closed if t.r_multiple]
        avg_r_multiple = sum(r_multiples) / len(r_multiples) if r_multiples else 0
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Hold times
        hold_times = [t.hold_time_minutes for t in closed if t.hold_time_minutes]
        avg_hold_time = sum(hold_times) / len(hold_times) if hold_times else 0
        
        win_hold_times = [t.hold_time_minutes for t in wins if t.hold_time_minutes]
        avg_win_hold_time = sum(win_hold_times) / len(win_hold_times) if win_hold_times else 0
        
        loss_hold_times = [t.hold_time_minutes for t in losses if t.hold_time_minutes]
        avg_loss_hold_time = sum(loss_hold_times) / len(loss_hold_times) if loss_hold_times else 0
        
        # AI stats
        ai_trades = [t for t in closed if t.ai_managed]
        ai_managed_trades = len(ai_trades)
        ai_wins = sum(1 for t in ai_trades if t.realized_pnl and t.realized_pnl > 0)
        ai_win_rate = ai_wins / ai_managed_trades if ai_managed_trades > 0 else 0
        ai_avg_adjustments = sum(t.ai_adjustments_count for t in ai_trades) / ai_managed_trades if ai_managed_trades > 0 else 0
        
        breakeven_moves = sum(1 for t in trades if t.moved_to_breakeven)
        trailing_stops = sum(1 for t in trades if t.trailing_stop_activated)
        partial_exits = sum(1 for t in trades if t.partial_exit_taken)
        
        # By type
        stock_trades = sum(1 for t in trades if t.trade_type == TradeType.STOCK.value)
        crypto_trades = sum(1 for t in trades if t.trade_type == TradeType.CRYPTO.value)
        option_trades = sum(1 for t in trades if t.trade_type == TradeType.OPTION.value)
        
        # By strategy
        strategy_stats = {}
        strategies = set(t.strategy for t in closed)
        for strategy in strategies:
            strat_trades = [t for t in closed if t.strategy == strategy]
            strat_wins = sum(1 for t in strat_trades if t.realized_pnl and t.realized_pnl > 0)
            strat_pnl = sum(t.realized_pnl for t in strat_trades if t.realized_pnl)
            
            strategy_stats[strategy] = {
                'trades': len(strat_trades),
                'wins': strat_wins,
                'win_rate': strat_wins / len(strat_trades) if strat_trades else 0,
                'total_pnl': strat_pnl,
                'avg_pnl': strat_pnl / len(strat_trades) if strat_trades else 0
            }
        
        return JournalStats(
            total_trades=total_trades,
            open_trades=open_trades,
            closed_trades=closed_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            breakeven_trades=breakeven_trades,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win=largest_win,
            largest_loss=largest_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_r_multiple=avg_r_multiple,
            expectancy=expectancy,
            avg_hold_time_minutes=avg_hold_time,
            avg_win_hold_time=avg_win_hold_time,
            avg_loss_hold_time=avg_loss_hold_time,
            ai_managed_trades=ai_managed_trades,
            ai_win_rate=ai_win_rate,
            ai_avg_adjustments=ai_avg_adjustments,
            breakeven_moves=breakeven_moves,
            trailing_stops=trailing_stops,
            partial_exits=partial_exits,
            stock_trades=stock_trades,
            crypto_trades=crypto_trades,
            option_trades=option_trades,
            strategy_stats=strategy_stats
        )
    
    def export_to_csv(self, output_path: str, start_date: Optional[datetime] = None):
        """Export trades to CSV"""
        trades = self.get_trades(start_date=start_date, limit=10000)
        
        if not trades:
            logger.warning("No trades to export")
            return
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'trade_id', 'trade_type', 'symbol', 'side', 'entry_time', 'entry_price',
                'quantity', 'position_size_usd', 'stop_loss', 'take_profit', 'risk_pct',
                'reward_pct', 'risk_reward_ratio', 'strategy', 'setup_type', 'timeframe',
                'exit_time', 'exit_price', 'exit_reason', 'realized_pnl', 'pnl_pct',
                'hold_time_minutes', 'r_multiple', 'ai_managed', 'ai_adjustments_count',
                'moved_to_breakeven', 'trailing_stop_activated', 'partial_exit_taken',
                'max_favorable_pct', 'max_adverse_pct', 'broker', 'order_id', 'notes',
                'tags', 'status'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for trade in trades:
                row = asdict(trade)
                # Convert datetime objects
                row['entry_time'] = row['entry_time'].isoformat() if row['entry_time'] else None
                row['exit_time'] = row['exit_time'].isoformat() if row['exit_time'] else None
                row['created_at'] = row['created_at'].isoformat() if row['created_at'] else None
                # Convert lists
                row['tags'] = ','.join(row['tags']) if row['tags'] else ''
                row['ai_decisions'] = ''  # Skip for CSV
                writer.writerow(row)
        
        logger.info(f"📊 Exported {len(trades)} trades to {output_path}")


# Singleton instance
_unified_journal: Optional[UnifiedTradeJournal] = None


def get_unified_journal() -> UnifiedTradeJournal:
    """Get or create singleton journal"""
    global _unified_journal
    if _unified_journal is None:
        _unified_journal = UnifiedTradeJournal()
    return _unified_journal


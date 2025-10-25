"""Backtesting framework for EMA Reclaim + Fibonacci extension strategy."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import yfinance as yf
from analyzers.technical import TechnicalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single backtest trade"""
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    stop_loss: float = 0.0
    target_t1: Optional[float] = None
    target_t2: Optional[float] = None
    target_t3: Optional[float] = None
    position_size: float = 1.0
    
    # Trade outcome
    exit_reason: str = ""  # "T1", "T2", "T3", "STOP", "TIME"
    pnl: float = 0.0
    pnl_pct: float = 0.0
    held_days: int = 0
    
    # Setup characteristics
    had_reclaim: bool = False
    had_alignment: bool = False
    had_strong_rs: bool = False
    demarker_value: Optional[float] = None
    confidence_score: float = 0.0
    
    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.exit_date is None
    
    def close(self, exit_date: datetime, exit_price: float, reason: str):
        """Close the trade"""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.position_size
        self.pnl_pct = ((exit_price / self.entry_price) - 1) * 100
        self.held_days = (exit_date - self.entry_date).days


@dataclass
class BacktestResults:
    """Results from a backtest run"""
    ticker: str
    start_date: datetime
    end_date: datetime
    trades: List[Trade] = field(default_factory=list)
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    
    avg_hold_days: float = 0.0
    
    # Setup-specific stats
    reclaim_trades: int = 0
    reclaim_win_rate: float = 0.0
    aligned_trades: int = 0
    aligned_win_rate: float = 0.0
    triple_threat_trades: int = 0
    triple_threat_win_rate: float = 0.0
    
    def calculate_metrics(self):
        """Calculate performance metrics from trades"""
        if not self.trades:
            return
        
        self.total_trades = len(self.trades)
        
        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]
        
        self.winning_trades = len(winning)
        self.losing_trades = len(losing)
        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        self.total_pnl = sum(t.pnl for t in self.trades)
        self.total_pnl_pct = sum(t.pnl_pct for t in self.trades)
        
        self.avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        self.avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        self.avg_win_pct = np.mean([t.pnl_pct for t in winning]) if winning else 0
        self.avg_loss_pct = np.mean([t.pnl_pct for t in losing]) if losing else 0
        
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        self.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        self.avg_hold_days = np.mean([t.held_days for t in self.trades]) if self.trades else 0
        
        # Calculate drawdown
        equity_curve = np.cumsum([t.pnl for t in self.trades])
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = running_max - equity_curve
        self.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        self.max_drawdown_pct = (self.max_drawdown / (running_max[np.argmax(drawdown)] + 1)) * 100 if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (simplified, assuming daily returns)
        returns = [t.pnl_pct for t in self.trades]
        if returns and np.std(returns) > 0:
            self.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
        
        # Setup-specific stats
        reclaim_trades = [t for t in self.trades if t.had_reclaim]
        self.reclaim_trades = len(reclaim_trades)
        self.reclaim_win_rate = (len([t for t in reclaim_trades if t.pnl > 0]) / len(reclaim_trades) * 100) if reclaim_trades else 0
        
        aligned_trades = [t for t in self.trades if t.had_alignment]
        self.aligned_trades = len(aligned_trades)
        self.aligned_win_rate = (len([t for t in aligned_trades if t.pnl > 0]) / len(aligned_trades) * 100) if aligned_trades else 0
        
        triple_trades = [t for t in self.trades if t.had_reclaim and t.had_alignment and t.had_strong_rs]
        self.triple_threat_trades = len(triple_trades)
        self.triple_threat_win_rate = (len([t for t in triple_trades if t.pnl > 0]) / len(triple_trades) * 100) if triple_trades else 0
    
    def print_summary(self):
        """Print backtest results summary"""
        print("\n" + "="*80)
        print(f"BACKTEST RESULTS: {self.ticker}")
        print(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print("="*80)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Trades: {self.total_trades}")
        print(f"  Win Rate: {self.win_rate:.1f}% ({self.winning_trades}W / {self.losing_trades}L)")
        print(f"  Total P&L: ${self.total_pnl:.2f} ({self.total_pnl_pct:.2f}%)")
        print(f"  Avg Win: ${self.avg_win:.2f} ({self.avg_win_pct:.2f}%)")
        print(f"  Avg Loss: ${self.avg_loss:.2f} ({self.avg_loss_pct:.2f}%)")
        print(f"  Profit Factor: {self.profit_factor:.2f}")
        print(f"  Max Drawdown: ${self.max_drawdown:.2f} ({self.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"  Avg Hold: {self.avg_hold_days:.1f} days")
        
        print(f"\n🎯 SETUP-SPECIFIC STATS:")
        print(f"  EMA Reclaim Trades: {self.reclaim_trades} (Win Rate: {self.reclaim_win_rate:.1f}%)")
        print(f"  Timeframe Aligned: {self.aligned_trades} (Win Rate: {self.aligned_win_rate:.1f}%)")
        print(f"  Triple Threat: {self.triple_threat_trades} (Win Rate: {self.triple_threat_win_rate:.1f}%)")
        
        print("="*80 + "\n")


class EMAFibonacciBacktester:
    """Backtester for EMA Reclaim + Fibonacci strategy"""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 position_size_pct: float = 0.10,  # 10% of capital per trade
                 max_hold_days: int = 15,  # Max holding period
                 use_fibonacci_targets: bool = True,
                 require_reclaim: bool = False,
                 require_alignment: bool = False,
                 require_strong_rs: bool = False,
                 min_confidence: float = 60.0):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            max_hold_days: Maximum days to hold a trade
            use_fibonacci_targets: Use Fibonacci T1/T2/T3 targets
            require_reclaim: Only trade EMA reclaim setups
            require_alignment: Only trade when timeframes aligned
            require_strong_rs: Only trade stocks with RS > 60
            min_confidence: Minimum confidence score to enter
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_hold_days = max_hold_days
        self.use_fibonacci_targets = use_fibonacci_targets
        self.require_reclaim = require_reclaim
        self.require_alignment = require_alignment
        self.require_strong_rs = require_strong_rs
        self.min_confidence = min_confidence
    
    def backtest(self, ticker: str, start_date: str, end_date: str) -> BacktestResults:
        """
        Run backtest on a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            BacktestResults object
        """
        logger.info(f"Backtesting {ticker} from {start_date} to {end_date}")
        
        try:
            # Fetch historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.error(f"No data for {ticker}")
                return BacktestResults(ticker, datetime.now(), datetime.now())
            
            results = BacktestResults(
                ticker=ticker,
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date)
            )
            
            current_capital = self.initial_capital
            open_trade: Optional[Trade] = None
            
            # Iterate through each day
            for i in range(50, len(hist)):  # Start after enough data for indicators
                current_date = hist.index[i]
                df_slice = hist.iloc[:i+1]
                
                # Check if we need to close an existing trade
                if open_trade:
                    current_price = hist['Close'].iloc[i]
                    days_held = (current_date - open_trade.entry_date).days
                    
                    # Check exit conditions
                    exit_reason = None
                    exit_price = current_price
                    
                    # 1. Stop loss hit
                    if current_price <= open_trade.stop_loss:
                        exit_reason = "STOP"
                        exit_price = open_trade.stop_loss
                    
                    # 2. Fibonacci targets (if using)
                    elif self.use_fibonacci_targets and open_trade.target_t3:
                        if current_price >= open_trade.target_t3:
                            exit_reason = "T3"
                        elif current_price >= open_trade.target_t2:
                            exit_reason = "T2"
                        elif current_price >= open_trade.target_t1:
                            exit_reason = "T1"
                    
                    # 3. Max hold time
                    elif days_held >= self.max_hold_days:
                        exit_reason = "TIME"
                    
                    # Close trade if exit condition met
                    if exit_reason:
                        open_trade.close(current_date, exit_price, exit_reason)
                        results.trades.append(open_trade)
                        current_capital += open_trade.pnl
                        open_trade = None
                        continue
                
                # Check for entry signals (only if no open trade)
                if not open_trade:
                    # Calculate indicators
                    ema8 = TechnicalAnalyzer.ema(df_slice['Close'], 8)
                    ema21 = TechnicalAnalyzer.ema(df_slice['Close'], 21)
                    dem = TechnicalAnalyzer.demarker(df_slice, period=14)
                    ema_ctx = TechnicalAnalyzer.detect_ema_power_zone_and_reclaim(df_slice, ema8, ema21)
                    fib_targets = TechnicalAnalyzer.compute_fib_extensions_from_swing(df_slice)
                    
                    # Simulate timeframe alignment and RS (simplified for backtest)
                    # In real backtest, you'd fetch this data properly
                    simulated_aligned = ema_ctx.get('power_zone')  # Simplified
                    simulated_rs = 65 if ema_ctx.get('power_zone') else 50  # Simplified
                    
                    # Check entry conditions
                    has_reclaim = ema_ctx.get('is_reclaim', False)
                    has_alignment = simulated_aligned
                    has_strong_rs = simulated_rs > 60
                    demarker_val = dem.iloc[-1] if not dem.empty else 0.5
                    
                    # Simple confidence score
                    confidence = 50
                    if has_reclaim:
                        confidence += 20
                    if has_alignment:
                        confidence += 15
                    if has_strong_rs:
                        confidence += 15
                    
                    # Apply filters
                    should_enter = True
                    if self.require_reclaim and not has_reclaim:
                        should_enter = False
                    if self.require_alignment and not has_alignment:
                        should_enter = False
                    if self.require_strong_rs and not has_strong_rs:
                        should_enter = False
                    if confidence < self.min_confidence:
                        should_enter = False
                    
                    # Additional entry filter: DeMarker pullback or reclaim
                    if should_enter and not (has_reclaim or demarker_val <= 0.35):
                        should_enter = False
                    
                    # Enter trade
                    if should_enter:
                        entry_price = hist['Close'].iloc[i]
                        ema21_val = ema21.iloc[-1] if not ema21.empty else entry_price * 0.95
                        
                        # Set stop just below 21 EMA
                        stop_loss = ema21_val * 0.98
                        
                        # Position sizing
                        position_value = current_capital * self.position_size_pct
                        position_size = position_value / entry_price
                        
                        # Set targets
                        t1 = t2 = t3 = None
                        if self.use_fibonacci_targets and fib_targets:
                            t1 = fib_targets.get('T1_1272')
                            t2 = fib_targets.get('T2_1618')
                            t3 = fib_targets.get('T3_2618', fib_targets.get('T3_200'))
                        
                        open_trade = Trade(
                            entry_date=current_date,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            target_t1=t1,
                            target_t2=t2,
                            target_t3=t3,
                            position_size=position_size,
                            had_reclaim=has_reclaim,
                            had_alignment=has_alignment,
                            had_strong_rs=has_strong_rs,
                            demarker_value=demarker_val,
                            confidence_score=confidence
                        )
            
            # Close any remaining open trade
            if open_trade:
                open_trade.close(hist.index[-1], hist['Close'].iloc[-1], "END")
                results.trades.append(open_trade)
            
            # Calculate metrics
            results.calculate_metrics()
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest error for {ticker}: {e}")
            return BacktestResults(ticker, datetime.now(), datetime.now())
    
    def backtest_multiple(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, BacktestResults]:
        """Backtest multiple tickers"""
        results = {}
        for ticker in tickers:
            results[ticker] = self.backtest(ticker, start_date, end_date)
        return results
    
    def optimize_parameters(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """
        Simple parameter optimization using grid search.
        Tests different combinations of filters and returns best.
        """
        logger.info(f"Optimizing parameters for {ticker}")
        
        best_sharpe = -999
        best_params = {}
        best_results = None
        
        # Test different filter combinations
        for req_reclaim in [False, True]:
            for req_align in [False, True]:
                for req_rs in [False, True]:
                    for min_conf in [50, 60, 70, 80]:
                        bt = EMAFibonacciBacktester(
                            require_reclaim=req_reclaim,
                            require_alignment=req_align,
                            require_strong_rs=req_rs,
                            min_confidence=min_conf
                        )
                        
                        results = bt.backtest(ticker, start_date, end_date)
                        
                        if results.sharpe_ratio > best_sharpe and results.total_trades >= 5:
                            best_sharpe = results.sharpe_ratio
                            best_params = {
                                'require_reclaim': req_reclaim,
                                'require_alignment': req_align,
                                'require_strong_rs': req_rs,
                                'min_confidence': min_conf
                            }
                            best_results = results
        
        return {
            'best_params': best_params,
            'best_results': best_results
        }

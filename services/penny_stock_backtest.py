"""
Penny Stock Backtesting Module

Backtest penny stock trading rules to validate stop/target logic
and measure historical expectancy.
"""

from loguru import logger
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf



@dataclass
class BacktestTrade:
    """Individual backtest trade result"""
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    stop_loss: float
    target: float
    exit_reason: str  # 'STOP_HIT', 'TARGET_HIT', 'TIME_LIMIT'
    pnl: float
    pnl_pct: float
    days_held: int
    atr_at_entry: float


@dataclass
class BacktestResults:
    """Complete backtest results"""
    ticker: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy: float
    expectancy_pct: float
    profit_factor: float
    max_drawdown: float
    avg_holding_period: float
    trades: List[BacktestTrade]
    equity_curve: List[float]


class PennyStockBacktester:
    """Backtest penny stock trading strategies"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
    
    def backtest_atr_strategy(self, ticker: str, start_date: str, end_date: str,
                              stop_multiplier: float = 1.5, rr_ratio: float = 2.0,
                              atr_period: int = 14, max_holding_days: int = 30) -> Optional[BacktestResults]:
        """
        Backtest ATR-based stop/target strategy.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            stop_multiplier: ATR multiplier for stop (default 1.5)
            rr_ratio: Risk/reward ratio (default 2.0)
            atr_period: ATR period (default 14)
            max_holding_days: Max days to hold position (default 30)
            
        Returns:
            BacktestResults or None if insufficient data
        """
        try:
            # Fetch historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty or len(hist) < atr_period * 2:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            # Calculate ATR
            hist['ATR'] = self._calculate_atr_series(hist, atr_period)
            
            # Simulate trades
            trades = []
            in_position = False
            entry_date = None
            entry_price = None
            stop_loss = None
            target = None
            entry_atr = None
            
            equity = self.initial_capital
            equity_curve = [equity]
            
            for i in range(atr_period, len(hist)):
                date = hist.index[i]
                row = hist.iloc[i]
                price = row['Close']
                atr = row['ATR']
                
                if not in_position:
                    # Entry logic: Simple - enter when ATR is calculated
                    # In production, add entry conditions (momentum, volume, etc.)
                    entry_date = date
                    entry_price = price
                    entry_atr = atr
                    
                    # Calculate stop and target
                    stop_distance = atr * stop_multiplier
                    stop_loss = entry_price - stop_distance
                    target = entry_price + (stop_distance * rr_ratio)
                    
                    in_position = True
                    
                else:
                    # Check exit conditions
                    days_held = (date - entry_date).days
                    
                    # Stop loss hit
                    if row['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'STOP_HIT'
                        pnl = (exit_price - entry_price) * 100  # Assume 100 shares
                        
                        trade = BacktestTrade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            target=target,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_pct=(exit_price / entry_price - 1) * 100,
                            days_held=days_held,
                            atr_at_entry=entry_atr
                        )
                        trades.append(trade)
                        equity += pnl
                        in_position = False
                    
                    # Target hit
                    elif row['High'] >= target:
                        exit_price = target
                        exit_reason = 'TARGET_HIT'
                        pnl = (exit_price - entry_price) * 100
                        
                        trade = BacktestTrade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            target=target,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_pct=(exit_price / entry_price - 1) * 100,
                            days_held=days_held,
                            atr_at_entry=entry_atr
                        )
                        trades.append(trade)
                        equity += pnl
                        in_position = False
                    
                    # Max holding period exceeded
                    elif days_held >= max_holding_days:
                        exit_price = price
                        exit_reason = 'TIME_LIMIT'
                        pnl = (exit_price - entry_price) * 100
                        
                        trade = BacktestTrade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            target=target,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_pct=(exit_price / entry_price - 1) * 100,
                            days_held=days_held,
                            atr_at_entry=entry_atr
                        )
                        trades.append(trade)
                        equity += pnl
                        in_position = False
                
                equity_curve.append(equity)
            
            # Calculate statistics
            if not trades:
                return None
            
            return self._calculate_statistics(ticker, trades, equity_curve, 
                                             datetime.strptime(start_date, '%Y-%m-%d'),
                                             datetime.strptime(end_date, '%Y-%m-%d'))
            
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            return None
    
    def backtest_percentage_strategy(self, ticker: str, start_date: str, end_date: str,
                                    stop_pct: float = 8.0, target_pct: float = 16.0,
                                    max_holding_days: int = 30) -> Optional[BacktestResults]:
        """
        Backtest fixed percentage stop/target strategy.
        
        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            stop_pct: Stop loss percentage (default 8%)
            target_pct: Target percentage (default 16%)
            max_holding_days: Max days to hold position (default 30)
            
        Returns:
            BacktestResults or None if insufficient data
        """
        try:
            # Fetch historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty or len(hist) < 20:
                logger.warning(f"Insufficient data for {ticker}")
                return None
            
            # Simulate trades
            trades = []
            in_position = False
            
            equity = self.initial_capital
            equity_curve = [equity]
            
            for i in range(20, len(hist)):
                date = hist.index[i]
                row = hist.iloc[i]
                price = row['Close']
                
                if not in_position:
                    entry_date = date
                    entry_price = price
                    
                    stop_loss = entry_price * (1 - stop_pct / 100)
                    target = entry_price * (1 + target_pct / 100)
                    
                    in_position = True
                    
                else:
                    days_held = (date - entry_date).days
                    
                    # Stop loss hit
                    if row['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'STOP_HIT'
                        pnl = (exit_price - entry_price) * 100
                        
                        trade = BacktestTrade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            target=target,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_pct=(exit_price / entry_price - 1) * 100,
                            days_held=days_held,
                            atr_at_entry=0.0
                        )
                        trades.append(trade)
                        equity += pnl
                        in_position = False
                    
                    # Target hit
                    elif row['High'] >= target:
                        exit_price = target
                        exit_reason = 'TARGET_HIT'
                        pnl = (exit_price - entry_price) * 100
                        
                        trade = BacktestTrade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            target=target,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_pct=(exit_price / entry_price - 1) * 100,
                            days_held=days_held,
                            atr_at_entry=0.0
                        )
                        trades.append(trade)
                        equity += pnl
                        in_position = False
                    
                    # Max holding period
                    elif days_held >= max_holding_days:
                        exit_price = price
                        exit_reason = 'TIME_LIMIT'
                        pnl = (exit_price - entry_price) * 100
                        
                        trade = BacktestTrade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date,
                            exit_price=exit_price,
                            stop_loss=stop_loss,
                            target=target,
                            exit_reason=exit_reason,
                            pnl=pnl,
                            pnl_pct=(exit_price / entry_price - 1) * 100,
                            days_held=days_held,
                            atr_at_entry=0.0
                        )
                        trades.append(trade)
                        equity += pnl
                        in_position = False
                
                equity_curve.append(equity)
            
            if not trades:
                return None
            
            return self._calculate_statistics(ticker, trades, equity_curve,
                                             datetime.strptime(start_date, '%Y-%m-%d'),
                                             datetime.strptime(end_date, '%Y-%m-%d'))
            
        except Exception as e:
            logger.error(f"Error backtesting {ticker}: {e}")
            return None
    
    @staticmethod
    def _calculate_atr_series(hist: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR series"""
        high = hist['High']
        low = hist['Low']
        close = hist['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def _calculate_statistics(self, ticker: str, trades: List[BacktestTrade],
                             equity_curve: List[float], start_date: datetime,
                             end_date: datetime) -> BacktestResults:
        """Calculate backtest statistics"""
        
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        total_trades = len(trades)
        
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        
        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss)
        expectancy = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
        expectancy_pct = (win_rate / 100 * avg_win_pct) + ((100 - win_rate) / 100 * avg_loss_pct)
        
        # Profit factor = Gross Profit / Gross Loss
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        # Average holding period
        avg_holding = np.mean([t.days_held for t in trades])
        
        return BacktestResults(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=round(win_rate, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            avg_win_pct=round(avg_win_pct, 2),
            avg_loss_pct=round(avg_loss_pct, 2),
            expectancy=round(expectancy, 2),
            expectancy_pct=round(expectancy_pct, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown=round(max_dd, 2),
            avg_holding_period=round(avg_holding, 1),
            trades=trades,
            equity_curve=equity_curve
        )
    
    @staticmethod
    def format_backtest_results(results: BacktestResults) -> str:
        """Format backtest results for display"""
        
        output = f"""
=== BACKTEST RESULTS: {results.ticker} ===
Period: {results.start_date.date()} to {results.end_date.date()}

TRADE STATISTICS:
  Total Trades: {results.total_trades}
  Winning Trades: {results.winning_trades}
  Losing Trades: {results.losing_trades}
  Win Rate: {results.win_rate}%

PERFORMANCE:
  Average Win: ${results.avg_win:.2f} ({results.avg_win_pct:.2f}%)
  Average Loss: ${results.avg_loss:.2f} ({results.avg_loss_pct:.2f}%)
  Expectancy: ${results.expectancy:.2f} ({results.expectancy_pct:.2f}%)
  Profit Factor: {results.profit_factor:.2f}
  Max Drawdown: {results.max_drawdown:.2f}%
  Avg Holding Period: {results.avg_holding_period:.1f} days

INTERPRETATION:
  {"✅ POSITIVE EXPECTANCY" if results.expectancy > 0 else "❌ NEGATIVE EXPECTANCY"}
  {"✅ PROFIT FACTOR > 1.5" if results.profit_factor > 1.5 else "⚠️ LOW PROFIT FACTOR"}
  {"✅ WIN RATE > 50%" if results.win_rate > 50 else "⚠️ WIN RATE < 50%"}
"""
        return output


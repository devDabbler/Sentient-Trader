"""
GS Quant Integration Layer for Sentient Trader.
Provides institutional-grade risk analytics and backtesting for stocks and options.

Features:
- Options Greeks calculation (Black-Scholes with GS Quant fallback)
- Portfolio-level risk metrics (VaR, aggregated Greeks)
- Strategy backtesting (WARRIOR_SCALPING, SLOW_SCALPER, MICRO_SWING, OPTIONS)
- Sharpe/Sortino ratio calculation
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import math
import logging

# GS Quant imports (optional - graceful fallback)
try:
    from gs_quant.markets.portfolio import Portfolio
    from gs_quant.risk import Price, DeltaGamma, VaR
    from gs_quant.backtests.core import Backtest
    from gs_quant.backtests.strategy import Strategy
    from gs_quant.instrument import EqOption
    from gs_quant.markets import PricingContext
    GS_QUANT_AVAILABLE = True
except ImportError:
    GS_QUANT_AVAILABLE = False

logger = logging.getLogger(__name__)


class OptionType(Enum):
    """Option type enum"""
    CALL = "call"
    PUT = "put"


@dataclass
class Greeks:
    """Options Greeks container"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho
        }


@dataclass
class RiskMetrics:
    """Portfolio-level risk metrics"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    var_95: float  # 95% Value at Risk (1-day)
    var_99: float  # 99% Value at Risk (1-day)
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    beta: Optional[float] = None
    correlation_spy: Optional[float] = None
    portfolio_value: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'total_delta': self.total_delta,
            'total_gamma': self.total_gamma,
            'total_theta': self.total_theta,
            'total_vega': self.total_vega,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'beta': self.beta,
            'correlation_spy': self.correlation_spy,
            'portfolio_value': self.portfolio_value
        }


@dataclass
class BacktestResult:
    """Backtest results for strategy validation"""
    strategy_name: str
    ticker: str
    start_date: datetime
    end_date: datetime
    
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    volatility_pct: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    
    # Validation
    is_profitable: bool
    meets_sharpe_threshold: bool
    recommendation: str
    
    # Trade log
    trades: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'strategy_name': self.strategy_name,
            'ticker': self.ticker,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'total_return_pct': self.total_return_pct,
            'annualized_return_pct': self.annualized_return_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown_pct': self.max_drawdown_pct,
            'volatility_pct': self.volatility_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win_pct': self.avg_win_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'is_profitable': self.is_profitable,
            'meets_sharpe_threshold': self.meets_sharpe_threshold,
            'recommendation': self.recommendation
        }


class QuantAnalyticsService:
    """
    Integrates GS Quant capabilities with Sentient Trader.
    
    Provides:
    - Options Greeks calculation (Black-Scholes)
    - Portfolio-level risk metrics (VaR, Greeks aggregation)
    - Strategy backtesting (stocks and options)
    - Performance metrics (Sharpe, Sortino, Max Drawdown)
    
    Works for both stocks and options trading strategies.
    """
    
    def __init__(self):
        self.gs_quant_available = GS_QUANT_AVAILABLE
        self.backtest_cache: Dict[str, BacktestResult] = {}
        
        if not self.gs_quant_available:
            logger.info("QuantAnalyticsService running in pure Python mode (gs-quant not installed)")
        else:
            logger.info("QuantAnalyticsService initialized with GS Quant support")
    
    # =========================================================================
    # OPTIONS GREEKS
    # =========================================================================
    
    def calculate_greeks(
        self,
        underlying_price: float,
        strike: float,
        days_to_expiry: int,
        implied_volatility: float,
        risk_free_rate: float = 0.05,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0
    ) -> Greeks:
        """
        Calculate options Greeks using Black-Scholes model.
        Falls back to pure Python implementation if GS Quant unavailable.
        
        Args:
            underlying_price: Current price of the underlying asset
            strike: Strike price of the option
            days_to_expiry: Days until option expiration
            implied_volatility: Implied volatility (decimal, e.g., 0.30 for 30%)
            risk_free_rate: Risk-free interest rate (decimal)
            option_type: CALL or PUT
            dividend_yield: Dividend yield (decimal)
            
        Returns:
            Greeks dataclass with delta, gamma, theta, vega, rho
        """
        if self.gs_quant_available:
            return self._calculate_greeks_gs_quant(
                underlying_price, strike, days_to_expiry,
                implied_volatility, risk_free_rate, option_type
            )
        
        return self._calculate_greeks_black_scholes(
            underlying_price, strike, days_to_expiry,
            implied_volatility, risk_free_rate, option_type, dividend_yield
        )
    
    def _calculate_greeks_gs_quant(
        self,
        underlying_price: float,
        strike: float,
        days_to_expiry: int,
        implied_volatility: float,
        risk_free_rate: float,
        option_type: OptionType
    ) -> Greeks:
        """Calculate Greeks using GS Quant (when available)"""
        try:
            # GS Quant implementation - for now fallback to Black-Scholes
            # Full GS Quant requires authentication with Goldman Sachs
            return self._calculate_greeks_black_scholes(
                underlying_price, strike, days_to_expiry,
                implied_volatility, risk_free_rate, option_type, 0.0
            )
        except Exception as e:
            logger.error(f"GS Quant Greeks calculation failed: {e}")
            return self._calculate_greeks_black_scholes(
                underlying_price, strike, days_to_expiry,
                implied_volatility, risk_free_rate, option_type, 0.0
            )
    
    def _calculate_greeks_black_scholes(
        self,
        S: float,  # Underlying price
        K: float,  # Strike price
        T_days: int,  # Days to expiry
        sigma: float,  # Implied volatility
        r: float,  # Risk-free rate
        option_type: OptionType,
        q: float  # Dividend yield
    ) -> Greeks:
        """Pure Python Black-Scholes Greeks calculation"""
        from scipy.stats import norm
        
        # Convert days to years
        T = T_days / 365.0
        
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
        
        try:
            # Calculate d1 and d2
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            # Standard normal CDF and PDF
            N = norm.cdf
            n = norm.pdf
            
            if option_type == OptionType.CALL:
                delta = math.exp(-q * T) * N(d1)
                rho = K * T * math.exp(-r * T) * N(d2) / 100
            else:  # PUT
                delta = -math.exp(-q * T) * N(-d1)
                rho = -K * T * math.exp(-r * T) * N(-d2) / 100
            
            # Gamma (same for calls and puts)
            gamma = math.exp(-q * T) * n(d1) / (S * sigma * math.sqrt(T))
            
            # Vega (same for calls and puts) - per 1% change in volatility
            vega = S * math.exp(-q * T) * n(d1) * math.sqrt(T) / 100
            
            # Theta (daily decay)
            term1 = -(S * sigma * math.exp(-q * T) * n(d1)) / (2 * math.sqrt(T))
            if option_type == OptionType.CALL:
                term2 = q * S * math.exp(-q * T) * N(d1)
                term3 = r * K * math.exp(-r * T) * N(d2)
                theta = (term1 + term2 - term3) / 365
            else:
                term2 = q * S * math.exp(-q * T) * N(-d1)
                term3 = r * K * math.exp(-r * T) * N(-d2)
                theta = (term1 - term2 + term3) / 365
            
            return Greeks(
                delta=round(float(delta), 4),
                gamma=round(float(gamma), 6),
                theta=round(float(theta), 4),
                vega=round(float(vega), 4),
                rho=round(float(rho), 4)
            )
        except Exception as e:
            logger.error(f"Black-Scholes calculation error: {e}")
            return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)
    
    def calculate_option_price(
        self,
        underlying_price: float,
        strike: float,
        days_to_expiry: int,
        implied_volatility: float,
        risk_free_rate: float = 0.05,
        option_type: OptionType = OptionType.CALL,
        dividend_yield: float = 0.0
    ) -> float:
        """Calculate theoretical option price using Black-Scholes"""
        from scipy.stats import norm
        
        S, K, T_days, sigma, r, q = underlying_price, strike, days_to_expiry, implied_volatility, risk_free_rate, dividend_yield
        T = T_days / 365.0
        
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        
        try:
            d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            
            if option_type == OptionType.CALL:
                price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
            
            return round(float(price), 2)
        except Exception as e:
            logger.error(f"Option price calculation error: {e}")
            return 0.0
    
    # =========================================================================
    # PORTFOLIO RISK
    # =========================================================================
    
    def calculate_portfolio_risk(
        self,
        positions: List[Dict],
        historical_returns: Optional[List[float]] = None
    ) -> RiskMetrics:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            positions: List of position dicts with keys:
                - symbol, quantity, market_value, delta, gamma, theta, vega
            historical_returns: Optional list of historical daily returns for VaR
            
        Returns:
            RiskMetrics with aggregated Greeks and VaR
        """
        if not positions:
            return RiskMetrics(
                total_delta=0, total_gamma=0, total_theta=0, total_vega=0,
                var_95=0, var_99=0, max_drawdown=0, portfolio_value=0
            )
        
        # Aggregate Greeks
        total_delta = sum(p.get('delta', 0) * p.get('quantity', 1) for p in positions)
        total_gamma = sum(p.get('gamma', 0) * p.get('quantity', 1) for p in positions)
        total_theta = sum(p.get('theta', 0) * p.get('quantity', 1) for p in positions)
        total_vega = sum(p.get('vega', 0) * p.get('quantity', 1) for p in positions)
        
        # Portfolio value
        portfolio_value = sum(p.get('market_value', 0) for p in positions)
        
        # Calculate VaR
        if historical_returns and len(historical_returns) >= 20:
            var_95, var_99 = self._calculate_historical_var(historical_returns, portfolio_value)
        else:
            # Fallback: Parametric VaR with assumed volatility
            daily_vol = 0.02  # Assume 2% daily volatility
            var_95 = portfolio_value * daily_vol * 1.645  # 95% confidence
            var_99 = portfolio_value * daily_vol * 2.326  # 99% confidence
        
        # Max drawdown calculation
        max_drawdown = self._calculate_max_drawdown(historical_returns) if historical_returns else 0.0
        
        # Sharpe ratio (if we have returns)
        sharpe = self._calculate_sharpe_ratio(historical_returns) if historical_returns else None
        
        return RiskMetrics(
            total_delta=round(total_delta, 2),
            total_gamma=round(total_gamma, 6),
            total_theta=round(total_theta, 2),
            total_vega=round(total_vega, 2),
            var_95=round(var_95, 2),
            var_99=round(var_99, 2),
            max_drawdown=round(max_drawdown, 4),
            sharpe_ratio=round(sharpe, 2) if sharpe else None,
            portfolio_value=round(portfolio_value, 2)
        )
    
    def _calculate_historical_var(
        self,
        returns: List[float],
        portfolio_value: float
    ) -> Tuple[float, float]:
        """Calculate Historical VaR from returns series"""
        import numpy as np
        
        returns_array = np.array(returns)
        
        # Historical VaR: percentile of losses
        var_95 = -np.percentile(returns_array, 5) * portfolio_value
        var_99 = -np.percentile(returns_array, 1) * portfolio_value
        
        return max(0, var_95), max(0, var_99)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns series"""
        import numpy as np
        
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        return abs(float(np.min(drawdowns)))
    
    def _calculate_sharpe_ratio(
        self, 
        returns: List[float], 
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate annualized Sharpe ratio"""
        import numpy as np
        
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_sortino_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate annualized Sortino ratio (downside deviation only)"""
        import numpy as np
        
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)
        
        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
        return float(sortino)
    
    # =========================================================================
    # BACKTESTING
    # =========================================================================
    
    async def backtest_strategy(
        self,
        strategy_name: str,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        strategy_params: Optional[Dict] = None
    ) -> BacktestResult:
        """
        Backtest a trading strategy on historical data.
        
        Supports strategies:
        - WARRIOR_SCALPING: Gap & Go momentum (stocks)
        - SLOW_SCALPER: Intraday mean reversion (stocks)
        - MICRO_SWING: Key level rejections (stocks)
        - COVERED_CALL: Covered call premium collection (options)
        - CASH_SECURED_PUT: Cash-secured puts (options)
        
        Args:
            strategy_name: Name of the strategy to backtest
            ticker: Stock ticker symbol
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            strategy_params: Optional strategy-specific parameters
            
        Returns:
            BacktestResult with full performance metrics
        """
        cache_key = f"{strategy_name}_{ticker}_{start_date.date()}_{end_date.date()}"
        
        if cache_key in self.backtest_cache:
            logger.info(f"Returning cached backtest for {cache_key}")
            return self.backtest_cache[cache_key]
        
        logger.info(f"Running backtest: {strategy_name} on {ticker}")
        
        # Fetch historical data
        import yfinance as yf
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data is None or data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Flatten multi-level columns if present
            if data is not None and hasattr(data.columns, 'levels'):
                data.columns = data.columns.get_level_values(0)
            
            # Run strategy-specific backtest
            if strategy_name == "WARRIOR_SCALPING":
                result = self._backtest_warrior_scalping(ticker, data, initial_capital, strategy_params)
            elif strategy_name == "SLOW_SCALPER":
                result = self._backtest_slow_scalper(ticker, data, initial_capital, strategy_params)
            elif strategy_name == "MICRO_SWING":
                result = self._backtest_micro_swing(ticker, data, initial_capital, strategy_params)
            elif strategy_name == "COVERED_CALL":
                result = self._backtest_covered_call(ticker, data, initial_capital, strategy_params)
            elif strategy_name == "CASH_SECURED_PUT":
                result = self._backtest_cash_secured_put(ticker, data, initial_capital, strategy_params)
            else:
                result = self._backtest_generic(ticker, data, initial_capital, strategy_name, strategy_params)
            
            # Add metadata
            result.strategy_name = strategy_name
            result.ticker = ticker
            result.start_date = start_date
            result.end_date = end_date
            
            # Generate recommendation
            result.is_profitable = result.total_return_pct > 0
            result.meets_sharpe_threshold = result.sharpe_ratio >= 0.5
            
            if result.sharpe_ratio >= 1.0 and result.win_rate >= 0.55:
                result.recommendation = "STRONG_BUY"
            elif result.sharpe_ratio >= 0.5 and result.win_rate >= 0.50:
                result.recommendation = "FAVORABLE"
            elif result.sharpe_ratio >= 0 and result.is_profitable:
                result.recommendation = "NEUTRAL"
            else:
                result.recommendation = "CAUTION"
            
            self.backtest_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return BacktestResult(
                strategy_name=strategy_name,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                total_return_pct=0,
                annualized_return_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown_pct=0,
                volatility_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                is_profitable=False,
                meets_sharpe_threshold=False,
                recommendation="ERROR",
                trades=[]
            )
    
    def _backtest_warrior_scalping(
        self,
        ticker: str,
        data,
        initial_capital: float,
        params: Optional[Dict] = None
    ) -> BacktestResult:
        """
        Backtest WARRIOR_SCALPING strategy (Gap & Go momentum).
        
        Rules:
        - Entry: Gap up > 2% on high volume
        - Exit: 5% profit target or 2% stop loss
        - Time: First 2 hours of trading only
        """
        import numpy as np
        
        params = params or {}
        gap_threshold = params.get('gap_threshold', 2.0)
        volume_ratio = params.get('volume_ratio', 1.5)
        profit_target = params.get('profit_target', 5.0)
        stop_loss = params.get('stop_loss', 2.0)
        
        trades = []
        capital = initial_capital
        
        # Calculate daily gaps and volume
        data = data.copy()
        data['gap_pct'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1) * 100
        data['vol_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        for i in range(20, len(data)):
            row = data.iloc[i]
            
            # Entry signal: Gap up > threshold, volume > ratio
            if row['gap_pct'] > gap_threshold and row['vol_ratio'] > volume_ratio:
                entry_price = float(row['Open'])
                target = entry_price * (1 + profit_target / 100)
                stop = entry_price * (1 - stop_loss / 100)
                
                # Simulate exit (using high/low of day)
                high = float(row['High'])
                low = float(row['Low'])
                close = float(row['Close'])
                
                if high >= target:
                    exit_price = target
                    pnl_pct = profit_target
                elif low <= stop:
                    exit_price = stop
                    pnl_pct = -stop_loss
                else:
                    exit_price = close
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                
                trades.append({
                    'date': str(data.index[i].date()),
                    'entry': round(entry_price, 2),
                    'exit': round(exit_price, 2),
                    'pnl_pct': round(pnl_pct, 2)
                })
                
                capital *= (1 + pnl_pct / 100)
        
        return self._calculate_backtest_metrics(trades, initial_capital, capital, data)
    
    def _backtest_slow_scalper(
        self,
        ticker: str,
        data,
        initial_capital: float,
        params: Optional[Dict] = None
    ) -> BacktestResult:
        """Backtest SLOW_SCALPER strategy (mean reversion with Bollinger Bands)"""
        import numpy as np
        
        params = params or {}
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        stop_loss = params.get('stop_loss', 2.0)
        
        trades = []
        capital = initial_capital
        
        data = data.copy()
        data['sma'] = data['Close'].rolling(bb_period).mean()
        data['std'] = data['Close'].rolling(bb_period).std()
        data['lower_band'] = data['sma'] - bb_std * data['std']
        data['upper_band'] = data['sma'] + bb_std * data['std']
        
        in_position = False
        entry_price = 0
        
        for i in range(bb_period, len(data)):
            row = data.iloc[i]
            
            if not in_position:
                # Entry: Price touches lower Bollinger Band
                if float(row['Low']) <= float(row['lower_band']):
                    entry_price = float(row['lower_band'])
                    in_position = True
            else:
                # Exit: Price reaches SMA (mean reversion target) or stop loss
                target = float(row['sma'])
                stop = entry_price * (1 - stop_loss / 100)
                
                high = float(row['High'])
                low = float(row['Low'])
                close = float(row['Close'])
                
                if high >= target:
                    exit_price = target
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    in_position = False
                elif low <= stop:
                    exit_price = stop
                    pnl_pct = -stop_loss
                    in_position = False
                else:
                    continue  # Hold position
                
                if not in_position:
                    trades.append({
                        'date': str(data.index[i].date()),
                        'entry': round(entry_price, 2),
                        'exit': round(exit_price, 2),
                        'pnl_pct': round(pnl_pct, 2)
                    })
                    capital *= (1 + pnl_pct / 100)
        
        return self._calculate_backtest_metrics(trades, initial_capital, capital, data)
    
    def _backtest_micro_swing(
        self,
        ticker: str,
        data,
        initial_capital: float,
        params: Optional[Dict] = None
    ) -> BacktestResult:
        """Backtest MICRO_SWING strategy (key level rejections)"""
        import numpy as np
        
        params = params or {}
        lookback = params.get('lookback', 20)
        profit_target = params.get('profit_target', 3.0)
        stop_loss = params.get('stop_loss', 1.5)
        
        trades = []
        capital = initial_capital
        
        data = data.copy()
        data['support'] = data['Low'].rolling(lookback).min()
        data['resistance'] = data['High'].rolling(lookback).max()
        
        in_position = False
        entry_price = 0
        position_type = None  # 'long' or 'short'
        
        for i in range(lookback, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]
            
            if not in_position:
                # Long entry: Bounce off support
                if float(row['Low']) <= float(row['support']) * 1.01 and float(row['Close']) > float(row['Open']):
                    entry_price = float(row['Close'])
                    in_position = True
                    position_type = 'long'
            else:
                # Exit logic
                if position_type == 'long':
                    target = entry_price * (1 + profit_target / 100)
                    stop = entry_price * (1 - stop_loss / 100)
                    
                    high = float(row['High'])
                    low = float(row['Low'])
                    
                    if high >= target:
                        exit_price = target
                        pnl_pct = profit_target
                        in_position = False
                    elif low <= stop:
                        exit_price = stop
                        pnl_pct = -stop_loss
                        in_position = False
                
                if not in_position:
                    trades.append({
                        'date': str(data.index[i].date()),
                        'entry': round(entry_price, 2),
                        'exit': round(exit_price, 2),
                        'pnl_pct': round(pnl_pct, 2)
                    })
                    capital *= (1 + pnl_pct / 100)
        
        return self._calculate_backtest_metrics(trades, initial_capital, capital, data)
    
    def _backtest_covered_call(
        self,
        ticker: str,
        data,
        initial_capital: float,
        params: Optional[Dict] = None
    ) -> BacktestResult:
        """Backtest COVERED_CALL strategy (options premium collection)"""
        import numpy as np
        
        params = params or {}
        otm_pct = params.get('otm_pct', 5.0)  # Sell calls 5% OTM
        premium_yield = params.get('premium_yield', 1.5)  # Assume 1.5% monthly premium
        
        trades = []
        capital = initial_capital
        
        # Simulate monthly covered call writing
        monthly_data = data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        shares = int(initial_capital / float(monthly_data.iloc[0]['Open']) / 100) * 100
        if shares == 0:
            shares = 100
        
        for i in range(len(monthly_data) - 1):
            row = monthly_data.iloc[i]
            next_row = monthly_data.iloc[i + 1]
            
            stock_price = float(row['Close'])
            strike = stock_price * (1 + otm_pct / 100)
            premium = stock_price * (premium_yield / 100)
            
            # Check if called away
            max_price = float(next_row['High'])
            end_price = float(next_row['Close'])
            
            if max_price >= strike:
                # Called away at strike
                stock_pnl = (strike - stock_price) / stock_price * 100
                total_pnl = stock_pnl + premium_yield
            else:
                # Keep shares, collect premium
                stock_pnl = (end_price - stock_price) / stock_price * 100
                total_pnl = stock_pnl + premium_yield
            
            trades.append({
                'date': str(monthly_data.index[i].date()),
                'entry': round(stock_price, 2),
                'strike': round(strike, 2),
                'premium': round(premium, 2),
                'pnl_pct': round(total_pnl, 2)
            })
            
            capital *= (1 + total_pnl / 100)
        
        return self._calculate_backtest_metrics(trades, initial_capital, capital, data)
    
    def _backtest_cash_secured_put(
        self,
        ticker: str,
        data,
        initial_capital: float,
        params: Optional[Dict] = None
    ) -> BacktestResult:
        """Backtest CASH_SECURED_PUT strategy (options premium collection)"""
        import numpy as np
        
        params = params or {}
        otm_pct = params.get('otm_pct', 5.0)  # Sell puts 5% OTM
        premium_yield = params.get('premium_yield', 1.2)  # Assume 1.2% monthly premium
        
        trades = []
        capital = initial_capital
        
        # Simulate monthly cash-secured put writing
        monthly_data = data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        for i in range(len(monthly_data) - 1):
            row = monthly_data.iloc[i]
            next_row = monthly_data.iloc[i + 1]
            
            stock_price = float(row['Close'])
            strike = stock_price * (1 - otm_pct / 100)
            premium = stock_price * (premium_yield / 100)
            
            # Check if assigned
            min_price = float(next_row['Low'])
            
            if min_price <= strike:
                # Assigned at strike, now own shares at strike price
                end_price = float(next_row['Close'])
                assignment_loss = (end_price - strike) / strike * 100
                total_pnl = premium_yield + assignment_loss
            else:
                # Put expires worthless, keep premium
                total_pnl = premium_yield
            
            trades.append({
                'date': str(monthly_data.index[i].date()),
                'strike': round(strike, 2),
                'premium': round(premium, 2),
                'assigned': min_price <= strike,
                'pnl_pct': round(total_pnl, 2)
            })
            
            capital *= (1 + total_pnl / 100)
        
        return self._calculate_backtest_metrics(trades, initial_capital, capital, data)
    
    def _backtest_generic(
        self,
        ticker: str,
        data,
        initial_capital: float,
        strategy_name: str,
        params: Optional[Dict] = None
    ) -> BacktestResult:
        """Generic backtest using simple SMA crossover"""
        import numpy as np
        
        params = params or {}
        fast_period = params.get('fast_period', 10)
        slow_period = params.get('slow_period', 30)
        
        trades = []
        capital = initial_capital
        
        data = data.copy()
        data['sma_fast'] = data['Close'].rolling(fast_period).mean()
        data['sma_slow'] = data['Close'].rolling(slow_period).mean()
        
        in_position = False
        entry_price = 0
        
        for i in range(slow_period + 1, len(data)):
            row = data.iloc[i]
            prev_row = data.iloc[i - 1]
            
            # Golden cross: fast SMA crosses above slow SMA
            if not in_position:
                if float(prev_row['sma_fast']) <= float(prev_row['sma_slow']) and \
                   float(row['sma_fast']) > float(row['sma_slow']):
                    entry_price = float(row['Close'])
                    in_position = True
            else:
                # Death cross: fast SMA crosses below slow SMA
                if float(prev_row['sma_fast']) >= float(prev_row['sma_slow']) and \
                   float(row['sma_fast']) < float(row['sma_slow']):
                    exit_price = float(row['Close'])
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'date': str(data.index[i].date()),
                        'entry': round(entry_price, 2),
                        'exit': round(exit_price, 2),
                        'pnl_pct': round(pnl_pct, 2)
                    })
                    
                    capital *= (1 + pnl_pct / 100)
                    in_position = False
        
        return self._calculate_backtest_metrics(trades, initial_capital, capital, data)
    
    def _calculate_backtest_metrics(
        self,
        trades: List[Dict],
        initial_capital: float,
        final_capital: float,
        data
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics from trade list"""
        import numpy as np
        
        if not trades:
            return BacktestResult(
                strategy_name="",
                ticker="",
                start_date=datetime.now(),
                end_date=datetime.now(),
                total_return_pct=0,
                annualized_return_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown_pct=0,
                volatility_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                is_profitable=False,
                meets_sharpe_threshold=False,
                recommendation="NO_TRADES",
                trades=[]
            )
        
        # Extract PnL series
        pnl_list = [t['pnl_pct'] for t in trades]
        pnl_array = np.array(pnl_list)
        
        # Basic metrics
        total_return_pct = (final_capital - initial_capital) / initial_capital * 100
        
        # Annualized return (assuming 252 trading days)
        num_days = len(data)
        years = num_days / 252
        if years > 0 and final_capital > 0:
            annualized_return_pct = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return_pct = 0
        
        # Trade statistics
        winning_trades = len([p for p in pnl_list if p > 0])
        losing_trades = len([p for p in pnl_list if p < 0])
        total_trades = len(trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        
        avg_win_pct = np.mean(wins) if wins else 0
        avg_loss_pct = abs(np.mean(losses)) if losses else 0
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Risk metrics
        volatility_pct = float(np.std(pnl_array)) if len(pnl_array) > 1 else 0
        
        # Sharpe and Sortino (annualized)
        if volatility_pct > 0:
            sharpe_ratio = (np.mean(pnl_array) / volatility_pct) * np.sqrt(len(trades))
        else:
            sharpe_ratio = 0
        
        downside = pnl_array[pnl_array < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino_ratio = (np.mean(pnl_array) / np.std(downside)) * np.sqrt(len(trades))
        else:
            sortino_ratio = sharpe_ratio
        
        # Max drawdown
        cumulative = np.cumprod(1 + pnl_array / 100)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max * 100
        max_drawdown_pct = abs(float(np.min(drawdowns)))
        
        return BacktestResult(
            strategy_name="",
            ticker="",
            start_date=datetime.now(),
            end_date=datetime.now(),
            total_return_pct=round(total_return_pct, 2),
            annualized_return_pct=round(annualized_return_pct, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            sortino_ratio=round(sortino_ratio, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            volatility_pct=round(volatility_pct, 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2) if profit_factor != float('inf') else 999.99,
            avg_win_pct=round(float(avg_win_pct), 2),
            avg_loss_pct=round(float(avg_loss_pct), 2),
            is_profitable=total_return_pct > 0,
            meets_sharpe_threshold=sharpe_ratio >= 0.5,
            recommendation="",
            trades=trades
        )
    
    def clear_cache(self):
        """Clear the backtest cache"""
        self.backtest_cache.clear()
        logger.info("Backtest cache cleared")
    
    def get_available_strategies(self) -> List[Dict[str, str]]:
        """Return list of available strategies for backtesting"""
        return [
            {
                "name": "WARRIOR_SCALPING",
                "description": "Gap & Go momentum trading - Enter on gap up > 2% with high volume",
                "type": "stock",
                "timeframe": "intraday"
            },
            {
                "name": "SLOW_SCALPER",
                "description": "Mean reversion using Bollinger Bands - Buy at lower band, sell at mean",
                "type": "stock",
                "timeframe": "intraday"
            },
            {
                "name": "MICRO_SWING",
                "description": "Key level rejection trading - Trade bounces off support/resistance",
                "type": "stock",
                "timeframe": "swing"
            },
            {
                "name": "COVERED_CALL",
                "description": "Sell OTM calls against stock holdings for premium income",
                "type": "options",
                "timeframe": "monthly"
            },
            {
                "name": "CASH_SECURED_PUT",
                "description": "Sell OTM puts to collect premium or acquire stock at discount",
                "type": "options",
                "timeframe": "monthly"
            },
            {
                "name": "SMA_CROSSOVER",
                "description": "Generic SMA crossover strategy (golden/death cross)",
                "type": "stock",
                "timeframe": "swing"
            }
        ]


# Singleton instance for easy import
_quant_service: Optional[QuantAnalyticsService] = None


def get_quant_service() -> QuantAnalyticsService:
    """Get or create singleton QuantAnalyticsService instance"""
    global _quant_service
    if _quant_service is None:
        _quant_service = QuantAnalyticsService()
    return _quant_service

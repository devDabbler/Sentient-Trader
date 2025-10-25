"""Options chain integration with Fibonacci-based strike selection."""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class OptionStrike:
    """Represents an option strike with Greeks and pricing"""
    strike: float
    expiration: datetime
    dte: int  # Days to expiration
    
    # Call data
    call_bid: float = 0.0
    call_ask: float = 0.0
    call_last: float = 0.0
    call_volume: int = 0
    call_open_interest: int = 0
    call_iv: float = 0.0
    call_delta: float = 0.0
    call_theta: float = 0.0
    call_vega: float = 0.0
    
    # Put data
    put_bid: float = 0.0
    put_ask: float = 0.0
    put_last: float = 0.0
    put_volume: int = 0
    put_open_interest: int = 0
    put_iv: float = 0.0
    put_delta: float = 0.0
    put_theta: float = 0.0
    put_vega: float = 0.0
    
    # Metadata
    is_fib_target: bool = False
    fib_level: Optional[str] = None  # "T1", "T2", "T3", "C"
    distance_from_spot: float = 0.0  # % from current price
    
    def get_mid_price(self, option_type: str = "call") -> float:
        """Get mid price for call or put"""
        if option_type.lower() == "call":
            return (self.call_bid + self.call_ask) / 2 if self.call_bid and self.call_ask else self.call_last
        else:
            return (self.put_bid + self.put_ask) / 2 if self.put_bid and self.put_ask else self.put_last


@dataclass
class OptionsSpread:
    """Represents an options spread strategy"""
    strategy_name: str  # "Bull Call Spread", "Bull Put Spread", etc.
    description: str
    
    # Long leg
    long_strike: float
    long_call_or_put: str  # "call" or "put"
    long_price: float
    long_delta: float
    
    # Short leg
    short_strike: float
    short_call_or_put: str
    short_price: float
    short_delta: float
    
    # Spread metrics
    net_debit: float  # Negative for credit spread
    max_profit: float
    max_loss: float
    breakeven: float
    probability_profit: float  # Based on delta
    risk_reward_ratio: float
    
    expiration: datetime
    dte: int
    
    # Context
    is_fibonacci_based: bool = False
    target_level: Optional[str] = None  # Which Fib target this aims for


class FibonacciOptionsChain:
    """Options chain analyzer with Fibonacci integration"""
    
    def __init__(self, ticker: str):
        """Initialize with ticker symbol"""
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.current_price = 0.0
        self.options_dates = []
        self._fetch_current_price()
        self._fetch_options_dates()
    
    def _fetch_current_price(self):
        """Fetch current stock price"""
        try:
            hist = self.stock.history(period="1d")
            if not hist.empty:
                self.current_price = hist['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Error fetching price for {self.ticker}: {e}")
    
    def _fetch_options_dates(self):
        """Fetch available options expiration dates"""
        try:
            self.options_dates = list(self.stock.options)
        except Exception as e:
            logger.error(f"Error fetching options dates for {self.ticker}: {e}")
            self.options_dates = []
    
    def get_chain_for_dte(self, target_dte: int, tolerance: int = 7) -> Optional[pd.DataFrame]:
        """
        Get options chain closest to target DTE.
        
        Args:
            target_dte: Target days to expiration
            tolerance: Acceptable deviation in days
        
        Returns:
            DataFrame with options chain or None
        """
        if not self.options_dates:
            return None
        
        best_date = None
        best_diff = float('inf')
        
        for date_str in self.options_dates:
            exp_date = pd.to_datetime(date_str)
            dte = (exp_date - pd.Timestamp.now()).days
            diff = abs(dte - target_dte)
            
            if diff < best_diff and diff <= tolerance:
                best_diff = diff
                best_date = date_str
        
        if not best_date:
            # Just use closest available
            best_date = self.options_dates[0]
        
        try:
            chain = self.stock.option_chain(best_date)
            exp_date = pd.to_datetime(best_date)
            dte = (exp_date - pd.Timestamp.now()).days
            
            # Combine calls and puts
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            
            calls['type'] = 'call'
            puts['type'] = 'put'
            
            combined = pd.concat([calls, puts], ignore_index=True)
            combined['expiration'] = exp_date
            combined['dte'] = dte
            
            return combined
        except Exception as e:
            logger.error(f"Error fetching chain for {best_date}: {e}")
            return None
    
    def find_strikes_near_fibonacci(self, fib_targets: Dict[str, float], 
                                   target_dte: int = 45) -> Dict[str, OptionStrike]:
        """
        Find option strikes closest to Fibonacci levels.
        
        Args:
            fib_targets: Dict with keys like 'A', 'B', 'C', 'T1_1272', 'T2_1618', 'T3_200'
            target_dte: Target days to expiration
        
        Returns:
            Dict mapping Fib level to OptionStrike
        """
        chain = self.get_chain_for_dte(target_dte)
        if chain is None or chain.empty:
            return {}
        
        results = {}
        
        # Key Fibonacci levels to find
        levels = {
            'C': fib_targets.get('C'),  # Support/entry
            'T1': fib_targets.get('T1_1272'),  # First target
            'T2': fib_targets.get('T2_1618'),  # Second target
            'T3': fib_targets.get('T3_2618') or fib_targets.get('T3_200'),  # Third target
        }
        
        for level_name, level_price in levels.items():
            if level_price is None:
                continue
            
            # Find closest strike
            strikes = chain['strike'].unique()
            closest_strike = min(strikes, key=lambda x: abs(x - level_price))
            
            # Get option data for this strike
            strike_data = chain[chain['strike'] == closest_strike]
            
            if not strike_data.empty:
                exp_date = strike_data['expiration'].iloc[0]
                dte = strike_data['dte'].iloc[0]
                
                # Extract call data
                call_data = strike_data[strike_data['type'] == 'call']
                put_data = strike_data[strike_data['type'] == 'put']
                
                option_strike = OptionStrike(
                    strike=closest_strike,
                    expiration=exp_date,
                    dte=dte,
                    is_fib_target=True,
                    fib_level=level_name,
                    distance_from_spot=((closest_strike / self.current_price) - 1) * 100
                )
                
                # Populate call data
                if not call_data.empty:
                    option_strike.call_bid = call_data['bid'].iloc[0]
                    option_strike.call_ask = call_data['ask'].iloc[0]
                    option_strike.call_last = call_data['lastPrice'].iloc[0]
                    option_strike.call_volume = call_data['volume'].iloc[0]
                    option_strike.call_open_interest = call_data['openInterest'].iloc[0]
                    option_strike.call_iv = call_data.get('impliedVolatility', pd.Series([0])).iloc[0]
                    
                    # Greeks (if available)
                    if 'delta' in call_data.columns:
                        option_strike.call_delta = call_data['delta'].iloc[0]
                    if 'theta' in call_data.columns:
                        option_strike.call_theta = call_data['theta'].iloc[0]
                    if 'vega' in call_data.columns:
                        option_strike.call_vega = call_data['vega'].iloc[0]
                
                # Populate put data
                if not put_data.empty:
                    option_strike.put_bid = put_data['bid'].iloc[0]
                    option_strike.put_ask = put_data['ask'].iloc[0]
                    option_strike.put_last = put_data['lastPrice'].iloc[0]
                    option_strike.put_volume = put_data['volume'].iloc[0]
                    option_strike.put_open_interest = put_data['openInterest'].iloc[0]
                    option_strike.put_iv = put_data.get('impliedVolatility', pd.Series([0])).iloc[0]
                    
                    if 'delta' in put_data.columns:
                        option_strike.put_delta = put_data['delta'].iloc[0]
                    if 'theta' in put_data.columns:
                        option_strike.put_theta = put_data['theta'].iloc[0]
                    if 'vega' in put_data.columns:
                        option_strike.put_vega = put_data['vega'].iloc[0]
                
                results[level_name] = option_strike
        
        return results
    
    def suggest_fibonacci_spreads(self, fib_targets: Dict[str, float], 
                                 analysis, 
                                 target_dte: int = 45) -> List[OptionsSpread]:
        """
        Suggest options spreads based on Fibonacci targets and market context.
        
        Args:
            fib_targets: Fibonacci levels
            analysis: StockAnalysis object with trend/IV context
            target_dte: Target days to expiration
        
        Returns:
            List of suggested OptionsSpread objects
        """
        spreads = []
        
        # Get strikes near Fibonacci levels
        fib_strikes = self.find_strikes_near_fibonacci(fib_targets, target_dte)
        
        if not fib_strikes:
            return spreads
        
        # Get ATM strike
        chain = self.get_chain_for_dte(target_dte)
        if chain is None or chain.empty:
            return spreads
        
        strikes = sorted(chain['strike'].unique())
        atm_strike = min(strikes, key=lambda x: abs(x - self.current_price))
        atm_data = chain[chain['strike'] == atm_strike]
        
        # Strategy 1: Bull Call Spread (Buy ATM, Sell T1)
        if "UPTREND" in analysis.trend and 'T1' in fib_strikes:
            t1_strike_obj = fib_strikes['T1']
            
            # Get ATM call data
            atm_call = atm_data[atm_data['type'] == 'call']
            if not atm_call.empty:
                long_price = (atm_call['bid'].iloc[0] + atm_call['ask'].iloc[0]) / 2
                short_price = t1_strike_obj.get_mid_price('call')
                
                net_debit = long_price - short_price
                max_profit = (t1_strike_obj.strike - atm_strike) - net_debit
                max_loss = net_debit
                breakeven = atm_strike + net_debit
                
                # Estimate probability using delta
                long_delta = atm_call.get('delta', pd.Series([0.5])).iloc[0]
                short_delta = t1_strike_obj.call_delta
                prob_profit = abs(long_delta - short_delta) if short_delta else abs(long_delta) * 0.7
                
                spread = OptionsSpread(
                    strategy_name="Bull Call Spread (Fib T1)",
                    description=f"Buy {atm_strike} call, Sell {t1_strike_obj.strike} call (T1 target)",
                    long_strike=atm_strike,
                    long_call_or_put="call",
                    long_price=long_price,
                    long_delta=long_delta,
                    short_strike=t1_strike_obj.strike,
                    short_call_or_put="call",
                    short_price=short_price,
                    short_delta=short_delta,
                    net_debit=net_debit,
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven=breakeven,
                    probability_profit=prob_profit * 100,
                    risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                    expiration=t1_strike_obj.expiration,
                    dte=t1_strike_obj.dte,
                    is_fibonacci_based=True,
                    target_level="T1"
                )
                spreads.append(spread)
        
        # Strategy 2: Bull Put Spread (Sell puts at C level for EMA Reclaim setups)
        if analysis.ema_reclaim and analysis.iv_rank > 50 and 'C' in fib_strikes:
            c_strike_obj = fib_strikes['C']
            
            # Find a lower strike for the long put (5-10% OTM)
            lower_strike = min(strikes, key=lambda x: abs(x - (c_strike_obj.strike * 0.92)))
            lower_data = chain[(chain['strike'] == lower_strike) & (chain['type'] == 'put')]
            
            if not lower_data.empty:
                short_price = c_strike_obj.get_mid_price('put')
                long_price = (lower_data['bid'].iloc[0] + lower_data['ask'].iloc[0]) / 2
                
                net_credit = short_price - long_price
                max_profit = net_credit
                max_loss = (c_strike_obj.strike - lower_strike) - net_credit
                breakeven = c_strike_obj.strike - net_credit
                
                short_delta = c_strike_obj.put_delta
                long_delta = lower_data.get('delta', pd.Series([0])).iloc[0]
                prob_profit = abs(short_delta) if short_delta else 0.65
                
                spread = OptionsSpread(
                    strategy_name="Bull Put Spread (Fib C Support)",
                    description=f"Sell {c_strike_obj.strike} put (C level), Buy {lower_strike} put",
                    long_strike=lower_strike,
                    long_call_or_put="put",
                    long_price=long_price,
                    long_delta=long_delta,
                    short_strike=c_strike_obj.strike,
                    short_call_or_put="put",
                    short_price=short_price,
                    short_delta=short_delta,
                    net_debit=-net_credit,  # Negative for credit
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven=breakeven,
                    probability_profit=prob_profit * 100,
                    risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                    expiration=c_strike_obj.expiration,
                    dte=c_strike_obj.dte,
                    is_fibonacci_based=True,
                    target_level="C"
                )
                spreads.append(spread)
        
        # Strategy 3: Wide Call Spread (Buy ATM, Sell T2 for larger move)
        if "STRONG UPTREND" in analysis.trend and 'T2' in fib_strikes:
            t2_strike_obj = fib_strikes['T2']
            
            atm_call = atm_data[atm_data['type'] == 'call']
            if not atm_call.empty:
                long_price = (atm_call['bid'].iloc[0] + atm_call['ask'].iloc[0]) / 2
                short_price = t2_strike_obj.get_mid_price('call')
                
                net_debit = long_price - short_price
                max_profit = (t2_strike_obj.strike - atm_strike) - net_debit
                max_loss = net_debit
                breakeven = atm_strike + net_debit
                
                long_delta = atm_call.get('delta', pd.Series([0.5])).iloc[0]
                short_delta = t2_strike_obj.call_delta
                prob_profit = abs(long_delta - short_delta) if short_delta else abs(long_delta) * 0.5
                
                spread = OptionsSpread(
                    strategy_name="Bull Call Spread (Fib T2 Extended)",
                    description=f"Buy {atm_strike} call, Sell {t2_strike_obj.strike} call (T2 target)",
                    long_strike=atm_strike,
                    long_call_or_put="call",
                    long_price=long_price,
                    long_delta=long_delta,
                    short_strike=t2_strike_obj.strike,
                    short_call_or_put="call",
                    short_price=short_price,
                    short_delta=short_delta,
                    net_debit=net_debit,
                    max_profit=max_profit,
                    max_loss=max_loss,
                    breakeven=breakeven,
                    probability_profit=prob_profit * 100,
                    risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
                    expiration=t2_strike_obj.expiration,
                    dte=t2_strike_obj.dte,
                    is_fibonacci_based=True,
                    target_level="T2"
                )
                spreads.append(spread)
        
        return spreads
    
    def print_fibonacci_strikes(self, fib_strikes: Dict[str, OptionStrike]):
        """Print Fibonacci-based strikes in a formatted table"""
        if not fib_strikes:
            print("No Fibonacci strikes found.")
            return
        
        print(f"\n{'='*100}")
        print(f"FIBONACCI OPTIONS STRIKES: {self.ticker} @ ${self.current_price:.2f}")
        print(f"{'='*100}\n")
        
        print(f"{'Level':<8} {'Strike':<10} {'Distance':<12} {'DTE':<6} {'Call Bid/Ask':<18} {'Put Bid/Ask':<18} {'OI (C/P)'}")
        print(f"{'-'*8} {'-'*10} {'-'*12} {'-'*6} {'-'*18} {'-'*18} {'-'*15}")
        
        for level, strike_obj in fib_strikes.items():
            distance_str = f"{strike_obj.distance_from_spot:+.1f}%"
            call_ba = f"{strike_obj.call_bid:.2f}/{strike_obj.call_ask:.2f}"
            put_ba = f"{strike_obj.put_bid:.2f}/{strike_obj.put_ask:.2f}"
            oi_str = f"{strike_obj.call_open_interest}/{strike_obj.put_open_interest}"
            
            print(f"{level:<8} ${strike_obj.strike:<9.2f} {distance_str:<12} {strike_obj.dte:<6} {call_ba:<18} {put_ba:<18} {oi_str}")
        
        print(f"\n{'='*100}\n")
    
    def print_spread_suggestions(self, spreads: List[OptionsSpread]):
        """Print spread suggestions in formatted output"""
        if not spreads:
            print("No spread suggestions available.")
            return
        
        print(f"\n{'='*100}")
        print(f"FIBONACCI-BASED SPREAD SUGGESTIONS: {self.ticker}")
        print(f"{'='*100}\n")
        
        for i, spread in enumerate(spreads, 1):
            print(f"[{i}] {spread.strategy_name}")
            print(f"    {spread.description}")
            print(f"    Expiration: {spread.expiration.strftime('%Y-%m-%d')} ({spread.dte} DTE)")
            print(f"    Net {'Debit' if spread.net_debit > 0 else 'Credit'}: ${abs(spread.net_debit):.2f}")
            print(f"    Max Profit: ${spread.max_profit:.2f} | Max Loss: ${spread.max_loss:.2f}")
            print(f"    Breakeven: ${spread.breakeven:.2f}")
            print(f"    Risk/Reward: {spread.risk_reward_ratio:.2f} | Prob Profit: {spread.probability_profit:.1f}%")
            print()
        
        print(f"{'='*100}\n")

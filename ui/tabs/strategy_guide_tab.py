"""
Strategy Guide Tab
Educational content and strategy templates

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

# Import StrategyAdvisor with fallback
try:
    from analyzers.strategy import StrategyAdvisor
except ImportError:
    logger.debug("StrategyAdvisor not available")
    # Fallback - minimal mock with all required fields
    class StrategyAdvisor:
        STRATEGIES = {
            'iron_condor': {
                'name': 'Iron Condor',
                'description': 'Neutral strategy for range-bound markets',
                'experience': 'Intermediate',
                'risk_level': 'Limited',
                'max_loss': 'Width of spread - credit',
                'max_gain': 'Credit received',
                'best_conditions': ['Low volatility', 'Neutral market'],
                'best_for': ['Range-bound markets', 'Income generation'],
                'capital_required': 'Moderate',
                'capital_req': 'Moderate',  # Alternative field name
                'time_horizon': 'Short to medium term',
                'recommended_dte': '30-45 days',
                'adjustments': 'Can adjust strikes',
                'example': 'Example trade details',
                'typical_win_rate': '65-75%',
                'breakeven': 'Short strikes +/- credit',
                'greeks': 'Negative theta, positive vega',
                'when_to_use': 'When expecting range-bound movement',
                'when_to_avoid': 'During high volatility or trending markets'
            }
        }

def render_tab():
    """Main render function called from app.py"""
    st.header("Strategy Guide")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("📚 Complete Strategy Guide")
    
    for i, (strategy_key, strategy_info) in enumerate(StrategyAdvisor.STRATEGIES.items()):
        with st.expander(f"{strategy_info['name']} - {strategy_info['experience']} | {strategy_info['risk_level']} Risk"):
            st.write(f"**Description:** {strategy_info['description']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Risk Profile:**")
                st.write(f"• Risk: {strategy_info['risk_level']}")
                st.write(f"• Max Loss: {strategy_info['max_loss']}")
                st.write(f"• Max Gain: {strategy_info['max_gain']}")
                st.write(f"• Win Rate: {strategy_info['typical_win_rate']}")
            
            with col2:
                st.write("**Requirements:**")
                st.write(f"• Experience: {strategy_info['experience']}")
                st.write(f"• Capital: {strategy_info['capital_req']}")
                st.write("**Best For:**")
                for condition in strategy_info['best_for']:
                    st.write(f"• {condition}")

        st.divider()
        st.header("🎓 Options Education & Returns Calculator")
        st.write("Learn option terms, see example usages, and try a small returns calculator for common option structures.")

        # Use external pricing helpers (supports American option pricing via binomial tree)
        from services.options_pricing import black_scholes_price, binomial_american_price, greeks_finite_difference


        def calc_long_option_pnl(premium_paid: float, strike: float, underlying_price: float, contracts: int = 1, side: str = 'Call', use_bs: bool = False, days_to_expiry: int = 30, iv: float = 0.5, rf: float = 0.01):
            """Calculate basic P&L metrics for a long call/put bought for a single-leg option.

            Returns a dict with: position_value, pnl, roi_pct, breakeven
            """
            # position value for a call at expiry (intrinsic only)
            is_call = side.lower().startswith('c')
            intrinsic = max(0.0, underlying_price - strike) if is_call else max(0.0, strike - underlying_price)
            if use_bs:
                # Convert days to years
                T = max(days_to_expiry, 0) / 365.0
                theo = black_scholes_price(underlying_price, strike, T, rf, iv, is_call=is_call)
                position_value = theo * 100 * contracts
            else:
                position_value = intrinsic * 100 * contracts

            cost = premium_paid * 100 * contracts
            pnl = position_value - cost
            roi_pct = (pnl / cost * 100) if cost != 0 else 0.0
            breakeven = strike + premium_paid if is_call else strike - premium_paid
            return {
                'position_value': position_value,
                'pnl': pnl,
                'roi_pct': roi_pct,
                'breakeven': breakeven,
                'max_loss': -cost,
                'max_gain': 'Unlimited' if True else None
            }

        def calc_vertical_spread_pnl(premium_paid_long: float, premium_received_short: float, long_strike: float, short_strike: float, underlying_price: float, contracts: int = 1, is_call: bool = True, use_bs: bool = False, days_to_expiry: int = 30, iv_long: float = 0.5, iv_short: float = 0.5, rf: float = 0.01):
            """Calculate P&L for a vertical spread (debit or credit depending on premiums).

            Returns dict with pnl, max_loss, max_gain, roi_pct, breakeven
            """
            # net premium paid (debit positive means paid)
            net_premium = (premium_paid_long - premium_received_short)
            cost = net_premium * 100 * contracts

            # Use Black-Scholes theoretical pricing for legs if requested
            if use_bs:
                T = max(days_to_expiry, 0) / 365.0
                price_long = black_scholes_price(underlying_price, long_strike, T, rf, iv_long, is_call=is_call)
                price_short = black_scholes_price(underlying_price, short_strike, T, rf, iv_short, is_call=is_call)
                position_value = (price_long - price_short) * 100 * contracts if is_call else (price_short - price_long) * 100 * contracts
            else:
                # For a call vertical, intrinsic difference
                if is_call:
                    intrinsic_long = max(0.0, underlying_price - long_strike)
                    intrinsic_short = max(0.0, underlying_price - short_strike)
                else:
                    intrinsic_long = max(0.0, long_strike - underlying_price)
                    intrinsic_short = max(0.0, short_strike - underlying_price)

                # Position value at expiry = (intrinsic_long - intrinsic_short) * 100
                position_value = max(0.0, intrinsic_long - intrinsic_short) * 100 * contracts

            pnl = position_value - cost

            width = abs(short_strike - long_strike) * 100 * contracts
            # Max gain for a debit spread is width - cost (if you paid a net debit), for credit spread it's credit received minus assignment costs
            if net_premium >= 0:
                # net debit
                max_gain = max(0.0, width - cost)
                max_loss = -cost
            else:
                # net credit
                max_gain = -cost  # credit received
                max_loss = -(width + cost)  # worst-case if spread assigned against you

            roi_pct = (pnl / abs(cost) * 100) if cost != 0 else 0.0

            # Breakeven approximations
            if is_call:
                breakeven = long_strike + net_premium
            else:
                breakeven = long_strike - net_premium

            return {
                'position_value': position_value,
                'pnl': pnl,
                'max_gain': max_gain,
                'max_loss': max_loss,
                'roi_pct': roi_pct,
                'breakeven': breakeven,
                'net_premium': net_premium
            }


        # --- UI wiring: Load from selected strategy, persist small calc history ---
        if 'calc_history' not in st.session_state:
            st.session_state.calc_history = []


        edu_col1, edu_col2 = st.columns([1, 1])

        with edu_col1:
            st.subheader("Key Option Terms")
            st.markdown("""
            - Strike: The agreed exercise price for the option.
            - Premium: Price paid to buy the option (per share).
            - DTE: Days to expiration.
            - IV / IV Rank: Implied volatility; higher IV usually means higher option prices.
            - Intrinsic Value: The in-the-money amount (if any).
            - Time Value: Portion of premium attributable to time/volatility.
            - Breakeven: Underlying price at expiry where the trade neither makes nor loses money.
            """, unsafe_allow_html=True)

            st.subheader("When to use")
            st.write("Long calls/puts: directional plays with limited risk (premium). Good when expecting big moves.")
            st.write("Vertical spreads: reduce cost and cap profit; useful for defined-risk directional or neutral trades.")

        with edu_col2:
            st.subheader("Try the Returns Calculator")
            # Use a unique key for this educational calculator to avoid collisions with other widgets
            calc_type = st.selectbox("Structure", ["Long Option", "Vertical Spread"], key=f"tab5_edu_calc_type_selectbox_{i}")

            if calc_type == "Long Option":
                # Prefill from example_trade if available
                ex = st.session_state.get('example_trade', {})
                side = st.selectbox("Side", ["Call", "Put"], index=0, key=f'tab5_longoption_side_select_{i}')
                premium = st.number_input("Premium (per share, $)", min_value=0.0, value=float(ex.get('premium', 1.50)), step=0.01, format="%.2f", key=f"premium_{i}")
                strike = st.number_input("Strike ($)", min_value=0.01, value=float(ex.get('strike', 50.0)), step=0.01, format="%.2f", key=f"strike_{i}")
                underlying = st.number_input("Underlying Price at Expiry ($)", min_value=0.0, value=float(ex.get('underlying', strike + 5)), step=0.01, format="%.2f", key=f"underlying_{i}")
                contracts = st.number_input("Contracts", min_value=1, max_value=100, value=int(ex.get('qty', 1)), key=f"contracts_{i}")
                days_to_expiry = st.number_input("Days to Expiry", min_value=0, max_value=365, value=int(ex.get('dte', 30)), key=f"days_to_expiry_{i}")
                iv = st.number_input("Implied Volatility (annual %, e.g. 50 for 50%)", min_value=0.0, max_value=500.0, value=float(ex.get('iv_pct', 50.0)), step=0.1, key=f"iv_{i}") / 100.0
                model_choice = st.radio("Pricing model", ["American (binomial)", "European (Black-Scholes)"], index=0, key=f"model_choice_{i}")
                use_bs = (model_choice == "European (Black-Scholes)")
                # Allow user to tune binomial steps for American pricing
                # Use a session_state-backed slider so we can auto-apply recommended steps
                if 'binomial_steps' not in st.session_state:
                    st.session_state.binomial_steps = 300
                binomial_steps = st.slider("Binomial steps (American model accuracy)", min_value=10, max_value=5000, value=st.session_state.binomial_steps, step=10, key=f'binomial_steps_{i}', help="More steps -> higher accuracy but slower.")

                # Convergence score weighting: let user trade off time vs accuracy
                st.subheader("Convergence tuning")
                weight_time = st.slider("Weight: time vs accuracy (0 = ignore time, 1 = equal, >1 = favor time)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key=f"weight_time_{i}", help="Higher values increase the importance of time in the combined conv_score.")

                # Named wrapper for binomial pricer to pass into greeks wrapper (helps tracebacks)
                def binomial_pricer(S_loc: float, K_loc: float, T_loc: float, r_loc: float, sigma_loc: float, is_call_loc: bool = True):
                    return binomial_american_price(S_loc, K_loc, T_loc, r_loc, sigma_loc, is_call=is_call_loc, steps=binomial_steps)

                # Sensitivity explorer: quick bumps for IV and underlying to show P&L / delta
                st.subheader("Sensitivity explorer")
                col_a, col_b = st.columns(2)
                with col_a:
                    bump_iv = st.number_input("Bump IV (pts, e.g. 1 = +1%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1, key=f"bump_iv_{i}")
                with col_b:
                    bump_underlying = st.number_input("Bump underlying ($)", min_value=-1000.0, max_value=1000.0, value=0.0, step=0.01, key=f"bump_underlying_{i}")

                if st.button("Load from selected strategy/example", key=f"sensitivity_load_example_{i}"):
                    # If an example_trade exists, prefill fields (session state will supply next rerun)
                    if 'example_trade' in st.session_state:
                        ex2 = st.session_state.example_trade
                        st.session_state._calc_prefill = {
                            'side': 'Call',
                            'premium': ex2.get('premium', ex2.get('estimated_risk', 1.5) / 100.0),
                            'strike': ex2.get('strike'),
                            'underlying': ex2.get('strike'),
                            'qty': ex2.get('qty', 1),
                            'dte': 30,
                            'iv_pct': ex2.get('iv_rank', 48.0)
                        }
                        # We no longer call experimental_rerun for compatibility.
                        st.success("Example trade loaded into session state. Adjust values if needed then click Calculate.")

                if st.button("Calculate Long Option P&L", key=f"calculate_long_{i}"):
                    res = calc_long_option_pnl(premium, strike, underlying, int(contracts), side, use_bs, int(days_to_expiry), float(iv), rf=0.01)
                    st.write(f"Position value: ${res['position_value']:.2f}")
                    st.write(f"P&L: ${res['pnl']:.2f}")
                    st.write(f"ROI: {res['roi_pct']:.1f}%")
                    st.write(f"Breakeven: ${res['breakeven']:.2f}")
                    st.write(f"Max Loss: ${res['max_loss']:.2f}")
                    # Show greeks depending on model
                    T = max(int(days_to_expiry), 0) / 365.0
                    is_call_flag = side.lower().startswith('c')
                    if use_bs:
                        # greeks_finite_difference will detect black_scholes_price and return analytic greeks
                        greeks = greeks_finite_difference(black_scholes_price, underlying, strike, T, 0.01, iv, is_call=is_call_flag)
                        model_note = "European Black-Scholes (analytic greeks)"
                    else:
                        # Binomial American greeks via finite diffs around the named binomial_pricer
                        greeks = greeks_finite_difference(binomial_pricer, underlying, strike, T, 0.01, iv, is_call=is_call_flag)
                        model_note = f"American (binomial CRR, steps={binomial_steps}) — supports early exercise"

                    # vega in our API is per 1 vol point (1% = +1). Also show per 0.01 (decimal) for clarity.
                    vega_per_1pt = greeks['vega']
                    vega_per_decimal = greeks['vega'] / 100.0
                    # Absolute dollar vega for the given number of contracts (per 1 vol pt)
                    contracts_int = int(contracts)
                    vega_dollars = vega_per_1pt * 100.0 * contracts_int
                    # percent of cost (cost = premium * 100 * contracts)
                    cost = premium * 100.0 * contracts_int
                    pct_of_cost = (vega_dollars / cost * 100.0) if cost != 0 else None

                    st.write(f"Delta: {greeks['delta']:.3f} • Gamma: {greeks['gamma']:.4f} • Vega (per 1 vol pt): {vega_per_1pt:.2f} • Vega (per 0.01 decimal): {vega_per_decimal:.4f} • Theta/day: {greeks['theta']:.2f}")
                    st.write(f"Vega absolute: ${vega_dollars:.2f} per 1 vol pt for {contracts_int} contract(s)" + (f" • {pct_of_cost:.1f}% of cost" if pct_of_cost is not None else ""))
                    st.info(f"Note: {model_note}. Results are approximations; Black-Scholes assumes European exercise and constant vol. American pricing uses a binomial tree numeric method.")
                    # Save to history
                    st.session_state.calc_history.append({
                        'type': 'long', 'side': side, 'premium': premium, 'strike': strike,
                        'underlying': underlying, 'contracts': int(contracts), 'dte': int(days_to_expiry), 'iv': iv, 'use_bs': bool(use_bs), 'binomial_steps': int(binomial_steps),
                        'result_pnl': res['pnl'], 'result_roi': res['roi_pct'], 'timestamp': datetime.now().isoformat()
                    })

                    # Sensitivity outputs: simple bump calculations
                    if bump_iv != 0.0 or bump_underlying != 0.0:
                        bumped_iv = iv + (bump_iv / 100.0)
                        bumped_under = underlying + bump_underlying
                        # Price with bumps
                        if use_bs:
                            base_price = black_scholes_price(underlying, strike, T, 0.01, iv, is_call=is_call_flag)
                            bumped_price = black_scholes_price(bumped_under, strike, T, 0.01, bumped_iv, is_call=is_call_flag)
                        else:
                            base_price = binomial_american_price(underlying, strike, T, 0.01, iv, is_call=is_call_flag, steps=binomial_steps)
                            bumped_price = binomial_american_price(bumped_under, strike, T, 0.01, bumped_iv, is_call=is_call_flag, steps=binomial_steps)

                        pnl_base = (base_price - premium) * 100 * int(contracts)
                        pnl_bumped = (bumped_price - premium) * 100 * int(contracts)
                        st.subheader("Sensitivity results")
                        st.write(f"Base theoretical price: ${base_price:.2f} — P&L: ${pnl_base:.2f}")
                        st.write(f"Bumped price: ${bumped_price:.2f} — P&L: ${pnl_bumped:.2f}")

                    # Micro-benchmark: convergence test for binomial pricer
                    if not use_bs:
                        st.subheader("Binomial convergence micro-benchmark")

                        # Benchmark caps
                        max_steps_allowed = st.number_input("Max benchmark steps cap", min_value=100, max_value=10000, value=5000, step=100, key=f"max_steps_allowed_{i}")
                        max_runtime_seconds = st.number_input("Max benchmark runtime (s)", min_value=1.0, max_value=120.0, value=10.0, step=1.0, key=f"max_runtime_seconds_{i}")

                        if 'benchmark_status' not in st.session_state:
                            st.session_state.benchmark_status = {'running': False, 'results': None, 'started_at': None}

                        def _run_benchmark_background(underlying_loc, strike_loc, T_loc, iv_loc, is_call_loc, step_list_loc, max_runtime_loc):
                            import time as _time
                            res = []
                            t_start = _time.perf_counter()
                            for s in step_list_loc:
                                # Check runtime cap
                                if (_time.perf_counter() - t_start) > max_runtime_loc:
                                    break
                                # Respect step cap
                                if s > max_steps_allowed:
                                    break
                                t0 = _time.perf_counter()
                                p = binomial_american_price(underlying_loc, strike_loc, T_loc, 0.01, iv_loc, is_call=is_call_loc, steps=s)
                                t1 = _time.perf_counter()
                                res.append({'steps': s, 'price': p, 'time_s': t1 - t0})

                            # Save results and mark not running
                            st.session_state.benchmark_status['results'] = res
                            st.session_state.benchmark_status['running'] = False
                            st.session_state.benchmark_status['started_at'] = None

                        # Start benchmark async
                        tol_pct = st.number_input("Tolerance (relative %)", min_value=1e-6, max_value=100.0, value=0.1, step=0.01, key=f"tol_pct_{i}", help="Stop when relative price change between successive runs is below this percentage (e.g. 0.1 = 0.1%).")
                        st.caption("Tolerance is relative to the previous run's price. Lower values require tighter convergence and more computation. For very small option prices, consider increasing tolerance to avoid excessive runs.")

                        if st.button("Run convergence benchmark (background)", key=f"run_benchmark_{i}"):
                            import threading as _threading
                            # Build step list, include selected binomial_steps and some canonical points
                            step_list = [50, 200, 500, 1000, max(2000, binomial_steps)]
                            step_list = sorted(list(dict.fromkeys(step_list)))
                            st.session_state.benchmark_status['running'] = True
                            st.session_state.benchmark_status['results'] = None
                            st.session_state.benchmark_status['started_at'] = __import__('time').time()
                            # initialize incremental results container
                            st.session_state.benchmark_status['results'] = []
                            st.session_state.benchmark_status['running'] = True
                            st.session_state.benchmark_status['started_at'] = __import__('time').time()

                            # wrap the background runner to append interim results
                            def _bg_runner_append(*args, **kwargs):
                                import time as _time
                                # call into the original runner but append results as they come
                                underlying_loc, strike_loc, T_loc, iv_loc, is_call_loc, steps_loc, max_runtime_loc = args
                                t_start = _time.perf_counter()
                                for s in steps_loc:
                                    if (_time.perf_counter() - t_start) > max_runtime_loc:
                                        break
                                    if s > max_steps_allowed:
                                        break
                                    t0 = _time.perf_counter()
                                    p = binomial_american_price(underlying_loc, strike_loc, T_loc, 0.01, iv_loc, is_call=is_call_loc, steps=s)
                                    t1 = _time.perf_counter()
                                    # append result incrementally
                                    st.session_state.benchmark_status['results'].append({'steps': s, 'price': p, 'time_s': t1 - t0})
                                # finished
                                st.session_state.benchmark_status['running'] = False
                                st.session_state.benchmark_status['started_at'] = None

                            th = _threading.Thread(target=_bg_runner_append, args=(underlying, strike, T, iv, is_call_flag, step_list, float(max_runtime_seconds)), daemon=True)
                            th.start()

                        # Adaptive benchmark: auto-increase steps until price change < tol or caps reached
                        growth_strategy = st.selectbox("Adaptive growth strategy", ["doubling", "multiply", "additive"], index=0, key=f'tab5_growth_strategy_select_{i}', help="How step counts increase between iterations")
                        growth_param = st.number_input("Growth parameter", min_value=1.1, max_value=10.0, value=2.0, step=0.1, key=f"growth_param_{i}", help="Multiplier for 'multiply' strategy, ignored for doubling; for additive, this is the additive step size")

                        if st.button("Run adaptive convergence (background)", key=f"run_adaptive_{i}"):
                            import threading as _threading
                            def _run_adaptive(underlying_loc, strike_loc, T_loc, iv_loc, is_call_loc, start_steps, tol_loc, max_runtime_loc, strategy, param):
                                import time as _time
                                res = []
                                t_start = _time.perf_counter()
                                # Start from start_steps and double until tolerance met
                                s = max(50, start_steps)
                                last_price = None
                                while True:
                                    if (_time.perf_counter() - t_start) > max_runtime_loc:
                                        break
                                    if s > max_steps_allowed:
                                        break
                                    t0 = _time.perf_counter()
                                    p = binomial_american_price(underlying_loc, strike_loc, T_loc, 0.01, iv_loc, is_call=is_call_loc, steps=s)
                                    t1 = _time.perf_counter()
                                    res.append({'steps': s, 'price': p, 'time_s': t1 - t0})
                                    if last_price is not None:
                                        # use relative change: |p - last| / |last| < tol_loc
                                        if last_price != 0:
                                            rel = abs(p - last_price) / abs(last_price)
                                        else:
                                            rel = abs(p - last_price)
                                        if rel < tol_loc:
                                            break
                                    last_price = p
                                    # Increase according to chosen strategy
                                    if strategy == 'doubling':
                                        if s < 100:
                                            s *= 2
                                        else:
                                            s = int(s * 1.5)
                                    elif strategy == 'multiply':
                                        s = int(s * float(param))
                                    else:  # additive
                                        s = int(s + float(param))

                                st.session_state.benchmark_status['results'] = res
                                st.session_state.benchmark_status['running'] = False
                                st.session_state.benchmark_status['started_at'] = None

                            st.session_state.benchmark_status['running'] = True
                            st.session_state.benchmark_status['results'] = None
                            st.session_state.benchmark_status['started_at'] = __import__('time').time()
                            # pass tolerance as decimal fraction to the adaptive runner
                            th2 = _threading.Thread(target=_run_adaptive, args=(underlying, strike, T, iv, is_call_flag, binomial_steps, float(tol_pct) / 100.0, float(max_runtime_seconds), growth_strategy, float(growth_param)), daemon=True)
                            th2.start()

                        # Show status
                        auto_refresh = st.checkbox("Auto-refresh progress", value=True)

                        if st.session_state.benchmark_status.get('running'):
                            st.info("Benchmark running in background...")

                        # Show intermediate results table if present
                        results = st.session_state.benchmark_status.get('results') or []
                        if results:
                            import pandas as _pd
                            df_partial = _pd.DataFrame(results)
                            st.write("Intermediate results:")
                            st.table(df_partial)

                        # Auto-refresh logic: try to rerun if available, otherwise show Refresh button
                        # Only auto-refresh if explicitly enabled by user
                        if auto_refresh and st.session_state.benchmark_status.get('running'):
                            rerun_fn = getattr(st, 'rerun', None)
                            if callable(rerun_fn):
                                import time as _time
                                _time.sleep(0.5)
                                try:
                                    # Limit reruns to prevent performance issues
                                    if st.session_state.get('benchmark_refresh_count', 0) < 100:
                                        st.session_state.benchmark_refresh_count = st.session_state.get('benchmark_refresh_count', 0) + 1
                                        rerun_fn()
                                except Exception:
                                    pass
                            else:
                                if st.button("Refresh progress", key=f"refresh_progress_{i}"):
                                    pass

                        # If benchmark finished, show final results and plots
                        if not st.session_state.benchmark_status.get('running'):
                            results_final = st.session_state.benchmark_status.get('results') or []
                            if results_final:
                                # Baseline is last price
                                baseline = results_final[-1]['price']
                                st.write("Convergence results (price, time):")
                                for r in results_final:
                                    st.write(f"steps={r['steps']:5d}  price=${r['price']:.4f}  dt={r['time_s']*1000:.1f}ms  delta={r['price']-baseline:+.4f}")

                                # Plot price vs steps and time vs steps and show tolerance line
                                try:
                                    import pandas as _pd
                                    df = _pd.DataFrame(results_final)
                                    df = df.set_index('steps')

                                    # Price convergence plot with tolerance line
                                    try:
                                        import matplotlib.pyplot as _plt
                                        fig, ax = _plt.subplots()
                                        ax.plot(df.index, df['price'], marker='o')
                                        # add tolerance line around baseline (use list of colors and accepted linestyle)
                                        # draw tolerance band as relative fraction of baseline
                                        tol_frac = float(tol_pct) / 100.0
                                        ax.hlines([baseline * (1.0 + tol_frac), baseline * (1.0 - tol_frac)], xmin=df.index.min(), xmax=df.index.max(), colors=['r', 'r'], linestyles='dashed')
                                        ax.set_xlabel('steps')
                                        ax.set_ylabel('price')
                                        ax.set_title('Price vs Steps (convergence)')
                                        st.pyplot(fig)
                                    except Exception:
                                        # Fallback to simple Streamlit line chart
                                        st.line_chart(df['price'])

                                    # Time plot
                                    st.line_chart(df['time_s'])

                                    # Combined convergence metric: relative change * time per (unit steps)
                                    try:
                                        dfc = df.copy()
                                        # relative absolute change between successive prices
                                        dfc['rel_change'] = dfc['price'].pct_change().abs().fillna(0.0)
                                        # avoid divide by zero for index cast
                                        idx_vals = _pd.Series(dfc.index.astype(float)).replace(0.0, 1.0).values
                                        dfc['time_per_step'] = dfc['time_s'] / idx_vals
                                        # convergence score: rel_change * time_per_step (lower is better)
                                        # apply user weight: conv_score = rel_change * (time_per_step ** weight_time)
                                        dfc['conv_score'] = dfc['rel_change'] * (dfc['time_per_step'] ** float(weight_time))
                                        # show table with new metrics
                                        st.write("Convergence metrics (lower conv_score is better):")
                                        st.dataframe(dfc[['price', 'rel_change', 'time_per_step', 'conv_score']])
                                        # recommend step minimizing conv_score
                                        best_idx = dfc['conv_score'].idxmin()
                                        best_row = dfc.loc[best_idx]
                                        st.success(f"Recommended steps: {int(best_idx)} (conv_score={best_row['conv_score']:.6g}, price=${best_row['price']:.4f})")
                                        if st.button("Auto-apply recommended steps"):
                                            # set the session_state slider value to recommended steps (clamped to slider bounds)
                                            new_steps = int(best_idx)
                                            new_steps = max(10, min(5000, new_steps))
                                            st.session_state['binomial_steps'] = new_steps
                                            rerun_fn = getattr(st, 'rerun', None)
                                            if callable(rerun_fn):
                                                try:
                                                    rerun_fn()
                                                except Exception:
                                                    pass
                                        # plot conv_score
                                        st.line_chart(dfc['conv_score'])
                                    except Exception:
                                        pass
                                except Exception:
                                    st.write("Benchmark finished but plotting failed.")
                            else:
                                st.write("Benchmark finished but returned no results (capped). Try increasing caps.")

            else:
                # Vertical spread inputs
                is_call = st.radio("Type", ["Call Vertical", "Put Vertical"], key=f"is_call_{i}") == "Call Vertical"
                ex = st.session_state.get('example_trade', {})
                long_strike = st.number_input("Long Strike ($)", min_value=0.01, value=float(ex.get('long_strike', 48.0)), step=0.01, format="%.2f", key=f"long_strike_{i}")
                short_strike = st.number_input("Short Strike ($)", min_value=0.01, value=float(ex.get('short_strike', 52.0)), step=0.01, format="%.2f", key=f"short_strike_{i}")
                premium_long = st.number_input("Premium paid for long leg ($)", min_value=0.0, value=float(ex.get('premium_long', 2.00)), step=0.01, format="%.2f", key=f"premium_long_{i}")
                premium_short = st.number_input("Premium received for short leg ($)", min_value=0.0, value=float(ex.get('premium_short', 0.50)), step=0.01, format="%.2f", key=f"premium_short_{i}")
                underlying_price = st.number_input("Underlying Price at Expiry ($)", min_value=0.0, value=float(ex.get('underlying', (long_strike + short_strike) / 2)), step=0.01, format="%.2f", key=f"underlying_price_{i}")
                contracts_vs = st.number_input("Contracts", min_value=1, max_value=100, value=int(ex.get('qty', 1)), key=f"contracts_vs_{i}")
                days_to_expiry = st.number_input("Days to Expiry", min_value=0, max_value=365, value=int(ex.get('dte', 30)), key=f"days_to_expiry_vs_{i}")
                iv_long = st.number_input("IV Long (%)", min_value=0.0, max_value=500.0, value=float(ex.get('iv_long', 50.0)), step=0.1, key=f"iv_long_{i}") / 100.0
                iv_short = st.number_input("IV Short (%)", min_value=0.0, max_value=500.0, value=float(ex.get('iv_short', 45.0)), step=0.1, key=f"iv_short_{i}") / 100.0
                use_bs_vs = st.checkbox("Use Black-Scholes for theoretical leg pricing", value=False, key=f"use_bs_vs_{i}")

                if st.button("Calculate Vertical Spread P&L", key=f"calculate_vertical_{i}"):
                    res = calc_vertical_spread_pnl(premium_long, premium_short, long_strike, short_strike, underlying_price, int(contracts_vs), is_call=is_call, use_bs=bool(use_bs_vs), days_to_expiry=int(days_to_expiry), iv_long=float(iv_long), iv_short=float(iv_short), rf=0.01)
                    st.write(f"Position value: ${res['position_value']:.2f}")
                    st.write(f"P&L: ${res['pnl']:.2f}")
                    st.write(f"Net Premium (paid): ${res['net_premium']*100:.2f}")
                    st.write(f"Max Gain: ${res['max_gain']:.2f}")
                    st.write(f"Max Loss: ${res['max_loss']:.2f}")
                    st.write(f"ROI: {res['roi_pct']:.1f}%")
                    st.write(f"Breakeven: ${res['breakeven']:.2f}")
                    st.info("Note: This simplifies assignment and ignores early assignment, transaction costs and margin effects.")
                    st.session_state.calc_history.append({
                        'type': 'vertical', 'long_strike': long_strike, 'short_strike': short_strike,
                        'premium_long': premium_long, 'premium_short': premium_short, 'underlying': underlying_price,
                        'contracts': int(contracts_vs), 'dte': int(days_to_expiry), 'iv_long': iv_long, 'iv_short': iv_short,
                        'use_bs': bool(use_bs_vs), 'result_pnl': res['pnl'], 'result_roi': res['roi_pct'], 'timestamp': datetime.now().isoformat()
                    })

            # Calculation history and export
            st.subheader("Calculation History")
            if st.session_state.calc_history:
                hist_df = pd.DataFrame(st.session_state.calc_history)
                st.dataframe(hist_df.sort_values('timestamp', ascending=False).reset_index(drop=True), width='stretch')
                if st.button("Export History CSV", key=f"export_csv_{i}"):
                    # Add UTF-8 BOM for Excel compatibility
                    csv = '\ufeff' + hist_df.to_csv(index=False)
                    st.download_button("Download CSV", csv.encode('utf-8-sig'), "calc_history.csv", "text/csv")
            else:
                st.info("No calculations yet — run one above to populate history.")



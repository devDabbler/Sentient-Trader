"""
Quant Analytics Tab
Options Greeks, Portfolio Risk, and Strategy Backtesting

Provides institutional-grade analytics for stocks and options trading.
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio

# Import the quant analytics service
try:
    from services.quant_analytics import (
        QuantAnalyticsService, 
        get_quant_service,
        OptionType,
        Greeks,
        RiskMetrics,
        BacktestResult
    )
    QUANT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Quant analytics not available: {e}")
    QUANT_AVAILABLE = False


def render_tab():
    """Main render function for Quant Analytics tab"""
    st.header("📊 Quant Analytics")
    st.caption("Institutional-grade risk analytics and backtesting for stocks & options")
    
    if not QUANT_AVAILABLE:
        st.error("Quant Analytics service is not available. Check logs for import errors.")
        return
    
    # Get quant service
    quant = get_quant_service()
    
    # Show GS Quant status
    if quant.gs_quant_available:
        st.success("✅ GS Quant library available (institutional analytics)")
    else:
        st.info("ℹ️ Running in pure Python mode (Black-Scholes implementation)")
    
    # Create sub-tabs
    subtab1, subtab2, subtab3 = st.tabs([
        "🎯 Options Greeks Calculator",
        "⚠️ Portfolio Risk Dashboard", 
        "📈 Strategy Backtester"
    ])
    
    with subtab1:
        render_greeks_calculator(quant)
    
    with subtab2:
        render_risk_dashboard(quant)
    
    with subtab3:
        render_backtester(quant)


def render_greeks_calculator(quant: QuantAnalyticsService):
    """Render the Options Greeks Calculator section"""
    st.subheader("Options Greeks Calculator")
    st.write("Calculate Delta, Gamma, Theta, Vega, and Rho for any option contract.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option Parameters**")
        underlying_price = st.number_input(
            "Underlying Price ($)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Current price of the underlying stock"
        )
        
        strike = st.number_input(
            "Strike Price ($)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Strike price of the option"
        )
        
        days_to_expiry = st.number_input(
            "Days to Expiration",
            min_value=1,
            max_value=730,
            value=30,
            help="Number of days until option expires"
        )
        
        implied_volatility = st.slider(
            "Implied Volatility (%)",
            min_value=5,
            max_value=200,
            value=30,
            help="Implied volatility as percentage"
        ) / 100.0
    
    with col2:
        st.markdown("**Additional Parameters**")
        option_type = st.radio(
            "Option Type",
            ["CALL", "PUT"],
            horizontal=True
        )
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.25,
            help="Annual risk-free interest rate"
        ) / 100.0
        
        dividend_yield = st.slider(
            "Dividend Yield (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.25,
            help="Annual dividend yield"
        ) / 100.0
    
    if st.button("📊 Calculate Greeks", type="primary", key="calc_greeks"):
        with st.spinner("Calculating..."):
            opt_type = OptionType.CALL if option_type == "CALL" else OptionType.PUT
            
            greeks = quant.calculate_greeks(
                underlying_price=underlying_price,
                strike=strike,
                days_to_expiry=days_to_expiry,
                implied_volatility=implied_volatility,
                risk_free_rate=risk_free_rate,
                option_type=opt_type,
                dividend_yield=dividend_yield
            )
            
            # Also calculate theoretical price
            option_price = quant.calculate_option_price(
                underlying_price=underlying_price,
                strike=strike,
                days_to_expiry=days_to_expiry,
                implied_volatility=implied_volatility,
                risk_free_rate=risk_free_rate,
                option_type=opt_type,
                dividend_yield=dividend_yield
            )
            
            st.success("✅ Greeks calculated successfully!")
            
            # Display results
            st.markdown("---")
            st.markdown(f"### Theoretical {option_type} Option Price: **${option_price:.2f}**")
            
            # Greeks metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            
            with m1:
                delta_color = "normal" if -1 <= greeks.delta <= 1 else "off"
                st.metric("Delta (Δ)", f"{greeks.delta:.4f}", help="Price sensitivity to underlying movement")
            
            with m2:
                st.metric("Gamma (Γ)", f"{greeks.gamma:.6f}", help="Delta sensitivity to underlying movement")
            
            with m3:
                theta_display = f"${greeks.theta:.4f}"
                st.metric("Theta (Θ)", theta_display, help="Daily time decay (negative = losing value)")
            
            with m4:
                st.metric("Vega (ν)", f"{greeks.vega:.4f}", help="Sensitivity to 1% change in IV")
            
            with m5:
                st.metric("Rho (ρ)", f"{greeks.rho:.4f}", help="Sensitivity to interest rate change")
            
            # Interpretation
            with st.expander("📖 Greeks Interpretation", expanded=True):
                interpretations = []
                
                if opt_type == OptionType.CALL:
                    if greeks.delta > 0.7:
                        interpretations.append("**Delta > 0.7**: Deep ITM call - behaves almost like stock")
                    elif greeks.delta > 0.5:
                        interpretations.append("**Delta 0.5-0.7**: ITM call - good directional exposure")
                    elif greeks.delta > 0.3:
                        interpretations.append("**Delta 0.3-0.5**: ATM call - balanced risk/reward")
                    else:
                        interpretations.append("**Delta < 0.3**: OTM call - leveraged bet, higher risk")
                else:
                    if greeks.delta < -0.7:
                        interpretations.append("**Delta < -0.7**: Deep ITM put - strong downside protection")
                    elif greeks.delta < -0.5:
                        interpretations.append("**Delta -0.5 to -0.7**: ITM put - good hedge")
                    elif greeks.delta < -0.3:
                        interpretations.append("**Delta -0.3 to -0.5**: ATM put - balanced")
                    else:
                        interpretations.append("**Delta > -0.3**: OTM put - cheap insurance")
                
                if greeks.gamma > 0.05:
                    interpretations.append("**High Gamma**: Delta will change rapidly with price movement")
                
                if greeks.theta < -0.10:
                    interpretations.append("**High Theta Decay**: Losing significant value daily - time is against you")
                
                if greeks.vega > 0.20:
                    interpretations.append("**High Vega**: Very sensitive to IV changes - volatility crush risk")
                
                for interp in interpretations:
                    st.markdown(f"• {interp}")


def render_risk_dashboard(quant: QuantAnalyticsService):
    """Render the Portfolio Risk Dashboard section"""
    st.subheader("Portfolio Risk Dashboard")
    st.write("Analyze portfolio-level risk metrics including VaR and aggregated Greeks.")
    
    # Initialize positions in session state
    if 'quant_positions' not in st.session_state:
        st.session_state.quant_positions = []
    
    # Add position form
    with st.expander("➕ Add Position", expanded=len(st.session_state.quant_positions) == 0):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.text_input("Symbol", value="SPY", key="pos_symbol").upper()
            quantity = st.number_input("Quantity", min_value=1, value=100, key="pos_qty")
        
        with col2:
            market_value = st.number_input("Market Value ($)", min_value=0.0, value=10000.0, key="pos_value")
            delta = st.number_input("Delta (per share)", value=1.0, step=0.1, key="pos_delta", 
                                   help="Use 1.0 for stocks, or option delta")
        
        with col3:
            gamma = st.number_input("Gamma", value=0.0, step=0.01, key="pos_gamma")
            theta = st.number_input("Theta (daily)", value=0.0, step=0.01, key="pos_theta")
            vega = st.number_input("Vega", value=0.0, step=0.01, key="pos_vega")
        
        if st.button("Add Position", key="add_pos"):
            st.session_state.quant_positions.append({
                'symbol': symbol,
                'quantity': quantity,
                'market_value': market_value,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega
            })
            st.success(f"Added {quantity} {symbol}")
            st.rerun()
    
    # Display current positions
    if st.session_state.quant_positions:
        st.markdown("### Current Positions")
        
        # Create a table
        import pandas as pd
        df = pd.DataFrame(st.session_state.quant_positions)
        df['Total Delta'] = df['delta'] * df['quantity']
        df['Total Theta'] = df['theta'] * df['quantity']
        
        st.dataframe(df, use_container_width=True)
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear All", key="clear_positions"):
                st.session_state.quant_positions = []
                st.rerun()
        
        # Calculate risk metrics
        st.markdown("---")
        st.markdown("### Risk Metrics")
        
        risk = quant.calculate_portfolio_risk(st.session_state.quant_positions)
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("Portfolio Value", f"${risk.portfolio_value:,.2f}")
            st.metric("Total Delta", f"{risk.total_delta:,.2f}")
        
        with m2:
            st.metric("VaR (95%)", f"${risk.var_95:,.2f}", 
                     help="Maximum expected loss with 95% confidence (1-day)")
            st.metric("Total Gamma", f"{risk.total_gamma:,.6f}")
        
        with m3:
            st.metric("VaR (99%)", f"${risk.var_99:,.2f}",
                     help="Maximum expected loss with 99% confidence (1-day)")
            st.metric("Total Theta", f"${risk.total_theta:,.2f}/day")
        
        with m4:
            st.metric("Max Drawdown", f"{risk.max_drawdown * 100:.2f}%")
            st.metric("Total Vega", f"{risk.total_vega:,.2f}")
        
        # Risk interpretation
        with st.expander("📖 Risk Interpretation"):
            st.markdown(f"""
            **Portfolio Summary:**
            - **Net Delta Exposure**: {risk.total_delta:,.2f} 
              {"(Bullish)" if risk.total_delta > 0 else "(Bearish)" if risk.total_delta < 0 else "(Neutral)"}
            - **Daily Theta Decay**: ${abs(risk.total_theta):,.2f} 
              {"(Collecting premium)" if risk.total_theta > 0 else "(Paying premium)"}
            - **Volatility Exposure**: {risk.total_vega:,.2f} Vega
              {"(Benefits from IV expansion)" if risk.total_vega > 0 else "(Benefits from IV contraction)"}
            
            **Value at Risk (VaR):**
            - There is a 5% chance of losing more than **${risk.var_95:,.2f}** in a single day
            - There is a 1% chance of losing more than **${risk.var_99:,.2f}** in a single day
            """)
    else:
        st.info("Add positions above to calculate portfolio risk metrics.")
        
        # Show sample portfolio
        if st.button("Load Sample Portfolio"):
            st.session_state.quant_positions = [
                {'symbol': 'SPY', 'quantity': 100, 'market_value': 45000, 'delta': 1.0, 'gamma': 0, 'theta': 0, 'vega': 0},
                {'symbol': 'SPY 450C', 'quantity': 2, 'market_value': 1200, 'delta': 0.65, 'gamma': 0.02, 'theta': -0.15, 'vega': 0.25},
                {'symbol': 'AAPL', 'quantity': 50, 'market_value': 9500, 'delta': 1.0, 'gamma': 0, 'theta': 0, 'vega': 0},
            ]
            st.rerun()


def render_backtester(quant: QuantAnalyticsService):
    """Render the Strategy Backtester section"""
    st.subheader("Strategy Backtester")
    st.write("Backtest trading strategies on historical data with comprehensive metrics.")
    
    # Get available strategies
    strategies = quant.get_available_strategies()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strategy Selection**")
        
        strategy_options = {s['name']: f"{s['name']} - {s['description']}" for s in strategies}
        selected_strategy = st.selectbox(
            "Strategy",
            options=list(strategy_options.keys()),
            format_func=lambda x: strategy_options[x],
            key="bt_strategy"
        )
        
        # Show strategy details
        selected_info = next((s for s in strategies if s['name'] == selected_strategy), None)
        if selected_info:
            st.caption(f"Type: {selected_info['type'].upper()} | Timeframe: {selected_info['timeframe'].upper()}")
        
        ticker = st.text_input("Ticker Symbol", value="SPY", key="bt_ticker").upper()
        
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            key="bt_capital"
        )
    
    with col2:
        st.markdown("**Backtest Period**")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start = st.date_input("Start Date", value=start_date, key="bt_start")
        with date_col2:
            end = st.date_input("End Date", value=end_date, key="bt_end")
        
        # Strategy-specific parameters
        st.markdown("**Strategy Parameters**")
        
        params = {}
        if selected_strategy == "WARRIOR_SCALPING":
            params['gap_threshold'] = st.slider("Gap Threshold (%)", 1.0, 10.0, 2.0, 0.5)
            params['volume_ratio'] = st.slider("Volume Ratio (x avg)", 1.0, 5.0, 1.5, 0.25)
            params['profit_target'] = st.slider("Profit Target (%)", 2.0, 15.0, 5.0, 0.5)
            params['stop_loss'] = st.slider("Stop Loss (%)", 1.0, 10.0, 2.0, 0.5)
        
        elif selected_strategy == "SLOW_SCALPER":
            params['bb_period'] = st.slider("Bollinger Period", 10, 50, 20)
            params['bb_std'] = st.slider("Bollinger Std Dev", 1.0, 3.0, 2.0, 0.25)
            params['stop_loss'] = st.slider("Stop Loss (%)", 1.0, 10.0, 2.0, 0.5)
        
        elif selected_strategy == "MICRO_SWING":
            params['lookback'] = st.slider("Lookback Period", 10, 50, 20)
            params['profit_target'] = st.slider("Profit Target (%)", 1.0, 10.0, 3.0, 0.5)
            params['stop_loss'] = st.slider("Stop Loss (%)", 0.5, 5.0, 1.5, 0.25)
        
        elif selected_strategy in ["COVERED_CALL", "CASH_SECURED_PUT"]:
            params['otm_pct'] = st.slider("OTM Distance (%)", 2.0, 15.0, 5.0, 0.5)
            params['premium_yield'] = st.slider("Assumed Monthly Premium (%)", 0.5, 3.0, 1.5, 0.25)
    
    if st.button("🚀 Run Backtest", type="primary", key="run_backtest"):
        with st.spinner(f"Backtesting {selected_strategy} on {ticker}..."):
            try:
                # Run async backtest
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    quant.backtest_strategy(
                        strategy_name=selected_strategy,
                        ticker=ticker,
                        start_date=datetime.combine(start, datetime.min.time()),
                        end_date=datetime.combine(end, datetime.min.time()),
                        initial_capital=float(initial_capital),
                        strategy_params=params
                    )
                )
                
                loop.close()
                
                if result.recommendation == "ERROR":
                    st.error("Backtest failed. Check ticker symbol and date range.")
                    return
                
                if result.total_trades == 0:
                    st.warning("No trades generated. Try adjusting parameters or a different date range.")
                    return
                
                st.success(f"✅ Backtest complete! {result.total_trades} trades analyzed.")
                
                # Display results
                st.markdown("---")
                
                # Recommendation banner
                rec_colors = {
                    "STRONG_BUY": "🟢",
                    "FAVORABLE": "🟡",
                    "NEUTRAL": "⚪",
                    "CAUTION": "🔴",
                    "NO_TRADES": "⚫"
                }
                rec_icon = rec_colors.get(result.recommendation, "⚫")
                st.markdown(f"### {rec_icon} Recommendation: **{result.recommendation}**")
                
                # Key metrics
                st.markdown("### Performance Metrics")
                
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    delta_color = "normal" if result.total_return_pct >= 0 else "inverse"
                    st.metric(
                        "Total Return",
                        f"{result.total_return_pct:+.2f}%",
                        delta=f"${initial_capital * result.total_return_pct / 100:,.2f}"
                    )
                
                with m2:
                    st.metric("Annualized Return", f"{result.annualized_return_pct:+.2f}%")
                
                with m3:
                    sharpe_color = "off" if result.sharpe_ratio < 0.5 else "normal"
                    st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                
                with m4:
                    st.metric("Sortino Ratio", f"{result.sortino_ratio:.2f}")
                
                m5, m6, m7, m8 = st.columns(4)
                
                with m5:
                    st.metric("Win Rate", f"{result.win_rate * 100:.1f}%")
                
                with m6:
                    st.metric("Profit Factor", f"{result.profit_factor:.2f}x")
                
                with m7:
                    st.metric("Max Drawdown", f"-{result.max_drawdown_pct:.2f}%")
                
                with m8:
                    st.metric("Volatility", f"{result.volatility_pct:.2f}%")
                
                # Trade statistics
                st.markdown("### Trade Statistics")
                
                t1, t2, t3, t4 = st.columns(4)
                
                with t1:
                    st.metric("Total Trades", result.total_trades)
                
                with t2:
                    st.metric("Winning Trades", result.winning_trades)
                
                with t3:
                    st.metric("Losing Trades", result.losing_trades)
                
                with t4:
                    st.metric("Avg Win / Avg Loss", f"{result.avg_win_pct:.2f}% / {result.avg_loss_pct:.2f}%")
                
                # Trade log
                if result.trades:
                    with st.expander(f"📋 Trade Log ({len(result.trades)} trades)", expanded=False):
                        import pandas as pd
                        trades_df = pd.DataFrame(result.trades)
                        
                        # Color code PnL
                        def color_pnl(val):
                            if isinstance(val, (int, float)):
                                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                                return f'color: {color}'
                            return ''
                        
                        if 'pnl_pct' in trades_df.columns:
                            styled_df = trades_df.style.applymap(color_pnl, subset=['pnl_pct'])
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.dataframe(trades_df, use_container_width=True)
                
                # Save result to session state for reference
                st.session_state.last_backtest_result = result
                
            except Exception as e:
                logger.error(f"Backtest error: {e}", exc_info=True)
                st.error(f"Backtest failed: {str(e)}")
    
    # Clear cache option
    st.markdown("---")
    if st.button("🗑️ Clear Backtest Cache", key="clear_bt_cache"):
        quant.clear_cache()
        st.success("Backtest cache cleared!")

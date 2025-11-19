"""
Generate Signal Tab
Manual signal generation and analysis

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

# Import with fallback
try:
    from src.integrations.trading_config import get_trading_mode_manager, TradingMode, switch_to_paper_mode, switch_to_production_mode
except ImportError:
    logger.debug("Trading config not available, using fallback")
    # Fallback
    class TradingMode:
        PAPER = "PAPER"
        PRODUCTION = "PRODUCTION"
    
    class MockTradingModeManager:
        def is_paper_mode(self):
            return True
        def get_current_mode(self):
            return TradingMode.PAPER
    
    def get_trading_mode_manager():
        return MockTradingModeManager()
    
    def switch_to_paper_mode():
        logger.info("Switched to paper mode (fallback)")
    
    def switch_to_production_mode():
        logger.info("Switched to production mode (fallback)")

# Helper function fallbacks
def calculate_dte(expiry_date):
    """Calculate days to expiration"""
    if isinstance(expiry_date, str):
        expiry = datetime.strptime(expiry_date, '%Y-%m-%d')
    else:
        expiry = expiry_date
    return (expiry - datetime.now()).days

try:
    from clients.option_alpha import OptionAlphaClient
except ImportError:
    logger.debug("OptionAlphaClient not available")
    OptionAlphaClient = None

def render_tab():
    """Main render function called from app.py"""
    st.header("Generate Signal")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("📊 Generate Trading Signal")
    
    # Get current trading mode
    mode_manager = get_trading_mode_manager()
    paper_mode = mode_manager.is_paper_mode()
    
    # Show current trading mode with switch option
    col1, col2 = st.columns([3, 1])
    with col1:
        if paper_mode:
            st.info("🔒 **Paper Trading Mode** - Signals will be logged only")
        else:
            st.warning("⚠️ **LIVE TRADING MODE** - Real trades will be executed!")
    with col2:
        if paper_mode:
            if st.button("Switch to Live Trading", type="primary"):
                if switch_to_production_mode():
                    st.success("Switched to Live Trading Mode!")
                    st.rerun()
                else:
                    st.error("Failed to switch to Live Trading Mode")
        else:
            if st.button("Switch to Paper Trading"):
                if switch_to_paper_mode():
                    st.success("Switched to Paper Trading Mode!")
                    st.rerun()
                else:
                    st.error("Failed to switch to Paper Trading Mode")
    
    # Check if we have a selected strategy template
    selected_template_id = st.session_state.get('selected_template')
    selected_strategy = st.session_state.get('selected_strategy')
    selected_ticker = st.session_state.get('selected_ticker', 'N/A')
    
    if selected_template_id:
        try:
            from models.option_strategy_templates import template_manager
            template = template_manager.get_template(selected_template_id)
            
            if template:
                st.success(f"🎯 **Using Strategy Template:** {template.name}")
                
                # Show template details in an expandable section
                with st.expander("📋 Template Details", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Direction", template.direction)
                        st.metric("Risk Level", template.risk_level)
                    with col2:
                        st.metric("Capital Required", template.capital_requirement)
                        if template.typical_win_rate:
                            st.metric("Win Rate", template.typical_win_rate)
                    with col3:
                        st.metric("IV Rank", template.ideal_iv_rank)
                        st.metric("Type", template.strategy_type)
                    
                    st.markdown(f"**Description:** {template.description}")
                    st.markdown(f"**Max Loss:** {template.max_loss}")
                    st.markdown(f"**Max Gain:** {template.max_gain}")
                    
                    if template.setup_steps:
                        st.write("**Setup Steps:**")
                        for i, step in enumerate(template.setup_steps, 1):
                            st.write(f"{i}. {step}")
                    
                    if template.warnings:
                        st.write("**⚠️ Warnings:**")
                        for warning in template.warnings:
                            st.warning(warning)
                    
                    if template.option_alpha_compatible:
                        st.success(f"✅ Option Alpha Compatible - Action: `{template.option_alpha_action}`")
                
                # Pre-fill strategy selection
                if template.option_alpha_action:
                    st.session_state.selected_strategy = template.option_alpha_action
                    selected_strategy = template.option_alpha_action
                
                st.divider()
            else:
                st.warning("Selected template not found. Please select a valid template.")
        except Exception as e:
            st.error(f"Error loading template: {e}")
    
    if selected_strategy:
        st.info(f"💡 Using recommended strategy: **{selected_strategy}** for **{selected_ticker}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol",
            value=st.session_state.get('selected_ticker', 'SOFI'),
            key='signal_ticker_input',
            help="Ticker symbol (e.g., AAPL). If you loaded a recommended strategy this may be prefilled."
        )
        ticker = ticker_input.upper() if isinstance(ticker_input, str) and ticker_input else ''
        
        # Get allowed strategies with fallback
        if hasattr(st.session_state, 'config') and st.session_state.config:
            allowed = list(st.session_state.config.allowed_strategies or [])
        else:
            allowed = ['SELL_PUT', 'SELL_CALL', 'BUY_PUT', 'BUY_CALL']
        
        default_idx = 0
        sel = st.session_state.get('selected_strategy')
        try:
            if sel in allowed:
                default_idx = allowed.index(sel)
        except Exception:
            default_idx = 0

        action = st.selectbox(
            "Strategy",
            options=allowed,
            index=default_idx,
            key='signal_strategy_select',
            help="Choose the option strategy to send. Select from recommended or custom strategies."
        )
        
        # Prefill from example_trade when available
        example = st.session_state.get('example_trade') or {}
        default_expiry = example.get('expiry', (datetime.now() + timedelta(days=30)).date())
        expiry_date = st.date_input(
            "Expiration Date",
            value=default_expiry,
            min_value=datetime.now().date(),
            max_value=(datetime.now() + timedelta(days=365)).date(),
            key='signal_expiry_date',
            help="Expiration date for the option contract. Be mindful of DTE (days to expiration)."
        )

        default_strike = example.get('strike', 9.0)
        strike = st.number_input("Strike Price", min_value=0.0, value=float(default_strike), step=0.5, format="%.2f", key='signal_strike_price')
        st.caption("Strike price for the option(s). Use analysis support/resistance as a reference.")
    
    with col2:
        qty = st.number_input("Quantity (contracts)", min_value=1, max_value=10, value=int(example.get('qty', 2)), key='signal_qty')
        st.caption("Number of contracts (1 contract = 100 shares). Keep within your capital limits.")
        # Use example or analysis IV if available
        # Determine a safe numeric default for IV rank (never None)
        _ex_iv = example.get('iv_rank') if isinstance(example.get('iv_rank'), (int, float)) else None
        _curr_iv = None
        if st.session_state.current_analysis and getattr(st.session_state.current_analysis, 'iv_rank', None) is not None:
            try:
                _curr_iv = float(st.session_state.current_analysis.iv_rank)
            except Exception:
                _curr_iv = None

        if _ex_iv is not None:
            default_iv = float(_ex_iv)
        elif _curr_iv is not None:
            default_iv = _curr_iv
        else:
            default_iv = 48.0

        iv_rank = st.slider("IV Rank (%)", 0, 100, int(default_iv))
        st.caption("Implied Volatility Rank — helps decide premium selling vs buying strategies.")

        estimated_risk = st.number_input("Estimated Risk ($)", min_value=0.0, value=float(example.get('estimated_risk', 200.0)), step=50.0)
        st.caption("Estimated maximum risk for the trade (approx). Used by guardrails.")

        llm_score = st.slider("AI Confidence", 0.0, 1.0, float(example.get('llm_score', 0.77)), 0.01)
        st.caption("AI confidence score for this signal (0.0 low → 1.0 high). Use as guidance, not final truth.")
    
    note = st.text_area(
        "Signal Note",
        value=f"AI-score={llm_score}; IVR={iv_rank}; Strategy={action}",
        help="Additional context"
    )
    
    # Add analysis function
    def calculate_roi_analysis(ticker: str, action: str, strike: float, qty: int, expiry_date, iv_rank: float, estimated_risk: float) -> Dict:
        """Calculate ROI and profit scenarios for the trade"""
        try:
            import yfinance as yf
            from services.options_pricing import black_scholes_price
            
            # Get current stock price
            stock = yf.Ticker(ticker)
            stock_info = stock.info
            current_price = stock_info.get('currentPrice') or stock_info.get('regularMarketPrice') or stock_info.get('previousClose', 0)
            
            if not current_price:
                # Try to get from history
                hist = stock.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    return {'error': f'Could not fetch current price for {ticker}'}
            
            current_price = float(current_price)
            dte = calculate_dte(expiry_date.strftime('%Y-%m-%d'))
            
            # Estimate option premium using Black-Scholes (simplified)
            # Convert IV rank to implied volatility (rough estimate: 0.15 + (iv_rank/100) * 0.35)
            iv = 0.15 + (iv_rank / 100) * 0.35
            T = max(dte, 1) / 365.0
            rf = 0.05  # Risk-free rate estimate
            
            is_call = 'CALL' in action.upper()
            is_sell = 'SELL' in action.upper()
            
            # Calculate theoretical premium
            if is_call:
                premium_per_share = black_scholes_price(current_price, strike, T, rf, iv, is_call=True)
            else:
                premium_per_share = black_scholes_price(current_price, strike, T, rf, iv, is_call=False)
            
            # Adjust for selling (you receive premium)
            if is_sell:
                premium_received = premium_per_share * 100 * qty
            else:
                premium_received = -premium_per_share * 100 * qty  # Negative = cost
            
            # Calculate metrics based on strategy
            if action == 'SELL_CALL':  # Covered Call
                capital_required = current_price * 100 * qty  # Need to own 100 shares per contract
                premium_income = premium_per_share * 100 * qty
                
                # Determine if strike is above or below current price
                strike_above_current = strike > current_price
                
                if strike_above_current:
                    # Normal covered call: Strike above current price
                    # Max profit occurs when stock is called away at strike
                    # Profit = premium + (strike - current_price) * 100 shares
                    max_profit = premium_income + (strike - current_price) * 100 * qty
                    max_profit_pct = (max_profit / capital_required) * 100
                    max_profit_scenario_price = strike
                    max_profit_description = f'Stock called away at ${strike:.2f}'
                else:
                    # Deep ITM call: Strike below current price (will almost certainly be exercised)
                    # When strike < current, you're selling shares below market value
                    # Max profit = premium received + (strike - current_price) * 100 shares
                    # This will typically result in a very small profit or even a loss
                    max_profit = premium_income + (strike - current_price) * 100 * qty
                    max_profit_pct = (max_profit / capital_required) * 100
                    max_profit_scenario_price = strike
                    max_profit_description = f'Stock called away at ${strike:.2f} (deep ITM - likely early assignment)'
                
                # Max loss: Stock goes to $0 (limited, not unlimited!)
                # Loss = (current_price * 100 * qty) - premium_received
                max_loss = (current_price * 100 * qty) - premium_income
                max_loss_pct = (max_loss / capital_required) * 100
                
                # Breakeven = stock price - premium received per share
                # This is where total P&L = 0 (stock loss offset by premium)
                breakeven_price = current_price - premium_per_share
                
                # ROI scenarios
                scenarios = {
                    'best_case': {
                        'stock_price': max_profit_scenario_price,
                        'profit': max_profit,
                        'roi_pct': max_profit_pct,
                        'description': max_profit_description
                    },
                    'current_price': {
                        'stock_price': current_price,
                        'profit': premium_income if strike_above_current else premium_income,
                        'roi_pct': (premium_income / capital_required) * 100,
                        'description': f'Stock stays at ${current_price:.2f}'
                    },
                    'breakeven': {
                        'stock_price': breakeven_price,
                        'profit': 0,
                        'roi_pct': 0,
                        'description': f'Breakeven at ${breakeven_price:.2f}'
                    },
                    'worst_case': {
                        'stock_price': 0,
                        'profit': -max_loss,
                        'roi_pct': -max_loss_pct,
                        'description': f'Stock goes to $0 (max loss)'
                    }
                }
                
                return {
                    'strategy': 'Covered Call',
                    'ticker': ticker,
                    'current_stock_price': current_price,
                    'strike': strike,
                    'contracts': qty,
                    'dte': dte,
                    'capital_required': capital_required,
                    'premium_received': premium_income,
                    'premium_per_share': premium_per_share,
                    'max_profit': max_profit,
                    'max_profit_pct': max_profit_pct,
                    'max_loss': max_loss,
                    'max_loss_pct': max_loss_pct,
                    'breakeven_price': breakeven_price,
                    'roi_at_strike': max_profit_pct,
                    'scenarios': scenarios,
                    'iv_rank': iv_rank,
                    'estimated_iv': iv * 100,
                    'strike_above_current': strike_above_current
                }
            
            elif action == 'SELL_PUT':  # Cash-Secured Put
                capital_required = strike * 100 * qty  # Cash needed to secure
                premium_income = premium_per_share * 100 * qty
                
                max_profit = premium_income  # If stock stays above strike
                max_profit_pct = (max_profit / capital_required) * 100
                
                # Max loss if assigned
                max_loss = (strike - current_price) * 100 * qty - premium_income
                if max_loss < 0:
                    max_loss = 0  # Can't lose more than premium if stock goes up
                
                breakeven_price = strike - premium_per_share
                
                scenarios = {
                    'best_case': {
                        'stock_price': current_price,
                        'profit': max_profit,
                        'roi_pct': max_profit_pct,
                        'description': f'Stock stays above ${strike:.2f}'
                    },
                    'assigned': {
                        'stock_price': strike,
                        'profit': premium_income,
                        'roi_pct': max_profit_pct,
                        'description': f'Assigned at ${strike:.2f}'
                    },
                    'breakeven': {
                        'stock_price': breakeven_price,
                        'profit': 0,
                        'roi_pct': 0,
                        'description': f'Breakeven at ${breakeven_price:.2f}'
                    }
                }
                
                return {
                    'strategy': 'Cash-Secured Put',
                    'ticker': ticker,
                    'current_stock_price': current_price,
                    'strike': strike,
                    'contracts': qty,
                    'dte': dte,
                    'capital_required': capital_required,
                    'premium_received': premium_income,
                    'premium_per_share': premium_per_share,
                    'max_profit': max_profit,
                    'max_profit_pct': max_profit_pct,
                    'max_loss': max_loss,
                    'breakeven_price': breakeven_price,
                    'scenarios': scenarios,
                    'iv_rank': iv_rank,
                    'estimated_iv': iv * 100
                }
            
            else:
                # Generic calculation for other strategies
                return {
                    'strategy': action,
                    'ticker': ticker,
                    'current_stock_price': current_price,
                    'strike': strike,
                    'contracts': qty,
                    'dte': dte,
                    'premium_per_share': premium_per_share,
                    'estimated_risk': estimated_risk,
                    'iv_rank': iv_rank
                }
                
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return {'error': str(e)}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Validate", width='stretch'):
            if not hasattr(st.session_state, 'validator') or not st.session_state.validator:
                st.error("Validator not initialized")
            else:
                dte = calculate_dte(expiry_date.strftime('%Y-%m-%d'))
                
                signal = {
                    'ticker': ticker,
                    'action': action,
                    'expiry': expiry_date.strftime('%Y-%m-%d'),
                    'dte': dte,
                    'strike': strike,
                    'qty': qty,
                    'iv_rank': iv_rank,
                    'estimated_risk': estimated_risk,
                    'llm_score': llm_score,
                    'note': note
                }
                
                is_valid, message = st.session_state.validator.validate_signal(signal)
                
                if is_valid:
                    st.success(f"✅ {message}")
                    st.json(signal)
                else:
                    st.error(f"❌ {message}")
    
    with col2:
        analyze_clicked = st.button("📊 Analyze & Calculate ROI", width='stretch', type="secondary")
        
        if analyze_clicked:
            if not ticker:
                st.error("Please enter a ticker symbol")
            else:
                with st.spinner(f"Analyzing {ticker} {action} trade..."):
                    analysis = calculate_roi_analysis(ticker, action, strike, qty, expiry_date, iv_rank, estimated_risk)
                    
                    if 'error' in analysis:
                        st.error(f"❌ {analysis['error']}")
                    else:
                        st.success("✅ Analysis Complete!")
                        st.divider()
                        
                        # Display key metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Current Price", f"${analysis.get('current_stock_price', 0):.2f}")
                        with col_b:
                            st.metric("Strike Price", f"${strike:.2f}")
                        with col_c:
                            st.metric("Capital Required", f"${analysis.get('capital_required', 0):,.2f}")
                        with col_d:
                            st.metric("Premium Income", f"${analysis.get('premium_received', 0):,.2f}")
                        
                        st.divider()
                        
                        # Display ROI metrics
                        st.subheader("💰 Profit & ROI Analysis")
                        col_e, col_f, col_g = st.columns(3)
                        with col_e:
                            st.metric("Max Profit", f"${analysis.get('max_profit', 0):,.2f}", 
                                     f"{analysis.get('max_profit_pct', 0):.2f}% ROI")
                        with col_f:
                            max_loss = analysis.get('max_loss', 'N/A')
                            if isinstance(max_loss, (int, float)):
                                st.metric("Max Loss", f"${max_loss:,.2f}")
                            else:
                                st.metric("Max Loss", str(max_loss))
                        with col_g:
                            st.metric("Breakeven", f"${analysis.get('breakeven_price', 0):.2f}")
                        
                        # Display scenarios
                        if 'scenarios' in analysis:
                            st.divider()
                            st.subheader("📈 Profit Scenarios")
                            for scenario_name, scenario_data in analysis['scenarios'].items():
                                with st.expander(f"{scenario_data['description']} (${scenario_data['stock_price']:.2f})"):
                                    col_h, col_i = st.columns(2)
                                    with col_h:
                                        st.metric("Profit", f"${scenario_data['profit']:,.2f}")
                                    with col_i:
                                        st.metric("ROI", f"{scenario_data['roi_pct']:.2f}%")
                        
                        # Display additional info
                        st.divider()
                        st.subheader("ℹ️ Trade Details")
                        info_col1, info_col2 = st.columns(2)
                        with info_col1:
                            st.write(f"**Strategy:** {analysis.get('strategy', action)}")
                            st.write(f"**Contracts:** {qty}")
                            st.write(f"**Days to Expiry:** {analysis.get('dte', 0)}")
                        with info_col2:
                            st.write(f"**IV Rank:** {iv_rank}%")
                            st.write(f"**Estimated IV:** {analysis.get('estimated_iv', 0):.1f}%")
                            st.write(f"**Premium per Share:** ${analysis.get('premium_per_share', 0):.2f}")
                        
                        # Store analysis in session state
                        st.session_state.last_roi_analysis = analysis
    
    with col3:
        if st.button("🚀 Send Signal", width='stretch', type="primary"):
            dte = calculate_dte(expiry_date.strftime('%Y-%m-%d'))
            
            # Map action to Option Alpha format
            oa_action_mapping = {
                'SELL_PUT': 'BPS',  # Bull Put Spread
                'SELL_CALL': 'BCS', # Bear Call Spread
                'BUY_PUT': 'PUT',
                'BUY_CALL': 'CALL'
            }
            
            oa_action = oa_action_mapping.get(action, action)
            
            # Determine market condition based on IV rank
            market_condition = "high_vol" if iv_rank > 60 else "normal" if iv_rank > 30 else "low_vol"
            
            signal = {
                'symbol': 'SPX',  # Fixed to SPX for your bot
                'action': oa_action,
                'expiry': expiry_date.strftime('%Y-%m-%d'),
                'dte': dte,
                'strike': strike,
                'quantity': qty,
                'iv_rank': iv_rank,
                'market_condition': market_condition,
                'estimated_risk': estimated_risk,
                'llm_score': llm_score,
                'note': f"AI Analysis: {note}"
            }
            
            if not hasattr(st.session_state, 'validator') or not st.session_state.validator:
                st.error("Validator not initialized")
                is_valid, validation_msg = False, "Validator not available"
            else:
                is_valid, validation_msg = st.session_state.validator.validate_signal(signal)
            
            if not is_valid:
                st.error(f"❌ {validation_msg}")
            else:
                if paper_mode:
                    st.info("📝 Paper mode: Signal logged")
                    success = True
                    message = "Signal logged in paper trading mode"
                else:
                    if OptionAlphaClient is None:
                        st.error("OptionAlphaClient not available")
                        success = False
                        message = "OptionAlphaClient module not found"
                    else:
                        webhook_url = st.session_state.get('webhook_url', '')
                        if not webhook_url:
                            st.error("Webhook URL not configured")
                            success = False
                            message = "Missing webhook URL"
                        else:
                            client = OptionAlphaClient(webhook_url)
                            success, message = client.send_signal(signal)
                
                if success:
                    if hasattr(st.session_state, 'validator') and st.session_state.validator:
                        st.session_state.validator.record_order(signal)
                    st.session_state.signal_history.append({
                        **signal,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'Paper' if paper_mode else 'Live',
                        'result': message
                    })
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        
        # Reset button in same column
        if hasattr(st.session_state, 'validator') and st.session_state.validator:
            if st.button("🔄 Reset", width='stretch'):
                st.session_state.validator.reset_daily_counters()
                st.success("Counters reset!")
    
    st.divider()
    st.subheader("Current Status")
    
    m1, m2, m3, m4 = st.columns(4)
    
    # Show status only if validator and config are available
    if hasattr(st.session_state, 'validator') and hasattr(st.session_state, 'config') and st.session_state.validator and st.session_state.config:
        with m1:
            st.metric("Daily Orders", f"{st.session_state.validator.daily_orders}/{st.session_state.config.max_daily_orders}")
        with m2:
            st.metric("Daily Risk", f"${st.session_state.validator.daily_risk:.0f}/${st.session_state.config.max_daily_risk:.0f}")
        with m3:
            in_hours, _ = st.session_state.validator.is_trading_hours()
            st.metric("Trading Hours", "✅ Open" if in_hours else "❌ Closed")
        with m4:
            st.metric("Mode", "📝 Paper" if paper_mode else "🔴 Live")
    else:
        with m1:
            st.info("Validator not initialized")
        with m2:
            st.info("Config not available")
        with m3:
            st.info("Trading hours N/A")
        with m4:
            st.metric("Mode", "📝 Paper" if paper_mode else "🔴 Live")


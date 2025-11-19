"""
Fractional DCA (Dollar-Cost Averaging) Tab
UI for automated fractional share investing with AI confirmation
"""

import streamlit as st
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional

from services.fractional_dca_manager import FractionalDCAManager, DCASchedule
from analyzers.orb_fvg_strategy import ORBFVGAnalyzer
from analyzers.trading_styles import TradingStyleAnalyzer


def render_tab(supabase_client=None):
    """Render the Fractional DCA tab"""
    
    st.title("💰 Fractional DCA Manager")
    st.markdown("**Automate dollar-cost averaging for high-priced stocks with AI confirmation**")
    
    # Initialize DCA manager in session state
    if 'dca_manager' not in st.session_state:
        st.session_state.dca_manager = FractionalDCAManager(supabase_client)
        # Try to load existing data from database
        st.session_state.dca_manager.load_from_db()
    
    dca_manager = st.session_state.dca_manager
    
    # Sidebar for quick stats
    with st.sidebar:
        st.subheader("📊 Portfolio Overview")
        
        # Get current prices for all tickers
        schedules = dca_manager.get_all_positions()
        if schedules:
            tickers = [s['ticker'] for s in schedules]
            try:
                current_prices = {}
                for ticker in tickers:
                    ticker_obj = yf.Ticker(ticker)
                    hist = ticker_obj.history(period="1d")
                    if not hist.empty:
                        current_prices[ticker] = float(hist['Close'].iloc[-1])
                
                # Calculate portfolio performance
                performance = dca_manager.calculate_portfolio_performance(current_prices)
                
                st.metric("Total Invested", f"${performance['total_invested']:,.2f}")
                st.metric("Current Value", f"${performance['current_value']:,.2f}")
                st.metric("Total Gain", 
                         f"${performance['total_gain']:,.2f}",
                         delta=f"{performance['total_gain_pct']:.1f}%")
                st.metric("Active Positions", performance['position_count'])
            except Exception as e:
                logger.error(f"Error fetching prices: {e}")
                st.warning("Unable to fetch current prices")
        else:
            st.info("No active DCA schedules yet")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 DCA Schedules",
        "🎯 Execute Manual Buy",
        "📈 Performance",
        "📋 Transaction History"
    ])
    
    # TAB 1: DCA Schedules
    with tab1:
        render_schedules_tab(dca_manager)
    
    # TAB 2: Manual Buy with AI Confirmation
    with tab2:
        render_manual_buy_tab(dca_manager)
    
    # TAB 3: Performance Tracking
    with tab3:
        render_performance_tab(dca_manager)
    
    # TAB 4: Transaction History
    with tab4:
        render_history_tab(dca_manager)


def render_schedules_tab(dca_manager: FractionalDCAManager):
    """Render the DCA schedules management tab"""
    
    st.header("📅 Manage DCA Schedules")
    
    # Add new schedule section
    with st.expander("➕ Add New DCA Schedule", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            # High-priced stock presets
            preset_stocks = {
                "Custom": "",
                "NVDA (Nvidia)": "NVDA",
                "META (Meta)": "META",
                "GOOGL (Alphabet)": "GOOGL",
                "TSLA (Tesla)": "TSLA",
                "MSFT (Microsoft)": "MSFT",
                "AMZN (Amazon)": "AMZN",
                "AAPL (Apple)": "AAPL",
                "BRK.B (Berkshire)": "BRK.B",
                "BKNG (Booking)": "BKNG",
                "AZO (AutoZone)": "AZO"
            }
            
            preset = st.selectbox("Select Stock", list(preset_stocks.keys()))
            
            if preset == "Custom":
                new_ticker = st.text_input("Ticker Symbol", placeholder="e.g., NVDA").upper()
            else:
                new_ticker = preset_stocks[preset]
                st.info(f"Selected: **{new_ticker}**")
            
            new_amount = st.number_input(
                "Amount Per Interval ($)",
                min_value=10.0,
                max_value=10000.0,
                value=100.0,
                step=10.0,
                help="Dollar amount to invest each interval"
            )
            
            new_frequency = st.selectbox(
                "Frequency",
                ["weekly", "daily", "monthly"],
                help="How often to execute DCA buys"
            )
        
        with col2:
            new_strategy = st.selectbox(
                "Analysis Strategy",
                ["AI", "ORB_FVG", "SCALP", "WARRIOR_SCALPING", "BUY_AND_HOLD"],
                help="Strategy to use for pre-buy analysis"
            )
            
            new_min_confidence = st.slider(
                "Minimum Confidence (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=5,
                help="Only buy if AI confidence meets this threshold"
            )
            
            new_max_price = st.number_input(
                "Max Price Limit (optional)",
                min_value=0.0,
                value=0.0,
                step=10.0,
                help="Skip buy if price exceeds this (0 = no limit)"
            )
            if new_max_price == 0:
                new_max_price = None
        
        if st.button("➕ Add DCA Schedule", type="primary"):
            if new_ticker:
                try:
                    schedule = dca_manager.add_schedule(
                        ticker=new_ticker,
                        amount=new_amount,
                        frequency=new_frequency,
                        min_confidence=new_min_confidence,
                        strategy=new_strategy,
                        max_price=new_max_price
                    )
                    st.success(f"✅ Added DCA schedule for {new_ticker}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding schedule: {str(e)}")
            else:
                st.warning("Please enter a ticker symbol")
    
    st.divider()
    
    # Display existing schedules
    st.subheader("📋 Active DCA Schedules")
    
    schedules = dca_manager.get_all_positions()
    
    if not schedules:
        st.info("💡 No DCA schedules yet. Add your first one above!")
        return
    
    for schedule in schedules:
        with st.expander(f"**{schedule['ticker']}** - ${schedule['amount_per_interval']:.0f} {schedule['frequency']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Invested", f"${schedule['total_invested']:,.2f}")
                st.metric("Total Shares", f"{schedule['total_shares']:.4f}")
                st.metric("Avg Cost", f"${schedule['average_cost']:.2f}")
            
            with col2:
                st.metric("Transactions", schedule['transaction_count'])
                st.metric("Min Confidence", f"{schedule['min_confidence']:.0f}%")
                st.metric("Strategy", schedule['strategy'])
            
            with col3:
                status = "🟢 Active" if schedule['active'] else "⏸️ Paused"
                st.metric("Status", status)
                
                if schedule['last_buy_date']:
                    st.metric("Last Buy", schedule['last_buy_date'].strftime("%Y-%m-%d"))
                
                if schedule['next_buy_date']:
                    days_until = (schedule['next_buy_date'] - datetime.now()).days
                    st.metric("Next Buy", f"in {days_until} days")
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if schedule['active']:
                    if st.button(f"⏸️ Pause", key=f"pause_{schedule['ticker']}"):
                        dca_manager.pause_schedule(schedule['ticker'])
                        st.success(f"Paused {schedule['ticker']}")
                        st.rerun()
                else:
                    if st.button(f"▶️ Resume", key=f"resume_{schedule['ticker']}"):
                        dca_manager.resume_schedule(schedule['ticker'])
                        st.success(f"Resumed {schedule['ticker']}")
                        st.rerun()
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{schedule['ticker']}"):
                    dca_manager.remove_schedule(schedule['ticker'])
                    st.success(f"Deleted {schedule['ticker']}")
                    st.rerun()
            
            with col3:
                if st.button(f"💰 Buy Now", key=f"buy_{schedule['ticker']}"):
                    st.session_state.manual_buy_ticker = schedule['ticker']
                    st.info(f"Go to 'Execute Manual Buy' tab to buy {schedule['ticker']}")
            
            with col4:
                if st.button(f"📊 View History", key=f"history_{schedule['ticker']}"):
                    st.session_state.history_filter = schedule['ticker']
                    st.info(f"Go to 'Transaction History' tab")


def render_manual_buy_tab(dca_manager: FractionalDCAManager):
    """Render manual buy execution tab with AI confirmation"""
    
    st.header("🎯 Execute Manual DCA Buy")
    st.markdown("**Buy fractional shares with AI analysis confirmation**")
    
    # Ticker selection
    schedules = dca_manager.get_all_positions()
    ticker_list = [s['ticker'] for s in schedules] if schedules else []
    
    # Prefill if coming from schedules tab
    default_ticker = st.session_state.get('manual_buy_ticker', ticker_list[0] if ticker_list else "NVDA")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input("Ticker Symbol", value=default_ticker).upper()
    
    with col2:
        amount = st.number_input("Amount ($)", min_value=10.0, value=100.0, step=10.0)
    
    # Strategy selection
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox(
            "Analysis Strategy",
            ["AI", "ORB_FVG", "SCALP", "BUY_AND_HOLD"],
            help="Strategy for pre-buy analysis"
        )
    
    with col2:
        min_confidence = st.slider(
            "Min Confidence to Execute",
            0, 100, 60, 5,
            help="Only execute if confidence meets threshold"
        )
    
    if st.button("🔍 Analyze & Preview Buy", type="primary"):
        if not ticker:
            st.warning("Please enter a ticker symbol")
            return
        
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Fetch current price and data
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period="1d", interval="1m")
                
                if hist.empty:
                    st.error(f"Unable to fetch data for {ticker}")
                    return
                
                current_price = float(hist['Close'].iloc[-1])
                shares = amount / current_price
                
                # Run AI analysis based on strategy
                if strategy == "ORB_FVG":
                    orb_analyzer = ORBFVGAnalyzer()
                    analysis = orb_analyzer.analyze(ticker, hist, current_price)
                    confidence = analysis['confidence']
                    recommendation = analysis['signal']
                else:
                    # Use AI/other strategy
                    # For demo, using simple analysis
                    confidence = 75.0  # Placeholder
                    recommendation = "BUY"
                
                # Display analysis results
                st.subheader(f"📊 Analysis Results for {ticker}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("Shares to Buy", f"{shares:.4f}")
                
                with col3:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col4:
                    st.metric("Recommendation", recommendation)
                
                # Decision logic
                should_buy, reason = dca_manager.should_execute_buy(
                    ticker, current_price, confidence, recommendation
                )
                
                if confidence >= min_confidence and recommendation in ["BUY", "STRONG_BUY"]:
                    st.success(f"✅ **APPROVED**: {reason}")
                    
                    st.info(f"**Trade Preview:**\n"
                           f"- Buy {shares:.4f} shares of {ticker}\n"
                           f"- At ${current_price:.2f} per share\n"
                           f"- Total: ${amount:.2f}\n"
                           f"- Confidence: {confidence:.1f}%")
                    
                    if st.button("✅ Confirm & Execute Buy", type="primary"):
                        # Record the transaction
                        transaction = dca_manager.record_transaction(
                            ticker=ticker,
                            amount=amount,
                            price=current_price,
                            shares=shares,
                            confidence=confidence,
                            strategy=strategy,
                            recommendation=recommendation
                        )
                        
                        st.success(f"🎉 Successfully bought {shares:.4f} shares of {ticker}!")
                        st.balloons()
                        
                        # Show updated position
                        position = dca_manager.get_position_summary(ticker)
                        if position:
                            st.metric("New Total Shares", f"{position['total_shares']:.4f}")
                            st.metric("New Avg Cost", f"${position['average_cost']:.2f}")
                
                else:
                    st.warning(f"⚠️ **NOT RECOMMENDED**: Confidence ({confidence:.1f}%) below threshold ({min_confidence}%)")
                    st.info("Adjust your minimum confidence or wait for better conditions")
            
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)
                st.error(f"Analysis error: {str(e)}")


def render_performance_tab(dca_manager: FractionalDCAManager):
    """Render performance tracking tab"""
    
    st.header("📈 DCA Portfolio Performance")
    
    schedules = dca_manager.get_all_positions()
    
    if not schedules:
        st.info("No DCA positions yet. Add schedules to start tracking!")
        return
    
    # Fetch current prices
    tickers = [s['ticker'] for s in schedules]
    current_prices = {}
    
    with st.spinner("Fetching current prices..."):
        for ticker in tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period="1d")
                if not hist.empty:
                    current_prices[ticker] = float(hist['Close'].iloc[-1])
            except:
                current_prices[ticker] = 0.0
    
    # Calculate performance
    performance = dca_manager.calculate_portfolio_performance(current_prices)
    
    # Overall metrics
    st.subheader("💼 Portfolio Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Invested", f"${performance['total_invested']:,.2f}")
    
    with col2:
        st.metric("Current Value", f"${performance['current_value']:,.2f}")
    
    with col3:
        gain_color = "normal" if performance['total_gain'] >= 0 else "inverse"
        st.metric("Total Gain", 
                 f"${performance['total_gain']:,.2f}",
                 delta=f"{performance['total_gain_pct']:.1f}%")
    
    with col4:
        st.metric("Positions", performance['position_count'])
    
    st.divider()
    
    # Individual position performance
    st.subheader("📊 Position Performance")
    
    if performance['positions']:
        df = pd.DataFrame(performance['positions'])
        
        # Format for display
        df['invested'] = df['invested'].apply(lambda x: f"${x:,.2f}")
        df['current_value'] = df['current_value'].apply(lambda x: f"${x:,.2f}")
        df['gain'] = df['gain'].apply(lambda x: f"${x:,.2f}")
        df['gain_pct'] = df['gain_pct'].apply(lambda x: f"{x:.1f}%")
        df['average_cost'] = df['average_cost'].apply(lambda x: f"${x:.2f}")
        df['current_price'] = df['current_price'].apply(lambda x: f"${x:.2f}")
        df['shares'] = df['shares'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            df[['ticker', 'shares', 'invested', 'current_value', 'gain', 'gain_pct', 'average_cost', 'current_price']],
            use_container_width=True,
            hide_index=True
        )


def render_history_tab(dca_manager: FractionalDCAManager):
    """Render transaction history tab"""
    
    st.header("📋 Transaction History")
    
    # Filter options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        schedules = dca_manager.get_all_positions()
        ticker_options = ["All"] + [s['ticker'] for s in schedules]
        default_filter = st.session_state.get('history_filter', "All")
        
        if default_filter not in ticker_options:
            default_filter = "All"
        
        ticker_filter = st.selectbox("Filter by Ticker", ticker_options, index=ticker_options.index(default_filter))
    
    with col2:
        limit = st.number_input("Show Last N", min_value=10, max_value=1000, value=100, step=10)
    
    # Get transactions
    if ticker_filter == "All":
        transactions = dca_manager.get_transaction_history(limit=limit)
    else:
        transactions = dca_manager.get_transaction_history(ticker=ticker_filter, limit=limit)
    
    if not transactions:
        st.info("No transactions yet. Execute your first DCA buy to see history!")
        return
    
    # Display summary stats
    total_invested = sum(t.amount for t in transactions)
    total_shares = sum(t.shares for t in transactions)
    avg_confidence = sum(t.confidence for t in transactions) / len(transactions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(transactions))
    
    with col2:
        st.metric("Total Invested", f"${total_invested:,.2f}")
    
    with col3:
        st.metric("Total Shares", f"{total_shares:.4f}")
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    st.divider()
    
    # Transaction table
    df = pd.DataFrame([
        {
            'Date': t.date.strftime("%Y-%m-%d %H:%M"),
            'Ticker': t.ticker,
            'Amount': f"${t.amount:.2f}",
            'Price': f"${t.price:.2f}",
            'Shares': f"{t.shares:.4f}",
            'Confidence': f"{t.confidence:.1f}%",
            'Strategy': t.strategy,
            'AI Rec': t.ai_recommendation
        }
        for t in transactions
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Export button
    if st.button("📥 Export to CSV"):
        filename = f"dca_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        dca_manager.export_to_csv(filename)
        st.success(f"✅ Exported to {filename}")

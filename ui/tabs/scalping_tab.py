"""
Scalping/Day Trade Tab
High-frequency scalping and day trading tools

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple

def render_tab():
    """Main render function called from app.py"""
    st.header("Scalping/Day Trade")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    st.header("⚡ Scalping & Day Trading Dashboard")
    st.write("Quick entry/exit interface for stock day trading and scalping. Works with both Tradier and IBKR.")
    st.info("💡 **Perfect for:** Blue chips, penny stocks, runners, and high-momentum plays. Get instant scalping signals!")
    
    # Quick Scalping Analyzer
    with st.expander("⚡ Quick Scalping Analyzer - Instant Signals", expanded=True):
        st.write("Get instant scalping signals for ANY ticker - optimized for 1-5 minute trades.")
        
        col_scalp1, col_scalp2, col_scalp3 = st.columns([2, 1, 1])
        
        with col_scalp1:
            scalp_ticker = st.text_input(
                "Ticker to Scalp",
                value="SPY",
                help="Enter any ticker: SPY, QQQ, AAPL, penny stocks, or runners"
            ).upper()
        
        with col_scalp2:
            scalp_mode = st.selectbox(
                "Scalping Mode",
                options=["Standard", "Penny Stock", "Runner/Momentum"],
                help="Penny Stock = tighter stops, Runner = momentum-based"
            )
        
        with col_scalp3:
            st.write("")
            st.write("")
            scalp_analyze_btn = st.button("⚡ Get Scalp Signal", type="primary", width="stretch")
        
        if scalp_analyze_btn and scalp_ticker:
            with st.status(f"⚡ Analyzing {scalp_ticker} for scalping...", expanded=True) as scalp_status:
                st.write("📊 Fetching real-time data...")
                
                try:
                    # Get analysis with scalp trading style
                    analysis = ComprehensiveAnalyzer.analyze_stock(scalp_ticker, "SCALP")
                    
                    if analysis:
                        scalp_status.update(label=f"✅ Scalp analysis complete for {scalp_ticker}", state="complete")
                        
                        # Detect characteristics
                        is_penny = is_penny_stock(analysis.price)
                        volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
                        is_runner = volume_vs_avg > 200 and abs(analysis.change_pct) > 10
                        
                        # Auto-adjust mode
                        if scalp_mode == "Standard" and is_penny:
                            st.info("💡 Auto-detected penny stock - using tighter stops")
                            scalp_mode = "Penny Stock"
                        elif scalp_mode == "Standard" and is_runner:
                            st.info("💡 Auto-detected runner - using momentum strategy")
                            scalp_mode = "Runner/Momentum"
                        
                        # Calculate scalping parameters based on mode
                        if scalp_mode == "Penny Stock":
                            stop_pct = 3.0  # Tight 3% stop
                            target_pct = 5.0  # Quick 5% target
                            risk_label = "HIGH"
                        elif scalp_mode == "Runner/Momentum":
                            stop_pct = 2.0  # Very tight 2% stop
                            target_pct = 8.0  # Larger 8% target for runners
                            risk_label = "VERY HIGH"
                        else:  # Standard
                            stop_pct = 0.5  # Standard 0.5% stop
                            target_pct = 1.0  # Standard 1% target
                            risk_label = "MEDIUM"
                        
                        # Calculate levels
                        entry_price = analysis.price
                        stop_loss = entry_price * (1 - stop_pct/100)
                        target_price = entry_price * (1 + target_pct/100)
                        
                        # Determine signal
                        signal = "NEUTRAL"
                        signal_color = "🟡"
                        confidence = 50
                        
                        if analysis.rsi < 40 and analysis.macd_signal == "BULLISH" and volume_vs_avg > 50:
                            signal = "BUY"
                            signal_color = "🟢"
                            confidence = 75
                        elif analysis.rsi > 60 and analysis.macd_signal == "BEARISH":
                            signal = "SELL"
                            signal_color = "🔴"
                            confidence = 70
                        elif analysis.trend == "BULLISH" and analysis.rsi < 60:
                            signal = "BUY"
                            signal_color = "🟢"
                            confidence = 65
                        elif analysis.trend == "BEARISH" and analysis.rsi > 40:
                            signal = "SELL"
                            signal_color = "🔴"
                            confidence = 60
                        
                        # Display signal
                        st.markdown(f"## {signal_color} SCALP SIGNAL: {signal}")
                        
                        # Metrics
                        scalp_col1, scalp_col2, scalp_col3, scalp_col4 = st.columns(4)
                        
                        with scalp_col1:
                            st.metric("Entry Price", f"${entry_price:.4f}" if is_penny else f"${entry_price:.2f}")
                            st.caption(f"Current: ${analysis.price:.4f}" if is_penny else f"${analysis.price:.2f}")
                        
                        with scalp_col2:
                            st.metric("Target", f"${target_price:.4f}" if is_penny else f"${target_price:.2f}")
                            st.caption(f"🎯 +{target_pct:.1f}%")
                        
                        with scalp_col3:
                            st.metric("Stop Loss", f"${stop_loss:.4f}" if is_penny else f"${stop_loss:.2f}")
                            st.caption(f"🛑 -{stop_pct:.1f}%")
                        
                        with scalp_col4:
                            st.metric("Confidence", f"{confidence}%")
                            st.metric("Risk", risk_label)
                        
                        # Additional info
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            st.write("**Technical Indicators:**")
                            st.write(f"• RSI: {analysis.rsi:.1f} {'🟢 Oversold' if analysis.rsi < 30 else '🔴 Overbought' if analysis.rsi > 70 else '🟡 Neutral'}")
                            st.write(f"• MACD: {analysis.macd_signal}")
                            st.write(f"• Trend: {analysis.trend}")
                            st.write(f"• Volume: {volume_vs_avg:+.0f}% vs avg")
                        
                        with info_col2:
                            st.write("**Scalping Strategy:**")
                            if scalp_mode == "Penny Stock":
                                st.write("• ⚡ Quick in/out (1-5 min)")
                                st.write("• 🛑 Tight 3% stop loss")
                                st.write("• 🎯 5% profit target")
                                st.write("• ⚠️ High risk - small size!")
                            elif scalp_mode == "Runner/Momentum":
                                st.write("• 🚀 Ride the momentum")
                                st.write("• 🛑 Very tight 2% stop")
                                st.write("• 🎯 8% profit target")
                                st.write("• ⚠️ Exit on volume drop!")
                            else:
                                st.write("• ⚡ Standard scalp (1-3 min)")
                                st.write("• 🛑 0.5% stop loss")
                                st.write("• 🎯 1% profit target")
                                st.write("• 📊 Watch L2 order book")
                        
                        # Warning for risky setups
                        if signal == "NEUTRAL":
                            st.warning("⚠️ No clear scalping setup right now. Wait for better entry or try another ticker.")
                        elif confidence < 65:
                            st.info("💡 Moderate confidence - consider reducing position size or waiting for confirmation.")
                        
                        # Quick action buttons
                        action_col1, action_col2 = st.columns(2)
                        with action_col1:
                            if st.button(f"📋 Copy {signal} Order to Form", width="stretch"):
                                st.session_state['scalp_prefill_symbol'] = scalp_ticker
                                st.session_state['scalp_prefill_side'] = signal
                                st.session_state['scalp_prefill_entry'] = entry_price
                                st.session_state['scalp_prefill_target'] = target_price
                                st.session_state['scalp_prefill_stop'] = stop_loss
                                st.success("✅ Copied to order form below!")
                        
                        with action_col2:
                            if st.button("🔄 Refresh Signal", width="stretch"):
                                st.rerun()
                    
                    else:
                        scalp_status.update(label=f"❌ Could not analyze {scalp_ticker}", state="error")
                        st.error(f"Unable to fetch data for {scalp_ticker}. Check ticker symbol.")
                
                except Exception as e:
                    scalp_status.update(label="❌ Analysis failed", state="error")
                    st.error(f"Error: {e}")
    
    # AI Autopilot Section
    with st.expander("🤖 AI Trading Autopilot - Get Smart Signals", expanded=False):
        st.write("Let AI analyze technicals, news, sentiment, and social media to recommend the best trades.")
        
        col_ai1, col_ai2, col_ai3 = st.columns(3)
        
        with col_ai1:
            ai_symbols = st.text_input(
                "Symbols to Analyze (comma-separated)",
                value="SPY,QQQ,AAPL,TSLA,NVDA",
                help="Enter stock symbols to get AI recommendations"
            )
        
        with col_ai2:
            ai_risk = st.selectbox(
                "Risk Tolerance",
                options=["LOW", "MEDIUM", "HIGH"],
                index=1,
                help="Your risk appetite"
            )
        
        with col_ai3:
            ai_provider = st.selectbox(
                "AI Provider",
                options=["openrouter", "openai", "anthropic"],
                index=0,
                help="OpenRouter is free!"
            )
        
        if st.button("🧠 Generate AI Signals", type="primary", width="stretch"):
            symbols_list = [s.strip().upper() for s in ai_symbols.split(',') if s.strip()]
            
            if not symbols_list:
                st.error("Please enter at least one symbol")
            else:
                with st.status("🤖 AI analyzing market data...", expanded=True) as status:
                    try:
                        # Import and initialize AI signal generator
                        from services.ai_trading_signals import create_ai_signal_generator
                        
                        st.write("Initializing AI engine...")
                        ai_generator = create_ai_signal_generator(provider=ai_provider)  # noqa: F841
                        
                        # Verify AI generator is ready and immediately test functionality
                        if not ai_generator or not hasattr(ai_generator, 'batch_analyze'):
                            raise Exception("Failed to initialize AI signal generator or missing batch_analyze method")
                        
                        # Collect data for each symbol
                        st.write(f"Gathering data for {len(symbols_list)} symbols...")
                        
                        technical_data_dict = {}
                        news_data_dict = {}
                        sentiment_data_dict = {}
                        
                        for symbol in symbols_list:
                            try:
                                st.write(f"Analyzing {symbol}...")
                                
                                # Get comprehensive analysis
                                # ComprehensiveAnalyzer and NewsAnalyzer are already defined globally
                                # Using OPTIONS as default for signal generation
                                analysis = ComprehensiveAnalyzer.analyze_stock(symbol, "OPTIONS")
                                
                                if analysis:
                                    technical_data_dict[symbol] = {
                                        'price': analysis.price,
                                        'change_pct': analysis.change_pct,
                                        'rsi': analysis.rsi,
                                        'macd_signal': analysis.macd_signal,
                                        'trend': analysis.trend,
                                        'volume': analysis.volume,
                                        'avg_volume': analysis.avg_volume,
                                        'support': analysis.support,
                                        'resistance': analysis.resistance,
                                        'iv_rank': analysis.iv_rank
                                    }
                                    
                                    news_data_dict[symbol] = analysis.recent_news
                                    
                                    sentiment_data_dict[symbol] = {
                                        'score': analysis.sentiment_score,
                                        'signals': analysis.sentiment_signals
                                    }
                            except Exception as e:
                                analysis = ComprehensiveAnalyzer.analyze_stock(symbol)
                                if not analysis:
                                    st.write(f"⚠️ Error analyzing {symbol}: Could not retrieve analysis.")
                                    continue
                        
                        st.write("Running AI analysis...")
                        
                        # Get account balance
                        account_balance = 10000.0  # Default
                        try:
                            if 'tradier_client' in st.session_state and st.session_state.tradier_client:
                                success, bal_data = st.session_state.tradier_client.get_account_balance()
                                if success and isinstance(bal_data, dict):
                                    # Tradier returns { 'balances': { 'total_cash': ... } }
                                    b = bal_data.get('balances') or {}
                                    account_balance = float(b.get('total_cash') or 0.0)
                        except Exception:
                            pass
                        
                        # Generate signals using the AI generator
                        signals = ai_generator.batch_analyze(
                            symbols=symbols_list,
                            technical_data_dict=technical_data_dict,
                            news_data_dict=news_data_dict,
                            sentiment_data_dict=sentiment_data_dict,
                            account_balance=account_balance,
                            risk_tolerance=ai_risk
                        )
                        
                        status.update(label=f"✅ AI analysis complete! Found {len(signals)} signals", state="complete")
                        
                        if signals:
                            st.success(f"🎯 AI found {len(signals)} high-confidence trading opportunities!")
                            
                            # Display signals
                            for idx, signal in enumerate(signals, 1):
                                with st.container():
                                    # Signal header with color
                                    signal_color = "🟢" if signal.signal == "BUY" else "🔴" if signal.signal == "SELL" else "⚪"
                                    
                                    col_sig1, col_sig2, col_sig3 = st.columns([2, 1, 1])
                                    
                                    with col_sig1:
                                        st.markdown(f"### {signal_color} {idx}. {signal.symbol} - {signal.signal}")
                                        st.write(f"**AI Reasoning:** {signal.reasoning}")
                                    
                                    with col_sig2:
                                        st.metric("Confidence", f"{signal.confidence:.0f}%")
                                        st.metric("Risk Level", signal.risk_level)
                                    
                                    with col_sig3:
                                        st.metric("Position Size", f"{signal.position_size} shares")
                                        st.metric("Time Horizon", signal.time_horizon)
                                    
                                    # Trading details
                                    col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                                    
                                    with col_detail1:
                                        if signal.entry_price:
                                            st.write(f"**Entry:** ${signal.entry_price:.2f}")
                                    
                                    with col_detail2:
                                        if signal.target_price:
                                            st.write(f"**Target:** ${signal.target_price:.2f}")
                                            profit_pct = ((signal.target_price - signal.entry_price) / signal.entry_price * 100) if signal.entry_price else 0
                                            st.write(f"📈 {profit_pct:+.1f}%")
                                    
                                    with col_detail3:
                                        if signal.stop_loss:
                                            st.write(f"**Stop:** ${signal.stop_loss:.2f}")
                                            loss_pct = ((signal.stop_loss - signal.entry_price) / signal.entry_price * 100) if signal.entry_price else 0
                                            st.write(f"📉 {loss_pct:.1f}%")
                                    
                                    with col_detail4:
                                        potential_profit = (signal.target_price - signal.entry_price) * signal.position_size if signal.entry_price and signal.target_price else 0
                                        st.write(f"**Potential:** ${potential_profit:,.0f}")
                                    
                                    # AI Scores
                                    st.write("**AI Analysis Scores:**")
                                    score_cols = st.columns(4)
                                    
                                    with score_cols[0]:
                                        st.metric("Technical", f"{signal.technical_score:.0f}/100")
                                    with score_cols[1]:
                                        st.metric("Sentiment", f"{signal.sentiment_score:.0f}/100")
                                    with score_cols[2]:
                                        st.metric("News", f"{signal.news_score:.0f}/100")
                                    with score_cols[3]:
                                        st.metric("Social", f"{signal.social_score:.0f}/100")
                                    
                                    # Quick execute buttons
                                    col_exec1, col_exec2 = st.columns(2)
                                    
                                    with col_exec1:
                                        if st.button(f"✅ Execute {signal.signal} Order", key=f"exec_{signal.symbol}_{idx}", type="primary", width="stretch"):
                                            st.session_state[f'execute_signal_{signal.symbol}'] = signal
                                            st.success(f"Ready to execute! Go to order entry below to place {signal.signal} order for {signal.symbol}")
                                    
                                    with col_exec2:
                                        if st.button(f"📋 Copy to Order Form", key=f"copy_{signal.symbol}_{idx}", width="stretch"):
                                            # Pre-fill order form
                                            st.session_state['ai_prefill_symbol'] = signal.symbol
                                            st.session_state['ai_prefill_qty'] = signal.position_size
                                            st.session_state['ai_prefill_side'] = signal.signal
                                            st.success(f"Copied to order form! Scroll down to execute.")
                                    
                                    st.divider()
                        else:
                            st.warning("🤔 AI didn't find any high-confidence signals right now.")
                            st.info("""
**Why no signals?**
- **Market is closed** - After-hours data is less reliable
- **Conservative AI** - The AI is being cautious (good thing!)
- **Current symbols** - Try popular tickers: SPY, QQQ, AAPL, MSFT, NVDA, TSLA
- **Risk tolerance** - Try changing from MEDIUM to LOW for more signals

**Tips:**
- Run during market hours (9:30 AM - 4:00 PM ET) for best results
- Use highly liquid stocks (SPY, QQQ, major tech stocks)
- Check back when market conditions improve
                            """)
                    
                    except Exception as e:
                        status.update(label="❌ AI analysis failed", state="error")
                        st.error(f"Error: {e}")
    
    st.divider()
    
    # Platform selection
    st.subheader("🔌 Select Trading Platform")
    
    col_platform1, col_platform2 = st.columns(2)
    
    with col_platform1:
        scalp_platform = st.radio(
            "Trading Platform",
            options=["Tradier", "IBKR"],
            horizontal=True,
            help="Choose which broker to use for scalping"
        )
    
    with col_platform2:
        # Use session state to track auto-refresh but ensure it defaults to False
        auto_refresh = st.toggle("Auto-refresh positions", value=st.session_state.get('auto_refresh_enabled', False), 
                                help="Automatically refresh every 5 seconds", key='auto_refresh_toggle')
        # Update session state only if changed
        st.session_state.auto_refresh_enabled = auto_refresh
    
    st.divider()
    
    # Check connection status
    if scalp_platform == "Tradier":
        # Initialize if not exists
        if 'tradier_client' not in st.session_state:
            st.session_state.tradier_client = None
        
        # Check if client exists and is valid
        if st.session_state.tradier_client is None:
            st.warning("⚠️ Tradier not connected.")
            
            # Try to initialize from environment
            col_init1, col_init2 = st.columns(2)
            
            with col_init1:
                if st.button("🔗 Connect to Tradier", type="primary", width="stretch"):
                    try:
                        from src.integrations.tradier_client import create_tradier_client_from_env
                        # Get current trading mode and create client
                        mode_manager = get_trading_mode_manager()
                        client = create_tradier_client_from_env()
                        if client:
                            st.session_state.tradier_client = client
                            logger.info("Tradier client connected successfully")
                            logger.info("Trading mode: %s", mode_manager.get_mode().value)
                            st.success(f"✅ Connected to Tradier ({mode_manager.get_mode().value.title()} Mode)!")
                            st.rerun()
                        else:
                            st.error("Failed to initialize Tradier client. Check your .env file.")
                            logger.error("Tradier client initialization returned None")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                        logger.error(f"Tradier connection error: {e}", exc_info=True)
            
            with col_init2:
                st.info("Or go to **🏦 Tradier Account** tab to configure.")
            
            st.stop()
        
        tradier_client = st.session_state.tradier_client
        
        # Validate client has required attributes
        if not hasattr(tradier_client, 'get_account_balance'):
            st.error("⚠️ Tradier client is not properly initialized.")
            if st.button("🔄 Reinitialize Client"):
                st.session_state.tradier_client = None
                st.rerun()
            st.stop()
        
        # Account summary
        st.subheader("💼 Account Summary")
        try:
            balance = tradier_client.get_account_balance()
            if balance and hasattr(balance, 'total_equity'):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Account Value", f"${balance.total_equity:,.2f}")
                with col2:
                    st.metric("Cash Available", f"${balance.total_cash:,.2f}")
                with col3:
                    st.metric("Buying Power", f"${balance.option_buying_power:,.2f}")
                with col4:
                    day_trade_buying_power = getattr(balance, 'day_trade_buying_power', balance.option_buying_power)
                    st.metric("Day Trade Power", f"${day_trade_buying_power:,.2f}")
            else:
                st.warning("Unable to fetch account balance. Please check connection.")
        except Exception as e:
            st.error(f"Error fetching account balance: {str(e)}")
            logger.error(f"Account balance error: {e}", exc_info=True)
        
        st.divider()
        
        # Quick order entry
        st.subheader("🎯 Quick Order Entry")
        
        # Check for AI prefill
        ai_symbol = st.session_state.get('ai_prefill_symbol', 'SPY')
        ai_qty = st.session_state.get('ai_prefill_qty', 100)
        ai_side = st.session_state.get('ai_prefill_side', 'BUY')
        
        # Show AI recommendation if available
        if 'ai_prefill_symbol' in st.session_state:
            st.info(f"🤖 AI Recommendation loaded: {ai_side} {ai_qty} shares of {ai_symbol}")
        
        col_entry1, col_entry2, col_entry3 = st.columns([2, 1, 1])
        
        with col_entry1:
            scalp_symbol = st.text_input("Symbol", value=ai_symbol, key="scalp_symbol_tradier").upper()
            
        with col_entry2:
            scalp_quantity = st.number_input("Shares", min_value=1, value=ai_qty, step=1, key="scalp_qty_tradier")
        
        with col_entry3:
            side_index = 0 if ai_side == "BUY" else 1
            scalp_side = st.selectbox("Side", options=["BUY", "SELL"], index=side_index, key="scalp_side_tradier")
        
        col_order1, col_order2, col_order3 = st.columns(3)
        
        with col_order1:
            if st.button("🚀 Market Order", type="primary", width="stretch", key="market_tradier"):
                if scalp_symbol:
                    try:
                        with st.spinner(f"Placing market order: {scalp_side} {scalp_quantity} {scalp_symbol}..."):
                            order = tradier_client.place_equity_order(
                                symbol=scalp_symbol,
                                side=scalp_side.lower(),
                                quantity=scalp_quantity,
                                order_type='market',
                                duration='day'
                            )
                            if order:
                                st.success(f"✅ Order placed! ID: {order.get('id', 'N/A')}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Order failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col_order2:
            limit_price = st.number_input("Limit $", min_value=0.01, value=100.0, step=0.01, key="limit_price_tradier")
            if st.button("📊 Limit Order", width="stretch", key="limit_tradier"):
                if scalp_symbol:
                    try:
                        order = tradier_client.place_equity_order(
                            symbol=scalp_symbol,
                            side=scalp_side.lower(),
                            quantity=scalp_quantity,
                            order_type='limit',
                            duration='day',
                            price=limit_price
                        )
                        if order:
                            st.success(f"✅ Limit order placed at ${limit_price}")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col_order3:
            stop_price = st.number_input("Stop $", min_value=0.01, value=100.0, step=0.01, key="stop_price_tradier")
            if st.button("🛑 Stop Order", width="stretch", key="stop_tradier"):
                if scalp_symbol:
                    try:
                        order = tradier_client.place_equity_order(
                            symbol=scalp_symbol,
                            side=scalp_side.lower(),
                            quantity=scalp_quantity,
                            order_type='stop',
                            duration='day',
                            stop=stop_price
                        )
                        if order:
                            st.success(f"✅ Stop order placed at ${stop_price}")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.divider()
        
        # Current positions
        st.subheader("📊 Current Positions")
        
        col_pos1, col_pos2 = st.columns([3, 1])
        
        with col_pos2:
            if st.button("🔄 Refresh", width="stretch", key="refresh_pos_tradier"):
                st.rerun()
        
        try:
            success, positions = tradier_client.get_positions()
            
            if not success:
                st.warning("⚠️ Unable to fetch positions from Tradier API")
                positions = []
            
            if positions and isinstance(positions, list) and len(positions) > 0:
                positions_data = []
                
                for pos in positions:
                    if not isinstance(pos, dict):
                        continue
                    
                    # Get current quote
                    try:
                        symbol = pos.get('symbol', '')
                        if not symbol:
                            continue
                        
                        quote = tradier_client.get_quote(symbol)
                        if isinstance(quote, dict):
                            current_price = float(quote.get('last', 0))
                        else:
                            current_price = 0
                    except Exception as e:
                        logger.warning(f"Error getting quote for {symbol}: {e}")
                        current_price = 0
                    
                    try:
                        cost_basis = float(pos.get('cost_basis', 0))
                        quantity = float(pos.get('quantity', 0))
                        avg_price = cost_basis / quantity if quantity != 0 else 0
                        current_value = current_price * quantity
                        pnl = current_value - cost_basis
                        pnl_pct = (pnl / cost_basis * 100) if cost_basis != 0 else 0
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error calculating position metrics: {e}")
                        continue
                    
                    positions_data.append({
                        'Symbol': pos['symbol'],
                        'Qty': int(quantity),
                        'Avg Price': f"${avg_price:.2f}",
                        'Current': f"${current_price:.2f}",
                        'Value': f"${current_value:,.2f}",
                        'P&L': f"${pnl:,.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%",
                        '_pnl_raw': pnl  # Hidden column for styling
                    })
                
                # Display positions
                # pandas already imported at module level
                df_positions = pd.DataFrame(positions_data)
                
                # Style the dataframe
                def color_pnl(val):
                    if 'P&L' in val.name or 'P&L %' in val.name:
                        return ['color: green' if '+' in str(x) or (isinstance(x, (int, float)) and x > 0) else 'color: red' if '-' in str(x) or (isinstance(x, (int, float)) and x < 0) else '' for x in val]
                    return ['' for _ in val]
                
                # Remove hidden column before display
                display_df = df_positions.drop(columns=['_pnl_raw'])
                st.dataframe(display_df, width="stretch", height=300)
                
                # Quick close buttons
                st.write("**Quick Actions:**")
                cols = st.columns(min(len(positions), 4))
                
                for idx, pos in enumerate(positions[:4]):
                    with cols[idx]:
                        symbol = pos['symbol']
                        qty = int(float(pos.get('quantity', 0)))
                        
                        if st.button(f"❌ Close {symbol}", key=f"close_{symbol}_tradier", width="stretch"):
                            side = 'sell' if qty > 0 else 'buy'
                            try:
                                order = tradier_client.place_equity_order(
                                    symbol=symbol,
                                    side=side,
                                    quantity=abs(qty),
                                    order_type='market',
                                    duration='day'
                                )
                                if order:
                                    st.success(f"✅ Closing {symbol}")
                                    time.sleep(1)
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.info("No open positions")
        
        except Exception as e:
            st.error(f"Error fetching positions: {e}")
            logger.error(f"Positions error details: {type(positions) if 'positions' in locals() else 'undefined'}", exc_info=True)
        
        st.divider()
        
        # Open orders
        st.subheader("📝 Open Orders")
        
        try:
            success, orders = tradier_client.get_orders()
            
            if not success:
                st.warning("⚠️ Unable to fetch orders from Tradier API")
                orders = []
            
            if orders and isinstance(orders, list) and len(orders) > 0:
                orders_data = []
                
                for order in orders:
                    if not isinstance(order, dict):
                        continue
                    
                    if order.get('status') not in ['filled', 'canceled', 'rejected']:
                        orders_data.append({
                            'ID': order.get('id', 'N/A'),
                            'Symbol': order.get('symbol', 'N/A'),
                            'Side': order.get('side', 'N/A').upper(),
                            'Qty': order.get('quantity', 0),
                            'Type': order.get('type', 'N/A').upper(),
                            'Price': f"${order.get('price', 0):.2f}" if order.get('price') else 'N/A',
                            'Status': order.get('status', 'N/A').upper()
                        })
                
                if orders_data:
                    df_orders = pd.DataFrame(orders_data)
                    st.dataframe(df_orders, width="stretch")
                    
                    # Cancel orders
                    col_cancel1, col_cancel2 = st.columns([2, 1])
                    
                    with col_cancel1:
                        order_id_cancel = st.text_input("Order ID to cancel", key="cancel_order_id_tradier")
                    
                    with col_cancel2:
                        st.write("")
                        st.write("")
                        if st.button("❌ Cancel Order", key="cancel_order_tradier"):
                            if order_id_cancel:
                                try:
                                    result = tradier_client.cancel_order(int(order_id_cancel))
                                    if result:
                                        st.success(f"Order {order_id_cancel} cancelled")
                                        time.sleep(1)
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                else:
                    st.info("No pending orders")
            else:
                st.info("No open orders")
        
        except Exception as e:
            st.error(f"Error fetching orders: {e}")
            logger.error(f"Orders error details: {type(orders) if 'orders' in locals() else 'undefined'}", exc_info=True)
    
    elif scalp_platform == "IBKR":
        # Check IBKR connection
        if not st.session_state.get('ibkr_connected') or not st.session_state.get('ibkr_client'):
            st.warning("⚠️ IBKR not connected. Go to **📈 IBKR Trading** tab to connect.")
            st.stop()
        
        try:
            from src.integrations.ibkr_client import IBKRClient
            ibkr_client = st.session_state.ibkr_client
            
            # Account summary
            st.subheader("💼 Account Summary")
            try:
                account_info = ibkr_client.get_account_info()
                if account_info:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Net Liquidation", f"${account_info.net_liquidation:,.2f}")
                    with col2:
                        st.metric("Cash", f"${account_info.total_cash_value:,.2f}")
                    with col3:
                        st.metric("Buying Power", f"${account_info.buying_power:,.2f}")
                    with col4:
                        st.metric("Day Trades Left", "Unlimited" if account_info.is_pdt and account_info.net_liquidation >= 25000 else str(account_info.day_trades_remaining))
            except Exception as e:
                st.error(f"Error fetching account: {e}")
            
            st.divider()
            
            # Quick order entry (same as Tradier but using IBKR client)
            st.subheader("🎯 Quick Order Entry")
            
            col_entry1, col_entry2, col_entry3 = st.columns([2, 1, 1])
            
            with col_entry1:
                scalp_symbol_ibkr = st.text_input("Symbol", value="SPY", key="scalp_symbol_ibkr").upper()
                
            with col_entry2:
                scalp_quantity_ibkr = st.number_input("Shares", min_value=1, value=100, step=1, key="scalp_qty_ibkr")
            
            with col_entry3:
                scalp_side_ibkr = st.selectbox("Side", options=["BUY", "SELL"], key="scalp_side_ibkr")
            
            col_order1, col_order2, col_order3 = st.columns(3)
            
            with col_order1:
                if st.button("🚀 Market Order", type="primary", width="stretch", key="market_ibkr"):
                    try:
                        order = ibkr_client.place_market_order(scalp_symbol_ibkr, scalp_side_ibkr, scalp_quantity_ibkr)
                        if order:
                            st.success(f"✅ Order placed! ID: {order.order_id}")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col_order2:
                limit_price_ibkr = st.number_input("Limit $", min_value=0.01, value=100.0, step=0.01, key="limit_price_ibkr")
                if st.button("📊 Limit Order", width="stretch", key="limit_ibkr"):
                    try:
                        order = ibkr_client.place_limit_order(scalp_symbol_ibkr, scalp_side_ibkr, scalp_quantity_ibkr, limit_price_ibkr)
                        if order:
                            st.success(f"✅ Limit order placed")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            with col_order3:
                stop_price_ibkr = st.number_input("Stop $", min_value=0.01, value=100.0, step=0.01, key="stop_price_ibkr")
                if st.button("🛑 Stop Order", width="stretch", key="stop_ibkr"):
                    try:
                        order = ibkr_client.place_stop_order(scalp_symbol_ibkr, scalp_side_ibkr, scalp_quantity_ibkr, stop_price_ibkr)
                        if order:
                            st.success(f"✅ Stop order placed")
                            time.sleep(1)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.divider()
            
            # Positions (similar to Tradier)
            st.subheader("📊 Current Positions")
            
            col_pos1, col_pos2 = st.columns([3, 1])
            
            with col_pos2:
                if st.button("🔄 Refresh", width="stretch", key="refresh_pos_ibkr"):
                    st.rerun()
            
            try:
                positions = ibkr_client.get_positions()
                
                if positions:
                    positions_data = []
                    for pos in positions:
                        positions_data.append({
                            'Symbol': pos.symbol,
                            'Qty': int(pos.position),
                            'Avg Price': f"${pos.avg_cost:.2f}",
                            'Current': f"${pos.market_price:.2f}",
                            'Value': f"${pos.market_value:,.2f}",
                            'P&L': f"${pos.unrealized_pnl:,.2f}"
                        })
                    
                    # pandas already imported at module level
                    df_positions = pd.DataFrame(positions_data)
                    st.dataframe(df_positions, width="stretch", height=300)
                    
                    # Quick close
                    st.write("**Quick Actions:**")
                    cols = st.columns(min(len(positions), 4))
                    for idx, pos in enumerate(positions[:4]):
                        with cols[idx]:
                            if st.button(f"❌ Close {pos.symbol}", key=f"close_{pos.symbol}_ibkr", width="stretch"):
                                if ibkr_client.flatten_position(pos.symbol):
                                    st.success(f"✅ Closing {pos.symbol}")
                                    time.sleep(1)
                                    st.rerun()
                else:
                    st.info("No open positions")
            
            except Exception as e:
                st.error(f"Error: {e}")
            
        except ImportError:
            st.error("IBKR client not available")
    
    # Auto-refresh functionality - only rerun if explicitly enabled
    # This prevents constant background reloads
    # The toggle state is already stored in auto_refresh variable, no need for redundant check
    if auto_refresh:
        # Add a timestamp check to prevent immediate reruns
        last_refresh = st.session_state.get('last_auto_refresh_time', 0)
        current_time = time.time()
        # Only refresh if at least 5 seconds have passed
        if current_time - last_refresh >= 5:
            st.session_state['last_auto_refresh_time'] = current_time
            time.sleep(5)
            st.rerun()


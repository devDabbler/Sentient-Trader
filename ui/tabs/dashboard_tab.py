"""
Dashboard Tab
Main dashboard with stock analysis, signal generation, and quick execution

Extracted from app.py for modularization
"""
import streamlit as st
import time
from loguru import logger
from typing import Dict, List, Optional, Tuple

def render_tab():
    """Main render function called from app.py"""
    st.header("Dashboard")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    # Render only the selected tab content
    logger.info(f"🏁 TAB1 RENDERING - Session state: show_quick_trade={st.session_state.get('show_quick_trade', 'NOT SET')}, has_analysis={st.session_state.get('current_analysis') is not None}")
    st.header("🔍 Comprehensive Stock Intelligence")
    st.write("Get real-time analysis including news, catalysts, technical indicators, and IV metrics.")
    st.info("💡 **Works with ALL stocks:** Blue chips, penny stocks (<$5), OTC stocks, and runners. Automatically detects momentum plays!")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_ticker = st.text_input(
            "Enter Ticker Symbol to Analyze", 
            value="SOFI",
            help="Enter any ticker: AAPL, TSLA, penny stocks (SNDL, GNUS), or OTC stocks"
        ).upper()
    
    with col2:
        trading_style_display = st.selectbox(
            "Trading Style",
            options=[
                "🤖 AI Analysis",
                "📊 Day Trade", 
                "📈 Swing Trade", 
                "⚡ Scalp", 
                "⚔️ Warrior Scalping",
                "💎 Buy & Hold", 
                "🎯 Options",
                "📊 ORB+FVG (15min)",
                "📈 EMA Crossover + Heikin Ashi",
                "📊 RSI + Stochastic + Hammer",
                "🎯 Fisher RSI Multi-Indicator",
                "📉 MACD + Volume + RSI",
                "🔥 Aggressive Scalping"
            ],
            index=0,
            help="Select your trading style for personalized recommendations"
        )
        # Map display names to internal codes
        style_map = {
            "🤖 AI Analysis": "AI",
            "📊 Day Trade": "DAY_TRADE",
            "📈 Swing Trade": "SWING_TRADE",
            "⚡ Scalp": "SCALP",
            "⚔️ Warrior Scalping": "WARRIOR_SCALPING",
            "💎 Buy & Hold": "BUY_HOLD",
            "🎯 Options": "OPTIONS",
            "📊 ORB+FVG (15min)": "ORB_FVG",
            "📈 EMA Crossover + Heikin Ashi": "EMA_HEIKIN_ASHI",
            "📊 RSI + Stochastic + Hammer": "RSI_STOCHASTIC_HAMMER",
            "🎯 Fisher RSI Multi-Indicator": "FISHER_RSI",
            "📉 MACD + Volume + RSI": "MACD_VOLUME_RSI",
            "🔥 Aggressive Scalping": "AGGRESSIVE_SCALPING"
        }
        trading_style = style_map[trading_style_display]
    
    with col3:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Analyze Stock", type="primary", width="stretch")
    
    # Quick examples with style descriptions
    st.caption("**Examples:** AAPL (blue chip) | SNDL (penny stock) | SPY (ETF) | TSLA (volatile) | Any OTC stock")
    
    # Dynamic caption based on selected style
    style_descriptions = {
        "AI": "🤖 **AI Analysis:** Machine learning-powered analysis with confidence scores and multi-factor signals",
        "DAY_TRADE": "💡 **Day Trade:** Intraday equity trades, exit by market close (0.5-3% targets)",
        "SWING_TRADE": "💡 **Swing Trade:** Multi-day equity holds, 3-10 day timeframe (5-15% targets)",
        "SCALP": "💡 **Scalp:** Ultra-short term, seconds to minutes (0.1-0.5% targets, high risk)",
        "WARRIOR_SCALPING": "⚔️ **Warrior Scalping:** Aggressive momentum scalping with gap analysis (1-3% targets)",
        "BUY_HOLD": "💡 **Buy & Hold:** Long-term investing, 6+ months (20%+ annual targets)",
        "OPTIONS": "💡 **Options:** Calls, puts, spreads based on IV and trend analysis",
        "ORB_FVG": "📊 **ORB+FVG (15min):** Opening Range Breakout with Fair Value Gap confirmation (1-2R targets, proven $6.5k/month)",
        "EMA_HEIKIN_ASHI": "📈 **EMA Crossover + Heikin Ashi:** Freqtrade strategy with EMA 20/50/100 crossovers",
        "RSI_STOCHASTIC_HAMMER": "📊 **RSI + Stochastic + Hammer:** Freqtrade strategy with oversold signals and candlestick patterns",
        "FISHER_RSI": "🎯 **Fisher RSI Multi-Indicator:** Freqtrade strategy combining Fisher RSI with MFI and Stochastic",
        "MACD_VOLUME_RSI": "📉 **MACD + Volume + RSI:** Freqtrade strategy with MACD crossovers and volume confirmation",
        "AGGRESSIVE_SCALPING": "🔥 **Aggressive Scalping:** Freqtrade strategy with fast EMA crosses and tight stops (1-3% targets)"
    }
    st.caption(style_descriptions.get(trading_style, "Select a trading style for analysis"))
    
    # Quick Trade Modal - AT TOP so it's immediately visible when Execute button is clicked
    logger.info(f"🔍 Checking modal display: show_quick_trade={st.session_state.get('show_quick_trade', False)}")
    if st.session_state.get('show_quick_trade', False):
        logger.info("🚀 DISPLAYING QUICK TRADE MODAL AT TOP OF TAB1")
        st.divider()
        st.header("🚀 Execute Trade")
        
        # Get the selected recommendation and analysis
        selected_rec = st.session_state.get('selected_recommendation', None)
        analysis = st.session_state.get('current_analysis', None)
        
        if not analysis:
            logger.error("❌ Modal error: No analysis data in session state")
            st.error("❌ Analysis data not available. Please analyze a stock first.")
            if st.button("Close"):
                st.session_state.show_quick_trade = False
                # Modal close needs rerun to update UI
                st.rerun()
        else:
            logger.info(f"✅ Modal has analysis data: ticker={analysis.ticker}, price={analysis.price}")
            if selected_rec:
                logger.info(f"✅ Modal has recommendation: {selected_rec.get('type')} - {selected_rec.get('strategy', 'N/A')}")
                st.subheader(f"📋 {selected_rec['type']} - {selected_rec.get('strategy', selected_rec.get('action', ''))}")
            else:
                st.subheader(f"📋 Quick Trade: {st.session_state.get('quick_trade_ticker', 'N/A')}")
            
            # Check if Tradier is connected
            if not st.session_state.tradier_client:
                st.error("❌ Tradier not connected. Please configure in the 🏦 Tradier Account tab.")
                if st.button("Close", key="close_no_tradier"):
                    st.session_state.show_quick_trade = False
                    # Modal close needs rerun to update UI
                    st.rerun()
            else:
                verdict_action = st.session_state.get('quick_trade_verdict', 'N/A')
                st.success(f"✅ Tradier Connected | Verdict: **{verdict_action}**")
                
                # Show AI recommendation details if available
                if selected_rec:
                    st.info(f"**AI Reasoning:** {selected_rec.get('reasoning', 'N/A')}")
                    if selected_rec['type'] == 'STOCK':
                        st.caption(f"Stop Loss: ${selected_rec['stop_loss']:.2f} | Target: ${selected_rec['target']:.2f} | Hold: {selected_rec['hold_time']}")
                    else:
                        st.caption(f"Strike: {selected_rec.get('strike_suggestion', 'N/A')} | DTE: {selected_rec.get('dte_suggestion', 'N/A')}")
                
                trade_col1, trade_col2 = st.columns(2)
                
                with trade_col1:
                    st.write("**Order Configuration:**")
                    
                    # Pre-fill based on recommendation
                    if selected_rec and selected_rec['type'] == 'STOCK':
                        default_symbol = selected_rec['symbol']
                        default_action = selected_rec['action'].lower().replace('_', '_')
                        default_qty = 10
                        default_type = selected_rec['order_type']
                        default_price = selected_rec.get('price', st.session_state.get('quick_trade_price', analysis.price))
                    elif selected_rec and selected_rec['type'] == 'OPTION':
                        default_symbol = selected_rec['symbol']
                        default_action = selected_rec['action']
                        default_qty = selected_rec.get('quantity', 1)
                        default_type = "limit"
                        default_price = st.session_state.get('quick_trade_price', analysis.price)
                    else:
                        default_symbol = st.session_state.get('quick_trade_ticker', analysis.ticker)
                        default_action = "buy"
                        default_qty = 10
                        default_type = "market"
                        default_price = st.session_state.get('quick_trade_price', analysis.price)
                    
                    trade_symbol = st.text_input("Symbol", value=default_symbol, key="modal_trade_symbol")
                    
                    # Determine if this is an options trade
                    is_options_trade = selected_rec and selected_rec['type'] == 'OPTION'
                    
                    if is_options_trade:
                        st.warning("⚠️ **Options Trade:** You'll need to specify the exact option symbol (e.g., AAPL250117C150)")
                        trade_class = st.selectbox("Order Class", ["option", "equity"], index=0, key="modal_trade_class")
                        
                        if trade_class == "option":
                            st.info(f"💡 **Suggested Strike:** {selected_rec.get('strike_suggestion', 'N/A')}")
                            st.info(f"💡 **Suggested Expiration:** {selected_rec.get('dte_suggestion', 'N/A')}")
                            
                            # Add helpful information about finding options symbols
                            with st.expander("📋 How to find valid options symbols", expanded=False):
                                st.markdown("""
                                **Since Tradier sandbox has limited options data, here's how to find valid symbols:**
                                
                                1. **Tradier Web Platform**: 
                                   - Go to [sandbox.tradier.com](https://sandbox.tradier.com)
                                   - Search for SOFI options
                                   - Copy the exact symbol format
                                
                                2. **Yahoo Finance**:
                                   - Search "SOFI options"
                                   - Look for the symbol format: `SOFI251126C00025000`
                                
                                3. **Common SOFI Strike Prices** (as of recent):
                                   - $20, $22.50, $25, $27.50, $30, $32.50, $35
                                
                                4. **Common Expiration Dates**:
                                   - Weekly: Every Friday
                                   - Monthly: Third Friday of each month
                                
                                **Symbol Format**: `SOFI + YYMMDD + C/P + 8-digit strike`
                                - Example: `SOFI251126C00025000` = SOFI $25 Call expiring 11/26/25
                                """)
                            
                            # Generate options contract symbol automatically
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Auto-generate options symbol if we have the required data
                                auto_generated_symbol = ""
                                if selected_rec and selected_rec.get('strike_suggestion') and selected_rec.get('dte_suggestion'):
                                    try:
                                        # Parse strike price and round to nearest $0.50 or $1.00
                                        strike = float(selected_rec['strike_suggestion'].replace('$', '').split()[0])
                                        
                                        # Round to nearest $0.50 for better strike price availability
                                        if strike < 10:
                                            strike = round(strike * 2) / 2  # Round to nearest $0.50
                                        else:
                                            strike = round(strike)  # Round to nearest $1.00
                                        
                                        # Calculate expiration date (DTE from today)
                                        dte_text = selected_rec['dte_suggestion'].split()[0]
                                        # Handle range format like "30-45" by taking the first number
                                        if '-' in dte_text:
                                            dte = int(dte_text.split('-')[0])
                                        else:
                                            dte = int(dte_text)
                                        
                                        # Round to common options expiration dates (Fridays)
                                        exp_date = datetime.now() + timedelta(days=dte)
                                        # Find the next Friday (options typically expire on Fridays)
                                        days_until_friday = (4 - exp_date.weekday()) % 7
                                        if days_until_friday == 0 and exp_date.weekday() != 4:  # If today is not Friday
                                            days_until_friday = 7
                                        exp_date = exp_date + timedelta(days=days_until_friday)
                                        
                                        # Determine option type (P for PUT, C for CALL)
                                        option_type = "P" if "PUT" in selected_rec.get('strategy', '') else "C"
                                        
                                        # Format: SYMBOL + YYMMDD + P/C + 8-digit strike (padded with zeros)
                                        auto_generated_symbol = f"{trade_symbol.upper()}{exp_date.strftime('%y%m%d')}{option_type}{int(strike * 1000):08d}"
                                        
                                        # Set the session state before creating the widget
                                        if 'modal_option_symbol' not in st.session_state or not st.session_state['modal_option_symbol']:
                                            st.session_state['modal_option_symbol'] = auto_generated_symbol
                                    except:
                                        pass
                                
                                # Use temp generated symbol if available, otherwise use existing value
                                default_value = st.session_state.get('temp_generated_symbol', st.session_state.get('modal_option_symbol', ''))
                                if st.session_state.get('temp_generated_symbol'):
                                    # Clear the temp value after using it
                                    st.session_state['modal_option_symbol'] = st.session_state['temp_generated_symbol']
                                    del st.session_state['temp_generated_symbol']
                                
                                option_symbol = st.text_input(
                                    "Options Contract Symbol", 
                                    value=st.session_state.get('modal_option_symbol', ''), 
                                    placeholder="e.g., SOFI250117P00029000",
                                    help="Enter the full options contract symbol (e.g., SOFI250117P00029000 for SOFI $29 Put expiring 01/17/25)",
                                    key="modal_option_symbol"
                                )
                            
                            with col2:
                                if st.button("🔧 Auto-Generate", help="Generate options symbol from strike and DTE"):
                                    if selected_rec and selected_rec.get('strike_suggestion') and selected_rec.get('dte_suggestion'):
                                        try:
                                            # Parse strike price and round to nearest $0.50 or $1.00
                                            strike = float(selected_rec['strike_suggestion'].replace('$', '').split()[0])
                                            
                                            # Round to nearest $0.50 for better strike price availability
                                            if strike < 10:
                                                strike = round(strike * 2) / 2  # Round to nearest $0.50
                                            else:
                                                strike = round(strike)  # Round to nearest $1.00
                                            
                                            # Calculate expiration date (DTE from today)
                                            dte_text = selected_rec['dte_suggestion'].split()[0]
                                            # Handle range format like "30-45" by taking the first number
                                            if '-' in dte_text:
                                                dte = int(dte_text.split('-')[0])
                                            else:
                                                dte = int(dte_text)
                                            
                                            # Round to common options expiration dates (Fridays)
                                            exp_date = datetime.now() + timedelta(days=dte)
                                            # Find the next Friday (options typically expire on Fridays)
                                            days_until_friday = (4 - exp_date.weekday()) % 7
                                            if days_until_friday == 0 and exp_date.weekday() != 4:  # If today is not Friday
                                                days_until_friday = 7
                                            exp_date = exp_date + timedelta(days=days_until_friday)
                                            
                                            # Determine option type (P for PUT, C for CALL)
                                            option_type = "P" if "PUT" in selected_rec.get('strategy', '') else "C"
                                            
                                            # Format: SYMBOL + YYMMDD + P/C + 8-digit strike (padded with zeros)
                                            generated_symbol = f"{trade_symbol.upper()}{exp_date.strftime('%y%m%d')}{option_type}{int(strike * 1000):08d}"
                                            # Store the generated symbol in a temporary session state key
                                            st.session_state['temp_generated_symbol'] = generated_symbol
                                            # Symbol generation needs rerun to populate input field
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error generating symbol: {e}")
                                    else:
                                        st.error("Need strike and DTE suggestions to auto-generate")
                                
                                # Add validation button
                                if st.button("✅ Validate Symbol", help="Check if the options symbol exists"):
                                    if st.session_state.get('modal_option_symbol'):
                                        symbol = st.session_state['modal_option_symbol']
                                        
                                        # Basic format validation first
                                        if len(symbol) < 15:
                                            st.error("❌ Options symbol too short. Expected format: SYMBOL + YYMMDD + C/P + 8-digit strike")
                                        elif not any(c.isdigit() for c in symbol):
                                            st.error("❌ Options symbol must contain numbers for date and strike")
                                        elif not any(c in ['C', 'P'] for c in symbol):
                                            st.error("❌ Options symbol must contain 'C' for Call or 'P' for Put")
                                        else:
                                            with st.spinner("Validating options symbol..."):
                                                success, message = st.session_state.tradier_client.validate_options_symbol(symbol)
                                                if success:
                                                    st.success(f"✅ {message}")
                                                else:
                                                    # Check if it's an API limitation
                                                    if "API limitation" in message:
                                                        st.warning(f"⚠️ {message}")
                                                        st.info("💡 The symbol format looks correct. You can proceed with the trade, but verify the symbol exists on your broker's platform.")
                                                    else:
                                                        st.error(f"❌ {message}")
                                    else:
                                        st.error("Please enter an options symbol first")
                            
                            trade_side = st.selectbox("Action", 
                                                    ["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"],
                                                    index=0 if 'buy' in default_action else 1,
                                                    key="modal_trade_side")
                            trade_quantity = st.number_input("Contracts", min_value=1, value=default_qty, step=1, key="modal_trade_qty")
                        else:
                            trade_side = st.selectbox("Action", ["buy", "sell", "sell_short", "buy_to_cover"], key="modal_trade_side2")
                            trade_quantity = st.number_input("Quantity (shares)", min_value=1, value=default_qty, step=1, key="modal_trade_qty2")
                            option_symbol = None
                    else:
                        trade_class = "equity"
                        if default_action == "SELL_SHORT":
                            side_index = 2
                        elif default_action == "BUY":
                            side_index = 0
                        else:
                            side_index = 0
                        
                        trade_side = st.selectbox("Action", 
                                                ["buy", "sell", "sell_short", "buy_to_cover"],
                                                index=side_index,
                                                key="modal_trade_side3")
                        trade_quantity = st.number_input("Quantity (shares)", min_value=1, value=default_qty, step=1, key="modal_trade_qty3")
                    
                    # Bracket order eligibility check (for display purposes)
                    can_use_bracket = (trade_class == "equity" and trade_side in ["buy", "sell"])
                    
                    trade_type = st.selectbox("Order Type", 
                                            ["market", "limit"],
                                            index=0 if default_type == "market" else 1,
                                            key="modal_trade_type",
                                            help="💡 Select 'limit' to enable automatic bracket orders with stop-loss & take-profit")
                    
                    if trade_type == "limit":
                        trade_price = st.number_input("Limit Price", 
                                                     min_value=0.01, 
                                                     value=float(default_price) if default_price else float(analysis.price),
                                                     step=0.01,
                                                     format="%.2f",
                                                     key="modal_trade_price")
                        
                        # Show bracket order preview
                        if can_use_bracket:
                            # Get stop/target for preview
                            if selected_rec and selected_rec['type'] == 'STOCK':
                                preview_stop = selected_rec['stop_loss']
                                preview_target = selected_rec['target']
                            else:
                                preview_stop = analysis.support
                                preview_target = analysis.resistance
                            
                            # Validate and adjust
                            if trade_side == "buy":
                                if preview_stop >= trade_price:
                                    preview_stop = trade_price * 0.97
                                if preview_target <= trade_price:
                                    preview_target = trade_price * 1.05
                            else:
                                if preview_stop <= trade_price:
                                    preview_stop = trade_price * 1.03
                                if preview_target >= trade_price:
                                    preview_target = trade_price * 0.95
                            
                            st.success(f"🎯 **BRACKET ORDER ACTIVE**")
                            st.info(f"✅ Entry: ${trade_price:.2f} | 🎯 Target: ${preview_target:.2f} | 🛑 Stop: ${preview_stop:.2f}")
                        else:
                            st.info(f"Limit order will execute when price reaches ${trade_price:.2f}")
                    else:
                        trade_price = None
                        st.warning(f"⚠️ Market orders execute immediately - **bracket orders NOT available**")
                        st.info(f"💡 To enable automatic stop-loss & take-profit, change to 'limit' order type")
                
                with trade_col2:
                    st.write("**Order Summary:**")
                    
                    # Show bracket order mode indicator
                    will_use_bracket = (
                        trade_class == "equity" and 
                        trade_type == "limit" and 
                        trade_side in ["buy", "sell"] and
                        trade_price is not None
                    )
                    if will_use_bracket:
                        st.success("🎯 **BRACKET MODE**: Auto stop-loss & take-profit enabled")
                    else:
                        st.info("📊 **SIMPLE ORDER MODE**")
                    
                    st.divider()
                    
                    # Calculate estimated cost
                    if is_options_trade:
                        st.warning("Options pricing requires real-time quote - estimate not available")
                        estimated_cost = "TBD"
                    else:
                        if trade_type == "limit" and trade_price:
                            estimated_cost = trade_price * trade_quantity
                        else:
                            estimated_cost = analysis.price * trade_quantity
                        st.metric("Estimated Cost", f"${estimated_cost:,.2f}")
                    
                    st.metric("Verdict", verdict_action)
                    
                    if selected_rec:
                        st.metric("AI Confidence", f"{selected_rec['confidence']:.0f}/100")
                    
                    # Risk warning based on verdict
                    if verdict_action in ["AVOID / WAIT", "CAUTIOUS BUY"]:
                        st.warning("⚠️ Analysis suggests caution with this trade!")
                    elif verdict_action == "STRONG BUY":
                        st.success("✅ Analysis shows strong confidence!")
                    
                    if selected_rec and selected_rec['type'] == 'STOCK':
                        st.caption(f"**Stop Loss:** ${selected_rec['stop_loss']:.2f}")
                        st.caption(f"**Target:** ${selected_rec['target']:.2f}")
                    elif selected_rec and selected_rec['type'] == 'OPTION':
                        st.caption(f"**Max Profit:** {selected_rec.get('max_profit', 'N/A')}")
                        st.caption(f"**Max Risk:** {selected_rec.get('max_risk', 'N/A')}")
                    else:
                        st.caption(f"**Stop Loss Suggestion:** ${analysis.support:.2f}")
                        st.caption(f"**Target Suggestion:** ${analysis.resistance:.2f}")
                
                # Place order button
                st.write("")
                confirm_col1, confirm_col2 = st.columns(2)
                
                with confirm_col1:
                    if st.button("✅ Place Order", type="primary", width="stretch", key="modal_place_order"):
                        with st.spinner("Placing order..."):
                            try:
                                # Validate required fields
                                if not trade_symbol:
                                    st.error("❌ Please enter a symbol")
                                    st.stop()
                                elif trade_quantity <= 0:
                                    st.error("❌ Quantity must be greater than 0")
                                    st.stop()
                                elif trade_type == "limit" and (not trade_price or trade_price <= 0):
                                    st.error("❌ Please enter a valid limit price")
                                    st.stop()
                                elif trade_class == "option" and (not st.session_state.get('modal_option_symbol', '')):
                                    st.error("❌ Please enter the options contract symbol (e.g., SOFI250117P00029000)")
                                    st.stop()
                                
                                # Determine if we can use bracket orders
                                # Bracket orders require: equity class, limit entry, buy/sell side, and stop/target prices
                                use_bracket = (
                                    trade_class == "equity" and 
                                    trade_type == "limit" and 
                                    trade_side in ["buy", "sell"] and
                                    trade_price is not None
                                )
                                
                                if use_bracket:
                                    # Get stop loss and target prices
                                    if selected_rec and selected_rec['type'] == 'STOCK':
                                        stop_loss = selected_rec['stop_loss']
                                        target = selected_rec['target']
                                    else:
                                        # Use technical support/resistance levels
                                        stop_loss = analysis.support
                                        target = analysis.resistance
                                    
                                    # Validate that stop/target make sense for the order direction
                                    if trade_side == "buy":
                                        # For buy orders: stop should be below entry, target above
                                        if stop_loss >= trade_price:
                                            stop_loss = round(trade_price * 0.97, 2)  # Default 3% stop
                                        if target <= trade_price:
                                            target = round(trade_price * 1.05, 2)  # Default 5% target
                                    else:
                                        # For sell orders: stop should be above entry, target below
                                        if stop_loss <= trade_price:
                                            stop_loss = round(trade_price * 1.03, 2)
                                        if target >= trade_price:
                                            target = round(trade_price * 0.95, 2)
                                    logger.info(f"🎯 Placing bracket order: {trade_symbol} {trade_side} {trade_quantity} @ ${trade_price} (SL: ${stop_loss:.2f}, Target: ${target:.2f})")
                                    
                                    # Prepare bracket order parameters
                                    bracket_params = {
                                        "symbol": trade_symbol.upper(),
                                        "side": trade_side,
                                        "quantity": trade_quantity,
                                        "entry_price": trade_price,
                                        "take_profit_price": target,
                                        "stop_loss_price": stop_loss,
                                        "duration": 'gtc',  # Use GTC for bracket orders
                                        "tag": f"AIREC{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                    }
                                    
                                    # Add option_symbol if this is an options trade
                                    if trade_class == "option" and st.session_state.get('modal_option_symbol'):
                                        bracket_params["option_symbol"] = st.session_state['modal_option_symbol'].upper()
                                    
                                    success, result = st.session_state.tradier_client.place_bracket_order(**bracket_params)
                                else:
                                    # Fallback to regular order for market orders or options
                                    order_data = {
                                        "class": trade_class,
                                        "side": trade_side,
                                        "quantity": str(trade_quantity),
                                        "type": trade_type,
                                        "duration": "day",
                                        "tag": f"AIREC{datetime.now().strftime('%Y%m%d%H%M%S')}"
                                    }
                                    
                                    # Use appropriate symbol field based on trade class
                                    if trade_class == "option" and st.session_state.get('modal_option_symbol'):
                                        order_data["option_symbol"] = st.session_state['modal_option_symbol'].upper()
                                        trade_symbol_display = st.session_state['modal_option_symbol']
                                    else:
                                        order_data["symbol"] = trade_symbol.upper()
                                        trade_symbol_display = trade_symbol
                                    
                                    if trade_type == "limit" and trade_price:
                                        order_data["price"] = str(trade_price)
                                    
                                    # Explain why bracket wasn't used
                                    reason = "market order" if trade_type == "market" else "options trade" if trade_class != "equity" else "non-standard side"
                                    logger.info(f"🚀 Placing REGULAR order ({reason}): {trade_symbol_display} {trade_side} {trade_quantity} @ {trade_type}")
                                    success, result = st.session_state.tradier_client.place_order(order_data)
                                
                                if success:
                                    order_id = result.get('order', {}).get('id', 'Unknown')
                                    if use_bracket:
                                        st.success(f"🎉 Bracket order placed successfully! Order ID: {order_id}")
                                        st.info(f"✅ Entry: ${trade_price} | 🎯 Target: ${target:.2f} | 🛑 Stop: ${stop_loss:.2f}")
                                    else:
                                        st.success(f"🎉 Order placed successfully! Order ID: {order_id}")
                                    st.json(result)
                                    
                                    # Log the trade
                                    logger.info(f"AI recommendation executed: {trade_symbol} {trade_side} {trade_quantity} @ {trade_type}")
                                    
                                    # Clear the modal after successful order
                                    if st.button("Close & Refresh", key="close_success"):
                                        st.session_state.show_quick_trade = False
                                        st.session_state.selected_recommendation = None
                                        # Modal close needs rerun to update UI
                                        st.rerun()
                                else:
                                    st.error(f"❌ Order failed: {result.get('error', 'Unknown error')}")
                                    st.json(result)
                            except Exception as e:
                                st.error(f"❌ Error placing order: {str(e)}")
                                logger.error(f"Quick trade error: {e}", exc_info=True)
                
                with confirm_col2:
                    if st.button("❌ Cancel", width="stretch", key="modal_cancel"):
                        st.session_state.show_quick_trade = False
                        st.session_state.selected_recommendation = None
                        # Modal close needs rerun to update UI
                        st.rerun()
        st.divider()
    
    if analyze_btn and search_ticker:
        # Clear previous analysis from session state
        if 'analysis' in st.session_state:
            del st.session_state['analysis']

        # Use new st.status for better progress indication
        with st.status(f"🔍 Analyzing {search_ticker}...", expanded=True) as status:
            st.write("📊 Fetching market data...")
            time.sleep(0.5)  # Simulate processing time
            
            st.write("📈 Calculating technical indicators...")
            time.sleep(0.5)
            
            st.write("📰 Analyzing news sentiment...")
            time.sleep(0.5)
            
            st.write("🎯 Identifying catalysts...")
            time.sleep(0.5)
            
            st.write("📄 Fetching SEC filings (8-K, 10-Q, 10-K)...")
            time.sleep(0.3)
            
            st.write(f"🤖 Generating {trading_style_display} recommendations...")
            analysis = ComprehensiveAnalyzer.analyze_stock(search_ticker, trading_style)
            
            # Fetch SEC filings and enhanced catalyst data
            sec_filings = []
            enhanced_catalysts = []
            if analysis:
                try:
                    logger.info(f"📄 Fetching SEC filings for {search_ticker}...")
                    # Create a temporary SEC detector instance (we don't need alert system for just fetching)
                    from services.event_detectors.base_detector import BaseEventDetector
                    
                    # Get company CIK for SEC filings
                    try:
                        stock = yf.Ticker(search_ticker)
                        info = stock.info
                        # Try to get CIK from info or look it up
                        cik = None
                        if 'cik' in info:
                            cik = str(info['cik']).zfill(10)  # CIK should be 10 digits
                        
                        # If no CIK in info, try to look it up from SEC
                        if not cik:
                            logger.info(f"🔍 CIK not found in yfinance info, looking up from SEC...")
                            try:
                                import requests
                                url = "https://www.sec.gov/files/company_tickers.json"
                                headers = {'User-Agent': "Sentient Trader/1.0 (trading@example.com)"}
                                response = requests.get(url, headers=headers, timeout=10)
                                response.raise_for_status()
                                companies = response.json()
                                for company in companies.values():
                                    if company.get('ticker', '').upper() == search_ticker.upper():
                                        cik = str(company.get('cik_str', '')).zfill(10)
                                        logger.info(f"✅ Found CIK for {search_ticker}: {cik}")
                                        break
                            except Exception as lookup_error:
                                logger.warning(f"Could not lookup CIK for {search_ticker}: {lookup_error}")
                        
                        # Log CIK status
                        if cik:
                            logger.info(f"✅ CIK found for {search_ticker}: {cik}")
                        else:
                            logger.warning(f"⚠️ No CIK available for {search_ticker}, skipping SEC filings")
                        
                        if cik:
                            # Create SEC detector instance (using None for alert_system since we just want data)
                            class TempSECDetector:
                                def __init__(self):
                                    self.user_agent = "Sentient Trader/1.0 (trading@example.com)"
                                
                                def get_company_cik(self, ticker: str):
                                    # Already have it
                                    return cik
                                
                                def get_recent_filings(self, ticker: str, cik: str, hours_back: int = 168):
                                    """Get recent SEC filings (last 7 days)"""
                                    try:
                                        import requests
                                        from datetime import datetime, timedelta
                                        
                                        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
                                        headers = {'User-Agent': self.user_agent}
                                        
                                        response = requests.get(url, headers=headers, timeout=10)
                                        response.raise_for_status()
                                        
                                        data = response.json()
                                        recent_filings = data.get('filings', {}).get('recent', {})
                                        
                                        if not recent_filings:
                                            return []
                                        
                                        filings = []
                                        cutoff_time = datetime.now() - timedelta(hours=hours_back)
                                        
                                        filing_dates = recent_filings.get('filingDate', [])
                                        form_types = recent_filings.get('form', [])
                                        accession_numbers = recent_filings.get('accessionNumber', [])
                                        primary_documents = recent_filings.get('primaryDocument', [])
                                        
                                        # Check last 20 filings
                                        for i in range(min(len(filing_dates), 20)):
                                            try:
                                                filing_date = datetime.strptime(filing_dates[i], '%Y-%m-%d')
                                                
                                                if filing_date >= cutoff_time:
                                                    form_type = form_types[i]
                                                    accession = accession_numbers[i]
                                                    primary_doc = primary_documents[i] if i < len(primary_documents) else ''
                                                    
                                                    # Build filing URL
                                                    accession_clean = accession.replace('-', '')
                                                    cik_clean = cik.lstrip('0')  # Remove leading zeros
                                                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{primary_doc}"
                                                    
                                                    # Get filing description
                                                    filing_descriptions = {
                                                        '8-K': 'Material Event Report',
                                                        '8-K/A': 'Amended Material Event',
                                                        '4': 'Insider Trading Statement',
                                                        '10-Q': 'Quarterly Report',
                                                        '10-K': 'Annual Report',
                                                        '10-Q/A': 'Amended Quarterly Report',
                                                        '10-K/A': 'Amended Annual Report',
                                                        'S-1': 'IPO Registration',
                                                        'S-3': 'Securities Registration',
                                                        'DEF 14A': 'Proxy Statement'
                                                    }
                                                    
                                                    filings.append({
                                                        'ticker': ticker,
                                                        'form_type': form_type,
                                                        'filing_date': filing_date.strftime('%Y-%m-%d'),
                                                        'description': filing_descriptions.get(form_type, form_type),
                                                        'url': filing_url,
                                                        'days_ago': (datetime.now() - filing_date).days,
                                                        'is_critical': form_type in ['8-K', '8-K/A', '4', 'S-1']
                                                    })
                                            except Exception as e:
                                                logger.debug(f"Error parsing filing {i}: {e}")
                                                continue
                                        
                                        return sorted(filings, key=lambda x: x['filing_date'], reverse=True)[:10]  # Last 10
                                        
                                    except Exception as e:
                                        logger.error(f"Error fetching SEC filings: {e}")
                                        return []
                            
                            sec_detector = TempSECDetector()
                            sec_filings = sec_detector.get_recent_filings(search_ticker, cik, hours_back=168)  # Last 7 days
                            logger.info(f"✅ Retrieved {len(sec_filings)} recent SEC filings for {search_ticker}")
                            
                            # Analyze filings for catalysts
                            if sec_filings:
                                for filing in sec_filings:
                                    if filing['form_type'] == '8-K':
                                        # Parse 8-K for material events
                                        try:
                                            # Note: Full parsing would require fetching filing content
                                            # For now, we'll just flag 8-Ks as material events
                                            enhanced_catalysts.append({
                                                'type': 'SEC Filing - 8-K',
                                                'date': filing['filing_date'],
                                                'days_away': -filing['days_ago'],  # Negative means in the past
                                                'impact': 'HIGH',
                                                'description': f"Material event filing: {filing['description']}",
                                                'filing_url': filing['url'],
                                                'is_critical': filing['is_critical']
                                            })
                                        except Exception as e:
                                            logger.debug(f"Error parsing 8-K: {e}")
                    except Exception as e:
                        logger.warning(f"Could not fetch SEC filings for {search_ticker}: {e}")
                        sec_filings = []
                except Exception as e:
                    logger.error(f"Error fetching SEC filings data for {search_ticker}: {e}", exc_info=True)
                    sec_filings = []
            
            # Store filings in session state
            st.session_state.sec_filings = sec_filings
            st.session_state.enhanced_catalysts = enhanced_catalysts
            
            # Run unified penny stock analysis if applicable
            penny_stock_analysis = None
            if analysis and is_penny_stock(analysis.price):
                logger.info(f"💰 PENNY STOCK DETECTED: {search_ticker} @ ${analysis.price:.2f} (< ${PENNY_THRESHOLDS.MAX_PENNY_STOCK_PRICE})")
                st.write("💰 Running enhanced penny stock analysis...")
                try:
                    unified_analyzer = UnifiedPennyStockAnalysis()
                    logger.info(f"✅ UnifiedPennyStockAnalysis initialized for {search_ticker}")
                    
                    # Map trading style for penny stock analysis
                    penny_style_map = {
                        "DAY_TRADE": "SCALP",
                        "SWING_TRADE": "SWING",
                        "SCALP": "SCALP",
                        "BUY_HOLD": "POSITION",
                        "OPTIONS": "SWING"
                    }
                    penny_trading_style = penny_style_map.get(trading_style, "SWING")
                    logger.info(f"📊 Trading style mapped: {trading_style} -> {penny_trading_style} for penny stock analysis")
                    
                    logger.info(f"🔍 Starting comprehensive penny stock analysis for {search_ticker}...")
                    penny_stock_analysis = unified_analyzer.analyze_comprehensive(
                        ticker=search_ticker,
                        trading_style=penny_trading_style,
                        include_backtest=False,  # Skip backtest for speed
                        check_options=(trading_style == "OPTIONS")
                    )
                    
                    if penny_stock_analysis:
                        if 'error' in penny_stock_analysis:
                            logger.error(f"❌ Penny stock analysis error for {search_ticker}: {penny_stock_analysis['error']}")
                            st.error(f"⚠️ Penny stock analysis encountered an error: {penny_stock_analysis['error']}")
                        else:
                            logger.info(f"✅ Penny stock analysis completed for {search_ticker}")
                            logger.info(f"   Classification: {penny_stock_analysis.get('classification', 'N/A')}")
                            logger.info(f"   ATR Stop: ${penny_stock_analysis.get('atr_stop_loss', 'N/A')} ({penny_stock_analysis.get('atr_stop_pct', 0):.1f}%)")
                            logger.info(f"   Risk Level: {penny_stock_analysis.get('risk_level', 'N/A')}")
                            
                            if 'final_recommendation' in penny_stock_analysis:
                                final_rec = penny_stock_analysis['final_recommendation']
                                logger.info(f"   Final Decision: {final_rec.get('decision', 'N/A')}")
                    else:
                        logger.warning(f"⚠️ Penny stock analysis returned None for {search_ticker}")
                    
                    st.session_state.penny_stock_analysis = penny_stock_analysis
                    logger.info(f"💾 Penny stock analysis stored in session state for {search_ticker}")
                except Exception as e:
                    logger.error(f"❌ ERROR running unified penny stock analysis for {search_ticker}: {e}", exc_info=True)
                    st.error(f"⚠️ Error running enhanced penny stock analysis: {str(e)}")
                    penny_stock_analysis = None
            else:
                if analysis:
                    logger.info(f"ℹ️ {search_ticker} @ ${analysis.price:.2f} is NOT a penny stock (>= $5.0)")
                else:
                    logger.warning(f"⚠️ No analysis available for {search_ticker} to check penny stock status")

            # --- Generate Premium AI Trading Signal ---
            st.session_state.ai_trading_signal = None
            if analysis:
                st.write("🤖 Generating Premium AI Trading Signal with Gemini...")
                signal_generator = AITradingSignalGenerator()

                # Prepare data for the signal generator
                technical_data = {
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
                news_data = analysis.recent_news
                sentiment_data = {
                    'score': analysis.sentiment_score,
                    'signals': analysis.sentiment_signals
                }
                social_data = None  # Social sentiment not available in StockAnalysis

                # Generate the signal using the configured premium model
                ai_signal = signal_generator.generate_signal(
                    symbol=analysis.ticker,
                    technical_data=technical_data,
                    news_data=news_data,
                    sentiment_data=sentiment_data,
                    social_data=social_data
                )
                st.session_state.ai_trading_signal = ai_signal
            # ----------------------------------------

            if analysis:
                status.update(label=f"✅ Analysis complete for {search_ticker}", state="complete")
            else:
                status.update(label=f"❌ Analysis failed for {search_ticker}", state="error")
            
            if analysis:
                logger.info(f"💾 Storing analysis in session state: {analysis.ticker} @ ${analysis.price:.2f}")
                st.session_state.current_analysis = analysis
                logger.info(f"✅ Analysis stored. Quick trade flag status: {st.session_state.get('show_quick_trade', False)}")
                
                # Detect penny stock and runner characteristics
                is_penny_stock_flag = is_penny_stock(analysis.price)
                is_otc = analysis.ticker.endswith(('.OTC', '.PK', '.QB'))
                volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
                is_runner = volume_vs_avg > 200 and analysis.change_pct > 10  # 200%+ volume spike and 10%+ gain
                
                # Get unified penny stock analysis if available
                penny_stock_analysis = st.session_state.get('penny_stock_analysis')
                if is_penny_stock_flag:
                    if penny_stock_analysis:
                        logger.info(f"✅ Found penny stock analysis in session state for {analysis.ticker}")
                    else:
                        logger.warning(f"⚠️ Penny stock detected but no enhanced analysis found for {analysis.ticker}")
                
                # Header metrics
                st.success(f"✅ Analysis complete for {analysis.ticker}")

                # --- Display Premium AI Trading Signal ---
                if 'ai_trading_signal' in st.session_state and st.session_state.ai_trading_signal:
                    signal = st.session_state.ai_trading_signal
                    st.subheader("🤖 Premium AI Trading Signal (Gemini)")
                    
                    signal_color = "green" if signal.signal == "BUY" else "red" if signal.signal == "SELL" else "orange"
                    st.markdown(f"## <span style='color:{signal_color};'>{signal.signal}</span>", unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{signal.confidence:.1f}%")
                    with col2:
                        st.metric("Risk Level", signal.risk_level)
                    with col3:
                        st.metric("Time Horizon", signal.time_horizon.replace('_', ' ').title())

                    with st.expander("View AI Reasoning and Price Targets"):
                        st.write(f"**Reasoning:** {signal.reasoning}")
                        st.write("-")
                        st.metric("Entry Price", f"${signal.entry_price:.2f}" if signal.entry_price else "N/A")
                        st.metric("Target Price", f"${signal.target_price:.2f}" if signal.target_price else "N/A")
                        st.metric("Stop Loss", f"${signal.stop_loss:.2f}" if signal.stop_loss else "N/A")
                    st.divider()
                # ----------------------------------------
                
                # Special alerts for penny stocks and runners
                if is_runner:
                    st.warning(f"🚀 **RUNNER DETECTED!** {volume_vs_avg:+.0f}% volume spike with {analysis.change_pct:+.1f}% price move!")
                
                if is_penny_stock_flag:
                    if penny_stock_analysis and 'classification' in penny_stock_analysis:
                        classification = penny_stock_analysis.get('classification', 'PENNY_STOCK')
                        if classification == 'LOW_PRICED':
                            st.info(f"💰 **LOW-PRICED STOCK** (${analysis.price:.2f}) - Price < $5 but market cap suggests established company. Moderate risk.")
                        else:
                            st.warning(f"💰 **{classification}** (${analysis.price:.4f}) - High risk/high reward. Use enhanced risk management.")
                    else:
                        st.info(f"💰 **PENNY STOCK** (${analysis.price:.4f}) - High risk/high reward. Use caution and proper position sizing.")
                
                if is_otc:
                    st.warning("⚠️ **OTC STOCK** - Lower liquidity, wider spreads, higher risk. Limited data may be available.")
                
                metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
                
                with metric_col1:
                    price_display = f"${analysis.price:.4f}" if is_penny_stock_flag else f"${analysis.price:.2f}"
                    st.metric("Price", price_display, f"{analysis.change_pct:+.2f}%")
                with metric_col2:
                    st.metric("Trend", analysis.trend)
                with metric_col3:
                    st.metric("Confidence", f"{int(analysis.confidence_score)}%")
                with metric_col4:
                    st.metric("IV Rank", f"{analysis.iv_rank}%")
                with metric_col5:
                    volume_indicator = "🔥" if volume_vs_avg > 100 else "📊"
                    st.metric(f"{volume_indicator} Volume", f"{analysis.volume:,}", f"{volume_vs_avg:+.1f}%")
                
                # Runner Metrics (if detected)
                if is_runner or volume_vs_avg > 100:
                    st.subheader("🚀 Runner / Momentum Metrics")
                    
                    runner_col1, runner_col2, runner_col3, runner_col4 = st.columns(4)
                    
                    with runner_col1:
                        st.metric("Volume Spike", f"{volume_vs_avg:+.0f}%")
                        if volume_vs_avg > 300:
                            st.caption("🔥 EXTREME volume!")
                        elif volume_vs_avg > 200:
                            st.caption("🔥 Very high volume")
                        else:
                            st.caption("📈 Elevated volume")
                    
                    with runner_col2:
                        st.metric("Price Change", f"{analysis.change_pct:+.2f}%")
                        if abs(analysis.change_pct) > 20:
                            st.caption("🚀 Major move!")
                        elif abs(analysis.change_pct) > 10:
                            st.caption("📈 Strong move")
                    
                    with runner_col3:
                        # Calculate momentum score
                        momentum_score = min(100, (abs(analysis.change_pct) * 2 + volume_vs_avg / 5))
                        st.metric("Momentum Score", f"{momentum_score:.0f}/100")
                        if momentum_score > 80:
                            st.caption("🔥 HOT!")
                        elif momentum_score > 60:
                            st.caption("🔥 Strong")
                    
                    with runner_col4:
                        # Risk level for runners
                        runner_risk = "EXTREME" if is_penny_stock_flag and volume_vs_avg > 300 else "VERY HIGH" if volume_vs_avg > 200 else "HIGH"
                        st.metric("Runner Risk", runner_risk)
                        st.caption("⚠️ Use stops!")
                    
                    if is_runner:
                        st.warning("""
**Runner Trading Tips:**
- ✅ Use tight stop losses (3-5%)
- ✅ Take profits quickly (don't be greedy)
- ✅ Watch for volume decline (exit signal)
- ✅ Avoid chasing - wait for pullbacks
- ❌ Don't hold overnight (high gap risk)
                        """)
                
                # Technical Indicators
                st.subheader("📊 Technical Indicators")
                
                tech_col1, tech_col2, tech_col3 = st.columns(3)
                
                with tech_col1:
                    st.metric("RSI (14)", f"{analysis.rsi:.1f}")
                    if analysis.rsi < 30:
                        st.caption("🟢 Oversold - potential buy")
                    elif analysis.rsi > 70:
                        st.caption("🔴 Overbought - potential sell")
                    else:
                        st.caption("🟡 Neutral")
                
                with tech_col2:
                    st.metric("MACD Signal", analysis.macd_signal)
                    if analysis.macd_signal == "BULLISH":
                        st.caption("🟢 Bullish momentum")
                    elif analysis.macd_signal == "BEARISH":
                        st.caption("🔴 Bearish momentum")
                    else:
                        st.caption("🟡 Neutral momentum")
                
                with tech_col3:
                    st.metric("Support", f"${analysis.support}")
                    st.metric("Resistance", f"${analysis.resistance}")
                
                # IV Analysis
                st.subheader("📈 Implied Volatility Analysis")
                
                iv_col1, iv_col2, iv_col3 = st.columns(3)
                
                with iv_col1:
                    st.metric("IV Rank", f"{analysis.iv_rank}%")
                    if analysis.iv_rank > 60:
                        st.caption("🔥 High IV - Great for selling premium")
                    elif analysis.iv_rank < 40:
                        st.caption("❄️ Low IV - Good for buying options")
                    else:
                        st.caption("➡️ Moderate IV")
                
                with iv_col2:
                    st.metric("IV Percentile", f"{analysis.iv_percentile}%")
                
                with iv_col3:
                    if analysis.iv_rank > 50:
                        st.info("💡 Consider: Selling puts, covered calls, iron condors")
                    else:
                        st.info("💡 Consider: Buying calls/puts, debit spreads")
                
                # Entropy Analysis (Market Noise Detection)
                st.subheader("🔬 Entropy Analysis (Market Noise Detection)")
                
                entropy_col1, entropy_col2, entropy_col3 = st.columns(3)
                
                with entropy_col1:
                    entropy_value = analysis.entropy if analysis.entropy is not None else 50.0
                    st.metric("Entropy Score", f"{entropy_value:.1f}/100")
                    if entropy_value < 30:
                        st.caption("✅ Highly Structured - Ideal for trading")
                    elif entropy_value < 50:
                        st.caption("✅ Structured - Good patterns")
                    elif entropy_value < 70:
                        st.caption("⚠️ Mixed - Trade with caution")
                    else:
                        st.caption("❌ Noisy - High risk/choppy")
                
                with entropy_col2:
                    entropy_state = analysis.entropy_state if analysis.entropy_state else "UNKNOWN"
                    st.metric("Market State", entropy_state)
                    st.caption("Pattern predictability")
                
                with entropy_col3:
                    entropy_signal = analysis.entropy_signal if analysis.entropy_signal else "CAUTION"
                    signal_emoji = {"FAVORABLE": "✅", "CAUTION": "⚠️", "AVOID": "❌"}
                    st.metric("Trade Signal", f"{signal_emoji.get(entropy_signal, '⚠️')} {entropy_signal}")
                    if entropy_signal == "FAVORABLE":
                        st.caption("🟢 Low entropy - Trade normally")
                    elif entropy_signal == "CAUTION":
                        st.caption("🟡 Moderate entropy - Reduce size")
                    else:
                        st.caption("🔴 High entropy - Avoid or skip")
                
                # Entropy explanation
                with st.expander("ℹ️ What is Entropy?"):
                    st.write("""
                    **Entropy measures market unpredictability and noise:**
                    
                    - **Low Entropy (< 30)**: Clear patterns, predictable moves → Trade with confidence
                    - **Medium Entropy (30-70)**: Some noise, mixed signals → Trade with caution
                    - **High Entropy (> 70)**: Random/choppy price action → Avoid or reduce size significantly
                    
                    Entropy helps filter out false signals and whipsaws by identifying when the market is 
                    too noisy for reliable pattern recognition.
                    """)
                
                # Catalysts
                st.subheader("📅 Upcoming Catalysts")
                
                # Combine regular catalysts with enhanced catalysts from SEC filings
                all_catalysts = list(analysis.catalysts) if analysis.catalysts else []
                enhanced_catalysts = st.session_state.get('enhanced_catalysts', [])
                if enhanced_catalysts:
                    all_catalysts.extend(enhanced_catalysts)
                    logger.info(f"✅ Added {len(enhanced_catalysts)} enhanced catalysts from SEC filings")
                
                if all_catalysts:
                    for catalyst in all_catalysts:
                        impact_color = {
                            'HIGH': '🔴',
                            'MEDIUM': '🟡',
                            'LOW': '🟢'
                        }.get(catalyst['impact'], '⚪')
                        
                        # Format days away display
                        days_away = catalyst.get('days_away', 'N/A')
                        if isinstance(days_away, int):
                            if days_away < 0:
                                days_text = f"{abs(days_away)} days ago"
                            elif days_away == 0:
                                days_text = "Today"
                            else:
                                days_text = f"{days_away} days away"
                        else:
                            days_text = str(days_away)
                        
                        expander_title = f"{impact_color} {catalyst['type']} - {catalyst['date']} ({days_text})"
                        
                        with st.expander(expander_title):
                            st.write(f"**Impact Level:** {catalyst['impact']}")
                            st.write(f"**Details:** {catalyst['description']}")
                            
                            # Add filing URL if available
                            if 'filing_url' in catalyst:
                                st.write(f"[📄 View SEC Filing]({catalyst['filing_url']})")
                            
                            if catalyst['type'] == 'Earnings Report' and isinstance(days_away, int) and days_away >= 0 and days_away <= 7:
                                st.warning("⚠️ Earnings within 7 days - expect high volatility!")
                            
                            if catalyst.get('is_critical'):
                                st.error("🔴 **CRITICAL FILING** - Review immediately for material events")
                else:
                    st.info("No major catalysts identified in the next 60 days")
                
                # SEC Filings Section
                sec_filings = st.session_state.get('sec_filings', [])
                if sec_filings:
                    st.subheader("📄 Recent SEC Filings (Last 7 Days)")
                    logger.info(f"📄 Displaying {len(sec_filings)} SEC filings for {analysis.ticker}")
                    
                    filings_col1, filings_col2 = st.columns([3, 1])
                    
                    with filings_col1:
                        for filing in sec_filings[:10]:  # Show last 10
                            filing_icon = "🔴" if filing['is_critical'] else "🟡"
                            filing_desc = f"{filing_icon} **{filing['form_type']}** - {filing['description']}"
                            
                            with st.expander(f"{filing_desc} ({filing['filing_date']}, {filing['days_ago']} days ago)"):
                                st.write(f"**Filing Type:** {filing['form_type']}")
                                st.write(f"**Description:** {filing['description']}")
                                st.write(f"**Filing Date:** {filing['filing_date']}")
                                
                                if filing['is_critical']:
                                    st.error("⚠️ **CRITICAL FILING** - Material event (8-K) or significant filing")
                                
                                if filing.get('url'):
                                    st.write(f"[📄 View Filing on SEC.gov]({filing['url']})")
                    
                    with filings_col2:
                        critical_count = sum(1 for f in sec_filings if f['is_critical'])
                        total_count = len(sec_filings)
                        
                        st.metric("Total Filings", total_count)
                        if critical_count > 0:
                            st.metric("Critical Filings", critical_count, delta="Review Required")
                        
                        # Filing type breakdown
                        filing_types = {}
                        for f in sec_filings:
                            form_type = f['form_type']
                            filing_types[form_type] = filing_types.get(form_type, 0) + 1
                        
                        if filing_types:
                            st.write("**Filing Types:**")
                            for form_type, count in sorted(filing_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                                st.caption(f"{form_type}: {count}")
                else:
                    logger.info(f"ℹ️ No recent SEC filings found for {analysis.ticker} in the last 7 days")
                
                # News & Sentiment
                st.subheader("📰 Recent News & Sentiment")
                
                # Add refresh button for news
                col_refresh, col_info = st.columns([1, 4])
                with col_refresh:
                    if st.button("🔄 Refresh News", help="Get the latest news and sentiment"):
                        # Clear cache for this ticker and set flag for refresh
                        get_cached_news.clear()
                        st.session_state[f'news_refreshed_{analysis.ticker}'] = True
                        # News refresh needs rerun to display updated data
                        st.rerun()
                
                with col_info:
                    if analysis.recent_news:
                        st.success(f"✅ Found {len(analysis.recent_news)} recent news articles")
                    else:
                        st.warning("⚠️ No recent news found - this may indicate low news volume or connectivity issues")
                
                sentiment_col1, sentiment_col2 = st.columns([1, 3])
                
                with sentiment_col1:
                    sentiment_label = "POSITIVE" if analysis.sentiment_score > 0.2 else "NEGATIVE" if analysis.sentiment_score < -0.2 else "NEUTRAL"
                    sentiment_color = "🟢" if analysis.sentiment_score > 0.2 else "🔴" if analysis.sentiment_score < -0.2 else "🟡"
                    
                    st.metric("News Sentiment", f"{sentiment_color} {sentiment_label}")
                    st.metric("Sentiment Score", f"{analysis.sentiment_score:.2f}")
                    
                    # Show sentiment signals if available
                    if hasattr(analysis, 'sentiment_signals') and analysis.sentiment_signals:
                        with st.expander("📊 Sentiment Analysis Details"):
                            for signal in analysis.sentiment_signals[:3]:  # Show top 3
                                st.write(signal)
                
                with sentiment_col2:
                    if analysis.recent_news:
                        st.write("**Latest News Articles:**")
                        for idx, article in enumerate(analysis.recent_news[:5]):
                            # Create a more informative expander
                            expander_title = f"📰 {article['title'][:70]}..." if len(article['title']) > 70 else f"📰 {article['title']}"
                            
                            with st.expander(expander_title):
                                col_pub, col_time = st.columns(2)
                                with col_pub:
                                    st.write(f"**Publisher:** {article['publisher']}")
                                with col_time:
                                    st.write(f"**Published:** {article['published']}")
                                
                                # Show summary if available
                                if article.get('summary'):
                                    st.write("**Summary:**")
                                    st.write(article['summary'])
                                
                                # Link to full article
                                if article.get('link'):
                                    st.write(f"[📖 Read Full Article]({article['link']})")
                                
                                # Show article type
                                if article.get('type'):
                                    st.caption(f"Type: {article['type']}")
                    else:
                        st.info("📭 No recent news found for this ticker. This could be due to:")
                        st.write("• Low news volume for this stock")
                        st.write("• Temporary connectivity issues")
                        st.write("• Yahoo Finance API limitations")
                        st.write("• Try refreshing the news or check back later")
                
                # Enhanced Penny Stock Analysis (if applicable)
                logger.info(f"🔍 Checking enhanced penny stock display: is_penny_stock={is_penny_stock_flag}, has_penny_analysis={penny_stock_analysis is not None}")
                if is_penny_stock_flag and penny_stock_analysis:
                    logger.info(f"✅ DISPLAYING Enhanced Penny Stock Analysis for {analysis.ticker}")
                    st.subheader("💰 Enhanced Penny Stock Analysis")
                    st.success(f"✅ Enhanced analysis available for {analysis.ticker} - Showing detailed results below")
                    
                    # Classification
                    if 'classification' in penny_stock_analysis:
                        classification = penny_stock_analysis.get('classification', 'UNKNOWN')
                        risk_level = penny_stock_analysis.get('risk_level', 'UNKNOWN')
                        
                        class_col1, class_col2 = st.columns(2)
                        with class_col1:
                            st.metric("Stock Classification", classification)
                        with class_col2:
                            st.metric("Risk Level", risk_level)
                    
                    # ATR-Based Stop Loss & Targets
                    if 'atr_stop_loss' in penny_stock_analysis and penny_stock_analysis['atr_stop_loss']:
                        st.subheader("🎯 ATR-Based Risk Management")
                        
                        stop_loss = penny_stock_analysis.get('atr_stop_loss')
                        target = penny_stock_analysis.get('atr_target')
                        stop_pct = penny_stock_analysis.get('atr_stop_pct', 0)
                        target_pct = penny_stock_analysis.get('atr_target_pct', 0)
                        rr_ratio = penny_stock_analysis.get('atr_risk_reward', 0)
                        
                        stop_col1, stop_col2, stop_col3 = st.columns(3)
                        with stop_col1:
                            st.metric("Stop Loss", f"${stop_loss:.4f}", f"{stop_pct:.1f}%")
                            if stop_pct > 12:
                                st.error("⚠️ STOP EXCEEDS 12% MAX - Consider skipping or reducing position")
                            elif stop_pct > 8:
                                st.warning("⚠️ Wide stop - Use smaller position size")
                            else:
                                st.success("✅ Acceptable stop width")
                        with stop_col2:
                            st.metric("Target", f"${target:.4f}", f"{target_pct:.1f}%")
                        with stop_col3:
                            st.metric("Risk/Reward", f"{rr_ratio:.1f}:1")
                            if rr_ratio >= 2.0:
                                st.success("✅ Good R/R ratio")
                            else:
                                st.warning("⚠️ R/R below 2:1")
                        
                        # Stop recommendation
                        if 'stop_recommendation' in penny_stock_analysis:
                            st.info(penny_stock_analysis['stop_recommendation'])
                    
                    # Stock Liquidity Check
                    if 'liquidity_check' in penny_stock_analysis:
                        liquidity = penny_stock_analysis['liquidity_check']
                        st.subheader("💧 Stock Liquidity Analysis")
                        
                        liq_col1, liq_col2, liq_col3 = st.columns(3)
                        with liq_col1:
                            overall_risk = liquidity.get('overall_risk', 'UNKNOWN')
                            risk_color = {
                                'CRITICAL': '🔴',
                                'HIGH': '🟠',
                                'MEDIUM': '🟡',
                                'LOW': '🟢'
                            }.get(overall_risk, '⚪')
                            st.metric("Overall Risk", f"{risk_color} {overall_risk}")
                        with liq_col2:
                            max_pos_pct = liquidity.get('max_position_pct_of_volume', 0)
                            st.metric("Max Position", f"{max_pos_pct:.1f}% of daily volume")
                        with liq_col3:
                            avg_vol = liquidity.get('avg_daily_volume', 0)
                            st.metric("Avg Volume", f"{avg_vol:,}")
                        
                        if overall_risk == "CRITICAL":
                            st.error("❌ **CRITICAL LIQUIDITY RISK** - Cannot execute safely. AVOID or use extreme caution.")
                        elif overall_risk == "HIGH":
                            st.warning("⚠️ **HIGH RISK** - Use limit orders only, small position size")
                        
                        if liquidity.get('warnings'):
                            for warning in liquidity['warnings']:
                                st.warning(warning)
                    
                    # Final Recommendation
                    if 'final_recommendation' in penny_stock_analysis:
                        final_rec = penny_stock_analysis['final_recommendation']
                        st.subheader("📊 Final Recommendation")
                        
                        decision = final_rec.get('decision', 'UNKNOWN')
                        emoji = final_rec.get('emoji', '⚠️')
                        reason = final_rec.get('reason', 'N/A')
                        
                        st.markdown(f"## {emoji} **{decision}**")
                        st.write(f"**Reason:** {reason}")
                        
                        if final_rec.get('blockers'):
                            st.error("**Blockers:**")
                            for blocker in final_rec['blockers']:
                                st.write(f"  {blocker}")
                        
                        if final_rec.get('warnings'):
                            st.warning("**Warnings:**")
                            for warning in final_rec['warnings']:
                                st.write(f"  {warning}")
                        
                        if final_rec.get('signals'):
                            st.success("**Positive Signals:**")
                            for signal in final_rec['signals']:
                                st.write(f"  {signal}")
                    
                    logger.info(f"✅ Enhanced penny stock analysis display completed for {analysis.ticker}")
                elif is_penny_stock_flag:
                    logger.warning(f"⚠️ Penny stock detected but enhanced analysis not available - using fallback display")
                    # Fallback to basic penny stock assessment if enhanced analysis not available
                    st.subheader("⚠️ Penny Stock Risk Assessment")
                    
                    risk_col1, risk_col2 = st.columns(2)
                    
                    with risk_col1:
                        st.warning("""
**Penny Stock Risks:**
- 🔴 High volatility (can swing 20-50%+ daily)
- 🔴 Low liquidity (harder to exit positions)
- 🔴 Wide bid-ask spreads (higher trading costs)
- 🔴 Manipulation risk (pump & dump schemes)
- 🔴 Limited financial data/transparency
- 🔴 Higher bankruptcy risk
                        """)
                    
                    with risk_col2:
                        st.success("""
**Penny Stock Trading Rules:**
- ✅ Never risk more than 1-2% of portfolio
- ✅ Use limit orders (avoid market orders)
- ✅ Set tight stop losses (5-10%)
- ✅ Take profits quickly (don't be greedy)
- ✅ Research company fundamentals
- ✅ Watch for unusual volume spikes
- ✅ Avoid stocks with no news/catalysts
                        """)
                
                # Timeframe-Specific Analysis
                st.subheader(f"⏰ {trading_style_display} Analysis")
                
                # Calculate timeframe-specific metrics
                if trading_style == "DAY_TRADE":
                    # Day trading focus: quick moves, tight stops
                    timeframe_score = 0
                    reasons = []
                    
                    # ENTROPY CHECK (CRITICAL FOR DAY TRADING)
                    entropy_value = analysis.entropy if analysis.entropy is not None else 50.0
                    if entropy_value < 30:
                        timeframe_score += 30
                        reasons.append(f"✅ LOW ENTROPY ({entropy_value:.0f}) - Clean price action, ideal for day trading")
                    elif entropy_value < 50:
                        timeframe_score += 15
                        reasons.append(f"✅ Moderate entropy ({entropy_value:.0f}) - Structured patterns present")
                    elif entropy_value < 70:
                        timeframe_score -= 10
                        reasons.append(f"⚠️ Moderate-high entropy ({entropy_value:.0f}) - Use wider stops and reduce size 30%")
                    else:
                        timeframe_score -= 25
                        reasons.append(f"❌ HIGH ENTROPY ({entropy_value:.0f}) - CHOPPY MARKET - Avoid day trading or reduce size 50%+")
                    
                    if volume_vs_avg > 100:
                        timeframe_score += 20
                        reasons.append(f"✅ High volume (+{volume_vs_avg:.0f}%) - good for day trading")
                    else:
                        reasons.append(f"⚠️ Volume only +{volume_vs_avg:.0f}% - may lack intraday momentum")
                    
                    if abs(analysis.change_pct) > 2:
                        timeframe_score += 15
                        reasons.append(f"✅ Strong intraday move ({analysis.change_pct:+.1f}%)")
                    else:
                        reasons.append("⚠️ Low intraday volatility - limited profit potential")
                    
                    if 30 < analysis.rsi < 70:
                        timeframe_score += 15
                        reasons.append("✅ RSI in tradeable range (not overbought/oversold)")
                    
                    if not is_penny_stock_flag:
                        timeframe_score += 10
                        reasons.append("✅ Not a penny stock - better liquidity for day trading")
                    else:
                        reasons.append("⚠️ Penny stock - higher risk, use smaller size")
                    
                    if analysis.trend != "NEUTRAL":
                        timeframe_score += 10
                        reasons.append(f"✅ Clear trend ({analysis.trend}) - easier to trade")
                    
                    st.metric("Day Trading Suitability", f"{timeframe_score}/100")
                    
                    for reason in reasons:
                        st.write(reason)
                    
                    # Overall verdict based on entropy-adjusted score
                    if entropy_value >= 70:
                        st.error("🔴 **NOT RECOMMENDED** for day trading - High entropy (choppy market) will cause whipsaws")
                    elif timeframe_score > 70:
                        st.success("🟢 **EXCELLENT** for day trading - strong setup!")
                    elif timeframe_score > 50:
                        st.info("🟡 **GOOD** for day trading - proceed with caution")
                    elif timeframe_score > 30:
                        st.warning("🟡 **MARGINAL** for day trading - not ideal; multiple divergent signals")
                    else:
                        st.error("🔴 **POOR** for day trading - consider swing/position trading instead")
                    
                    st.write("**Day Trading Strategy:**")
                    st.write(f"• 🎯 Entry: ${analysis.price:.2f}")
                    st.write(f"• 🛑 Stop: ${analysis.support:.2f} (support level)")
                    st.write(f"• 💰 Target: ${analysis.resistance:.2f} (resistance level)")
                    st.write(f"• ⏰ Hold time: Minutes to hours (close before market close)")
                    st.write(f"• 📊 Watch: Volume, L2 order book, momentum")
                
                elif trading_style == "SWING_TRADE":
                    # Swing trading focus: multi-day moves, catalysts
                    timeframe_score = 0
                    reasons = []
                    
                    if len(analysis.catalysts) > 0:
                        timeframe_score += 30
                        reasons.append(f"✅ {len(analysis.catalysts)} upcoming catalyst(s) - potential multi-day move")
                    else:
                        reasons.append("⚠️ No near-term catalysts - may lack swing momentum")
                    
                    if analysis.trend != "NEUTRAL":
                        timeframe_score += 25
                        reasons.append(f"✅ Strong {analysis.trend} trend - good for swing trading")
                    
                    if analysis.sentiment_score > 0.2:
                        timeframe_score += 20
                        reasons.append(f"✅ Positive sentiment ({analysis.sentiment_score:.2f}) - bullish setup")
                    elif analysis.sentiment_score < -0.2:
                        timeframe_score += 15
                        reasons.append(f"✅ Negative sentiment ({analysis.sentiment_score:.2f}) - bearish setup")
                    
                    if len(analysis.recent_news) > 3:
                        timeframe_score += 15
                        reasons.append(f"✅ Active news flow ({len(analysis.recent_news)} articles) - sustained interest")
                    
                    if not is_penny_stock_flag or (is_penny_stock_flag and volume_vs_avg > 200):
                        timeframe_score += 10
                        reasons.append("✅ Sufficient liquidity for swing trading")
                    else:
                        reasons.append("⚠️ Low liquidity - may be hard to exit position")
                    
                    st.metric("Swing Trading Suitability", f"{timeframe_score}/100")
                    
                    for reason in reasons:
                        st.write(reason)
                    
                    if timeframe_score > 70:
                        st.success("🟢 **EXCELLENT** for swing trading - strong multi-day setup!")
                    elif timeframe_score > 50:
                        st.info("🟡 **GOOD** for swing trading - monitor catalysts")
                    else:
                        st.warning("🔴 **POOR** for swing trading - better for day trading or long-term hold")
                    
                    st.write("**Swing Trading Strategy:**")
                    st.write(f"• 🎯 Entry: ${analysis.price:.2f} (current price)")
                    # Dynamic stop using 21 EMA if available
                    stop_val = None
                    if getattr(analysis, 'ema21', None):
                        try:
                            stop_val = float(analysis.ema21) * 0.99
                        except Exception:
                            stop_val = None
                    if stop_val is None:
                        stop_val = analysis.support * 0.95
                    st.write(f"• 🛑 Stop: ${stop_val:.2f} (below 21 EMA or support)")

                    # Fibonacci targets if present; fallback to resistance-based target
                    fib = getattr(analysis, 'fib_targets', None)
                    if isinstance(fib, dict) and fib.get('T1_1272'):
                        st.write("• 💰 Targets:")
                        st.write(f"   - T1 (127.2%): ${fib['T1_1272']:.2f} (take 25%)")
                        if fib.get('T2_1618'):
                            st.write(f"   - T2 (161.8%): ${fib['T2_1618']:.2f} (take 50%)")
                        last_t3 = fib.get('T3_2618') or fib.get('T3_200')
                        if last_t3:
                            st.write(f"   - T3 (200-261.8%): ${last_t3:.2f} (trail remaining)")
                        st.write("• 🧭 Move stop to breakeven after T1, trail below 21 EMA thereafter")
                    else:
                        st.write(f"• 💰 Target: ${analysis.resistance * 1.05:.2f} (5% above resistance)")

                    # Context badges
                    if getattr(analysis, 'ema_power_zone', None):
                        st.write("• ✅ 8>21 EMA Power Zone active")
                    if getattr(analysis, 'ema_reclaim', None):
                        st.write("• ✅ EMA Reclaim confirmed")
                    if getattr(analysis, 'demarker', None) is not None:
                        dem = float(analysis.demarker)
                        zone = "Neutral"
                        if dem <= 0.30:
                            zone = "Oversold"
                        elif dem >= 0.70:
                            zone = "Overbought"
                        st.write(f"• 📈 DeMarker(14): {dem:.2f} ({zone})")

                    st.write(f"• ⏰ Hold time: 2-10 days (watch for catalyst completion)")
                    st.write(f"• 📊 Watch: News, catalyst dates, trend continuation")
                    
                    if analysis.catalysts:
                        st.write("**Key Catalysts to Watch:**")
                        for cat in analysis.catalysts[:3]:
                            st.write(f"  • {cat['type']} on {cat['date']} ({cat.get('days_away', 'N/A')} days)")
                
                elif trading_style == "BUY_HOLD":  # Buy & Hold
                    # Position trading focus: fundamentals, long-term trends
                    timeframe_score = 0
                    reasons = []
                    
                    if analysis.trend == "BULLISH":
                        timeframe_score += 30
                        reasons.append("✅ Strong bullish trend - good for long-term hold")
                    elif analysis.trend == "BEARISH":
                        timeframe_score += 20
                        reasons.append("✅ Bearish trend - consider short or inverse position")
                    
                    if len(analysis.catalysts) > 2:
                        timeframe_score += 25
                        reasons.append(f"✅ Multiple catalysts ({len(analysis.catalysts)}) - sustained growth potential")
                    
                    if analysis.sentiment_score > 0.3:
                        timeframe_score += 20
                        reasons.append(f"✅ Very positive sentiment ({analysis.sentiment_score:.2f}) - market confidence")
                    
                    if not is_penny_stock_flag:
                        timeframe_score += 15
                        reasons.append("✅ Established stock - lower bankruptcy risk")
                    else:
                        reasons.append("⚠️ Penny stock - very high risk for long-term hold")
                    
                    if analysis.iv_rank < 50:
                        timeframe_score += 10
                        reasons.append(f"✅ Low IV ({analysis.iv_rank}%) - less volatility risk")
                    
                    st.metric("Position Trading Suitability", f"{timeframe_score}/100")
                    
                    for reason in reasons:
                        st.write(reason)
                    
                    if timeframe_score > 70:
                        st.success("🟢 **EXCELLENT** for position trading - strong long-term hold!")
                    elif timeframe_score > 50:
                        st.info("🟡 **GOOD** for position trading - monitor fundamentals")
                    else:
                        st.warning("🔴 **POOR** for position trading - better for short-term trades")
                    
                    if is_penny_stock_flag:
                        st.error("⚠️ **WARNING:** Penny stocks are extremely risky for long-term holds due to bankruptcy risk!")
                    
                    st.write("**Position Trading Strategy:**")
                    st.write(f"• 🎯 Entry: ${analysis.price:.2f} (current price or pullback)")
                    st.write(f"• 🛑 Stop: ${analysis.price * 0.85:.2f} (15% trailing stop)")
                    st.write(f"• 💰 Target: ${analysis.price * 1.30:.2f} (30%+ gain over time)")
                    st.write(f"• ⏰ Hold time: Weeks to months (review quarterly)")
                    st.write(f"• 📊 Watch: Earnings, fundamentals, sector trends, macro conditions")
                    
                    if not is_penny_stock:
                        st.info("💡 **Position Trading Tip:** Consider selling covered calls or cash-secured puts to generate income while holding.")
                
                elif trading_style == "AI":
                    # AI Analysis
                    st.write("🤖 **AI-Powered Analysis**")
                    ai_results = TradingStyleAnalyzer.analyze_ai_style(analysis, hist)
                    
                    # Display AI score and prediction
                    ai_col1, ai_col2, ai_col3 = st.columns(3)
                    with ai_col1:
                        st.metric("AI Score", f"{ai_results['score']}/100")
                    with ai_col2:
                        st.metric("ML Prediction", ai_results.get('ml_prediction', 'N/A'))
                    with ai_col3:
                        st.metric("Risk Level", ai_results.get('risk_level', 'UNKNOWN'))
                    
                    # Display signals
                    if ai_results.get('signals'):
                        st.write("**📊 AI Signals:**")
                        for signal in ai_results['signals']:
                            st.write(signal)
                    
                    # Display recommendations
                    if ai_results.get('recommendations'):
                        st.write("**💡 AI Recommendations:**")
                        for rec in ai_results['recommendations']:
                            st.write(rec)
                
                elif trading_style == "SCALP":
                    # Scalp Analysis
                    st.write("⚡ **Scalping Analysis**")
                    scalp_results = TradingStyleAnalyzer.analyze_scalp_style(analysis, hist)
                    
                    # Display scalp score and risk
                    scalp_col1, scalp_col2 = st.columns(2)
                    with scalp_col1:
                        st.metric("Scalping Score", f"{scalp_results['score']}/100")
                    with scalp_col2:
                        st.metric("Risk Level", scalp_results.get('risk_level', 'UNKNOWN'))
                    
                    # Display signals
                    if scalp_results.get('signals'):
                        st.write("**📊 Scalping Signals:**")
                        for signal in scalp_results['signals']:
                            st.write(signal)
                    
                    # Display targets
                    if scalp_results.get('targets'):
                        st.write("**🎯 Scalping Targets:**")
                        for target in scalp_results['targets']:
                            st.write(target)
                    
                    # Display recommendations
                    if scalp_results.get('recommendations'):
                        st.write("**💡 Scalping Strategy:**")
                        for rec in scalp_results['recommendations']:
                            st.write(rec)
                
                elif trading_style == "WARRIOR_SCALPING":
                    # Warrior Scalping Analysis
                    st.write("⚔️ **Warrior Scalping Analysis**")
                    warrior_results = TradingStyleAnalyzer.analyze_warrior_scalping_style(analysis, hist)
                    
                    # Display warrior score and setup type
                    warrior_col1, warrior_col2, warrior_col3 = st.columns(3)
                    with warrior_col1:
                        st.metric("Warrior Score", f"{warrior_results['score']}/100")
                    with warrior_col2:
                        st.metric("Setup Type", warrior_results.get('setup_type', 'N/A'))
                    with warrior_col3:
                        st.metric("Risk Level", warrior_results.get('risk_level', 'UNKNOWN'))
                    
                    # Display signals
                    if warrior_results.get('signals'):
                        st.write("**📊 Warrior Signals:**")
                        for signal in warrior_results['signals']:
                            st.write(signal)
                    
                    # Display targets
                    if warrior_results.get('targets'):
                        st.write("**🎯 Warrior Targets:**")
                        for target in warrior_results['targets']:
                            st.write(target)
                    
                    # Display recommendations
                    if warrior_results.get('recommendations'):
                        st.write("**💡 Warrior Strategy:**")
                        for rec in warrior_results['recommendations']:
                            st.write(rec)
                
                elif trading_style == "ORB_FVG":
                    # ORB+FVG Strategy Analysis
                    st.write("📊 **Opening Range Breakout + Fair Value Gap Analysis**")
                    
                    try:
                        from analyzers.orb_fvg_strategy import ORBFVGAnalyzer
                        
                        # Get intraday data (15-minute bars for opening range)
                        import yfinance as yf
                        ticker_obj = yf.Ticker(ticker)
                        # Get today's intraday data (1-minute bars for FVG detection)
                        intraday_data = ticker_obj.history(period="1d", interval="1m")
                        
                        if len(intraday_data) > 0:
                            orb_analyzer = ORBFVGAnalyzer()
                            orb_results = orb_analyzer.analyze(ticker, intraday_data, analysis.price)
                            
                            # Display key metrics
                            orb_col1, orb_col2, orb_col3 = st.columns(3)
                            with orb_col1:
                                st.metric("Signal", orb_results['signal'])
                            with orb_col2:
                                st.metric("Confidence", f"{orb_results['confidence']:.1f}%")
                            with orb_col3:
                                st.metric("Risk Level", orb_results['risk_level'])
                            
                            # Opening Range info
                            if orb_results.get('opening_range'):
                                orb_range = orb_results['opening_range']
                                st.info(f"📊 **Opening Range (First 15min):** ${orb_range['orl']:.2f} - ${orb_range['orh']:.2f} ({orb_range['range_pct']:.1f}%)")
                            
                            # FVG info
                            if orb_results.get('fvg_signal') != 'NEUTRAL':
                                st.success(f"🎯 **Fair Value Gap:** {orb_results['fvg_signal']} (Strength: {orb_results['fvg_strength']})")
                            
                            # Trade Setup
                            if orb_results['signal'] in ['BUY', 'SELL']:
                                st.write("**📈 Trade Setup:**")
                                setup_col1, setup_col2 = st.columns(2)
                                with setup_col1:
                                    st.write(f"**Entry:** ${orb_results['entry']:.2f}")
                                    st.write(f"**Stop Loss:** ${orb_results['stop_loss']:.2f}")
                                    st.write(f"**Risk:** ${orb_results['risk']:.2f}")
                                with setup_col2:
                                    st.write(f"**Target:** ${orb_results['target']:.2f}")
                                    st.write(f"**Reward:** ${orb_results['reward']:.2f}")
                                    st.write(f"**R:R Ratio:** 1:{orb_results['risk_reward_ratio']:.1f}")
                            
                            # Key Signals
                            if orb_results.get('key_signals'):
                                st.write("**📊 Key Signals:**")
                                for signal in orb_results['key_signals'][:5]:
                                    st.write(signal)
                            
                            # Recommendations
                            if orb_results.get('recommendations'):
                                st.write("**💡 Recommendations:**")
                                for rec in orb_results['recommendations'][:5]:
                                    st.write(rec)
                        else:
                            st.warning("⚠️ No intraday data available. ORB+FVG requires market hours data.")
                    except Exception as e:
                        logger.error(f"Error in ORB+FVG analysis: {e}", exc_info=True)
                        st.error(f"ORB+FVG analysis error: {str(e)}")
                
                elif trading_style in ["EMA_HEIKIN_ASHI", "RSI_STOCHASTIC_HAMMER", "FISHER_RSI", "MACD_VOLUME_RSI", "AGGRESSIVE_SCALPING"]:
                    # Freqtrade Strategy Analysis
                    st.write("⚡ **Freqtrade Strategy Analysis**")
                    
                    strategy_names = {
                        "EMA_HEIKIN_ASHI": "EMA Crossover + Heikin Ashi",
                        "RSI_STOCHASTIC_HAMMER": "RSI + Stochastic + Hammer",
                        "FISHER_RSI": "Fisher RSI Multi-Indicator",
                        "MACD_VOLUME_RSI": "MACD + Volume + RSI",
                        "AGGRESSIVE_SCALPING": "Aggressive Scalping"
                    }
                    
                    st.info(f"📊 **Strategy:** {strategy_names.get(trading_style, trading_style)}")
                    
                    # Calculate strategy suitability score
                    strategy_score = 50
                    signals = []
                    
                    # Common indicators for freqtrade strategies
                    if 30 < analysis.rsi < 70:
                        strategy_score += 15
                        signals.append("✅ RSI in optimal range")
                    elif analysis.rsi < 30:
                        strategy_score += 10
                        signals.append("✅ RSI oversold - potential buy signal")
                    elif analysis.rsi > 70:
                        strategy_score += 10
                        signals.append("✅ RSI overbought - potential sell signal")
                    
                    if analysis.macd_signal == "BULLISH":
                        strategy_score += 15
                        signals.append("✅ MACD bullish crossover")
                    elif analysis.macd_signal == "BEARISH":
                        strategy_score += 10
                        signals.append("⚠️ MACD bearish crossover")
                    
                    if volume_vs_avg > 1.5:
                        strategy_score += 15
                        signals.append(f"✅ Volume surge: {volume_vs_avg:.1f}x average")
                    
                    if analysis.trend != "NEUTRAL":
                        strategy_score += 10
                        signals.append(f"✅ Clear trend: {analysis.trend}")
                    
                    # Strategy-specific scoring
                    if trading_style == "AGGRESSIVE_SCALPING":
                        if abs(analysis.change_pct) > 2:
                            strategy_score += 10
                            signals.append(f"✅ High volatility: {analysis.change_pct:+.2f}%")
                    elif trading_style == "EMA_HEIKIN_ASHI":
                        if analysis.trend == "BULLISH":
                            strategy_score += 10
                            signals.append("✅ Heikin Ashi bullish setup")
                    
                    strategy_score = min(100, strategy_score)
                    
                    # Display results
                    strat_col1, strat_col2 = st.columns(2)
                    with strat_col1:
                        st.metric("Strategy Score", f"{strategy_score}/100")
                    with strat_col2:
                        if strategy_score >= 75:
                            st.metric("Signal", "🟢 STRONG BUY")
                        elif strategy_score >= 60:
                            st.metric("Signal", "🟡 BUY")
                        elif strategy_score >= 40:
                            st.metric("Signal", "⚪ HOLD")
                        else:
                            st.metric("Signal", "🔴 SELL")
                    
                    if signals:
                        st.write("**📊 Strategy Signals:**")
                        for signal in signals:
                            st.write(signal)
                    
                    # Targets
                    target1 = analysis.price * 1.02
                    target2 = analysis.price * 1.05
                    target3 = analysis.price * 1.10
                    st.write("**🎯 Profit Targets:**")
                    st.write(f"T1: ${target1:.2f} (+2%)")
                    st.write(f"T2: ${target2:.2f} (+5%)")
                    st.write(f"T3: ${target3:.2f} (+10%)")
                    
                    # Stop loss
                    stop_loss = analysis.price * 0.98
                    st.write(f"**🛑 Stop Loss:** ${stop_loss:.2f} (-2%)")
                
                # AI Recommendation
                st.subheader(f"🤖 AI Trading Recommendation - {trading_style_display}")
                
                recommendation_box = st.container()
                with recommendation_box:
                    # ENTROPY OVERRIDE: Block day trading/scalping recommendations if entropy is too high
                    entropy_value = analysis.entropy if analysis.entropy is not None else 50.0
                    
                    if trading_style in ["DAY_TRADE", "SCALP"] and entropy_value >= 70:
                        st.error("❌ **DAY TRADING/SCALPING NOT RECOMMENDED**")
                        st.warning(f"""
                        **High Entropy Alert ({entropy_value:.0f}/100):**
                        
                        The market is currently too choppy and unpredictable for day trading or scalping. 
                        High entropy means random price movements that will cause whipsaws and false signals.
                        
                        **Recommendation:** 
                        - ⏸️ Skip this trade for day trading
                        - 🔄 Consider swing trading or options strategies instead
                        - ⏰ Wait for entropy to drop below 50 before day trading
                        - 📊 Current state: {analysis.entropy_state}
                        """)
                    elif trading_style == "DAY_TRADE" and entropy_value >= 50:
                        st.warning(f"""
                        **⚠️ Moderate Entropy Warning ({entropy_value:.0f}/100):**
                        
                        Market noise is elevated. Day trading is risky in these conditions.
                        
                        **If you still trade:**
                        - Reduce position size by 50%
                        - Use wider stops (1.5-2x normal)
                        - Take profits quickly
                        - Expect lower win rate
                        """)
                        st.markdown("---")
                        st.markdown(f"**{trading_style_display} Strategy (Use Caution):**")
                        st.markdown(analysis.recommendation)
                    else:
                        # Add penny stock context to recommendation
                        if is_penny_stock_flag and trading_style in ["BUY_HOLD", "SWING_TRADE"]:
                            st.warning("⚠️ **Penny Stock Alert:** High risk/high reward - use proper position sizing and tight stops")
                        
                        # Display recommendation with proper formatting
                        if trading_style == "OPTIONS":
                            st.info(f"**Options Strategy:**\n\n{analysis.recommendation}")
                        else:
                            # For equity strategies, use markdown for better formatting
                            st.markdown(f"**{trading_style_display} Strategy:**")
                            st.markdown(analysis.recommendation)
                
                # ML-Enhanced Confidence Analysis (MOVED UP - More Prominent)
                st.subheader(f"🧠 ML-Enhanced Confidence Analysis for {trading_style_display}")
                st.write(f"**Advanced multi-factor analysis** using 50+ alpha factors optimized for **{trading_style_display}** strategy.")
                
                # Calculate ML analysis BEFORE showing it
                ml_analysis_available = False
                alpha_factors = None
                ml_prediction_score = 0
                ml_confidence_level = "UNKNOWN"
                ml_strategy_notes = []
                
                try:
                    alpha_calc = AlphaFactorCalculator()
                    alpha_factors = alpha_calc.calculate_factors(search_ticker)
                    
                    if alpha_factors:
                        ml_analysis_available = True
                        
                        # Extract key factors
                        momentum = alpha_factors.get('return_20d', 0) * 100
                        momentum_5d = alpha_factors.get('return_5d', 0) * 100
                        momentum_1d = alpha_factors.get('return_1d', 0) * 100
                        vol_ratio = alpha_factors.get('volume_5d_ratio', 1)
                        rsi = alpha_factors.get('rsi_14', 50)
                        volatility = alpha_factors.get('volatility_20d', 0) * 100
                        macd = alpha_factors.get('macd', 0)
                        macd_signal = alpha_factors.get('macd_signal', 0)
                        
                        # TRADING STYLE SPECIFIC ML SCORING
                        ml_score = 50  # baseline
                        
                        if trading_style == "DAY_TRADE":
                            # Day Trading: Focus on intraday momentum, volume, and volatility
                            st.caption("🎯 ML optimized for intraday moves and quick profits")
                            
                            # Intraday momentum (35%)
                            if momentum_1d > 2:
                                ml_score += 20
                                ml_strategy_notes.append(f"✅ Strong intraday momentum (+{momentum_1d:.1f}%)")
                            elif momentum_1d > 1:
                                ml_score += 12
                                ml_strategy_notes.append(f"✅ Good intraday momentum (+{momentum_1d:.1f}%)")
                            elif momentum_1d < -2:
                                ml_score -= 15
                                ml_strategy_notes.append(f"⚠️ Negative intraday momentum ({momentum_1d:.1f}%)")
                            
                            # Volume is critical for day trading (30%)
                            if vol_ratio > 2.0:
                                ml_score += 20
                                ml_strategy_notes.append(f"✅ Exceptional volume ({vol_ratio:.1f}x avg)")
                            elif vol_ratio > 1.5:
                                ml_score += 15
                                ml_strategy_notes.append(f"✅ High volume ({vol_ratio:.1f}x avg)")
                            elif vol_ratio < 0.8:
                                ml_score -= 15
                                ml_strategy_notes.append(f"⚠️ Low volume ({vol_ratio:.1f}x avg)")
                            
                            # Volatility is good for day trading (20%)
                            if 2 < volatility < 5:
                                ml_score += 12
                                ml_strategy_notes.append(f"✅ Good volatility for day trading ({volatility:.1f}%)")
                            elif volatility > 5:
                                ml_score += 8
                                ml_strategy_notes.append(f"⚡ High volatility - use tight stops ({volatility:.1f}%)")
                            elif volatility < 1:
                                ml_score -= 10
                                ml_strategy_notes.append(f"⚠️ Low volatility - limited profit potential ({volatility:.1f}%)")
                            
                            # RSI for entry timing (15%)
                            if 30 < rsi < 70:
                                ml_score += 8
                                ml_strategy_notes.append(f"✅ RSI in tradeable range ({rsi:.0f})")
                            elif rsi < 30:
                                ml_score += 5
                                ml_strategy_notes.append(f"🟢 Oversold - bounce opportunity (RSI {rsi:.0f})")
                            elif rsi > 70:
                                ml_score -= 5
                                ml_strategy_notes.append(f"🔴 Overbought - reversal risk (RSI {rsi:.0f})")
                        
                        elif trading_style == "SWING_TRADE":
                            # Swing Trading: Focus on multi-day trends and momentum
                            st.caption("🎯 ML optimized for 3-10 day holds and trend continuation")
                            
                            # Multi-day momentum (35%)
                            if momentum_5d > 5:
                                ml_score += 20
                                ml_strategy_notes.append(f"✅ Strong 5-day momentum (+{momentum_5d:.1f}%)")
                            elif momentum_5d > 2:
                                ml_score += 12
                                ml_strategy_notes.append(f"✅ Good 5-day momentum (+{momentum_5d:.1f}%)")
                            elif momentum_5d < -5:
                                ml_score -= 15
                                ml_strategy_notes.append(f"⚠️ Negative 5-day trend ({momentum_5d:.1f}%)")
                            
                            # Trend consistency (25%)
                            if analysis.trend in ["STRONG UPTREND", "UPTREND"] and momentum > 0:
                                ml_score += 15
                                ml_strategy_notes.append(f"✅ Consistent uptrend (20d: +{momentum:.1f}%)")
                            elif analysis.trend in ["STRONG DOWNTREND", "DOWNTREND"] and momentum < 0:
                                ml_score += 10
                                ml_strategy_notes.append(f"✅ Consistent downtrend (short opportunity)")
                            
                            # Volume confirmation (20%)
                            if vol_ratio > 1.3:
                                ml_score += 12
                                ml_strategy_notes.append(f"✅ Volume supports swing ({vol_ratio:.1f}x avg)")
                            elif vol_ratio < 0.7:
                                ml_score -= 10
                                ml_strategy_notes.append(f"⚠️ Weak volume for swing ({vol_ratio:.1f}x avg)")
                            
                            # RSI for swing entries (20%)
                            if analysis.trend in ["UPTREND", "STRONG UPTREND"] and rsi < 50:
                                ml_score += 12
                                ml_strategy_notes.append(f"✅ Pullback in uptrend (RSI {rsi:.0f})")
                            elif rsi < 30:
                                ml_score += 8
                                ml_strategy_notes.append(f"🟢 Oversold - reversal setup (RSI {rsi:.0f})")
                        
                        elif trading_style == "SCALP":
                            # Scalping: Ultra-short term, high volume, tight spreads
                            st.caption("🎯 ML optimized for seconds-to-minutes holds")
                            
                            # Extreme intraday momentum (40%)
                            if abs(momentum_1d) > 3:
                                ml_score += 25
                                ml_strategy_notes.append(f"✅ Extreme momentum for scalping ({momentum_1d:+.1f}%)")
                            elif abs(momentum_1d) > 1.5:
                                ml_score += 15
                                ml_strategy_notes.append(f"✅ Good scalp momentum ({momentum_1d:+.1f}%)")
                            else:
                                ml_score -= 20
                                ml_strategy_notes.append(f"⚠️ Insufficient momentum for scalping ({momentum_1d:+.1f}%)")
                            
                            # Volume is CRITICAL for scalping (35%)
                            if vol_ratio > 3.0:
                                ml_score += 25
                                ml_strategy_notes.append(f"✅ Exceptional liquidity ({vol_ratio:.1f}x avg)")
                            elif vol_ratio > 2.0:
                                ml_score += 18
                                ml_strategy_notes.append(f"✅ High liquidity ({vol_ratio:.1f}x avg)")
                            elif vol_ratio < 1.5:
                                ml_score -= 25
                                ml_strategy_notes.append(f"❌ Insufficient volume for scalping ({vol_ratio:.1f}x avg)")
                            
                            # High volatility needed (25%)
                            if volatility > 4:
                                ml_score += 15
                                ml_strategy_notes.append(f"✅ High volatility for scalps ({volatility:.1f}%)")
                            elif volatility < 2:
                                ml_score -= 15
                                ml_strategy_notes.append(f"⚠️ Low volatility - limited scalp range ({volatility:.1f}%)")
                        
                        elif trading_style == "BUY_HOLD":
                            # Buy & Hold: Focus on long-term trends and stability
                            st.caption("🎯 ML optimized for 6+ month holds and fundamental strength")
                            
                            # Long-term trend (40%)
                            if momentum > 15:
                                ml_score += 25
                                ml_strategy_notes.append(f"✅ Strong long-term uptrend (+{momentum:.1f}%)")
                            elif momentum > 8:
                                ml_score += 18
                                ml_strategy_notes.append(f"✅ Good long-term trend (+{momentum:.1f}%)")
                            elif momentum < -10:
                                ml_score -= 20
                                ml_strategy_notes.append(f"⚠️ Long-term downtrend ({momentum:.1f}%)")
                            
                            # Trend stability (30%)
                            if analysis.trend in ["STRONG UPTREND", "UPTREND"]:
                                ml_score += 18
                                ml_strategy_notes.append(f"✅ Stable uptrend for long-term hold")
                            elif analysis.trend in ["STRONG DOWNTREND", "DOWNTREND"]:
                                ml_score -= 15
                                ml_strategy_notes.append(f"⚠️ Downtrend - not ideal for buy & hold")
                            
                            # Lower volatility preferred (15%)
                            if volatility < 2.5:
                                ml_score += 10
                                ml_strategy_notes.append(f"✅ Low volatility - stable hold ({volatility:.1f}%)")
                            elif volatility > 5:
                                ml_score -= 8
                                ml_strategy_notes.append(f"⚠️ High volatility - risky for long hold ({volatility:.1f}%)")
                            
                            # RSI for value entry (15%)
                            if rsi < 40:
                                ml_score += 10
                                ml_strategy_notes.append(f"✅ Undervalued entry (RSI {rsi:.0f})")
                            elif rsi > 70:
                                ml_score -= 8
                                ml_strategy_notes.append(f"⚠️ Overvalued (RSI {rsi:.0f})")
                        
                        else:  # OPTIONS
                            # Options: Focus on IV, trend, and volatility
                            st.caption("🎯 ML optimized for options strategies based on IV and trend")
                            
                            # Trend strength for directional plays (30%)
                            if analysis.trend in ["STRONG UPTREND", "UPTREND"] and momentum > 5:
                                ml_score += 18
                                ml_strategy_notes.append(f"✅ Strong trend for calls (+{momentum:.1f}%)")
                            elif analysis.trend in ["STRONG DOWNTREND", "DOWNTREND"] and momentum < -5:
                                ml_score += 15
                                ml_strategy_notes.append(f"✅ Strong trend for puts ({momentum:.1f}%)")
                            
                            # IV rank consideration (25%)
                            if analysis.iv_rank > 60:
                                ml_score += 15
                                ml_strategy_notes.append(f"✅ High IV ({analysis.iv_rank}%) - sell premium")
                            elif analysis.iv_rank < 40:
                                ml_score += 12
                                ml_strategy_notes.append(f"✅ Low IV ({analysis.iv_rank}%) - buy options")
                            
                            # Volatility for options (25%)
                            if 2 < volatility < 5:
                                ml_score += 12
                                ml_strategy_notes.append(f"✅ Good volatility for options ({volatility:.1f}%)")
                            elif volatility > 6:
                                ml_score += 8
                                ml_strategy_notes.append(f"⚡ High vol - expensive options ({volatility:.1f}%)")
                            
                            # MACD for timing (20%)
                            if macd > macd_signal:
                                ml_score += 10
                                ml_strategy_notes.append(f"✅ MACD bullish crossover")
                            elif macd < macd_signal:
                                ml_score -= 5
                                ml_strategy_notes.append(f"⚠️ MACD bearish")
                        
                        ml_prediction_score = max(0, min(100, ml_score))
                        
                        # Determine confidence level
                        if ml_prediction_score >= 80:
                            ml_confidence_level = "VERY HIGH"
                        elif ml_prediction_score >= 70:
                            ml_confidence_level = "HIGH"
                        elif ml_prediction_score >= 55:
                            ml_confidence_level = "MEDIUM"
                        else:
                            ml_confidence_level = "LOW"
                except Exception as e:
                    logger.error(f"Error calculating ML analysis: {e}")
                    ml_analysis_available = False
                
                # Display ML Analysis prominently
                if ml_analysis_available and alpha_factors:
                    # Show ML Score prominently
                    ml_col1, ml_col2, ml_col3 = st.columns([2, 1, 1])
                    
                    with ml_col1:
                        st.metric(
                            "🧠 ML Prediction Score",
                            f"{ml_prediction_score:.0f}/100",
                            help="Machine Learning confidence based on 50+ alpha factors"
                        )
                        if ml_confidence_level == "VERY HIGH":
                            st.success(f"✅ **{ml_confidence_level} CONFIDENCE** - Strong ML signals align with this trade")
                        elif ml_confidence_level == "HIGH":
                            st.info(f"✅ **{ml_confidence_level} CONFIDENCE** - Good ML signals support this trade")
                        elif ml_confidence_level == "MEDIUM":
                            st.warning(f"⚠️ **{ml_confidence_level} CONFIDENCE** - Mixed ML signals, proceed with caution")
                        else:
                            st.error(f"❌ **{ml_confidence_level} CONFIDENCE** - Weak ML signals, high risk")
                    
                    with ml_col2:
                        st.metric("Factors Analyzed", f"{len(alpha_factors)}")
                        st.caption("Alpha factors calculated")
                    
                    with ml_col3:
                        # Agreement between ML and traditional analysis
                        agreement_score = abs(ml_prediction_score - analysis.confidence_score)
                        if agreement_score < 15:
                            st.metric("System Agreement", "✅ Strong")
                            st.caption("ML & Technical align")
                        elif agreement_score < 30:
                            st.metric("System Agreement", "⚠️ Moderate")
                            st.caption("Some divergence")
                        else:
                            st.metric("System Agreement", "❌ Weak")
                            st.caption("Significant divergence")
                    
                    # Display ML Strategy-Specific Insights
                    if ml_strategy_notes:
                        st.write(f"**🎯 ML Insights for {trading_style_display}:**")
                        for note in ml_strategy_notes:
                            st.write(f"• {note}")
                    
                    # Key ML Factors
                    st.write("**Key ML Signals:**")
                    col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
                    
                    with col_ml1:
                        momentum = alpha_factors.get('return_20d', 0) * 100
                        st.metric("20-Day Momentum", f"{momentum:+.1f}%")
                        if momentum > 10:
                            st.caption("🔥 Strong uptrend")
                        elif momentum < -10:
                            st.caption("❄️ Strong downtrend")
                        else:
                            st.caption("➡️ Neutral")
                    
                    with col_ml2:
                        vol_ratio = alpha_factors.get('volume_5d_ratio', 1)
                        st.metric("Volume Signal", f"{vol_ratio:.2f}x")
                        if vol_ratio > 1.5:
                            st.caption("🔥 High activity")
                        elif vol_ratio < 0.7:
                            st.caption("❄️ Low activity")
                        else:
                            st.caption("➡️ Normal")
                    
                    with col_ml3:
                        rsi = alpha_factors.get('rsi_14', 50)
                        st.metric("RSI (14)", f"{rsi:.1f}")
                        if rsi > 70:
                            st.caption("⚠️ Overbought")
                        elif rsi < 30:
                            st.caption("✅ Oversold")
                        else:
                            st.caption("➡️ Neutral")
                    
                    with col_ml4:
                        volatility = alpha_factors.get('volatility_20d', 0) * 100
                        st.metric("20-Day Volatility", f"{volatility:.1f}%")
                        if volatility > 4:
                            st.caption("⚡ High vol")
                        elif volatility < 1.5:
                            st.caption("💤 Low vol")
                        else:
                            st.caption("➡️ Moderate")
                    
                    # Show detailed factors in expander
                    with st.expander("🔬 View All 50+ Alpha Factors (Advanced)"):
                        st.info("These are the same factors used by quantitative hedge funds for algorithmic trading.")
                        
                        # Group factors by category
                        price_factors = {k: v for k, v in alpha_factors.items() if 'return' in k or 'ma' in k or 'price' in k}
                        volume_factors = {k: v for k, v in alpha_factors.items() if 'volume' in k}
                        tech_factors = {k: v for k, v in alpha_factors.items() if k in ['rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'bollinger_position']}
                        momentum_factors = {k: v for k, v in alpha_factors.items() if 'momentum' in k or 'rs_' in k}
                        vol_factors = {k: v for k, v in alpha_factors.items() if 'volatility' in k or 'hl_' in k}
                        
                        # Use stateful navigation instead of st.tabs() to prevent reruns
                        if 'alpha_factors_tab' not in st.session_state:
                            st.session_state.alpha_factors_tab = "💰 Price"
                        
                        # Tab selector using radio buttons (no rerun on selection)
                        alpha_tab_selector = st.radio(
                            "Factor Category",
                            options=["💰 Price", "📊 Volume", "📈 Technical", "🚀 Momentum", "⚡ Volatility"],
                            horizontal=True,
                            key="alpha_factors_tab_selector",
                            label_visibility="collapsed"
                        )
                        
                        # Update session state if changed
                        if alpha_tab_selector != st.session_state.alpha_factors_tab:
                            st.session_state.alpha_factors_tab = alpha_tab_selector
                        
                        # Render the selected tab
                        if st.session_state.alpha_factors_tab == "💰 Price":
                            for k, v in price_factors.items():
                                st.write(f"**{k}**: {v:.4f}")
                        elif st.session_state.alpha_factors_tab == "📊 Volume":
                            for k, v in volume_factors.items():
                                st.write(f"**{k}**: {v:.4f}")
                        elif st.session_state.alpha_factors_tab == "📈 Technical":
                            for k, v in tech_factors.items():
                                st.write(f"**{k}**: {v:.4f}")
                        elif st.session_state.alpha_factors_tab == "🚀 Momentum":
                            for k, v in momentum_factors.items():
                                st.write(f"**{k}**: {v:.4f}")
                        elif st.session_state.alpha_factors_tab == "⚡ Volatility":
                            for k, v in vol_factors.items():
                                st.write(f"**{k}**: {v:.4f}")
                else:
                    st.warning("⚠️ ML analysis unavailable for this ticker. Using traditional technical analysis only.")
                    ml_prediction_score = analysis.confidence_score
                    ml_confidence_level = "N/A"
                
                st.divider()
                
                # COMPREHENSIVE VERDICT - Final Decision Summary
                st.header("📋 COMPREHENSIVE TRADING VERDICT")
                st.write(f"**Complete analysis summary for {analysis.ticker} using {trading_style_display} approach**")
                
                # Calculate overall verdict score
                verdict_score = 0
                verdict_factors = []
                
                # Technical Analysis Score (30%)
                tech_score = analysis.confidence_score
                verdict_score += tech_score * 0.30
                verdict_factors.append(("Technical Analysis", tech_score, 30))
                
                # ML Analysis Score (30%)
                if ml_analysis_available:
                    verdict_score += ml_prediction_score * 0.30
                    verdict_factors.append(("ML Prediction", ml_prediction_score, 30))
                else:
                    verdict_score += tech_score * 0.30  # Fallback to technical
                    verdict_factors.append(("ML Prediction", tech_score, 30))
                
                # Sentiment Score (20%)
                sentiment_score_normalized = (analysis.sentiment_score + 1) * 50  # Convert -1 to 1 range to 0-100
                verdict_score += sentiment_score_normalized * 0.20
                verdict_factors.append(("News Sentiment", sentiment_score_normalized, 20))
                
                # Catalyst Score (20%)
                catalyst_score = min(100, len(analysis.catalysts) * 25)  # 25 points per catalyst, max 100
                verdict_score += catalyst_score * 0.20
                verdict_factors.append(("Catalysts", catalyst_score, 20))
                
                verdict_score = round(verdict_score, 1)
                
                # Determine final recommendation
                if verdict_score >= 75:
                    verdict_color = "success"
                    verdict_emoji = "🟢"
                    verdict_action = "STRONG BUY"
                    verdict_message = "Excellent opportunity with strong signals across all analysis methods."
                    position_size = "Standard to Large (2-5% of portfolio)"
                elif verdict_score >= 60:
                    verdict_color = "info"
                    verdict_emoji = "🟢"
                    verdict_action = "BUY"
                    verdict_message = "Good opportunity with positive signals. Proceed with confidence."
                    position_size = "Standard (1-3% of portfolio)"
                elif verdict_score >= 45:
                    verdict_color = "warning"
                    verdict_emoji = "🟡"
                    verdict_action = "CAUTIOUS BUY"
                    verdict_message = "Mixed signals. Consider smaller position or wait for better setup."
                    position_size = "Small (0.5-1.5% of portfolio)"
                else:
                    verdict_color = "error"
                    verdict_emoji = "🔴"
                    verdict_action = "AVOID / WAIT"
                    verdict_message = "Weak signals across multiple analysis methods. High risk."
                    position_size = "None - Skip this trade"
                
                # Display Verdict
                if verdict_color == "success":
                    st.success(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                elif verdict_color == "info":
                    st.info(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                elif verdict_color == "warning":
                    st.warning(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                else:
                    st.error(f"### {verdict_emoji} VERDICT: {verdict_action} - Score: {verdict_score}/100")
                
                st.write(verdict_message)
                
                # Verdict Details
                verdict_col1, verdict_col2 = st.columns(2)
                
                with verdict_col1:
                    st.write("**📊 Score Breakdown:**")
                    for factor_name, factor_score, weight in verdict_factors:
                        score_bar = "█" * int(factor_score / 10) + "░" * (10 - int(factor_score / 10))
                        st.write(f"• **{factor_name}** ({weight}%): {factor_score:.0f}/100 {score_bar}")
                    
                    st.write("")
                    st.metric("Overall Verdict Score", f"{verdict_score:.0f}/100")
                
                with verdict_col2:
                    st.write("**✅ Action Plan:**")
                    st.write(f"• **Recommended Action:** {verdict_action}")
                    st.write(f"• **Position Size:** {position_size}")
                    st.write(f"• **Entry Price:** ${analysis.price:.2f}")
                    st.write(f"• **Stop Loss:** ${analysis.support:.2f} ({((analysis.support/analysis.price - 1) * 100):.1f}%)")
                    st.write(f"• **Target:** ${analysis.resistance:.2f} ({((analysis.resistance/analysis.price - 1) * 100):.1f}%)")
                    
                    # Risk/Reward
                    risk = abs(analysis.price - analysis.support)
                    reward = abs(analysis.resistance - analysis.price)
                    rr_ratio = reward / risk if risk > 0 else 0
                    st.write(f"• **Risk/Reward Ratio:** {rr_ratio:.2f}:1")
                    
                    if rr_ratio >= 2:
                        st.caption("✅ Excellent risk/reward")
                    elif rr_ratio >= 1.5:
                        st.caption("✅ Good risk/reward")
                    else:
                        st.caption("⚠️ Suboptimal risk/reward")
                
                # Key Considerations
                st.write("**⚠️ Key Considerations:**")
                considerations = []
                
                if is_penny_stock_flag:
                    considerations.append("🔴 **Penny Stock Risk:** High volatility, use tight stops and small position size")
                
                if is_runner:
                    considerations.append("🚀 **Runner Alert:** Extreme momentum - take profits quickly, don't chase")
                
                if analysis.earnings_days_away and analysis.earnings_days_away <= 7:
                    considerations.append(f"📅 **Earnings in {analysis.earnings_days_away} days:** Expect high volatility, consider closing before earnings")
                
                if analysis.iv_rank > 70:
                    considerations.append("⚡ **Very High IV:** Great for selling premium, expensive for buying options")
                elif analysis.iv_rank < 30:
                    considerations.append("💤 **Low IV:** Good for buying options, poor for selling premium")
                
                if analysis.sentiment_score < -0.3:
                    considerations.append("📰 **Negative Sentiment:** Market pessimism may create headwinds")
                elif analysis.sentiment_score > 0.3:
                    considerations.append("📰 **Positive Sentiment:** Market optimism supports the trade")
                
                if ml_analysis_available:
                    agreement_score = abs(ml_prediction_score - analysis.confidence_score)
                    if agreement_score > 30:
                        considerations.append("⚠️ **ML/Technical Divergence:** Significant disagreement between analysis methods - proceed carefully")
                
                if not considerations:
                    considerations.append("✅ No major risk factors identified - standard trading rules apply")
                
                for consideration in considerations:
                    st.write(f"• {consideration}")
                
                # Final Notes
                st.divider()
                # Ensure alpha_factors is defined
                if 'alpha_factors' not in locals():
                    alpha_factors = None
                
                # Get current timestamp safely
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                st.caption(f"""**Analysis completed at:** {current_time} | 
**Trading Style:** {trading_style_display} | 
**Data Source:** Yahoo Finance (Real-time) | 
**ML Factors:** {len(alpha_factors) if alpha_factors else 0} alpha factors analyzed""")
                
                # AI-POWERED TRADE RECOMMENDATIONS
                st.divider()
                st.header("🤖 AI Trade Recommendations")
                st.write(f"Based on your **{trading_style_display}** analysis and **{verdict_action}** verdict")
                
                # Generate AI-powered trade recommendations
                trade_recommendations = []
                
                # Only recommend if confidence is sufficient
                if verdict_score >= 45:
                    # STOCK TRADE RECOMMENDATION
                    if trading_style in ["DAY_TRADE", "SWING_TRADE", "BUY_HOLD"]:
                        stock_rec = {
                            "type": "STOCK",
                            "symbol": analysis.ticker,
                            "action": "BUY" if analysis.trend in ["UPTREND", "STRONG UPTREND"] else "SELL_SHORT",
                            "quantity": None,  # Will calculate based on position size
                            "order_type": "limit",  # Always use limit orders to enable bracket orders with stop-loss
                            "price": analysis.price,  # Set entry price for all trades
                            "stop_loss": analysis.support,
                            "target": analysis.resistance,
                            "hold_time": "Intraday" if trading_style == "DAY_TRADE" else "3-10 days" if trading_style == "SWING_TRADE" else "6+ months",
                            "confidence": verdict_score,
                            "reasoning": f"ML Score: {ml_prediction_score:.0f}/100, Trend: {analysis.trend}, RSI: {analysis.rsi:.0f}"
                        }
                        trade_recommendations.append(stock_rec)
                    
                    # OPTIONS TRADE RECOMMENDATIONS
                    if trading_style == "OPTIONS" or verdict_score >= 60:
                        # Determine best options strategy based on analysis
                        if analysis.iv_rank > 60:
                            # High IV - Sell premium
                            if analysis.trend in ["UPTREND", "STRONG UPTREND"]:
                                options_rec = {
                                    "type": "OPTION",
                                    "strategy": "SELL PUT",
                                    "symbol": analysis.ticker,
                                    "action": "sell_to_open",
                                    "option_type": "put",
                                    "strike_suggestion": f"${analysis.support:.2f} (ATM or slightly OTM)",
                                    "dte_suggestion": "30-45 DTE",
                                    "quantity": 1,
                                    "reasoning": f"High IV ({analysis.iv_rank}%) + Uptrend = Sell puts to collect premium",
                                    "max_profit": "Premium collected",
                                    "max_risk": "Strike - Premium (if assigned)",
                                    "confidence": verdict_score
                                }
                            else:
                                options_rec = {
                                    "type": "OPTION",
                                    "strategy": "IRON CONDOR",
                                    "symbol": analysis.ticker,
                                    "action": "multi_leg",
                                    "reasoning": f"High IV ({analysis.iv_rank}%) + Sideways = Iron Condor for range-bound profit",
                                    "strike_suggestion": f"Sell at ${analysis.support:.2f} and ${analysis.resistance:.2f}",
                                    "dte_suggestion": "30-45 DTE",
                                    "confidence": verdict_score - 10
                                }
                        elif analysis.iv_rank < 40:
                            # Low IV - Buy options
                            if analysis.trend in ["UPTREND", "STRONG UPTREND"]:
                                options_rec = {
                                    "type": "OPTION",
                                    "strategy": "BUY CALL",
                                    "symbol": analysis.ticker,
                                    "action": "buy_to_open",
                                    "option_type": "call",
                                    "strike_suggestion": f"${analysis.price * 1.02:.2f} (slightly OTM)",
                                    "dte_suggestion": "30-60 DTE",
                                    "quantity": 1,
                                    "reasoning": f"Low IV ({analysis.iv_rank}%) + Uptrend = Buy calls for directional move",
                                    "max_profit": "Unlimited",
                                    "max_risk": "Premium paid",
                                    "confidence": verdict_score
                                }
                            elif analysis.trend in ["DOWNTREND", "STRONG DOWNTREND"]:
                                options_rec = {
                                    "type": "OPTION",
                                    "strategy": "BUY PUT",
                                    "symbol": analysis.ticker,
                                    "action": "buy_to_open",
                                    "option_type": "put",
                                    "strike_suggestion": f"${analysis.price * 0.98:.2f} (slightly OTM)",
                                    "dte_suggestion": "30-60 DTE",
                                    "quantity": 1,
                                    "reasoning": f"Low IV ({analysis.iv_rank}%) + Downtrend = Buy puts for directional move",
                                    "max_profit": "Strike - Premium",
                                    "max_risk": "Premium paid",
                                    "confidence": verdict_score
                                }
                            else:
                                options_rec = None
                        else:
                            # Medium IV - Spreads
                            if analysis.trend in ["UPTREND", "STRONG UPTREND"]:
                                options_rec = {
                                    "type": "OPTION",
                                    "strategy": "BULL CALL SPREAD",
                                    "symbol": analysis.ticker,
                                    "action": "multi_leg",
                                    "reasoning": f"Medium IV ({analysis.iv_rank}%) + Uptrend = Bull call spread for defined risk",
                                    "strike_suggestion": f"Buy ${analysis.price:.2f}, Sell ${analysis.resistance:.2f}",
                                    "dte_suggestion": "30-45 DTE",
                                    "confidence": verdict_score - 5
                                }
                            else:
                                options_rec = None
                        
                        if options_rec:
                            trade_recommendations.append(options_rec)
                
                # Display recommendations
                if trade_recommendations:
                    for i, rec in enumerate(trade_recommendations, 1):
                        with st.expander(f"{'📈' if rec['type'] == 'STOCK' else '🎯'} Recommendation #{i}: {rec['type']} - {rec.get('strategy', rec.get('action', '').upper())}", expanded=True):
                            rec_col1, rec_col2 = st.columns([2, 1])
                            
                            with rec_col1:
                                if rec['type'] == 'STOCK':
                                    st.write(f"**Strategy:** {rec['action']} {rec['symbol']} stock")
                                    st.write(f"**Order Type:** {rec['order_type'].upper()}")
                                    if rec['price']:
                                        st.write(f"**Entry Price:** ${rec['price']:.2f}")
                                    st.write(f"**Stop Loss:** ${rec['stop_loss']:.2f} ({((rec['stop_loss']/analysis.price - 1) * 100):.1f}%)")
                                    st.write(f"**Target:** ${rec['target']:.2f} ({((rec['target']/analysis.price - 1) * 100):.1f}%)")
                                    st.write(f"**Hold Time:** {rec['hold_time']}")
                                    st.info(f"💡 **Why:** {rec['reasoning']}")
                                    
                                    # Calculate position size based on verdict
                                    if verdict_score >= 75:
                                        position_pct = "2-5%"
                                        shares_example = "20-50"
                                    elif verdict_score >= 60:
                                        position_pct = "1-3%"
                                        shares_example = "10-30"
                                    else:
                                        position_pct = "0.5-1.5%"
                                        shares_example = "5-15"
                                    
                                    st.caption(f"**Suggested Position Size:** {position_pct} of portfolio (~{shares_example} shares for $10k account)")
                                
                                else:  # OPTIONS
                                    st.write(f"**Strategy:** {rec['strategy']}")
                                    st.write(f"**Symbol:** {rec['symbol']}")
                                    st.write(f"**Strike:** {rec.get('strike_suggestion', 'See details')}")
                                    st.write(f"**Expiration:** {rec.get('dte_suggestion', '30-45 DTE')}")
                                    if 'max_profit' in rec:
                                        st.write(f"**Max Profit:** {rec['max_profit']}")
                                    if 'max_risk' in rec:
                                        st.write(f"**Max Risk:** {rec['max_risk']}")
                                    st.info(f"💡 **Why:** {rec['reasoning']}")
                                    st.caption(f"**Contracts:** Start with 1-2 contracts, scale based on experience")
                            
                            with rec_col2:
                                st.metric("Confidence", f"{rec['confidence']:.0f}/100")
                                
                                if rec['confidence'] >= 75:
                                    st.success("✅ HIGH CONFIDENCE")
                                elif rec['confidence'] >= 60:
                                    st.info("✅ GOOD CONFIDENCE")
                                else:
                                    st.warning("⚠️ MODERATE")
                                
                                # Execute button with callback - capture loop variables with default args
                                def execute_trade_callback(recommendation=rec, price=analysis.price, verdict=verdict_action, rec_num=i):
                                    logger.info(f"🔥 EXECUTE BUTTON CLICKED for recommendation #{rec_num}")
                                    logger.info(f"📊 Setting session state: symbol={recommendation['symbol']}, price={price}, verdict={verdict}")
                                    st.session_state.selected_recommendation = recommendation
                                    st.session_state.quick_trade_ticker = recommendation['symbol']
                                    st.session_state.quick_trade_price = price
                                    st.session_state.quick_trade_verdict = verdict
                                    st.session_state.show_quick_trade = True
                                    logger.info(f"✅ Session state set: show_quick_trade={st.session_state.show_quick_trade}")
                                
                                st.button(
                                    f"🚀 Execute This Trade", 
                                    key=f"execute_{i}", 
                                    width="stretch", 
                                    type="primary",
                                    on_click=execute_trade_callback
                                )
                else:
                    st.warning("⚠️ No trade recommendations - Verdict score too low. Consider waiting for a better setup.")
                
                # Other quick actions
                st.divider()
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    if st.button("🎯 Get More Strategy Ideas", width="stretch"):
                        st.session_state.goto_strategy_advisor = True
                        # Navigation action needs rerun
                        st.rerun()
                
                with action_col2:
                    if st.button("📊 View in Strategy Analyzer", width="stretch"):
                        st.session_state.analyzer_ticker = analysis.ticker
                        # Navigation action needs rerun
                        st.rerun()
                
            else:
                st.error(f"❌ Could not analyze {search_ticker}. Please check the ticker symbol.")
    
    elif st.session_state.current_analysis:
        st.info("💡 Previous analysis is displayed. Enter a new ticker and click Analyze to update.")


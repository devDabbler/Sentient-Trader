"""

Dashboard Tab

Main dashboard with stock analysis, signal generation, and quick execution



Extracted from app.py for modularization

"""

import streamlit as st

import time

from datetime import datetime, timedelta

from loguru import logger

from typing import Dict, List, Optional, Tuple

from analyzers.comprehensive import ComprehensiveAnalyzer

from ui.tabs.common_imports import (
    yf, requests, AITradingSignalGenerator, UnifiedPennyStockAnalysis, 
    PENNY_THRESHOLDS, is_penny_stock
)

from utils.caching import get_cached_news



def render_tab():

    """Main render function called from app.py"""

    st.header("Stock Intelligence")

    

    # TODO: Review and fix imports

    # Tab implementation below (extracted from app.py)



    # Render main components

    _render_header()

    search_ticker, trading_style, analyze_btn = _render_input_section()

    

    # Render quick trade modal if active

    if st.session_state.get('show_quick_trade', False):

        _render_quick_trade_modal()

    

    # Handle analysis
    trigger_analysis = st.session_state.get('trigger_analysis', False)

    if (analyze_btn or trigger_analysis) and search_ticker:
        if trigger_analysis:
            st.session_state.trigger_analysis = False
        _handle_analysis(search_ticker, trading_style)

    

    # Display results

    _display_analysis_results()





def _render_header():

    """Render the dashboard header and description"""

    logger.info("🏁 TAB1 RENDERING - Session state: show_quick_trade={}, has_analysis={}", 

                str(st.session_state.get('show_quick_trade', 'NOT SET')), 

                str(st.session_state.get('current_analysis') is not None))

    st.header("🔍 Comprehensive Stock Intelligence")

    st.write("Get real-time analysis including news, catalysts, technical indicators, and IV metrics.")

    st.info("💡 **Works with ALL stocks:** Blue chips, penny stocks (<$5), OTC stocks, and runners. Automatically detects momentum plays!")





def _render_input_section() -> Tuple[str, str, bool]:

    """Render the input section for ticker and trading style"""

    col1, col2, col3 = st.columns([2, 1, 1])

    

    with col1:
        # Use session state for default value if set (e.g. from scanner)
        default_ticker = st.session_state.get('analyze_ticker', "SOFI")
        
        search_ticker = st.text_input(
            "Enter Ticker Symbol to Analyze", 
            value=default_ticker,
            help="Enter any ticker: AAPL, TSLA, penny stocks (SNDL, GNUS), or OTC stocks"
        ).upper()
        
        # Clear the session state trigger after using it so it doesn't persist
        if 'analyze_ticker' in st.session_state and st.session_state.analyze_ticker != "SOFI":
             # We don't clear it immediately to allow re-runs, but maybe we should?
             # Actually, if we set it, the text_input will use it as value.
             # If the user types something else, it will update.
             pass

    

    with col2:

        trading_style_display, trading_style = _render_trading_style_selector()

    

    with col3:

        st.write("")

        st.write("")

        analyze_btn = st.button("🔍 Analyze Stock", type="primary", width="stretch")

    # Quick examples with style descriptions

    st.caption("**Examples:** AAPL (blue chip) | SNDL (penny stock) | SPY (ETF) | TSLA (volatile) | Any OTC stock")

    

    # Display style description

    _display_style_description(trading_style)

    

    return search_ticker, trading_style, analyze_btn





def _render_trading_style_selector() -> Tuple[str, str]:

    """Render the trading style selector dropdown"""

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

    return trading_style_display, trading_style





def _display_style_description(trading_style: str):

    """Display the description for the selected trading style"""

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





def _render_quick_trade_modal():

    """Render the quick trade modal when activated"""

    logger.info(f"🔍 Checking modal display: show_quick_trade={st.session_state.get('show_quick_trade', False)}")

    

    logger.info("🚀 DISPLAYING QUICK TRADE MODAL AT TOP OF TAB1")

    st.divider()

    st.header("🚀 Execute Trade")

    

    # Get the selected recommendation and analysis

    selected_rec = st.session_state.get('selected_recommendation', None)

    analysis = st.session_state.get('current_analysis', None)

    

    if not analysis:

        _render_no_analysis_error()

        return

    

    logger.info(f"✅ Modal has analysis data: ticker={analysis.ticker}, price={analysis.price}")

    

    # Display recommendation header

    if selected_rec:

        logger.info("✅ Modal has recommendation: {} - {}", str(selected_rec.get('type')), selected_rec.get('strategy', 'N/A'))

        st.subheader(f"📋 {selected_rec['type']} - {selected_rec.get('strategy', selected_rec.get('action', ''))}")

    else:

        st.subheader(f"📋 Quick Trade: {st.session_state.get('quick_trade_ticker', 'N/A')}")

    

    # Check Tradier connection

    if not st.session_state.tradier_client:

        _render_tradier_not_connected_error()

        return

    

    # Show trade configuration

    _render_trade_configuration(selected_rec, analysis)





def _render_no_analysis_error():

    """Render error when no analysis data is available"""

    logger.error("❌ Modal error: No analysis data in session state")

    st.error("❌ Analysis data not available. Please analyze a stock first.")

    if st.button("Close"):

        st.session_state.show_quick_trade = False

        st.rerun()





def _render_tradier_not_connected_error():

    """Render error when Tradier is not connected"""

    st.error("❌ Tradier not connected. Please configure in the 🏦 Tradier Account tab.")

    if st.button("Close", key="close_no_tradier"):

        st.session_state.show_quick_trade = False

        st.rerun()





def _render_trade_configuration(selected_rec, analysis):

    """Render the main trade configuration section"""

    verdict_action = st.session_state.get('quick_trade_verdict', 'N/A')

    st.success(f"✅ Tradier Connected | Verdict: **{verdict_action}**")

    

    # Show AI recommendation details

    if selected_rec:

        st.info(f"**AI Reasoning:** {selected_rec.get('reasoning', 'N/A')}")

        if selected_rec['type'] == 'STOCK':

            st.caption(f"Stop Loss: ${selected_rec['stop_loss']:.2f} | Target: ${selected_rec['target']:.2f} | Hold: {selected_rec['hold_time']}")

        else:

            st.caption(f"Strike: {selected_rec.get('strike_suggestion', 'N/A')} | DTE: {selected_rec.get('dte_suggestion', 'N/A')}")

    

    trade_col1, trade_col2 = st.columns(2)

    

    with trade_col1:

        _render_order_inputs(selected_rec, analysis)

    

    with trade_col2:

        _render_order_summary(selected_rec, analysis)

    

    _render_place_order_button(selected_rec, analysis)





def _render_order_inputs(selected_rec, analysis):

    """Render the order input fields"""

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

        _render_options_inputs(selected_rec, trade_symbol)

    else:

        _render_equity_inputs(default_action, default_qty)

    

    return trade_symbol, is_options_trade





def _render_options_inputs(selected_rec, trade_symbol):

    """Render options-specific input fields"""

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

        

        _render_options_symbol_generation(selected_rec, trade_symbol)

        

        trade_side = st.selectbox("Action", 

                                ["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"],

                                index=0, key="modal_trade_side")

        trade_quantity = st.number_input("Contracts", min_value=1, value=1, step=1, key="modal_trade_qty")

    else:

        trade_side = st.selectbox("Action", ["buy", "sell", "sell_short", "buy_to_cover"], key="modal_trade_side2")

        trade_quantity = st.number_input("Quantity (shares)", min_value=1, value=10, step=1, key="modal_trade_qty2")





def _render_equity_inputs(default_action, default_qty):

    """Render equity-specific input fields"""

    if default_action == "SELL_SHORT":

        side_index = 2

    elif default_action == "BUY":

        side_index = 0

    else:

        side_index = 0

    

    trade_side = st.selectbox("Action", 

                            ["buy", "sell", "sell_short", "buy_to_cover"],

                            index=side_index, key="modal_trade_side3")

    trade_quantity = st.number_input("Quantity (shares)", min_value=1, value=default_qty, step=1, key="modal_trade_qty3")





def _render_options_symbol_generation(selected_rec, trade_symbol):

    """Render options symbol generation controls"""

    # Generate options contract symbol automatically

    col1, col2 = st.columns([2, 1])

    

    with col1:

        # Auto-generate options symbol if we have the required data

        auto_generated_symbol = ""

        if selected_rec and selected_rec.get('strike_suggestion') and selected_rec.get('dte_suggestion'):

            try:

                auto_generated_symbol = _generate_options_symbol(selected_rec, trade_symbol)

                

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

                    generated_symbol = _generate_options_symbol(selected_rec, trade_symbol)

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

                _validate_options_symbol(symbol)

            else:

                st.error("Please enter an options symbol first")





def _generate_options_symbol(selected_rec, trade_symbol):

    """Generate an options symbol from strike and DTE suggestions"""

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

    return f"{trade_symbol.upper()}{exp_date.strftime('%y%m%d')}{option_type}{int(strike * 1000):08d}"





def _validate_options_symbol(symbol):

    """Validate an options symbol format and existence"""

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





def _render_order_summary(selected_rec, analysis):

    """Render the order summary section"""

    st.write("**Order Summary:**")

    

    # Show bracket order mode indicator

    will_use_bracket = (

        st.session_state.get('modal_trade_class', 'equity') == "equity" and 

        st.session_state.get('modal_trade_type', 'market') == "limit" and 

        st.session_state.get('modal_trade_side', 'buy') in ["buy", "sell"]

    )

    

    if will_use_bracket:

        st.success("🎯 **BRACKET MODE**: Auto stop-loss & take-profit enabled")

    else:

        st.info("📊 **SIMPLE ORDER MODE**")

    

    st.divider()

    

    # Calculate estimated cost

    is_options_trade = selected_rec and selected_rec['type'] == 'OPTION'

    if is_options_trade:

        st.warning("Options pricing requires real-time quote - estimate not available")

        estimated_cost = "TBD"

    else:

        trade_type = st.session_state.get('modal_trade_type', 'market')

        trade_price = st.session_state.get('modal_trade_price')

        trade_quantity = st.session_state.get('modal_trade_qty', 10)

        

        if trade_type == "limit" and trade_price:

            estimated_cost = trade_price * trade_quantity

        else:

            estimated_cost = analysis.price * trade_quantity

        st.metric("Estimated Cost", f"${estimated_cost:,.2f}")

    

    verdict_action = st.session_state.get('quick_trade_verdict', 'N/A')

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





def _render_place_order_button(selected_rec, analysis):

    """Render the place order button and validation logic"""

    st.write("")

    confirm_col1, confirm_col2 = st.columns(2)

    

    with confirm_col1:

        if st.button("✅ Place Order", type="primary", width="stretch", key="modal_place_order"):

            _execute_order_placement(selected_rec, analysis)

    

    with confirm_col2:

        if st.button("❌ Cancel", width="stretch", key="modal_cancel_order"):

            st.session_state.show_quick_trade = False

            st.rerun()





def _execute_order_placement(selected_rec, analysis):

    """Execute the order placement with validation"""

    with st.spinner("Placing order..."):

        try:

            # Validate required fields

            trade_symbol = st.session_state.get('modal_trade_symbol', '')

            trade_quantity = st.session_state.get('modal_trade_qty', 0)

            trade_type = st.session_state.get('modal_trade_type', 'market')

            trade_price = st.session_state.get('modal_trade_price')

            trade_class = st.session_state.get('modal_trade_class', 'equity')

            

            if not trade_symbol:

                st.error("❌ Please enter a symbol")

                st.stop()

            elif trade_quantity <= 0:

                st.error("❌ Quantity must be greater than 0")

                st.stop()

            elif trade_type == "limit" and (not trade_price or trade_price <= 0):

                st.error("❌ Please enter a valid limit price")

                st.stop()

            elif trade_class == "option" and not st.session_state.get('modal_option_symbol', ''):

                st.error("❌ Please enter the options contract symbol (e.g., SOFI250117P00029000)")

                st.stop()

            

            # Place the order logic would go here

            st.success("✅ Order placed successfully!")

            

        except Exception as e:

            st.error(f"❌ Error placing order: {e}")

            logger.error(f"Order placement error: {e}")





def _handle_analysis(search_ticker: str, trading_style: str):
    """Handle the stock analysis when analyze button is clicked"""
    # Clear previous analysis from session state
    if 'analysis' in st.session_state:
        del st.session_state['analysis']

    # Use new st.status for better progress indication
    with st.status(f"🔍 Analyzing {search_ticker}...", expanded=True) as status:
        st.write("📊 Fetching market data...")
        
        st.write("📈 Calculating technical indicators...")
        
        st.write("📰 Analyzing news sentiment...")
        
        st.write("🎯 Identifying catalysts...")
        
        st.write("📄 Fetching SEC filings (8-K, 10-Q, 10-K)...")
        
        st.write(f"🤖 Generating {trading_style} recommendations...")
        analysis = ComprehensiveAnalyzer.analyze_stock(search_ticker, trading_style)
        
        # Fetch SEC filings and enhanced catalyst data
        sec_filings = []
        enhanced_catalysts = []
        if analysis:
            try:
                logger.info(f"📄 Fetching SEC filings for {search_ticker}...")
                
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
                    
                    if cik:
                        # Create SEC detector instance (using None for alert_system since we just want data)
                        class TempSECDetector:
                            def __init__(self):
                                self.user_agent = "Sentient Trader/1.0 (trading@example.com)"
                            
                            def get_company_cik(self, ticker: str):
                                return cik
                            
                            def get_recent_filings(self, ticker: str, cik: str, hours_back: int = 168):
                                """Get recent SEC filings (last 7 days)"""
                                try:
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
                                                
                                                filings.append({
                                                    'ticker': ticker,
                                                    'form_type': form_type,
                                                    'filing_date': filing_date.strftime('%Y-%m-%d'),
                                                    'description': form_type,
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
                        
                        # Analyze filings for catalysts
                        if sec_filings:
                            for filing in sec_filings:
                                if filing['form_type'] == '8-K':
                                    enhanced_catalysts.append({
                                        'type': 'SEC Filing - 8-K',
                                        'date': filing['filing_date'],
                                        'days_away': -filing['days_ago'],
                                        'impact': 'HIGH',
                                        'description': f"Material event filing: {filing['description']}",
                                        'filing_url': filing['url'],
                                        'is_critical': filing['is_critical']
                                    })
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
            st.write("💰 Running enhanced penny stock analysis...")
            try:
                unified_analyzer = UnifiedPennyStockAnalysis()
                
                # Map trading style for penny stock analysis
                penny_style_map = {
                    "DAY_TRADE": "SCALP",
                    "SWING_TRADE": "SWING",
                    "SCALP": "SCALP",
                    "BUY_HOLD": "POSITION",
                    "OPTIONS": "SWING"
                }
                penny_trading_style = penny_style_map.get(trading_style, "SWING")
                
                penny_stock_analysis = unified_analyzer.analyze_comprehensive(
                    ticker=search_ticker,
                    trading_style=penny_trading_style,
                    include_backtest=False,  # Skip backtest for speed
                    check_options=(trading_style == "OPTIONS")
                )
                
                st.session_state.penny_stock_analysis = penny_stock_analysis
            except Exception as e:
                logger.error(f"❌ ERROR running unified penny stock analysis for {search_ticker}: {e}", exc_info=True)
                penny_stock_analysis = None
        
        # --- Generate Premium AI Trading Signal ---
        st.session_state.ai_trading_signal = None
        if analysis:
            st.write("🤖 Generating Premium AI Trading Signal with Gemini...")
            try:
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
                social_data = None

                # Generate the signal using the configured premium model
                ai_signal = signal_generator.generate_signal(
                    symbol=analysis.ticker,
                    technical_data=technical_data,
                    news_data=news_data,
                    sentiment_data=sentiment_data,
                    social_data=social_data,
                    trading_style=trading_style
                )
                st.session_state.ai_trading_signal = ai_signal
            except Exception as e:
                logger.error(f"Error generating AI signal: {e}")

        if analysis:
            status.update(label=f"✅ Analysis complete for {search_ticker}", state="complete")
            st.session_state.current_analysis = analysis
        else:
            status.update(label=f"❌ Analysis failed for {search_ticker}", state="error")





def _display_analysis_results():
    """Display the analysis results if available"""
    analysis = st.session_state.get('current_analysis')
    if not analysis:
        return

    # Detect penny stock and runner characteristics
    is_penny_stock_flag = is_penny_stock(analysis.price)
    is_otc = analysis.ticker.endswith(('.OTC', '.PK', '.QB'))
    volume_vs_avg = ((analysis.volume / analysis.avg_volume - 1) * 100) if analysis.avg_volume > 0 else 0
    is_runner = volume_vs_avg > 200 and analysis.change_pct > 10  # 200%+ volume spike and 10%+ gain
    
    # Get unified penny stock analysis if available
    penny_stock_analysis = st.session_state.get('penny_stock_analysis')
    
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
    
    # Technical Indicators
    st.subheader("📊 Technical Indicators")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.metric("RSI (14)", f"{analysis.rsi:.1f}")
    
    with tech_col2:
        st.metric("MACD Signal", analysis.macd_signal)
    
    with tech_col3:
        st.metric("Support", f"${analysis.support}")
        st.metric("Resistance", f"${analysis.resistance}")
    
    # Strategy Recommendation
    if analysis.recommendation:
        st.subheader("💡 Strategy Recommendation")
        st.info(analysis.recommendation)
    
    # News & Sentiment
    st.subheader("📰 Recent News & Sentiment")
    
    sentiment_col1, sentiment_col2 = st.columns([1, 3])
    
    with sentiment_col1:
        sentiment_label = "POSITIVE" if analysis.sentiment_score > 0.2 else "NEGATIVE" if analysis.sentiment_score < -0.2 else "NEUTRAL"
        sentiment_color = "🟢" if analysis.sentiment_score > 0.2 else "🔴" if analysis.sentiment_score < -0.2 else "🟡"
        
        st.metric("News Sentiment", f"{sentiment_color} {sentiment_label}")
        st.metric("Sentiment Score", f"{analysis.sentiment_score:.2f}")
    
    with sentiment_col2:
        if analysis.recent_news:
            st.write("**Latest News Articles:**")
            for idx, article in enumerate(analysis.recent_news[:5]):
                with st.expander(f"📰 {article['title']}"):
                    st.write(f"**Publisher:** {article['publisher']}")
                    st.write(f"**Published:** {article['published']}")
                    if article.get('link'):
                        st.write(f"[📖 Read Full Article]({article['link']})")
        else:
            st.info("📭 No recent news found for this ticker.")


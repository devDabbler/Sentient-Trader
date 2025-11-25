"""
Crypto Trading Tab
Cryptocurrency trading with Kraken integration, scanners, and position monitoring

Extracted from app.py for modularization
"""
import streamlit as st
from loguru import logger
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, cast
import os
import time
from datetime import datetime, timedelta
import pandas as pd

if TYPE_CHECKING:
    from services.ai_crypto_scanner import AICryptoScanner

# Import Kraken client with fallback
try:
    from clients.kraken_client import KrakenClient
except ImportError:
    logger.debug("KrakenClient not available")
    KrakenClient = None

# Cached Kraken client getter
@st.cache_resource
def get_kraken_client(api_key: str, api_secret: str):
    """Get cached Kraken client instance"""
    if KrakenClient is None:
        raise ImportError("KrakenClient not available")
    client = KrakenClient(api_key=api_key, api_secret=api_secret)
    success, message = client.validate_connection()
    if not success:
        raise ConnectionError(f"Kraken connection failed: {message}")
    return client

# Cached scanner getters
@st.cache_resource
def get_crypto_scanner(_kraken_client, _crypto_config):
    """Get cached CryptoOpportunityScanner"""
    try:
        from services.crypto_scanner import CryptoOpportunityScanner
        return CryptoOpportunityScanner(_kraken_client, _crypto_config)
    except Exception as e:
        logger.error(f"Failed to create CryptoOpportunityScanner: {e}")
        return None

@st.cache_resource
def get_ai_crypto_scanner(_kraken_client, _crypto_config):
    """Get cached AICryptoScanner"""
    try:
        from services.ai_crypto_scanner import AICryptoScanner
        return AICryptoScanner(_kraken_client, _crypto_config)
    except Exception as e:
        logger.error(f"Failed to create AICryptoScanner: {e}")
        return None

@st.cache_resource
def get_penny_crypto_scanner(_kraken_client, _crypto_config):
    """Get cached PennyCryptoScanner"""
    try:
        from services.penny_crypto_scanner import PennyCryptoScanner
        return PennyCryptoScanner(_kraken_client, _crypto_config)
    except Exception as e:
        logger.error(f"Failed to create PennyCryptoScanner: {e}")
        return None

@st.cache_resource
def get_sub_penny_discovery():
    """Get cached SubPennyDiscovery"""
    try:
        from services.sub_penny_discovery import SubPennyDiscovery
        return SubPennyDiscovery()
    except Exception as e:
        logger.error(f"Failed to create SubPennyDiscovery: {e}")
        return None

def render_tab():
    """Main render function called from app.py"""
    st.header("Crypto Trading")
    
    # TODO: Review and fix imports
    # Tab implementation below (extracted from app.py)


    # Ensure main tab state is preserved during reruns
    st.session_state.active_main_tab = "₿ Crypto Trading"
    
    st.header("₿ Cryptocurrency Trading (Kraken Integration)")
    st.write("Trade cryptocurrencies 24/7 with AI-powered signals and automated strategies.")
    
    # Check if Kraken is configured
    kraken_key = os.getenv('KRAKEN_API_KEY')
    kraken_secret = os.getenv('KRAKEN_API_SECRET')
    
    if not kraken_key or not kraken_secret:
        st.error("🔑 **Kraken API credentials not found!**")
        st.info("Please set up your Kraken API keys in the `.env` file.")
        st.markdown("""
        ### 🚀 Quick Setup Guide:
        
        1. **Create Kraken Account**: Visit [kraken.com](https://www.kraken.com/)
        2. **Generate API Keys**: Go to Settings > API
        3. **Set Permissions**: 
           - ✅ Query Funds
           - ✅ Query Orders
           - ✅ Create/Modify Orders (for live trading)
           - ❌ Withdraw Funds (keep disabled!)
        4. **Add to `.env` file**:
           ```
           KRAKEN_API_KEY=your_api_key_here
           KRAKEN_API_SECRET=your_private_key_here
           ```
        5. **Read Full Guide**: `documentation/KRAKEN_SETUP_GUIDE.md`
        
        ⚠️ **IMPORTANT**: Kraken has NO paper trading mode. Start with $100-200 "learning capital" 
        and use $20-30 position sizes for testing. See `documentation/CRYPTO_QUICK_START.md`
        
        🔒 **Security**: Never share your API keys! Store them only in `.env` file.
        """)
        return
    
    # Import crypto modules
    try:
        from clients.kraken_client import KrakenClient
        from services.crypto_scanner import CryptoOpportunityScanner
        from services.crypto_trading_signals import CryptoTradingSignalGenerator
        import config_crypto_trading as crypto_config
    except ImportError as e:
        st.error(f"Error importing crypto modules: {e}")
        st.info("Please ensure all crypto trading files are in place.")
        return
    
    # Initialize Kraken client (cached to avoid 4-5 second delay on reruns)
    try:
        kraken_client = get_kraken_client(api_key=kraken_key, api_secret=kraken_secret)
        st.success("✅ Connected to Kraken (cached)")
    except Exception as e:
        st.error(f"❌ **Failed to initialize Kraken client**: {e}")
        return
    
    # Initialize crypto watchlist manager
    if 'crypto_watchlist_manager' not in st.session_state:
        from services.crypto_watchlist_manager import CryptoWatchlistManager
        st.session_state.crypto_watchlist_manager = CryptoWatchlistManager()
    
    crypto_wl_manager = st.session_state.crypto_watchlist_manager
    
    # Use stateful navigation instead of st.tabs() to prevent automatic redirect
    if 'active_crypto_tab' not in st.session_state:
        st.session_state.active_crypto_tab = "📊 Dashboard"
    
    # Tab selector - STREAMLINED (was 12, now 8)
    tab_options = [
        "📊 Dashboard",  # Merged: Market Overview + Portfolio
        "🔍 Daily Scanner",  # Consolidated: All scanners (Penny, Sub-Penny, CoinGecko, Multi-Config)
        "⭐ My Watchlist",
        "⚡ Quick Trade",
        "🔔 Entry Monitors",
        "🤖 AI Position Monitor",
        "📓 Trade Journal",
        "🎯 DEX Launch Hunter"  # NEW: Early DEX token discovery (high risk)
    ]
    
    # --- New Tab Navigation Logic (using on_change callback) ---

    def update_active_crypto_tab():
        """Callback to update session state when radio button is changed by user."""
        st.session_state.active_crypto_tab = st.session_state.crypto_tab_selector

    # Determine the index for the radio button based on the single source of truth
    try:
        current_index = tab_options.index(st.session_state.active_crypto_tab)
    except (ValueError, KeyError):
        current_index = 1  # Default to Crypto Scanner if state is invalid
        st.session_state.active_crypto_tab = tab_options[current_index]

    # CRITICAL: Sync the radio selector key with active_crypto_tab for programmatic navigation
    # This ensures that when active_crypto_tab is set programmatically (e.g., from Daily Scanner),
    # the radio button reflects the correct tab on the next render
    if 'crypto_tab_selector' not in st.session_state:
        st.session_state.crypto_tab_selector = st.session_state.active_crypto_tab
    elif st.session_state.crypto_tab_selector != st.session_state.active_crypto_tab:
        # Programmatic navigation detected - sync the selector
        st.session_state.crypto_tab_selector = st.session_state.active_crypto_tab

    st.radio(
        "Select Crypto Feature:",
        tab_options,
        index=current_index,
        horizontal=True,
        key="crypto_tab_selector",
        on_change=update_active_crypto_tab  # Use the robust on_change callback
    )

    # The single source of truth for the active tab
    active_crypto_tab = st.session_state.active_crypto_tab
    
    st.divider()
    
    # ========== CHECK FOR MULTI-CONFIG BUTTON CLICKS (GLOBAL HANDLER) ==========
    # This catches button clicks from Daily Scanner and transfers setup to Quick Trade
    # Runs BEFORE tab rendering so it works regardless of which tab you navigate to
    
    # Debug: Log clicked flags in session state
    clicked_flags = [k for k in st.session_state.keys() if '_clicked' in str(k)]
    if clicked_flags:
        logger.info(f"🔘 CRYPTO TAB GLOBAL - Found clicked flags: {clicked_flags}")
    
    if 'multi_config_results' in st.session_state and st.session_state.multi_config_results is not None:
        results_df = st.session_state.multi_config_results
        logger.info(f"🔘 CRYPTO TAB GLOBAL - multi_config_results found with {len(results_df)} rows, index: {list(results_df.index)}")
        
        # Check if any "Use This Setup" button was clicked
        selected_config_idx = None
        for idx in results_df.index:
            # Check both button types: use_config_{idx} (best-per-pair) and use_filtered_{idx} (filtered results)
            flag_key = f'use_config_{idx}_clicked'
            flag_value = st.session_state.get(flag_key, False)
            logger.debug(f"🔘 CRYPTO TAB GLOBAL - Checking {flag_key} = {flag_value}")
            if flag_value:
                selected_config_idx = idx
                st.session_state[flag_key] = False  # Reset flag
                logger.info(f"🔘 CRYPTO TAB - Detected button click for config {idx} (best-per-pair)")
                break
            elif st.session_state.get(f'use_filtered_{idx}_clicked', False):
                selected_config_idx = idx
                st.session_state[f'use_filtered_{idx}_clicked'] = False  # Reset flag
                logger.info(f"🔘 CRYPTO TAB - Detected button click for filtered {idx} (filtered results)")
                break
        
        # If a config was selected, transfer to Quick Trade
        if selected_config_idx is not None:
            row = results_df.loc[selected_config_idx]
            pair = row.get('pair', 'UNKNOWN')
            trade_type = row.get('trade_type', 'UNKNOWN')
            
            logger.info(f"🔘 CRYPTO TAB - Transferring setup for {pair} - {trade_type}")
            
            # Store complete setup with REAL market data
            st.session_state.crypto_scanner_opportunity = {
                'symbol': row.get('pair', 'UNKNOWN'),
                'strategy': row.get('strategy', 'Unknown'),
                'confidence': row.get('ai_approved', False),
                'risk_level': 'Medium' if (row.get('leverage', 0) or 0) <= 2 else 'High',
                'score': row.get('ai_score', 0),
                'current_price': row.get('current_price', 0),
                'change_24h': row.get('change_24h', 0),
                'volume_ratio': (row.get('volume_24h', 0) or 0) / 1000000 if (row.get('volume_24h', 0) or 0) > 0 else 1.0,
                'volatility': row.get('volatility', 0),
                'reason': f"{row.get('trade_type', 'UNKNOWN')} recommended",
                'ai_reasoning': row.get('ai_recommendation', ''),
                'ai_confidence': 'High' if row.get('ai_confidence', 0) >= 75 else 'Medium' if row.get('ai_confidence', 0) >= 50 else 'Low',
                'ai_rating': row.get('ai_confidence', 0) / 10,
                'ai_risks': row.get('ai_risks', [])
            }
            
            st.session_state.crypto_quick_pair = row.get('pair', 'UNKNOWN')
            st.session_state.crypto_quick_trade_pair = row.get('pair', 'UNKNOWN')
            st.session_state.crypto_quick_direction = row.get('direction', 'BUY')
            st.session_state.crypto_trading_mode = row.get('trading_mode', 'Spot Trading')
            st.session_state.crypto_quick_leverage = row.get('leverage', 1)
            st.session_state.crypto_quick_position_size = row.get('position_size', 100)
            st.session_state.crypto_quick_stop_pct = row.get('stop_pct', 2.0)
            st.session_state.crypto_quick_target_pct = row.get('target_pct', 5.0)
            
            logger.info(f"📝 CRYPTO TAB - Session state set: pair={pair}, direction={row.get('direction', 'BUY')}, leverage={row.get('leverage', 1)}, position=${row.get('position_size', 100)}")
            
            # Switch to Quick Trade main tab AND Execute Trade subtab
            st.session_state.active_crypto_tab = "⚡ Quick Trade"
            st.session_state.quick_trade_subtab = "⚡ Execute Trade"
            
            # Set flag to show success message after rerun
            st.session_state.show_setup_success = {'pair': pair, 'trade_type': trade_type}
            
            logger.info(f"🔄 CRYPTO TAB - About to rerun with quick_trade_subtab set to: {st.session_state.quick_trade_subtab}")
            st.rerun()
    
    # Render only the active tab content
    logger.info(f"🔍 Rendering crypto tab: {active_crypto_tab}")
    if active_crypto_tab == "📊 Dashboard":
        st.subheader("📊 Crypto Dashboard - Market Overview & Portfolio")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("💡 **24/7 Trading**: Crypto markets never close! Trade anytime, any day.")
        
        with col2:
            if st.button("🔄 Refresh Data", key="crypto_refresh"):
                st.rerun()
        
        # Get account balance
        try:
            balances = kraken_client.get_account_balance()
            total_usd = kraken_client.get_total_balance_usd()
            
            st.markdown("### 💰 Account Balance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Value (USD)", f"${total_usd:,.2f}")
            
            # Find USD and crypto balances
            usd_balance = next((b for b in balances if b.currency in ['USD', 'ZUSD']), None)
            crypto_holdings = [b for b in balances if b.balance > 0 and b.currency not in ['USD', 'ZUSD']]
            
            with col2:
                if usd_balance:
                    st.metric("Available USD", f"${usd_balance.available:,.2f}")
                else:
                    st.metric("Available USD", "$0.00")
            
            with col3:
                st.metric("Crypto Assets", len(crypto_holdings))

        except Exception as e:
            st.error(f"Error fetching account data: {e}")

        # Show crypto holdings
        st.markdown("### 📊 Your Crypto Holdings")
        # Use the corrected portfolio analysis from the AI monitor tab
        if 'portfolio_analysis' in st.session_state and st.session_state.portfolio_analysis:
            analysis = st.session_state.portfolio_analysis
            positions = analysis.get('positions', [])
            
            if positions:
                import pandas as pd
                df = pd.DataFrame(positions)
                df_display = pd.DataFrame({
                    'Asset': df['pair'],
                    'Balance': df['volume'].map('{:,.4f}'.format),
                    'Avg. Entry': df['entry'].map('${:,.2f}'.format),
                    'Current Price': df['current'].map('${:,.2f}'.format),
                    'Value (USD)': df['value'].map('${:,.2f}'.format),
                    'P&L': df['pnl'].map('${:,.2f}'.format),
                    'P&L %': df['pnl_pct'].map('{:+.2f}%'.format),
                    'Allocation': df['allocation'].map('{:.2f}%'.format)
                })
                st.dataframe(df_display, hide_index=True, use_container_width=True)
            else:
                st.info("No tradeable crypto positions found.")
        else:
            st.info("Run the analysis in the 'AI Position Monitor' tab to see your holdings here.")
        
        # Show top crypto prices
        st.markdown("### 📈 Top Cryptocurrencies")
        
        major_pairs = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'XRP/USD', 'ADA/USD']
        
        price_data = []
        for pair in major_pairs:
            try:
                ticker = kraken_client.get_ticker_data(pair)
                if ticker:
                    change_pct = ((ticker['last_price'] - ticker['low_24h']) / ticker['low_24h']) * 100
                    price_data.append({
                        'Pair': pair,
                        'Price': f"${ticker['last_price']:,.2f}",
                        '24h High': f"${ticker['high_24h']:,.2f}",
                        '24h Low': f"${ticker['low_24h']:,.2f}",
                        '24h Change': f"{change_pct:.2f}%",
                        'Volume': f"{ticker['volume_24h']:,.2f}"
                    })
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching {pair}: {e}")
        
        if price_data:
            st.dataframe(price_data, width="stretch")
        
        # Quick navigation section
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Start Daily Scan", type="primary", use_container_width=True):
                st.session_state.active_crypto_tab = "🔍 Daily Scanner"
                st.rerun()
        
        with col2:
            if st.button("⚡ Quick Trade", use_container_width=True):
                st.session_state.active_crypto_tab = "⚡ Quick Trade"
                st.rerun()
        
        with col3:
            if st.button("🤖 AI Monitor", use_container_width=True):
                st.session_state.active_crypto_tab = "🤖 AI Position Monitor"
                st.rerun()
        
        # Settings section
        st.markdown("---")
        with st.expander("⚙️ Trading Preferences"):
            st.markdown("""
            **Your Crypto Trading Settings**
            
            Configure your preferences for crypto trading:
            - Default position size
            - Risk management rules
            - Notification preferences
            - Auto-trading settings
            
            *(Settings integration coming soon)*
            """)

    
    elif active_crypto_tab == "🔍 Daily Scanner":
        logger.info("🏁 DAILY SCANNER TAB RENDERING")
        
        # Import and render Daily Scanner UI
        try:
            logger.info("Step 1: Importing display_daily_scanner...")
            from ui.daily_scanner_ui import display_daily_scanner
            logger.info("✅ display_daily_scanner imported successfully")
            
            logger.info("Step 2: Importing AICryptoTradeReviewer...")
            from services.ai_crypto_trade_reviewer import AICryptoTradeReviewer
            logger.info("✅ AICryptoTradeReviewer imported successfully")
            
            # Initialize AI trade reviewer if not already
            logger.info("Step 3: Checking ai_trade_reviewer in session state...")
            if 'ai_trade_reviewer' not in st.session_state:
                logger.info("Step 3a: Initializing AICryptoTradeReviewer...")
                # Get LLM analyzer and supabase client from session state
                llm_analyzer = st.session_state.get('llm_analyzer')
                supabase_client = st.session_state.get('supabase_client')
                
                st.session_state.ai_trade_reviewer = AICryptoTradeReviewer(
                    llm_analyzer=llm_analyzer,
                    active_monitors=st.session_state.get('active_trade_monitors', {}),
                    supabase_client=supabase_client
                )
                logger.info("✅ AICryptoTradeReviewer initialized")
            else:
                logger.info("✅ Using existing ai_trade_reviewer from session state")
            
            # Restore monitors from session state
            logger.info("Step 4: Checking active_trade_monitors in session state...")
            if 'active_trade_monitors' in st.session_state:
                logger.info(f"Step 4a: Restoring monitors (type: {type(st.session_state.active_trade_monitors)})...")
                st.session_state.ai_trade_reviewer.active_monitors = st.session_state.active_trade_monitors
                logger.info("✅ Monitors restored")
            else:
                logger.info("✅ No monitors to restore")
            
            logger.info("Step 5: Calling display_daily_scanner...")
            display_daily_scanner(
                kraken_client=kraken_client,
                crypto_config=crypto_config,
                ai_trade_reviewer=st.session_state.ai_trade_reviewer
            )
            logger.info("✅ display_daily_scanner completed successfully")
            
        except Exception as e:
            st.error(f"Error loading Daily Scanner: {e}")
            logger.error("Daily Scanner error: {}", str(e), exc_info=True)
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    elif active_crypto_tab == "🔍 OLD_SCANNER_REMOVED":  # REMOVED - Consolidated into Daily Scanner
        st.info("⚠️ This scanner has been consolidated into the Daily Scanner tab")
        st.markdown("Please use **🔍 Daily Scanner** for all scanning needs.")
    
    elif active_crypto_tab == "🔍 Crypto Scanner":
        logger.info("🏁 CRYPTO_TAB2 (Scanner) RENDERING")
        st.subheader("🔍 Advanced Crypto Opportunity Scanner")
        st.write("**AI-powered scanner** to find the best crypto trading opportunities with multiple analysis modes.")
        
        # Initialize scanners (cached to avoid repeated initialization)
        crypto_scanner = get_crypto_scanner(kraken_client, crypto_config)
        ai_crypto_scanner = get_ai_crypto_scanner(kraken_client, crypto_config)

        if crypto_scanner is None:
            st.error("Unable to load the core crypto scanner. Please check logs for details.")
            return

        ai_scanner_available = ai_crypto_scanner is not None
        typed_ai_scanner: Optional["AICryptoScanner"] = cast("Optional[AICryptoScanner]", ai_crypto_scanner)

        # Analysis mode selector
        analysis_mode = st.radio(
            "🔬 Analysis Mode:",
            options=["⚡ Quick Scan (Technical Only)", "🧠 AI-Enhanced (LLM Analysis)", "🔥 Buzzing Cryptos", "🌶️ Hottest Cryptos", "💥 Breakout Detection"],
            horizontal=False,
            help="Choose analysis mode:\n- Quick: Fast technical analysis\n- AI-Enhanced: Adds LLM reasoning\n- Buzzing: High volume surges\n- Hottest: Strong momentum\n- Breakout: Technical breakouts"
        )
        
        use_ai = "AI-Enhanced" in analysis_mode
        
        ai_mode_available = ai_scanner_available

        if use_ai:
            with st.expander("ℹ️ What does AI-Enhanced include?", expanded=False):
                st.markdown("""
                **AI-Enhanced Mode** adds intelligent analysis:
                - **🤖 LLM Reasoning**: Natural language explanations
                - **🎯 Risk Assessment**: Crypto-specific risk analysis
                - **📊 Market Cycle**: Where we are in the cycle (accumulation, markup, etc.)
                - **💬 Social Narrative**: Current market sentiment
                - **⭐ AI Rating**: 0-10 confidence score from AI
                
                This provides **deeper insights** beyond pure technical analysis.
                """)

            if not ai_mode_available:
                st.warning(
                    "⚠️ AI-Enhanced mode requires the AI crypto scanner, but it failed to initialize. "
                    "Please verify OpenRouter credentials and dependencies."
                )
        
        st.divider()
        
        # Scan configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "Buzzing" in analysis_mode or "Hottest" in analysis_mode or "Breakout" in analysis_mode:
                top_n = st.number_input("Top N Results", min_value=5, max_value=20, value=10)
            else:
                scan_strategy = st.selectbox(
                    "Strategy",
                    ['ALL', 'SCALP', 'MOMENTUM', 'SWING'],
                    help="Select trading strategy to scan for"
                )
        
        with col2:
            if "Buzzing" not in analysis_mode and "Hottest" not in analysis_mode and "Breakout" not in analysis_mode:
                top_n = st.number_input("Top N Results", min_value=3, max_value=20, value=10)
        
        with col3:
            if "Buzzing" not in analysis_mode and "Hottest" not in analysis_mode and "Breakout" not in analysis_mode:
                min_score = st.slider("Min Score", min_value=50, max_value=90, value=60)
        
        # Filters
        with st.expander("🎚️ Advanced Filters", expanded=False):
            fcol1, fcol2 = st.columns(2)
            
            with fcol1:
                st.markdown("**Volume Filters**")
                min_volume_ratio = st.slider("Min Volume Ratio (x avg)", 1.0, 5.0, 1.0, 0.5, key="crypto_vol_ratio")
                
                st.markdown("**Momentum Filters**")
                min_momentum = st.slider("Min 24h Change %", 0.0, 10.0, 0.0, 1.0, key="crypto_momentum")
            
            with fcol2:
                st.markdown("**Volatility Filters**")
                max_volatility = st.slider("Max Volatility %", 5.0, 20.0, 20.0, 1.0, key="crypto_vol")
                
                if use_ai:
                    st.markdown("**AI Confidence Filter**")
                    ai_confidence_filter = st.selectbox(
                        "Min AI Confidence",
                        ['ALL', 'MEDIUM', 'HIGH'],
                        help="Filter by AI confidence level"
                    )
        
        # Scan button
        button_label = {
            "⚡ Quick Scan (Technical Only)": "🚀 Quick Scan",
            "🧠 AI-Enhanced (LLM Analysis)": "🧠 AI-Enhanced Scan",
            "🔥 Buzzing Cryptos": "🔥 Find Buzzing Cryptos",
            "🌶️ Hottest Cryptos": "🌶️ Find Hottest Cryptos",
            "💥 Breakout Detection": "💥 Find Breakouts"
        }
        
        # Initialize session state for scan results
        if 'crypto_scan_results' not in st.session_state:
            st.session_state.crypto_scan_results = None
        
        disable_scan = (analysis_mode == "🧠 AI-Enhanced (LLM Analysis)") and not ai_mode_available
        if st.button(button_label[analysis_mode], key="crypto_scan", type="primary", disabled=disable_scan):
            logger.info(f"🔍 CRYPTO SCAN BUTTON CLICKED - Mode: {analysis_mode}")
            print(f"\n{'='*80}\n🔍 CRYPTO SCAN BUTTON CLICKED - Mode: {analysis_mode}\n{'='*80}\n", flush=True)
            with st.spinner(f"Scanning crypto markets with {analysis_mode}..."):
                try:
                    opportunities = []
                    
                    # Execute appropriate scan based on mode
                    if analysis_mode == "⚡ Quick Scan (Technical Only)":
                        opportunities = crypto_scanner.scan_opportunities(
                            strategy=scan_strategy,
                            top_n=top_n,
                            min_score=min_score
                        )
                    
                    elif analysis_mode == "🧠 AI-Enhanced (LLM Analysis)":
                        if not ai_mode_available or typed_ai_scanner is None:
                            st.error("AI Crypto Scanner unavailable. Please fix the initialization issue above and try again.")
                            opportunities = []
                        else:
                            ai_conf_filter = None if ai_confidence_filter == 'ALL' else ai_confidence_filter
                            opportunities = typed_ai_scanner.scan_with_ai_confidence(
                                strategy=scan_strategy,
                                top_n=top_n,
                                min_score=min_score,
                                min_ai_confidence=ai_conf_filter
                            )

                    elif analysis_mode == "🔥 Buzzing Cryptos":
                        if use_ai and ai_scanner_available and typed_ai_scanner is not None:
                            opportunities = typed_ai_scanner.get_buzzing_cryptos(top_n=top_n)
                        elif use_ai and not ai_scanner_available:
                            st.warning("AI buzzing scan unavailable; falling back to technical scanner.")
                            opportunities = crypto_scanner.scan_buzzing_cryptos(
                                top_n=top_n,
                                min_volume_ratio=min_volume_ratio
                            )
                        else:
                            opportunities = crypto_scanner.scan_buzzing_cryptos(
                                top_n=top_n,
                                min_volume_ratio=min_volume_ratio
                            )
                    
                    elif analysis_mode == "🌶️ Hottest Cryptos":
                        if use_ai and ai_scanner_available and typed_ai_scanner is not None:
                            opportunities = typed_ai_scanner.get_hottest_cryptos(top_n=top_n)
                        elif use_ai and not ai_scanner_available:
                            st.warning("AI hottest scan unavailable; falling back to technical scanner.")
                            opportunities = crypto_scanner.scan_hottest_cryptos(
                                top_n=top_n,
                                min_momentum=min_momentum
                            )
                        else:
                            opportunities = crypto_scanner.scan_hottest_cryptos(
                                top_n=top_n,
                                min_momentum=min_momentum
                            )
                    
                    elif analysis_mode == "💥 Breakout Detection":
                        opportunities = crypto_scanner.scan_breakout_cryptos(top_n=top_n)
                    
                    # Apply filters
                    if opportunities:
                        # Volume ratio filter
                        if min_volume_ratio > 1.0:
                            opportunities = [opp for opp in opportunities if opp.volume_ratio >= min_volume_ratio]
                        
                        # Momentum filter
                        if min_momentum > 0:
                            opportunities = [opp for opp in opportunities if abs(opp.change_pct_24h) >= min_momentum]
                        
                        # Volatility filter
                        if max_volatility < 20.0:
                            opportunities = [opp for opp in opportunities if opp.volatility_24h <= max_volatility]
                    
                    # Store results in session state
                    st.session_state.crypto_scan_results = opportunities
                    st.session_state.just_scanned_crypto = True
                    # Keep user on Scanner tab after scan
                    st.session_state.active_crypto_tab = "🔍 Crypto Scanner"
                    logger.info(f"📊 Scan complete - Found {len(opportunities)} opportunities")
                    
                except Exception as e:
                    st.error(f"Scanner error: {e}")
                    logger.error("Crypto scanner error: {}", str(e), exc_info=True)
        
        # Display results from session state (persists across button clicks)
        if st.session_state.crypto_scan_results is not None:
            opportunities = st.session_state.crypto_scan_results
            
            if opportunities:
                logger.info(f"🎯 Rendering {len(opportunities)} crypto cards...")
                st.success(f"✅ Found {len(opportunities)} crypto opportunities!")
                
                # Summary metrics
                scol1, scol2, scol3, scol4 = st.columns(4)
                
                with scol1:
                    avg_score = sum(opp.score for opp in opportunities) / len(opportunities)
                    st.metric("Avg Score", f"{avg_score:.1f}/100")
                
                with scol2:
                    avg_vol_ratio = sum(opp.volume_ratio for opp in opportunities) / len(opportunities)
                    st.metric("Avg Volume Ratio", f"{avg_vol_ratio:.2f}x")
                
                with scol3:
                    high_conf = sum(1 for opp in opportunities if opp.confidence == 'HIGH')
                    st.metric("High Confidence", f"{high_conf}/{len(opportunities)}")
                
                with scol4:
                    if use_ai and hasattr(opportunities[0], 'ai_rating'):
                        avg_ai_rating = sum(opp.ai_rating for opp in opportunities) / len(opportunities)
                        st.metric("Avg AI Rating", f"{avg_ai_rating:.1f}/10")
                    else:
                        st.metric("Analysis Mode", "Technical" if not use_ai else "AI")
                
                st.divider()
                
                # Display each opportunity
                logger.info(f"🔁 Starting loop to render {len(opportunities)} expanders")
                for i, opp in enumerate(opportunities, 1):
                    logger.info(f"📋 Rendering card {i}/{len(opportunities)}: {opp.symbol}")
                    # Build expander title
                    title_parts = [
                        f"#{i}",
                        f"{opp.symbol}",
                        f"Score: {opp.score:.1f}",
                        f"{opp.confidence} Conf",
                        f"{opp.risk_level} Risk"
                    ]
                    
                    if use_ai and hasattr(opp, 'ai_rating'):
                        title_parts.append(f"AI: {opp.ai_rating:.1f}/10")
                    
                    expander_title = " | ".join(title_parts)
                    logger.info(f"🔽 Creating expander for {opp.symbol}: {expander_title}")
                    
                    with st.expander(expander_title, expanded=(i <= 3)):
                        logger.info(f"📂 Inside expander for {opp.symbol}, rendering button...")
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"${opp.current_price:,.2f}")
                            st.metric("Strategy", opp.strategy.upper())
                        
                        with col2:
                            direction = "🟢" if opp.change_pct_24h > 0 else "🔴"
                            st.metric("24h Change", f"{direction} {opp.change_pct_24h:.2f}%")
                            st.metric("Volatility", f"{opp.volatility_24h:.2f}%")
                        
                        with col3:
                            st.metric("Volume 24h", f"${opp.volume_24h:,.0f}")
                            st.metric("Vol Ratio", f"{opp.volume_ratio:.2f}x")
                        
                        with col4:
                            st.metric("Confidence", opp.confidence)
                            st.metric("Risk Level", opp.risk_level)
                        
                        st.divider()
                        
                        # Analysis
                        st.markdown("**📊 Technical Analysis:**")
                        st.info(opp.reason)
                        
                        # AI Analysis (if available)
                        if use_ai and hasattr(opp, 'ai_reasoning'):
                            st.divider()
                            
                            acol1, acol2 = st.columns(2)
                            
                            with acol1:
                                st.markdown("**🤖 AI Analysis:**")
                                st.markdown(f"**AI Confidence:** {opp.ai_confidence}")
                                st.markdown(f"**AI Rating:** {opp.ai_rating:.1f}/10")
                                st.markdown(f"**Reasoning:** {opp.ai_reasoning}")
                            
                            with acol2:
                                st.markdown("**⚠️ AI Risk Assessment:**")
                                st.warning(opp.ai_risks)
                                
                                if hasattr(opp, 'market_cycle_phase'):
                                    st.markdown(f"**Market Cycle:** {opp.market_cycle_phase}")
                                
                                if hasattr(opp, 'social_narrative') and opp.social_narrative:
                                    st.markdown(f"**Social Narrative:** {opp.social_narrative}")
                        
                        st.divider()
                        
                        # Action buttons
                        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                        
                        with bcol1:
                            button_key = f"save_wl_{i}"
                            logger.info(f"🔘 Creating button widget for {opp.symbol} with key={button_key}")
                            print(f"🔘 Creating button widget for {opp.symbol} with key={button_key}", flush=True)
                            button_clicked = st.button(f"⭐ Save to Watchlist", key=button_key)
                            logger.info(f"🎯 Button state for {opp.symbol}: clicked={button_clicked}")
                            print(f"🎯 Button state for {opp.symbol}: clicked={button_clicked}", flush=True)
                            if button_clicked:
                                logger.info(f"🔵 WATCHLIST BUTTON CLICKED for {opp.symbol}")
                                print(f"\n{'='*80}\n🔵🔵🔵 WATCHLIST BUTTON CLICKED for {opp.symbol} 🔵🔵🔵\n{'='*80}\n", flush=True)
                                try:
                                    with st.spinner(f"Saving {opp.symbol}..."):
                                        # Create opportunity data dict for watchlist
                                        opp_data = {
                                            'symbol': opp.symbol,
                                            'current_price': opp.current_price,
                                            'change_pct_24h': opp.change_pct_24h,
                                            'volume_24h': opp.volume_24h,
                                            'volume_ratio': opp.volume_ratio,
                                            'volatility_24h': opp.volatility_24h,
                                            'rsi': opp.rsi if hasattr(opp, 'rsi') else None,
                                            'momentum_score': opp.momentum_score if hasattr(opp, 'momentum_score') else None,
                                            'technical_score': opp.technical_score if hasattr(opp, 'technical_score') else None,
                                            'score': opp.score,
                                            'confidence': opp.confidence,
                                            'risk_level': opp.risk_level,
                                            'strategy': opp.strategy,
                                            'reason': opp.reason
                                        }
                                        logger.info(f"📦 Prepared data dict for {opp.symbol}: keys={list(opp_data.keys())}")
                                        logger.info(f"📊 Data values: price=${opp_data['current_price']}, confidence={opp_data['confidence']}, strategy={opp_data['strategy']}")
                                        
                                        # Check if crypto already exists in watchlist
                                        existing = crypto_wl_manager.get_crypto(opp.symbol)
                                        if existing:
                                            st.warning(f"⚠️ {opp.symbol} is already in your watchlist!")
                                            logger.info(f"⚠️ {opp.symbol} already exists in watchlist")
                                        else:
                                            logger.info(f"🔄 Calling crypto_wl_manager.add_crypto({opp.symbol}, opp_data)")
                                            success = crypto_wl_manager.add_crypto(opp.symbol, opp_data)
                                            logger.info(f"✨ add_crypto returned: {success}")
                                            
                                            if success:
                                                st.success(f"✅ Added {opp.symbol} to watchlist!")
                                                logger.info(f"✅ SUCCESS: {opp.symbol} added to watchlist")
                                            else:
                                                st.error(f"❌ Failed to add {opp.symbol} to watchlist")
                                                logger.warning(f"⚠️ FAILED: {opp.symbol} not added (returned False)")
                                except Exception as e:
                                    error_msg = f"Error saving {opp.symbol} to watchlist: {e}"
                                    st.error(f"❌ {error_msg}")
                                    logger.error("❌ EXCEPTION in watchlist save: {}", str(error_msg), exc_info=True)
                        
                        with bcol2:
                            if st.button(f"📊 Generate Signal", key=f"gen_signal_{i}"):
                                st.session_state.crypto_signal_symbol = opp.symbol
                                st.info(f"Navigate to Signal Generator tab to see {opp.symbol} signals!")
                        
                        with bcol3:
                            if st.button(f"💹 View Chart", key=f"view_chart_{i}"):
                                st.info(f"Chart viewing for {opp.symbol} - Coming soon!")
                        
                        with bcol4:
                            if st.button(f"⚡ Quick Trade", key=f"quick_trade_{i}"):
                                # Copy full trade setup to Quick Trade tab
                                from services.crypto_strategy_config import get_strategy_config
                                
                                st.session_state.crypto_quick_trade_pair = opp.symbol
                                
                                # Get standardized strategy configuration
                                strategy_key = opp.strategy.lower() if hasattr(opp, 'strategy') else 'momentum'
                                strategy_config = get_strategy_config(strategy_key)
                                
                                st.session_state.crypto_quick_stop_pct = strategy_config['stop_pct']
                                st.session_state.crypto_quick_target_pct = strategy_config['target_pct']
                                
                                # Set recommended direction based on analysis
                                # Most scanner results are BUY opportunities (long entries)
                                st.session_state.crypto_quick_direction = "BUY"
                                
                                # Store additional analysis data for AI review
                                st.session_state.crypto_scanner_opportunity = {
                                    'symbol': opp.symbol,
                                    'score': opp.score,
                                    'confidence': opp.confidence,
                                    'risk_level': opp.risk_level,
                                    'reason': opp.reason,
                                    'strategy': opp.strategy,
                                    'current_price': opp.current_price,
                                    'volume_ratio': opp.volume_ratio,
                                    'volatility': opp.volatility_24h,
                                    'change_24h': opp.change_pct_24h
                                }
                                
                                # Add AI analysis if available
                                if use_ai and hasattr(opp, 'ai_reasoning'):
                                    st.session_state.crypto_scanner_opportunity['ai_confidence'] = opp.ai_confidence
                                    st.session_state.crypto_scanner_opportunity['ai_rating'] = opp.ai_rating
                                    st.session_state.crypto_scanner_opportunity['ai_reasoning'] = opp.ai_reasoning
                                    st.session_state.crypto_scanner_opportunity['ai_risks'] = opp.ai_risks
                                
                                st.session_state.active_crypto_tab = "⚡ Quick Trade"
                                st.success(f"✅ {opp.symbol} setup copied to Quick Trade tab!")
                                logger.info(f"🎯 Quick Trade setup copied: {opp.symbol} | {opp.strategy.upper()} | Stop: {strategy_config['stop_pct']}% | Target: {strategy_config['target_pct']}%")
                                st.rerun()

            else:
                st.warning("No opportunities found matching your criteria. Try adjusting filters.")
        
        # Show scanner help
        with st.expander("❓ Scanner Help", expanded=False):
            st.markdown("""
            ### 🔍 Scanner Modes Explained
            
            **⚡ Quick Scan (Technical Only)**
            - Fast technical analysis based on price, volume, momentum
            - No AI overhead, great for quick checks
            - Use when you want fast results
            
            **🧠 AI-Enhanced (LLM Analysis)**
            - Adds LLM reasoning and risk assessment
            - Crypto-specific insights (24/7 market, social sentiment)
            - Best for high-confidence trades
            - Requires OpenRouter API key
            
            **🔥 Buzzing Cryptos**
            - Focus on volume surges (2x+ average)
            - Indicates strong interest and liquidity
            - Good for momentum plays
            
            **🌶️ Hottest Cryptos**
            - Focus on strong price momentum (3%+ moves)
            - Identifies trending assets
            - Good for breakout continuation
            
            **💥 Breakout Detection**
            - Technical breakouts (price > EMAs with volume)
            - High probability setups
            - Good entry points for new trends
            
            ### 🎯 Strategy Types
            
            - **SCALP**: Quick 1-3% moves, high frequency
            - **MOMENTUM**: Ride 5-10% trends with volume
            - **SWING**: Hold 1-7 days for larger moves
            - **ALL**: Mixed approach based on conditions
            
            ### ⚙️ Filter Tips
            
            - **Volume Ratio**: Higher = more interest, better liquidity
            - **Momentum**: Higher = stronger trend, more volatile
            - **Volatility**: Lower = more predictable, safer
            - **AI Confidence**: HIGH = AI agrees with technical analysis
            """)
    
    elif active_crypto_tab == "💰 OLD_PENNY_REMOVED":  # REMOVED - Integrated into Daily Scanner
        st.info("⚠️ Penny crypto scanning has been integrated into the Daily Scanner")
        st.markdown("Please use **🔍 Daily Scanner** → Select source: **💰 Penny Cryptos (<$1)**")
    
    elif active_crypto_tab == "💰 Penny Cryptos (<$1)":
        logger.info("🏁 CRYPTO_TAB3 (Penny Cryptos) RENDERING")
        st.subheader("💰 Penny Crypto Scanner - Monster Runners Under $1")
        st.write("**Find sub-$1 cryptocurrencies with extreme runner potential** - including sub-penny coins (0.0000000+)")
        
        # Initialize penny crypto scanner (cached to avoid re-initialization)
        penny_crypto_scanner = get_penny_crypto_scanner(kraken_client, crypto_config)

        if penny_crypto_scanner is None:
            st.error("Unable to load Penny Crypto Scanner. Please review logs for dependency issues.")
            return
        
        # Show scanner coverage
        scanner_stats = penny_crypto_scanner.get_scanner_stats()
        if scanner_stats:
            with st.expander("📊 Scanner Coverage & Method", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Coins Scanned", scanner_stats['total_coins_scanned'])
                    st.markdown("**Categories Covered:**")
                    for cat, count in scanner_stats['categories'].items():
                        st.caption(f"• {cat.replace('_', ' ').title()}: {count}")
                with col2:
                    st.markdown("**Scan Method:**")
                    st.caption(scanner_stats['scan_method'])
                    st.markdown("**Update Frequency:**")
                    st.caption(scanner_stats['update_frequency'])
                    st.warning(f"**Note:** {scanner_stats['note']}")
        
        # Scan mode selector
        # Preserve active tab state when radio changes
        if 'active_crypto_tab' not in st.session_state:
            st.session_state.active_crypto_tab = "💰 Penny Cryptos (<$1)"
        
        scan_mode = st.radio(
            "🎯 Scan Mode:",
            options=["💰 All Penny Cryptos (<$1)", "🔥 Sub-Penny Cryptos (<$0.01)"],
            horizontal=False,
            help="Choose scan mode:\n- All Penny: Cryptos under $1\n- Sub-Penny: Extreme runners under $0.01"
        )
        
        # Preserve tab state after radio change
        st.session_state.active_crypto_tab = "💰 Penny Cryptos (<$1)"
        
        st.divider()
        
        # Scan configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "Sub-Penny" in scan_mode:
                max_price = st.slider("Max Price", 0.001, 0.01, 0.01, 0.001, key="sub_penny_price")
            else:
                max_price = st.slider("Max Price", 0.01, 1.0, 1.0, 0.1, key="penny_price")
        
        with col2:
            top_n = st.number_input("Top N Results", min_value=5, max_value=30, value=15)
        
        with col3:
            min_runner_score = st.slider("Min Runner Score", 40, 100, 60, 5, key="runner_score")
        
        # Advanced filters
        with st.expander("🎚️ Advanced Filters", expanded=False):
            fcol1, fcol2 = st.columns(2)
            
            with fcol1:
                st.markdown("**Momentum Filters**")
                min_volume_ratio = st.slider("Min Volume Ratio (x avg)", 1.0, 5.0, 1.5, 0.5, key="penny_vol_ratio")
                min_volatility = st.slider("Min Volatility %", 1.0, 30.0, 5.0, 1.0, key="penny_vol_min")
            
            with fcol2:
                st.markdown("**Price Action**")
                min_momentum = st.slider("Min 24h Change %", 0.0, 50.0, 5.0, 5.0, key="penny_momentum")
                max_volatility = st.slider("Max Volatility %", 10.0, 50.0, 50.0, 5.0, key="penny_vol_max")
        
        # Initialize session state for penny scan results
        if 'penny_crypto_scan_results' not in st.session_state:
            st.session_state.penny_crypto_scan_results = None
        if 'trending_runners_results' not in st.session_state:
            st.session_state.trending_runners_results = None
        if 'sub_penny_discovery_results' not in st.session_state:
            st.session_state.sub_penny_discovery_results = None
        
        # Scan buttons
        col_scan1, col_scan2, col_scan3 = st.columns(3)
        
        with col_scan1:
            scan_penny = st.button("🚀 Scan Watchlist for Runners", key="penny_scan", type="primary")
        
        with col_scan2:
            scan_trending = st.button("🔥 Scan CoinGecko Trending", key="trending_scan", type="secondary")
        
        with col_scan3:
            scan_sub_penny = st.button("🔬 Discover Sub-Penny (<$0.01)", key="sub_penny_scan", type="secondary")
        
        # Handle sub-penny discovery
        if scan_sub_penny:
            logger.info("🔬 SUB-PENNY DISCOVERY BUTTON CLICKED")
            with st.spinner("🔬 Discovering ultra-low coins from CoinGecko (this may take 30-60s)..."):
                try:
                    import asyncio
                    # Get cached discovery engine
                    discovery = get_sub_penny_discovery()

                    if discovery is None:
                        raise RuntimeError("SubPennyDiscovery engine unavailable")

                    # Discover sub-penny runners
                    sub_penny_coins = asyncio.run(discovery.discover_sub_penny_runners(
                        max_price=0.01,
                        min_market_cap=0,
                        max_market_cap=10_000_000,
                        top_n=20,
                        sort_by="runner_potential"
                    ))
                    
                    st.session_state.sub_penny_discovery_results = sub_penny_coins
                    logger.info(f"📊 Sub-penny discovery complete - Found {len(sub_penny_coins)} opportunities")
                    
                except Exception as e:
                    st.error(f"Sub-penny discovery error: {e}")
                    logger.error("Sub-penny discovery error: {}", str(e), exc_info=True)
        
        # Handle trending runners scan
        if scan_trending:
            logger.info("🔍 TRENDING RUNNERS SCAN BUTTON CLICKED")
            with st.spinner("🔍 Fetching CoinGecko trending + sentiment analysis..."):
                try:
                    import asyncio
                    
                    # Get trending runners
                    trending_results = asyncio.run(
                        penny_crypto_scanner.scan_trending_runners(top_n=10)
                    )
                    
                    st.session_state.trending_runners_results = trending_results
                    logger.info(f"📊 Trending scan complete - Found {len(trending_results)} opportunities")
                    
                except Exception as e:
                    st.error(f"Trending scanner error: {e}")
                    logger.error("Trending crypto scanner error: {}", str(e), exc_info=True)
        
        # Scan button
        if scan_penny:
            logger.info(f"🔍 PENNY CRYPTO SCAN BUTTON CLICKED - Mode: {scan_mode}")
            with st.spinner(f"Scanning for penny cryptos with {scan_mode}..."):
                try:
                    if "Sub-Penny" in scan_mode:
                        opportunities = penny_crypto_scanner.scan_sub_penny_cryptos(
                            max_price=max_price,
                            top_n=top_n
                        )
                    else:
                        opportunities = penny_crypto_scanner.scan_penny_cryptos(
                            max_price=max_price,
                            top_n=top_n,
                            min_runner_score=min_runner_score
                        )
                    
                    # Apply advanced filters
                    if opportunities:
                        # Volume ratio filter
                        opportunities = [opp for opp in opportunities if opp.volume_ratio >= min_volume_ratio]
                        
                        # Momentum filter
                        opportunities = [opp for opp in opportunities if abs(opp.change_pct_24h) >= min_momentum]
                        
                        # Volatility filters
                        opportunities = [opp for opp in opportunities if opp.volatility_24h >= min_volatility and opp.volatility_24h <= max_volatility]
                    
                    # Store results
                    st.session_state.penny_crypto_scan_results = opportunities
                    logger.info(f"📊 Penny scan complete - Found {len(opportunities)} opportunities")
                    
                except Exception as e:
                    st.error(f"Scanner error: {e}")
                    logger.error("Penny crypto scanner error: {}", str(e), exc_info=True)
        
        # Display trending runners results
        if st.session_state.trending_runners_results is not None:
            trending_results = st.session_state.trending_runners_results
            if trending_results:
                logger.info(f"🎯 Rendering {len(trending_results)} trending runner cards...")
                st.success(f"✅ Found {len(trending_results)} trending monster runners from CoinGecko!")
                
                # Summary metrics
                tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                
                with tcol1:
                    avg_runner_score = sum(r['overall_runner_score'] for r in trending_results) / len(trending_results)
                    st.metric("Avg Runner Score", f"{avg_runner_score:.1f}/100")
                
                with tcol2:
                    bullish_count = sum(1 for r in trending_results if r['social_sentiment'].get('overall_sentiment') == 'BULLISH')
                    st.metric("Bullish Sentiment", f"{bullish_count}/{len(trending_results)}")
                
                with tcol3:
                    avg_trending_rank = sum(r['trending_score'] for r in trending_results) / len(trending_results)
                    st.metric("Avg Trending Rank", f"#{avg_trending_rank:.1f}")
                
                with tcol4:
                    st.metric("Data Source", "CoinGecko + Reddit + Twitter")
                
                st.divider()
                
                # Display each trending runner
                for i, result in enumerate(trending_results, 1):
                    symbol = result['symbol']
                    name = result['name']
                    trending_rank = result['trending_score']
                    runner_score = result['overall_runner_score']
                    social = result['social_sentiment']
                    runner_potential = result['runner_potential']
                    
                    # Build expander title
                    title_parts = [
                        f"#{i}",
                        f"🔥 {symbol}",
                        f"({name})",
                        f"Trending: #{trending_rank}",
                        f"Runner: {runner_score:.1f}",
                        f"{runner_potential['confidence']} Conf"
                    ]
                    
                    expander_title = " | ".join(title_parts)
                    
                    with st.expander(expander_title, expanded=(i <= 3)):
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Price (USD)", f"${result['price_usd']:.8f}")
                            st.metric("Market Cap Rank", result['market_cap_rank'] or "N/A")
                        
                        with col2:
                            st.metric("Trending Rank", f"#{trending_rank}")
                            st.metric("Trending Sentiment", result['trending_sentiment'])
                        
                        with col3:
                            st.metric("Overall Sentiment", social['overall_sentiment'])
                            st.metric("Sentiment Score", f"{social['overall_sentiment_score']:.2f}")
                        
                        with col4:
                            st.metric("Reddit Mentions", social['reddit_mentions'])
                            st.metric("Twitter Mentions", social['twitter_mentions'])
                        
                        st.divider()
                        
                        # Runner Potential Analysis
                        st.markdown("### 🚀 Monster Runner Potential")
                        st.metric("Runner Score", f"{runner_score:.1f}/100")
                        st.markdown(f"**Confidence:** {runner_potential['confidence']}")
                        st.markdown(f"**Recommendation:** {runner_potential['recommendation']}")
                        
                        # Signals
                        st.markdown("**Key Signals:**")
                        for signal in runner_potential['signals']:
                            st.caption(f"• {signal}")
                        
                        st.divider()
                        
                        # Technical data (if available)
                        if result['technical']:
                            tech = result['technical']
                            st.markdown("### 📊 Technical Analysis")
                            
                            tcol1, tcol2, tcol3 = st.columns(3)
                            
                            with tcol1:
                                st.metric("24h Change", f"{tech['change_24h']:+.2f}%")
                                st.metric("7d Change", f"{tech['change_7d']:+.2f}%")
                            
                            with tcol2:
                                st.metric("Volume 24h", f"${tech['volume_24h']:,.0f}")
                                st.metric("Vol Ratio", f"{tech['volume_ratio']:.2f}x")
                            
                            with tcol3:
                                st.metric("Volatility", f"{tech['volatility']:.2f}%")
                                st.metric("RSI", f"{tech['rsi']:.0f}")
                            
                            st.divider()
                        
                        # Action buttons
                        bcol1, bcol2, bcol3 = st.columns(3)
                        
                        with bcol1:
                            if st.button(f"⭐ Save to Watchlist", key=f"save_trending_{i}"):
                                try:
                                    with st.spinner(f"Saving {symbol}..."):
                                        opp_data = {
                                            'symbol': f"{symbol}/USD",
                                            'current_price': result['price_usd'],
                                            'change_pct_24h': result['technical']['change_24h'] if result['technical'] else 0,
                                            'volume_24h': result['technical']['volume_24h'] if result['technical'] else 0,
                                            'volume_ratio': result['technical']['volume_ratio'] if result['technical'] else 0,
                                            'volatility_24h': result['technical']['volatility'] if result['technical'] else 0,
                                            'rsi': result['technical']['rsi'] if result['technical'] else 0,
                                            'momentum_score': result['technical']['momentum_score'] if result['technical'] else 0,
                                            'score': runner_score,
                                            'confidence': runner_potential['confidence'],
                                            'risk_level': result['technical']['risk_level'] if result['technical'] else 'MEDIUM',
                                            'strategy': 'trending_runner',
                                            'reason': f"CoinGecko Trending #{trending_rank} | {' | '.join(runner_potential['signals'][:2])}"
                                        }
                                        
                                        # Check if crypto already exists in watchlist
                                        existing = crypto_wl_manager.get_crypto(f"{symbol}/USD")
                                        if existing:
                                            st.warning(f"⚠️ {symbol}/USD is already in your watchlist!")
                                        else:
                                            success = crypto_wl_manager.add_crypto(f"{symbol}/USD", opp_data)
                                            
                                            if success:
                                                st.success(f"✅ Added {symbol} to watchlist!")
                                            else:
                                                st.error(f"❌ Failed to add {symbol} to watchlist")
                                except Exception as e:
                                    st.error(f"❌ Error saving {symbol}: {e}")
                                    logger.error("Error saving trending runner to watchlist: {}", str(e), exc_info=True)
                        
                        with bcol2:
                            if st.button(f"📊 Generate Signal", key=f"gen_trending_signal_{i}"):
                                st.session_state.crypto_signal_symbol = f"{symbol}/USD"
                                st.info(f"Navigate to Signal Generator tab to see {symbol} signals!")
                        
                        with bcol3:
                            if st.button(f"⚡ Quick Trade", key=f"quick_trending_trade_{i}"):
                                st.session_state.crypto_quick_trade_pair = f"{symbol}/USD"
                                st.info(f"Quick trade setup for {symbol}")
            
            else:
                st.warning("No trending runners found. Try again later or check your CoinGecko API key.")
        
        # Display sub-penny discovery results
        if st.session_state.sub_penny_discovery_results is not None:
            sub_penny_coins = st.session_state.sub_penny_discovery_results
            if sub_penny_coins:
                logger.info(f"🎯 Rendering {len(sub_penny_coins)} sub-penny coins...")
                st.success(f"✅ Discovered {len(sub_penny_coins)} ultra-low coins under $0.01!")
                
                # Summary metrics
                scol1, scol2, scol3, scol4 = st.columns(4)
                
                with scol1:
                    avg_runner_score = sum(c.runner_potential_score for c in sub_penny_coins) / len(sub_penny_coins)
                    st.metric("Avg Runner Score", f"{avg_runner_score:.1f}/100")
                
                with scol2:
                    avg_price = sum(c.price_usd for c in sub_penny_coins) / len(sub_penny_coins)
                    st.metric("Avg Price", f"${avg_price:.8f}")
                
                with scol3:
                    avg_decimals = sum(c.price_decimals for c in sub_penny_coins) / len(sub_penny_coins)
                    st.metric("Avg Decimals", f"{avg_decimals:.1f}")
                
                with scol4:
                    total_market_cap = sum(c.market_cap for c in sub_penny_coins)
                    st.metric("Total Market Cap", f"${total_market_cap:,.0f}")
                
                st.divider()
                
                # Display each sub-penny coin
                for i, coin in enumerate(sub_penny_coins, 1):
                    # Build expander title
                    title_parts = [
                        f"#{i}",
                        f"🔬 {coin.symbol}",
                        f"${coin.price_usd:.{min(coin.price_decimals, 8)}f}",
                        f"Runner: {coin.runner_potential_score:.1f}",
                        f"MC: ${coin.market_cap:,.0f}"
                    ]
                    
                    expander_title = " | ".join(title_parts)
                    
                    with st.expander(expander_title, expanded=(i <= 3)):
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Price (USD)", f"${coin.price_usd:.{min(coin.price_decimals, 8)}f}")
                            st.metric("Decimals", f"{coin.price_decimals}")
                        
                        with col2:
                            st.metric("24h Change", f"{coin.change_24h:+.2f}%")
                            st.metric("7d Change", f"{coin.change_7d:+.2f}%")
                        
                        with col3:
                            st.metric("Market Cap", f"${coin.market_cap:,.0f}")
                            st.metric("Market Cap Rank", coin.market_cap_rank or "N/A")
                        
                        with col4:
                            st.metric("Volume 24h", f"${coin.volume_24h:,.0f}")
                            st.metric("Market Cap Change", f"{coin.market_cap_change_24h:+.2f}%")
                        
                        st.divider()
                        
                        # Recovery Potential
                        st.markdown("### 🚀 Recovery Potential")
                        if coin.ath > 0:
                            recovery = (coin.ath - coin.price_usd) / coin.price_usd * 100
                            st.metric("ATH Recovery Potential", f"+{recovery:.0f}%")
                            st.caption(f"ATH: ${coin.ath:.8f} | ATL: ${coin.atl:.8f}")
                        
                        st.divider()
                        
                        # Supply Analysis
                        st.markdown("### 📊 Supply Analysis")
                        scol1, scol2, scol3 = st.columns(3)
                        
                        with scol1:
                            st.metric("Circulating Supply", f"{coin.circulating_supply:,.0f}")
                        
                        with scol2:
                            st.metric("Total Supply", f"{coin.total_supply:,.0f}")
                        
                        with scol3:
                            if coin.total_supply > 0:
                                circ_pct = (coin.circulating_supply / coin.total_supply) * 100
                                st.metric("Circulating %", f"{circ_pct:.1f}%")
                        
                        st.divider()
                        
                        # Discovery Reason
                        st.markdown("### 💡 Why This Coin?")
                        st.info(coin.discovery_reason)
                        
                        st.divider()
                        
                        # Action buttons
                        bcol1, bcol2 = st.columns(2)
                        
                        with bcol1:
                            if st.button(f"⭐ Save to Watchlist", key=f"save_sub_penny_{i}"):
                                try:
                                    with st.spinner(f"Saving {coin.symbol}..."):
                                        opp_data = {
                                            'symbol': f"{coin.symbol}/USD",
                                            'current_price': coin.price_usd,
                                            'change_pct_24h': coin.change_24h,
                                            'volume_24h': coin.volume_24h,
                                            'volatility_24h': 0,  # Not available from CoinGecko
                                            'score': coin.runner_potential_score,
                                            'confidence': 'MEDIUM',
                                            'risk_level': 'HIGH',
                                            'strategy': 'sub_penny_runner',
                                            'reason': coin.discovery_reason
                                        }
                                        
                                        # Check if crypto already exists in watchlist
                                        existing = crypto_wl_manager.get_crypto(f"{coin.symbol}/USD")
                                        if existing:
                                            st.warning(f"⚠️ {coin.symbol}/USD is already in your watchlist!")
                                        else:
                                            success = crypto_wl_manager.add_crypto(f"{coin.symbol}/USD", opp_data)
                                            
                                            if success:
                                                st.success(f"✅ Added {coin.symbol} to watchlist!")
                                            else:
                                                st.error(f"❌ Failed to add {coin.symbol} to watchlist")
                                except Exception as e:
                                    st.error(f"❌ Error saving {coin.symbol}: {e}")
                                    logger.error("Error saving sub-penny to watchlist: {}", str(e), exc_info=True)
                        
                        with bcol2:
                            st.info(f"💡 Tip: Research {coin.symbol} on CoinGecko before trading")
            
            else:
                st.warning("No sub-penny coins found. Try adjusting filters or check back later.")
        
        # Display penny scan results
        if st.session_state.penny_crypto_scan_results is not None:
            opportunities = st.session_state.penny_crypto_scan_results
            if opportunities:
                logger.info(f"🎯 Rendering {len(opportunities)} penny crypto cards...")
                st.success(f"✅ Found {len(opportunities)} monster runner opportunities!")
                
                # Summary metrics
                scol1, scol2, scol3, scol4 = st.columns(4)
                
                with scol1:
                    avg_runner_score = sum(opp.runner_potential_score for opp in opportunities) / len(opportunities)
                    st.metric("Avg Runner Score", f"{avg_runner_score:.1f}/100")
                
                with scol2:
                    avg_vol_ratio = sum(opp.volume_ratio for opp in opportunities) / len(opportunities)
                    st.metric("Avg Volume Ratio", f"{avg_vol_ratio:.2f}x")
                
                with scol3:
                    high_conf = sum(1 for opp in opportunities if opp.confidence == 'HIGH')
                    st.metric("High Confidence", f"{high_conf}/{len(opportunities)}")
                
                with scol4:
                    avg_volatility = sum(opp.volatility_24h for opp in opportunities) / len(opportunities)
                    st.metric("Avg Volatility", f"{avg_volatility:.1f}%")
                
                st.divider()
                
                # Display each opportunity
                for i, opp in enumerate(opportunities, 1):
                    # Build expander title with price precision indicator
                    price_indicator = "🔬" if opp.price_decimals > 6 else "💰"
                    title_parts = [
                        f"#{i}",
                        f"{price_indicator} {opp.symbol}",
                        f"${opp.current_price:.{min(opp.price_decimals, 8)}f}",
                        f"Runner: {opp.runner_potential_score:.1f}",
                        f"{opp.confidence} Conf",
                        f"{opp.risk_level} Risk"
                    ]
                    
                    expander_title = " | ".join(title_parts)
                    
                    with st.expander(expander_title, expanded=(i <= 3)):
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Entry Price", f"${opp.current_price:.{min(opp.price_decimals, 8)}f}")
                            st.metric("Decimals", f"{opp.price_decimals}")
                        
                        with col2:
                            direction = "🟢" if opp.change_pct_24h > 0 else "🔴"
                            st.metric("24h Change", f"{direction} {opp.change_pct_24h:+.2f}%")
                            st.metric("7d Change", f"{opp.change_pct_7d:+.2f}%")
                        
                        with col3:
                            st.metric("Volume 24h", f"${opp.volume_24h:,.0f}")
                            st.metric("Vol Ratio", f"{opp.volume_ratio:.2f}x")
                        
                        with col4:
                            st.metric("Volatility", f"{opp.volatility_24h:.2f}%")
                            st.metric("RSI", f"{opp.rsi:.0f}")
                        
                        st.divider()
                        
                        # Monster Runner Targets
                        st.markdown("### 🎯 Monster Runner Targets")
                        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                        
                        with tcol1:
                            gain_1 = ((opp.target_1 - opp.entry_price) / opp.entry_price) * 100
                            st.metric("Target 1 (50%)", f"${opp.target_1:.{min(opp.price_decimals, 8)}f}", f"+{gain_1:.1f}%")
                        
                        with tcol2:
                            gain_2 = ((opp.target_2 - opp.entry_price) / opp.entry_price) * 100
                            st.metric("Target 2 (100%)", f"${opp.target_2:.{min(opp.price_decimals, 8)}f}", f"+{gain_2:.1f}%")
                        
                        with tcol3:
                            gain_3 = ((opp.target_3 - opp.entry_price) / opp.entry_price) * 100
                            st.metric("Target 3 (200%+)", f"${opp.target_3:.{min(opp.price_decimals, 8)}f}", f"+{gain_3:.1f}%")
                        
                        with tcol4:
                            st.metric("Momentum", f"{opp.momentum_score:.0f}/100")
                        
                        st.divider()
                        
                        # Analysis
                        st.markdown("**📊 Runner Potential Analysis:**")
                        st.info(opp.reason)
                        
                        st.divider()
                        
                        # Action buttons
                        bcol1, bcol2, bcol3 = st.columns(3)
                        
                        with bcol1:
                            if st.button(f"⭐ Save to Watchlist", key=f"save_penny_{i}"):
                                try:
                                    with st.spinner(f"Saving {opp.symbol}..."):
                                        opp_data = {
                                            'symbol': opp.symbol,
                                            'current_price': opp.current_price,
                                            'change_pct_24h': opp.change_pct_24h,
                                            'volume_24h': opp.volume_24h,
                                            'volume_ratio': opp.volume_ratio,
                                            'volatility_24h': opp.volatility_24h,
                                            'rsi': opp.rsi,
                                            'momentum_score': opp.momentum_score,
                                            'score': opp.runner_potential_score,
                                            'confidence': opp.confidence,
                                            'risk_level': opp.risk_level,
                                            'strategy': 'penny_runner',
                                            'reason': opp.reason
                                        }
                                        
                                        # Check if crypto already exists in watchlist
                                        existing = crypto_wl_manager.get_crypto(opp.symbol)
                                        if existing:
                                            st.warning(f"⚠️ {opp.symbol} is already in your watchlist!")
                                        else:
                                            success = crypto_wl_manager.add_crypto(opp.symbol, opp_data)
                                            
                                            if success:
                                                st.success(f"✅ Added {opp.symbol} to watchlist!")
                                            else:
                                                st.error(f"❌ Failed to add {opp.symbol} to watchlist")
                                except Exception as e:
                                    st.error(f"❌ Error saving {opp.symbol}: {e}")
                                    logger.error("Error saving penny crypto to watchlist: {}", str(e), exc_info=True)
                                    # Preserve tab state even on error
                                    st.session_state.active_crypto_tab = "💰 Penny Cryptos (<$1)"
                        
                        with bcol2:
                            if st.button(f"📊 Generate Signal", key=f"gen_penny_signal_{i}"):
                                st.session_state.crypto_signal_symbol = opp.symbol
                                st.info(f"Navigate to Signal Generator tab to see {opp.symbol} signals!")
                        
                        with bcol3:
                            if st.button(f"⚡ Quick Trade", key=f"quick_penny_trade_{i}"):
                                st.session_state.crypto_quick_trade_pair = opp.symbol
                                st.info(f"Quick trade setup for {opp.symbol}")
            
            else:
                st.warning("No penny cryptos found matching your criteria. Try adjusting filters.")
        
        # Show scanner help
        with st.expander("❓ Penny Crypto Scanner Help", expanded=False):
            st.markdown("""
            ### 💰 Penny Crypto Scanner Guide
            
            **What are Penny Cryptos?**
            - Cryptocurrencies trading under $1.00
            - Often have extreme volatility and runner potential
            - Can move 100%+ in hours or days
            
            **Sub-Penny Cryptos (🔬)**
            - Cryptos under $0.01 with highest runner potential
            - Display with extreme precision (0.0000000+)
            - Highest risk/reward ratio
            
            **Monster Runner Potential Score**
            - Combines: momentum, volume, volatility, RSI, price action
            - 60-75: MEDIUM potential
            - 75-85: HIGH potential
            - 85+: EXTREME potential
            
            **Key Signals for Runners:**
            - 🚀 EXTREME 24h moves (>15%)
            - 💥 Volume surges (3x+ average)
            - ⚡ High volatility (>15%)
            - 🎯 Oversold RSI (<30)
            - 📊 Strong 7-day trends
            
            **Risk Management:**
            - ⚠️ Penny cryptos are HIGHLY VOLATILE
            - Use EXTREME CAUTION with leverage
            - Set tight stop losses
            - Position size accordingly
            - Monitor 24/7 (crypto never sleeps)
            
            **Target Strategy:**
            - **Target 1 (50%)**: Quick scalp profit
            - **Target 2 (100%)**: Swing trade target
            - **Target 3 (200%+)**: Monster runner target
            
            **Best Practices:**
            - Start with small positions
            - Take profits at targets
            - Don't hold through resistance
            - Watch volume for confirmation
            - Monitor social sentiment
            """)
    
    elif active_crypto_tab == "⭐ My Watchlist":
        st.subheader("⭐ My Crypto Watchlist")
        
        # Import and render watchlist UI
        try:
            from ui.crypto_watchlist_ui import render_crypto_watchlist_tab
            render_crypto_watchlist_tab(crypto_wl_manager, kraken_client, crypto_config)
        except Exception as e:
            st.error(f"Error loading watchlist: {e}")
            logger.error("Crypto watchlist UI error: {}", str(e), exc_info=True)
    
    elif active_crypto_tab == "🔥 OLD_CMC_REMOVED":  # REMOVED - Features available in Daily Scanner
        st.info("⚠️ CoinMarketCap features have been integrated into the Daily Scanner")
        st.markdown("Please use **🔍 Daily Scanner** → Select source: **🔥 CoinGecko Trending**")
    
    elif active_crypto_tab == "🔥 CoinMarketCap Features":
        st.info("⚠️ CoinMarketCap features have been integrated into the Daily Scanner")
        st.markdown("Please use **🔍 Daily Scanner** → Select source: **🔥 CoinGecko Trending**")
    
    elif active_crypto_tab == "🎯 OLD_SIGNAL_REMOVED":  # REMOVED - Use Daily Scanner Tier 3
        st.info("⚠️ Signal generation has been integrated into the Daily Scanner workflow")
        st.markdown("Please use **🔍 Daily Scanner** → Run Tier 1-3 analysis for signals")
    
    elif active_crypto_tab == "🎯 Signal Generator":
        st.info("⚠️ Signal generation has been integrated into the Daily Scanner workflow")
        st.markdown("Please use **🔍 Daily Scanner** → Run Tier 1-3 analysis for signals")
    
    elif active_crypto_tab == "⚡ Quick Trade":
        # Import and render quick trade UI with cached scanners
        try:
            from ui.crypto_quick_trade_ui import render_quick_trade_tab
            
            # Get cached scanner instances to avoid duplicates
            penny_scanner = get_penny_crypto_scanner(kraken_client, crypto_config)
            crypto_scanner = get_crypto_scanner(kraken_client, crypto_config)
            ai_scanner = get_ai_crypto_scanner(kraken_client, crypto_config)
            sub_penny = get_sub_penny_discovery()
            
            # Get watchlist manager if available
            watchlist_manager = None
            if 'crypto_watchlist_manager' in st.session_state:
                watchlist_manager = st.session_state.crypto_watchlist_manager
            
            # Pass cached instances to avoid re-initialization
            render_quick_trade_tab(
                kraken_client, 
                crypto_config,
                penny_crypto_scanner=penny_scanner,
                crypto_opportunity_scanner=crypto_scanner,
                ai_crypto_scanner=ai_scanner,
                sub_penny_discovery=sub_penny,
                watchlist_manager=watchlist_manager
            )
        except Exception as e:
            st.error(f"Error loading quick trade: {e}")
            logger.error("Crypto quick trade UI error: {}", str(e), exc_info=True)
            # Ensure we stay on Quick Trade tab even after error
            st.session_state.active_crypto_tab = "⚡ Quick Trade"
    
    elif active_crypto_tab == "🔔 Entry Monitors":
        st.subheader("🔔 Entry Monitors - Waiting for Optimal Entry Timing")
        
        # Initialize AI Entry Assistant in session state (loads saved monitors from JSON)
        if 'ai_entry_assistant' not in st.session_state:
            try:
                from services.ai_entry_assistant import get_ai_entry_assistant
                from services.llm_strategy_analyzer import LLMStrategyAnalyzer
                
                logger.info("🔧 Initializing AI Entry Assistant for Entry Monitors tab...")
                llm_analyzer = LLMStrategyAnalyzer()
                entry_assistant = get_ai_entry_assistant(
                    kraken_client=kraken_client,
                    llm_analyzer=llm_analyzer,
                    check_interval_seconds=60,
                    enable_auto_entry=False
                )
                st.session_state.ai_entry_assistant = entry_assistant
                
                # Start monitoring if not running
                if not entry_assistant.is_running:
                    entry_assistant.start_monitoring()
                
                num_monitors = len(entry_assistant.opportunities)
                logger.info(f"✅ AI Entry Assistant initialized with {num_monitors} saved monitors")
            except Exception as e:
                logger.error(f"Failed to initialize AI Entry Assistant: {e}", exc_info=True)
        else:
            logger.debug("AI Entry Assistant already in session state")
        
        # Create sub-tabs for Entry Monitors and Approvals
        monitor_subtab1, monitor_subtab2 = st.tabs([
            "⏳ Monitored Entries",
            "✅ Pending Approvals"
        ])
        
        with monitor_subtab1:
            # Import entry monitors UI
            try:
                from ui.crypto_entry_monitors_ui import display_entry_monitors
                display_entry_monitors()
            except Exception as e:
                st.error(f"Error loading Entry Monitors: {e}")
                logger.error("Entry Monitors error: {}", str(e), exc_info=True)
        
        with monitor_subtab2:
            # Import approval UI
            try:
                from ui.crypto_approval_ui import display_pending_approvals
                display_pending_approvals()
            except Exception as e:
                st.error(f"Error loading Pending Approvals: {e}")
                logger.error("Pending Approvals error: {}", str(e), exc_info=True)
    
    elif active_crypto_tab == "🤖 AI Position Monitor":
        st.subheader("🤖 AI Position Monitor - Intelligent Trade Management")
        
        # Import AI monitor UI
        try:
            from ui.crypto_ai_monitor_ui import display_ai_position_monitor
            display_ai_position_monitor()
        except Exception as e:
            st.error(f"Error loading AI Position Monitor: {e}")
            logger.error("AI Monitor error: {}", str(e), exc_info=True)
    
    elif active_crypto_tab == "📓 Trade Journal":
        st.subheader("📓 Trade Journal - Learn from Every Trade")
        
        # Import journal UI
        try:
            from ui.trade_journal_ui import display_trade_journal
            display_trade_journal()
        except Exception as e:
            st.error(f"Error loading Trade Journal: {e}")
            logger.error("Journal error: {}", str(e), exc_info=True)
    
    elif active_crypto_tab == "🎯 DEX Launch Hunter":
        # Import and render DEX Hunter tab
        try:
            from ui.tabs.dex_hunter_tab import render_dex_hunter_tab
            render_dex_hunter_tab()
        except Exception as e:
            st.error(f"Error loading DEX Launch Hunter: {e}")
            logger.error("DEX Hunter error: {}", str(e), exc_info=True)
            st.info("💡 DEX Launch Hunter requires additional configuration. See Resources tab for setup guide.")
    
    elif active_crypto_tab == "📈 OLD_PORTFOLIO_REMOVED":  # REMOVED - Merged into Dashboard
        st.info("⚠️ Portfolio view has been merged into the Dashboard tab")
        st.markdown("Please use **📊 Dashboard** to view your portfolio")
    
    elif active_crypto_tab == "📈 Portfolio & Settings":
        st.subheader("📈 Crypto Portfolio & Settings")
        
        # Portfolio display - show all coins/positions
        st.markdown("### 💰 Your Portfolio")
        
        try:
            balances = kraken_client.get_account_balance()
            total_usd = kraken_client.get_total_balance_usd()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Portfolio Value", f"${total_usd:,.2f}")
            
            # Find USD and crypto balances
            usd_balance = next((b for b in balances if b.currency in ['USD', 'ZUSD']), None)
            crypto_holdings = [b for b in balances if b.balance > 0 and b.currency not in ['USD', 'ZUSD']]
            
            with col2:
                if usd_balance:
                    st.metric("Available USD", f"${usd_balance.available:,.2f}")
                else:
                    st.metric("Available USD", "$0.00")
            
            with col3:
                st.metric("Crypto Assets", len(crypto_holdings))
            
            with col4:
                total_crypto_value = total_usd - (usd_balance.available if usd_balance else 0)
                st.metric("Crypto Value", f"${total_crypto_value:,.2f}")
            
            # Show all crypto holdings with detailed info
            if crypto_holdings:
                st.markdown("#### 📊 All Your Crypto Holdings")
                
                # Add refresh button
                refresh_col1, refresh_col2 = st.columns([1, 4])
                with refresh_col1:
                    if st.button("🔄 Refresh Portfolio", width='stretch'):
                        st.rerun()
                
                holdings_data = []
                total_crypto_value_calc = 0.0
                
                for balance in crypto_holdings:
                    try:
                        # Strip .F suffix (futures) and other suffixes from currency name
                        currency = balance.currency.replace('.F', '').replace('.S', '').replace('.M', '')
                        pair = f"{currency}/USD"
                        ticker = kraken_client.get_ticker_data(pair)
                        
                        if ticker:
                            value_usd = balance.balance * ticker['last_price']
                            total_crypto_value_calc += value_usd
                            
                            # Calculate 24h change
                            change_24h = ((ticker['last_price'] - ticker['open_24h']) / ticker['open_24h'] * 100) if ticker.get('open_24h') else 0
                            
                            holdings_data.append({
                                'Asset': balance.currency,
                                'Balance': balance.balance,
                                'Available': balance.available,
                                'On Hold': balance.hold,
                                'Current Price': ticker['last_price'],
                                'Value (USD)': value_usd,
                                '24h Change %': change_24h,
                                '24h High': ticker.get('high_24h', 0),
                                '24h Low': ticker.get('low_24h', 0),
                                '24h Volume': ticker.get('volume_24h', 0)
                            })
                        else:
                            # If ticker not found, still show the balance
                            holdings_data.append({
                                'Asset': balance.currency,
                                'Balance': balance.balance,
                                'Available': balance.available,
                                'On Hold': balance.hold,
                                'Current Price': 'N/A',
                                'Value (USD)': 'N/A',
                                '24h Change %': 'N/A',
                                '24h High': 'N/A',
                                '24h Low': 'N/A',
                                '24h Volume': 'N/A'
                            })
                    except Exception as e:
                        logger.debug(f"Error processing balance for {balance.currency}: {e}")
                        holdings_data.append({
                            'Asset': balance.currency,
                            'Balance': balance.balance,
                            'Available': balance.available,
                            'On Hold': balance.hold,
                            'Current Price': 'N/A',
                            'Value (USD)': 'N/A',
                            '24h Change %': 'N/A',
                            '24h High': 'N/A',
                            '24h Low': 'N/A',
                            '24h Volume': 'N/A'
                        })
                
                if holdings_data:
                    # Sort by value (highest first)
                    holdings_data_sorted = sorted(
                        [h for h in holdings_data if isinstance(h.get('Value (USD)'), (int, float))],
                        key=lambda x: x.get('Value (USD)', 0),
                        reverse=True
                    )
                    
                    # Add holdings without value at the end
                    holdings_data_sorted.extend(
                        [h for h in holdings_data if not isinstance(h.get('Value (USD)'), (int, float))]
                    )
                    
                    # Display as dataframe with formatting
                    import pandas as pd
                    df = pd.DataFrame(holdings_data_sorted)
                    
                    # Format numeric columns
                    if 'Balance' in df.columns:
                        df['Balance'] = df['Balance'].apply(lambda x: f"{x:.8f}" if isinstance(x, (int, float)) else x)
                    if 'Available' in df.columns:
                        df['Available'] = df['Available'].apply(lambda x: f"{x:.8f}" if isinstance(x, (int, float)) else x)
                    if 'On Hold' in df.columns:
                        df['On Hold'] = df['On Hold'].apply(lambda x: f"{x:.8f}" if isinstance(x, (int, float)) else x)
                    if 'Current Price' in df.columns:
                        df['Current Price'] = df['Current Price'].apply(lambda x: f"${x:,.6f}" if isinstance(x, (int, float)) else x)
                    if 'Value (USD)' in df.columns:
                        df['Value (USD)'] = df['Value (USD)'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
                    if '24h Change %' in df.columns:
                        df['24h Change %'] = df['24h Change %'].apply(lambda x: f"{x:+.2f}%" if isinstance(x, (int, float)) else x)
                    if '24h High' in df.columns:
                        df['24h High'] = df['24h High'].apply(lambda x: f"${x:,.6f}" if isinstance(x, (int, float)) else x)
                    if '24h Low' in df.columns:
                        df['24h Low'] = df['24h Low'].apply(lambda x: f"${x:,.6f}" if isinstance(x, (int, float)) else x)
                    if '24h Volume' in df.columns:
                        df['24h Volume'] = df['24h Volume'].apply(lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x)
                    
                    st.dataframe(df, width='stretch', hide_index=True)
                    
                    # Show summary
                    if total_crypto_value_calc > 0:
                        st.info(f"📊 **Portfolio Summary**: {len([h for h in holdings_data_sorted if isinstance(h.get('Value (USD)'), (int, float))])} assets with total value of ${total_crypto_value_calc:,.2f}")
            else:
                st.info("💡 You don't have any crypto holdings yet. Start trading to build your portfolio!")
                
        except Exception as e:
            st.error(f"❌ Error fetching portfolio data: {e}")
            logger.error("Portfolio fetch error: {}", str(e), exc_info=True)
            st.info("💡 Make sure your Kraken API credentials are configured correctly and have balance read permissions.")
        
        st.divider()
        
        # Configuration display
        st.markdown("### ⚙️ Current Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Capital Management**")
            st.text(f"Total Capital: ${crypto_config.TOTAL_CAPITAL:,.2f}")
            st.text(f"Reserve Cash: {crypto_config.RESERVE_CASH_PCT}%")
            st.text(f"Max Position Size: {crypto_config.MAX_POSITION_SIZE_PCT}%")
        
        with col2:
            st.markdown("**Risk Management**")
            st.text(f"Risk per Trade: {crypto_config.RISK_PER_TRADE_PCT * 100}%")
            st.text(f"Max Daily Loss: {crypto_config.MAX_DAILY_LOSS_PCT * 100}%")
            st.text(f"Max Daily Orders: {crypto_config.MAX_DAILY_ORDERS}")
        
        st.markdown("### 📚 Resources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📖 Documentation**
            - [Kraken Setup Guide](documentation/KRAKEN_SETUP_GUIDE.md)
            - [Crypto Trading Config](config_crypto_trading.py)
            - [Trading Strategies](#)
            """)
        
        with col2:
            st.markdown("""
            **🛠️ Tools**
            - Test Kraken Connection
            - View Trading History
            - Export Trade Data
            """)
        
        with col3:
            st.markdown("""
            **⚠️ Safety**
            - Always use stop losses
            - Start with paper trading
            - Never risk more than 2%
            - Only invest what you can lose
            """)
        
        st.markdown("### 🎯 Strategy Recommendations")
        
        st.info("""
        **For Beginners** (Low Risk):
        - Strategy: CRYPTO_SWING
        - Pairs: BTC/USD, ETH/USD only
        - Position Size: 8-10%
        - Hold Time: 1-3 days
        - Take Profit: 6-8%
        - Stop Loss: 3%
        
        **For Active Traders** (Medium Risk):
        - Strategy: CRYPTO_SCALPING
        - Pairs: Top 5-10 by volume
        - Position Size: 10-12%
        - Hold Time: 15-30 minutes
        - Take Profit: 2-3%
        - Stop Loss: 1-1.5%
        
        **For Experienced** (High Risk):
        - Strategy: CRYPTO_MOMENTUM
        - Pairs: Volatile altcoins
        - Position Size: 8-12%
        - Hold Time: Few hours
        - Take Profit: 8-12%
        - Stop Loss: 4-5%
        """)
        
        st.warning("""
        ⚠️ **CRYPTO RISK WARNING**:
        
        Cryptocurrency trading is HIGHLY VOLATILE and RISKY:
        - Prices can swing 10%+ in hours
        - 24/7 market means no "safe" closing bell
        - Flash crashes and pumps are common
        - Regulation is still evolving
        - Exchanges can have downtime
        - **NO PAPER TRADING MODE on Kraken**
        
        **Start with "learning capital" ($100-200) you can afford to lose completely!**
        
        **Recommended for testing:**
        - Allocate only $100-200 initially
        - Use $20-30 position sizes
        - Make 10-20 small test trades
        - Expect to lose some money while learning
        - Scale up ONLY after consistent profitability
        
        This is REAL money from day 1. Trade conservatively!
        """)

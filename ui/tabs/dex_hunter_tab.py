"""
DEX Launch Hunter UI Tab

User interface for catching early token launches and pumps.
Separate from main crypto trading to manage high-risk DEX speculation.
"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from models.dex_models import (
    TokenLaunch, LaunchAlert, HunterConfig, WatchedWallet,
    Chain, RiskLevel, LaunchStage
)
from services.dex_launch_hunter import DexLaunchHunter
from services.smart_money_tracker import SmartMoneyTracker
from services.token_safety_analyzer import TokenSafetyAnalyzer
from clients.dexscreener_client import DexScreenerClient


def render_dex_hunter_tab():
    """Main render function for DEX Hunter tab"""
    
    st.title("🎯 DEX Launch Hunter")
    st.markdown("""
    **Catch early token launches on DEXs before they pump.**
    
    ⚠️ **EXTREMELY HIGH RISK** - This is for early DEX speculation, not your main trading strategy.
    Use Trust Wallet + PancakeSwap/Uniswap for execution.
    """)
    
    # Initialize hunter in session state
    if 'dex_hunter' not in st.session_state:
        config = HunterConfig()
        st.session_state.dex_hunter = DexLaunchHunter(config)
        st.session_state.dex_hunter_stats = {}
    
    hunter: DexLaunchHunter = st.session_state.dex_hunter
    
    # Sub-navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍 Launch Scanner",
        "🐋 Smart Money",
        "⚙️ Configuration",
        "🚨 Alerts",
        "📊 Portfolio",
        "📚 Resources"
    ])
    
    with tab1:
        render_launch_scanner(hunter)
    
    with tab2:
        render_smart_money(hunter)
    
    with tab3:
        render_configuration(hunter)
    
    with tab4:
        render_alerts(hunter)
    
    with tab5:
        render_portfolio()
    
    with tab6:
        render_resources()


def render_launch_scanner(hunter: DexLaunchHunter):
    """Launch scanner interface"""
    
    st.header("🔍 Launch Scanner")
    
    st.info("✨ **Using FREE DexScreener API** - No authentication required! Scans latest token profiles and boosted tokens.")
    
    # Scanner controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Scanner", type="primary"):
            with st.spinner("Starting scanner..."):
                try:
                    # Run one scan cycle immediately to test
                    asyncio.run(hunter.start_monitoring(continuous=False))
                    st.session_state.scanner_running = True
                    st.success("✅ Scanner completed one cycle! Check stats below.")
                    st.info("💡 For continuous monitoring, run scanner in background or set up scheduled scans.")
                except Exception as e:
                    st.error(f"Scanner error: {e}")
                    logger.error(f"Scanner error: {e}", exc_info=True)
    
    with col2:
        if st.button("⏸️ Stop Scanner"):
            st.session_state.scanner_running = False
            hunter.stop_monitoring()
            st.success("Scanner stopped")
    
    with col3:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Scanner status
    stats = hunter.get_stats()
    st.session_state.dex_hunter_stats = stats
    
    st.markdown("---")
    
    # Stats row
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Scanned", stats.get('total_scanned', 0))
    
    with stat_col2:
        st.metric("Discovered", stats.get('total_discovered', 0))
    
    with stat_col3:
        st.metric("Total Alerts", stats.get('total_alerts', 0))
    
    with stat_col4:
        st.metric("Blacklisted", stats.get('blacklisted_tokens', 0))
    
    st.markdown("---")
    
    # Quick scan button
    if st.button("⚡ Quick Scan Now", help="Scan for new launches RIGHT NOW (one cycle)"):
        with st.spinner("Scanning DexScreener for new launches..."):
            try:
                # Run single scan cycle
                asyncio.run(hunter.start_monitoring(continuous=False))
                st.success(f"✅ Scan complete! Found {hunter.total_scanned} tokens.")
                st.rerun()
            except Exception as e:
                st.error(f"Scan error: {e}")
                logger.error(f"Scan error: {e}", exc_info=True)
    
    st.markdown("---")
    
    # Manual token analysis
    st.subheader("🔍 Analyze Specific Token")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        contract_address = st.text_input(
            "Contract Address",
            placeholder="0x... or Solana address",
            help="Paste the token contract address to analyze"
        )
    
    with col2:
        chain = st.selectbox(
            "Chain",
            options=["ethereum", "bsc", "solana", "base", "arbitrum"],
            index=0
        )
    
    if st.button("🔬 Analyze Token", key="analyze_btn"):
        if contract_address:
            with st.spinner("Analyzing token..."):
                try:
                    chain_enum = Chain.ETH  # Map string to enum
                    if chain == "bsc":
                        chain_enum = Chain.BSC
                    elif chain == "solana":
                        chain_enum = Chain.SOLANA
                    elif chain == "base":
                        chain_enum = Chain.BASE
                    elif chain == "arbitrum":
                        chain_enum = Chain.ARBITRUM
                    
                    success, token = asyncio.run(
                        hunter.analyze_token(contract_address, chain_enum)
                    )
                    
                    if success and token:
                        st.success("✅ Analysis complete!")
                        display_token_details(token, hunter)
                    else:
                        st.error("❌ Failed to analyze token")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Token analysis error: {e}", exc_info=True)
        else:
            st.warning("Please enter a contract address")
    
    st.markdown("---")
    
    # Top opportunities
    st.subheader("🔥 Top Opportunities")
    
    opportunities = hunter.get_top_opportunities(limit=20)
    
    if opportunities:
        for token in opportunities:
            display_token_card(token, hunter)
    else:
        st.info("No opportunities found yet. Start the scanner or analyze tokens manually.")


def display_token_card(token: TokenLaunch, hunter: DexLaunchHunter):
    """Display token as an expandable card"""
    
    # Risk color coding
    risk_colors = {
        RiskLevel.SAFE: "🟢",
        RiskLevel.LOW: "🟡",
        RiskLevel.MEDIUM: "🟠",
        RiskLevel.HIGH: "🔴",
        RiskLevel.EXTREME: "⛔"
    }
    
    risk_emoji = risk_colors.get(token.risk_level, "⚪")
    
    # Timing indicators
    timing_emoji = {
        "ULTRA_FRESH": "🔥",  # Super fresh!
        "FRESH": "⚡",  # Fresh
        "EARLY": "✨",  # Still early
        "LATE": "⏰",  # Getting late
        "MISSED_PUMP": "💤"  # Too late
    }.get(token.launch_timing, "⚪")
    
    # Breakout indicator for super fresh coins
    breakout_badge = ""
    if token.breakout_potential >= 70:
        breakout_badge = " 🚀BREAKOUT"
    elif token.breakout_potential >= 50:
        breakout_badge = " 🌟PRIME"
    
    # Missed pump warning
    pump_status = ""
    if token.missed_pump_likely:
        pump_status = " ⚠️DUMPED"
    elif token.time_to_pump == "PRIME":
        pump_status = " 💎PRIME"
    elif token.time_to_pump == "HEATING":
        pump_status = " 🔥PUMPING"
    
    # Create expander with enhanced title
    with st.expander(
        f"{risk_emoji}{timing_emoji} **{token.symbol}**{breakout_badge}{pump_status} - "
        f"Score: {token.composite_score:.0f}/100 - ${token.price_usd:.8f} - "
        f"Age: {token.minutes_since_launch:.0f}min"
    ):
        # 🚨 LAUNCH TIMING SECTION (NEW!)
        st.markdown("### ⏰ Launch Timing")
        timing_col1, timing_col2, timing_col3, timing_col4 = st.columns(4)
        
        with timing_col1:
            timing_color = "green" if token.launch_timing in ["ULTRA_FRESH", "FRESH"] else "orange" if token.launch_timing == "EARLY" else "red"
            st.markdown(f"**Status:** :{timing_color}[{token.launch_timing}]")
            st.caption(f"{token.minutes_since_launch:.0f} minutes old")
        
        with timing_col2:
            pump_color = "green" if token.time_to_pump == "PRIME" else "orange" if token.time_to_pump in ["HEATING", "COOLING"] else "red"
            st.markdown(f"**Pump Status:** :{pump_color}[{token.time_to_pump}]")
            if token.missed_pump_likely:
                st.error("⚠️ Likely missed pump!")
        
        with timing_col3:
            st.metric("Timing Score", f"{token.timing_advantage_score:.0f}/100")
            st.caption("How early you are")
        
        with timing_col4:
            if token.breakout_potential > 0:
                breakout_color = "green" if token.breakout_potential >= 70 else "orange" if token.breakout_potential >= 50 else "gray"
                st.markdown(f"**Breakout:** :{breakout_color}[{token.breakout_potential:.0f}/100]")
                if token.breakout_potential >= 70:
                    st.success("🚀 HIGH breakout potential!")
            else:
                st.caption("N/A (not fresh)")
        
        st.markdown("---")
        
        # Scores
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Composite Score", f"{token.composite_score:.1f}/100")
            st.metric("Pump Potential", f"{token.pump_potential_score:.1f}/100")
            st.metric("Velocity", f"{token.velocity_score:.1f}/100")
        
        with col2:
            st.metric("Price", f"${token.price_usd:.8f}")
            st.metric("1h Change", f"{token.price_change_1h:+.1f}%")
            st.metric("24h Change", f"{token.price_change_24h:+.1f}%")
        
        with col3:
            st.metric("Liquidity", f"${token.liquidity_usd:,.0f}")
            st.metric("Volume 24h", f"${token.volume_24h:,.0f}")
            st.metric("Age", f"{token.age_hours:.1f}h")
        
        st.markdown("---")
        
        # Safety info
        if token.contract_safety:
            safety = token.contract_safety
            
            st.markdown(f"**Safety Score:** {safety.safety_score:.0f}/100")
            
            safety_col1, safety_col2 = st.columns(2)
            
            with safety_col1:
                st.markdown(f"""
                - Honeypot: {'❌ YES' if safety.is_honeypot else '✅ NO'}
                - Buy Tax: {safety.buy_tax:.1f}%
                - Sell Tax: {safety.sell_tax:.1f}%
                - LP Locked: {'✅ YES' if safety.lp_locked else '❌ NO'}
                """)
            
            with safety_col2:
                st.markdown(f"""
                - Renounced: {'✅ YES' if safety.is_renounced else '❌ NO'}
                - Mintable: {'❌ YES' if safety.is_mintable else '✅ NO'}
                - Blacklist: {'❌ YES' if safety.has_blacklist else '✅ NO'}
                - Hidden Owner: {'❌ YES' if safety.hidden_owner else '✅ NO'}
                """)
        
        # 🔍 VERIFICATION CHECKLIST (NEW!)
        st.markdown("---")
        st.markdown("### ✅ Manual Verification Checklist")
        st.warning("⚠️ **HIGH RISK** - Always manually verify before buying!")
        
        # Create checklist from alert if available (otherwise generate new one)
        checklist = hunter._create_verification_checklist(token)
        
        check_col1, check_col2 = st.columns(2)
        
        with check_col1:
            st.markdown("**🔍 What to Check:**")
            st.checkbox("✅ Liquidity > $5k and growing", key=f"check_liq_{token.contract_address}")
            st.checkbox("✅ 50+ holders (not concentrated)", key=f"check_holders_{token.contract_address}")
            st.checkbox("✅ Contract verified on explorer", key=f"check_contract_{token.contract_address}")
            st.checkbox("✅ Social media presence (Telegram/Twitter)", key=f"check_socials_{token.contract_address}")
            st.checkbox("✅ Volume trending up (not down)", key=f"check_volume_{token.contract_address}")
            st.checkbox("✅ Recent dev activity (not abandoned)", key=f"check_dev_{token.contract_address}")
        
        with check_col2:
            st.markdown("**🔗 Research Links:**")
            if checklist.dexscreener_url:
                st.link_button("📊 DexScreener Chart", checklist.dexscreener_url)
            if checklist.etherscan_url:
                explorer_name = "Etherscan" if token.chain.value == "ethereum" else "BSCScan" if token.chain.value == "bsc" else "Solscan"
                st.link_button(f"🔍 {explorer_name} Contract", checklist.etherscan_url)
            if checklist.holders_url:
                st.link_button("👥 Holder Distribution", checklist.holders_url)
            if checklist.twitter_search_url:
                st.link_button("🐦 Twitter Mentions", checklist.twitter_search_url)
            if checklist.telegram_search_url:
                st.link_button("💬 Telegram Search", checklist.telegram_search_url)
        
        # User notes
        notes = st.text_area(
            "📝 Your Notes:",
            placeholder="Add your research notes here...",
            key=f"notes_{token.contract_address}",
            height=80
        )
        
        # Action buttons
        st.markdown("---")
        
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("📊 View Chart", key=f"chart_{token.contract_address}"):
                if token.pairs:
                    st.markdown(f"[Open DexScreener]({token.pairs[0].url})")
        
        with btn_col2:
            if st.button("🔍 More Info", key=f"info_{token.contract_address}"):
                st.json({
                    "contract": token.contract_address,
                    "chain": token.chain.value,
                    "dex": token.primary_dex
                })
        
        with btn_col3:
            if st.button("⭐ Track", key=f"track_{token.contract_address}"):
                token.is_tracked = True
                st.success("Added to tracking list")
        
        with btn_col4:
            if st.button("🚫 Blacklist", key=f"blacklist_{token.contract_address}"):
                hunter.blacklisted_tokens.add(token.contract_address.lower())
                st.warning("Token blacklisted")


def display_token_details(token: TokenLaunch, hunter: DexLaunchHunter):
    """Display full token details"""
    
    st.subheader(f"{token.symbol} - Full Analysis")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Price", f"${token.price_usd:.8f}")
    
    with col2:
        st.metric("Market Cap", f"${token.market_cap:,.0f}" if token.market_cap > 0 else "N/A")
    
    with col3:
        st.metric("Liquidity", f"${token.liquidity_usd:,.0f}")
    
    with col4:
        st.metric("Volume 24h", f"${token.volume_24h:,.0f}")
    
    # Scores
    st.markdown("### 📊 Scores")
    
    score_col1, score_col2, score_col3, score_col4 = st.columns(4)
    
    with score_col1:
        st.metric("Composite", f"{token.composite_score:.0f}/100")
    
    with score_col2:
        st.metric("Pump Potential", f"{token.pump_potential_score:.0f}/100")
    
    with score_col3:
        st.metric("Velocity", f"{token.velocity_score:.0f}/100")
    
    with score_col4:
        st.metric("Risk", token.risk_level.value)
    
    # Price action
    st.markdown("### 📈 Price Action")
    
    price_col1, price_col2, price_col3 = st.columns(3)
    
    with price_col1:
        st.metric("5m Change", f"{token.price_change_5m:+.1f}%")
    
    with price_col2:
        st.metric("1h Change", f"{token.price_change_1h:+.1f}%")
    
    with price_col3:
        st.metric("24h Change", f"{token.price_change_24h:+.1f}%")
    
    # Safety analysis
    if token.contract_safety:
        st.markdown("### 🔒 Safety Analysis")
        
        safety = token.contract_safety
        
        st.progress(safety.safety_score / 100, f"Safety Score: {safety.safety_score:.0f}/100")
        
        safety_col1, safety_col2, safety_col3 = st.columns(3)
        
        with safety_col1:
            st.markdown(f"""
            **Contract Checks:**
            - Honeypot: {'❌ YES' if safety.is_honeypot else '✅ NO'}
            - Renounced: {'✅ YES' if safety.is_renounced else '❌ NO'}
            - Mintable: {'❌ YES' if safety.is_mintable else '✅ NO'}
            - Proxy: {'⚠️ YES' if safety.is_proxy else '✅ NO'}
            """)
        
        with safety_col2:
            st.markdown(f"""
            **Taxes & Fees:**
            - Buy Tax: {safety.buy_tax:.1f}%
            - Sell Tax: {safety.sell_tax:.1f}%
            - Can Change Tax: {'❌ YES' if safety.owner_can_change_tax else '✅ NO'}
            """)
        
        with safety_col3:
            st.markdown(f"""
            **Liquidity:**
            - LP Locked: {'✅ YES' if safety.lp_locked else '❌ NO'}
            - Lock Duration: {safety.lp_lock_duration_days or 'N/A'} days
            - Hidden Owner: {'❌ YES' if safety.hidden_owner else '✅ NO'}
            - Blacklist: {'❌ YES' if safety.has_blacklist else '✅ NO'}
            """)


def render_smart_money(hunter: DexLaunchHunter):
    """Smart money tracking interface"""
    
    st.header("🐋 Smart Money Tracker")
    
    tracker: SmartMoneyTracker = hunter.smart_money_tracker
    
    # Add wallet form
    with st.expander("➕ Add Wallet to Track"):
        col1, col2 = st.columns(2)
        
        with col1:
            wallet_address = st.text_input("Wallet Address")
            wallet_name = st.text_input("Wallet Name/Label")
            wallet_desc = st.text_area("Description (optional)")
        
        with col2:
            wallet_chain = st.selectbox(
                "Chain",
                options=["ethereum", "bsc", "solana", "base"],
                index=0
            )
            wallet_tags = st.multiselect(
                "Tags",
                options=["whale", "dev", "influencer", "early_adopter", "profitable"],
                default=["whale"]
            )
            min_tx_usd = st.number_input("Min Transaction Size ($)", value=1000.0, step=100.0)
        
        if st.button("Add Wallet"):
            if wallet_address and wallet_name:
                chain_enum = Chain.ETH
                if wallet_chain == "bsc":
                    chain_enum = Chain.BSC
                elif wallet_chain == "solana":
                    chain_enum = Chain.SOLANA
                elif wallet_chain == "base":
                    chain_enum = Chain.BASE
                
                tracker.add_wallet(
                    address=wallet_address,
                    name=wallet_name,
                    description=wallet_desc,
                    chain=chain_enum,
                    tags=wallet_tags,
                    min_transaction_usd=min_tx_usd
                )
                
                st.success(f"✅ Added {wallet_name} to tracking")
            else:
                st.warning("Please fill in address and name")
    
    st.markdown("---")
    
    # List tracked wallets
    st.subheader("📋 Tracked Wallets")
    
    wallets = tracker.get_all_wallets()
    
    if wallets:
        for wallet in wallets:
            with st.expander(f"🐋 {wallet.name} - {wallet.address[:10]}..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Address:** `{wallet.address}`
                    **Chain:** {wallet.chain.value}
                    **Tags:** {', '.join(wallet.tags)}
                    **Min TX:** ${wallet.min_transaction_usd:,.0f}
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Success Rate:** {wallet.success_rate:.1f}%
                    **Avg Multiple:** {wallet.avg_multiple:.1f}x
                    **Total Trades:** {wallet.total_trades}
                    **Last Activity:** {wallet.last_activity or 'N/A'}
                    """)
                
                if st.button("🗑️ Remove", key=f"remove_{wallet.address}"):
                    tracker.remove_wallet(wallet.address)
                    st.rerun()
    else:
        st.info("No wallets tracked yet. Add some successful wallets above!")


def render_configuration(hunter: DexLaunchHunter):
    """Configuration interface"""
    
    st.header("⚙️ Configuration")
    
    config: HunterConfig = hunter.config
    
    # Chains
    st.subheader("🌐 Enabled Chains")
    
    enabled_chains = st.multiselect(
        "Select chains to monitor",
        options=["ethereum", "bsc", "solana", "base", "arbitrum", "polygon"],
        default=[c.value for c in config.enabled_chains],
        help="Monitor these networks for new launches"
    )
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_liq = st.number_input(
            "Min Liquidity ($)",
            value=config.min_liquidity_usd,
            step=1000.0,
            help="Minimum liquidity to consider"
        )
        
        max_buy_tax = st.slider(
            "Max Buy Tax (%)",
            0, 50, int(config.max_buy_tax),
            help="Skip tokens with higher buy tax"
        )
        
        require_lp_lock = st.checkbox(
            "Require LP Locked",
            value=config.require_lp_locked,
            help="Only show tokens with locked liquidity"
        )
    
    with col2:
        max_liq = st.number_input(
            "Max Liquidity ($)",
            value=config.max_liquidity_usd,
            step=10000.0,
            help="Maximum liquidity (target early launches)"
        )
        
        max_sell_tax = st.slider(
            "Max Sell Tax (%)",
            0, 50, int(config.max_sell_tax),
            help="Skip tokens with higher sell tax"
        )
        
        min_lp_lock_days = st.number_input(
            "Min LP Lock Days",
            value=float(config.min_lp_lock_days),
            step=30.0,
            help="Minimum LP lock duration"
        )
    
    # Risk tolerance
    st.subheader("⚠️ Risk Tolerance")
    
    risk_level = st.select_slider(
        "Maximum Risk Level",
        options=["SAFE", "LOW", "MEDIUM", "HIGH", "EXTREME"],
        value=config.max_risk_level.value,
        help="Don't alert on tokens riskier than this"
    )
    
    # Scoring
    st.subheader("📊 Scoring Thresholds")
    
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        min_pump = st.slider(
            "Min Pump Potential",
            0, 100, int(config.min_pump_potential),
            help="Minimum pump potential score"
        )
    
    with score_col2:
        min_composite = st.slider(
            "Min Composite Score",
            0, 100, int(config.min_composite_score),
            help="Minimum overall score"
        )
    
    # Monitoring
    st.subheader("🔔 Monitoring")
    
    scan_interval = st.number_input(
        "Scan Interval (seconds)",
        value=float(config.scan_interval_seconds),
        min_value=30.0,
        max_value=600.0,
        step=30.0,
        help="How often to scan for new launches"
    )
    
    enable_discord = st.checkbox(
        "Enable Discord Alerts",
        value=config.enable_discord_alerts,
        help="Send alerts to Discord webhook"
    )
    
    if st.button("💾 Save Configuration", type="primary"):
        # Update config
        config.min_liquidity_usd = min_liq
        config.max_liquidity_usd = max_liq
        config.max_buy_tax = max_buy_tax
        config.max_sell_tax = max_sell_tax
        config.require_lp_locked = require_lp_lock
        config.min_lp_lock_days = int(min_lp_lock_days)
        config.min_pump_potential = min_pump
        config.min_composite_score = min_composite
        config.scan_interval_seconds = int(scan_interval)
        config.enable_discord_alerts = enable_discord
        
        st.success("✅ Configuration saved!")


def render_alerts(hunter: DexLaunchHunter):
    """Alerts interface"""
    
    st.header("🚨 Recent Alerts")
    
    alerts = hunter.get_recent_alerts(limit=50)
    
    if alerts:
        for alert in reversed(alerts):  # Newest first
            priority_emoji = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "⚪"
            }.get(alert.priority, "⚪")
            
            with st.expander(
                f"{priority_emoji} {alert.token.symbol} - {alert.priority} - "
                f"{alert.timestamp.strftime('%H:%M:%S')}"
            ):
                st.markdown(f"**{alert.message}**")
                
                if alert.reasoning:
                    st.markdown("**Reasons:**")
                    for reason in alert.reasoning:
                        st.markdown(f"- {reason}")
                
                st.markdown("---")
                
                display_token_details(alert.token, hunter)
    else:
        st.info("No alerts yet. Start the scanner to begin receiving alerts!")


def render_portfolio():
    """Portfolio tracking interface"""
    
    st.header("📊 DEX Portfolio")
    
    st.info("Portfolio tracking coming soon! For now, manage your DEX positions manually in Trust Wallet.")
    
    st.markdown("""
    **Track Your DEX Trades:**
    - Record entry/exit prices
    - Calculate P&L
    - Monitor active positions
    - View trade history
    """)


def render_resources():
    """Resources and guides"""
    
    st.header("📚 Resources & Guides")
    
    st.markdown("""
    ### 🎯 Essential Tools
    
    **DEX Screeners:**
    - [DexScreener](https://dexscreener.com/new) - New pairs feed
    - [DexTools](https://www.dextools.io/) - Live new pairs
    - [Moonarch](https://moonarch.app/) - BSC tokens
    
    **Safety Checks:**
    - [TokenSniffer](https://tokensniffer.com/) - Contract audit
    - [Honeypot.is](https://honeypot.is/) - Honeypot check
    - [RugCheck](https://rugcheck.xyz/) - Rug detection
    - [GoPlus Security](https://gopluslabs.io/) - Multi-check
    
    **Wallets & Trading:**
    - [Trust Wallet](https://trustwallet.com/) - Mobile wallet
    - [MetaMask](https://metamask.io/) - Browser wallet
    - [PancakeSwap](https://pancakeswap.finance/) - BSC DEX
    - [Uniswap](https://app.uniswap.org/) - ETH DEX
    - [Raydium](https://raydium.io/) - Solana DEX
    
    **Smart Money:**
    - [Nansen](https://www.nansen.ai/) - Wallet tracking (paid)
    - [Arkham](https://platform.arkhamintelligence.com/) - On-chain intel
    - [DeBank](https://debank.com/) - Wallet portfolio
    
    **Community:**
    - [CryptoTwitter](https://twitter.com/search?q=%23crypto) - Early calls
    - [r/CryptoMoonShots](https://reddit.com/r/cryptomoonshots) - Reddit
    - Telegram alpha groups (be careful of scams!)
    
    ---
    
    ### ⚠️ Risk Warning
    
    **DEX launch hunting is EXTREMELY RISKY:**
    - Most new tokens are scams or rugs
    - Even "safe" tokens can dump 99%
    - Only risk what you can afford to lose 100%
    - Never FOMO into a pump
    - Always check contract safety first
    - Set stop losses
    - Take profits on the way up
    - Don't chase pumps
    
    **Recommended Risk Management:**
    - Start with small positions ($50-$200)
    - Never more than 5% of portfolio in one DEX token
    - Take initial investment out at 2x
    - Let rest ride with stop loss
    - Track every trade for learning
    
    ---
    
    ### 📖 How to Use This Tool
    
    1. **Configure your filters** - Set risk tolerance, liquidity range, tax limits
    2. **Add smart money wallets** - Track successful traders
    3. **Start the scanner** - It will monitor DexScreener and alert you
    4. **Analyze manually** - Paste contract addresses to check safety
    5. **Check alerts** - Review promising launches
    6. **Execute in Trust Wallet** - Use PancakeSwap/Uniswap to buy
    7. **Track performance** - Record all trades
    
    ---
    
    ### 🚀 Best Practices
    
    **For Catching Early Pumps:**
    - Monitor DexScreener "New Pairs" every 15-30 minutes
    - Join 2-3 good Telegram alpha groups
    - Follow early-caller Twitter accounts
    - Check contract safety BEFORE buying
    - Look for: LP locked, low tax, renounced ownership
    - Buy small, sell half at 2x, ride the rest
    
    **Red Flags (Avoid These):**
    - Honeypot = can't sell
    - Buy/Sell tax > 15%
    - No LP lock or < 30 days
    - Mintable token
    - Hidden owner
    - Blacklist function
    - Too good to be true promises
    
    **Green Flags (Look For These):**
    - LP locked 6+ months
    - Renounced ownership
    - Low buy/sell tax (< 5%)
    - Growing community
    - Smart money accumulating
    - Real utility or narrative
    - Transparent team
    """)


# Helper function to run async code in Streamlit
def run_async(coro):
    """Run async function in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
